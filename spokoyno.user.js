// ==UserScript==
// @name         Spokoyno — 2ch WebM Companion
// @namespace    local.spokoyno
// @version      5.6.0
// @description  Tab-local video cache, fastest mirror, speed monitor and event-based screamer warning
// @updateURL    https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js
// @downloadURL  https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js
// @match        https://2ch.org/*/res/*.html*
// @match        https://2ch.su/*/res/*.html*
// @match        https://2ch.life/*/res/*.html*
// @connect      2ch.org
// @connect      2ch.su
// @connect      2ch.life
// @run-at       document-start
// @noframes
// @grant        GM_xmlhttpRequest
// @grant        GM_getTab
// @grant        GM_saveTab
// @grant        GM_getTabs
// @grant        GM_registerMenuCommand
// @grant        GM_openInTab
// ==/UserScript==

(async () => {
  'use strict';

  const MIRRORS = ['2ch.org', '2ch.su', '2ch.life'];
  const THREAD_PATH_RE = /^\/[^/]+\/res\/\d+\.html\/?$/;
  // Keep the v4 cache/state identifiers so an upgrade does not discard an existing tab cache.
  const PREFIX = 'tm2ch-media-v4:',
    CONCURRENCY = 6,
    MIRROR_TTL = 60_000,
    PROBE_BYTES = 256 * 1024;
  const PROBE_TIMEOUT = 8000,
    DOWNLOAD_TIMEOUT = 120_000,
    CLEANUP_INTERVAL = 15 * 60_000,
    ORPHAN_GRACE = 60 * 60_000,
    CACHE_RECONCILE_INTERVAL = 30_000,
    PERSIST_RETRY_INTERVAL = 24 * 60 * 60_000,
    MAX_DOWNLOAD_ATTEMPTS = 4,
    RETRY_BASE_DELAY = 2_000,
    TAB_SAVE_INTERVAL = 15_000,
    RECENT_DOWNLOADS = 8;
  const CACHE_META_URL = `${location.origin}/__tm2ch_cache_meta_v1__`;
  const MEDIA_EXT = /\.(?:mp4|webm|m4v|mov|ogv)$/i;
  const SCREAMER_REPORT_RE = /scream|скрим/i;
  const ANALYSIS_VERSION = 6,
    ANALYSIS_WINDOW = 0.05,
    ANALYSIS_TARGET_RATE = 16_000;
  const BASELINE_WINDOWS = 60,
    BASELINE_GAP = 2,
    MIN_BASELINE_WINDOWS = 20,
    EVENT_WINDOWS = 6;
  const EVENT_LOOKAHEAD = 10,
    SCREAMER_CONFIDENCE = 0.8;

  if (!THREAD_PATH_RE.test(location.pathname)) return;

  if (!window.caches) {
    console.error('[spokoyno] CacheStorage unavailable');
    return;
  }

  const gmGetTab = () => new Promise((r) => GM_getTab((x) => r(x || {})));
  const gmSaveTab = (t) => new Promise((r) => GM_saveTab(t, r));
  const gmGetTabs = () => new Promise((r) => GM_getTabs((x) => r(x || {})));
  const yieldMain = () => (window.scheduler?.yield ? window.scheduler.yield() : new Promise((r) => setTimeout(r, 0)));

  const tab = await gmGetTab();

  if (!tab.tm2chMediaV4) {
    const token = crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    tab.tm2chMediaV4 = {
      token,
      cacheName: `${PREFIX}${token}`,
      mirror: null,
      mirrorCheckedAt: 0,
      screamer: {}
    };
    await gmSaveTab(tab);
  }

  const state = tab.tm2chMediaV4;
  state.screamer ||= {};
  state.cacheOrigins ||= {};
  const originChangedFrom = state.lastOrigin && state.lastOrigin !== location.origin ? state.lastOrigin : null;
  state.lastOrigin = location.origin;
  state.cacheOrigins[location.origin] = { cacheName: state.cacheName, lastSeenAt: Date.now() };
  await gmSaveTab(tab);

  if (originChangedFrom) {
    console.warn(
      `[spokoyno] page origin changed from ${originChangedFrom} to ${location.origin}; their physical CacheStorage data is separate`
    );
  }

  const cacheName = state.cacheName;
  let cache = await caches.open(cacheName),
    mirrorPromise = null,
    running = 0;
  let decoderContext = null;

  const cached = new Set(),
    seen = new Set(),
    seenMedia = new Map(),
    queued = new Set(),
    queue = [];
  const activeDownloads = new Map(),
    activeRequests = new Map(),
    activePreloads = new Set(),
    failedMedia = new Set(),
    recentDownloads = [],
    retryTimers = new Map();
  const analysisResults = new Map(
    Object.entries(state.screamer).filter(([, r]) => r?.analysisVersion === ANALYSIS_VERSION)
  );
  const analysisQueued = new Set(),
    analysisQueue = [];
  const screamerBadges = new Map(),
    communityBadges = new Map(),
    communityReports = new Map(),
    attachmentFigures = new Map();

  let analysisRunning = false,
    analysisGeneration = 0,
    downloadGeneration = 0;
  let uiRoot = null,
    summaryBox = null,
    speedBox = null,
    uiScheduled = false,
    communityScanScheduled = false,
    cacheReconcilePromise = null,
    lastCacheReconcileAt = 0,
    lastCacheTouchAt = 0,
    storagePersistence = 'unknown',
    lastTabSaveAt = Date.now(),
    tabSaveTimer = null,
    tabStateDirty = false,
    tabSaveChain = Promise.resolve(),
    cacheOperationChain = Promise.resolve(),
    domPruneScheduled = false;

  try {
    for (const r of await cache.keys()) if (r.url !== CACHE_META_URL) cached.add(r.url);
  } catch (e) {
    console.warn('[spokoyno] cache restore failed:', e);
  }

  const mediaPath = (url) => {
    const u = new URL(url, location.href);
    return u.pathname;
  };

  const requestPath = (url) => {
    const u = new URL(url, location.href);
    return u.pathname + u.search;
  };

  // This canonicalizes mirrors only inside the current page origin's CacheStorage.
  const canonicalKey = (url) => location.origin + '/__tm2ch_cache_v4__' + mediaPath(url);
  const screamerKey = (url) => mediaPath(url);
  const mirrorUrl = (domain, path) => `https://${domain}${path}`;

  function saveTabSoon(delay = 500) {
    tabStateDirty = true;
    if (tabSaveTimer) return;
    const wait = Math.max(delay, TAB_SAVE_INTERVAL - (Date.now() - lastTabSaveAt));
    tabSaveTimer = setTimeout(() => {
      tabSaveTimer = null;
      flushTabState();
    }, wait);
  }

  async function flushTabState() {
    if (tabSaveTimer) {
      clearTimeout(tabSaveTimer);
      tabSaveTimer = null;
    }
    if (!tabStateDirty) return tabSaveChain;
    tabStateDirty = false;
    tabSaveChain = tabSaveChain
      .catch(() => {})
      .then(() => gmSaveTab(tab))
      .then(() => {
        lastTabSaveAt = Date.now();
      })
      .catch((e) => {
        tabStateDirty = true;
        console.warn('[spokoyno] tab-state save failed:', e);
      });
    await tabSaveChain;
    if (tabStateDirty) saveTabSoon(1000);
  }

  function withCacheLock(operation) {
    const result = cacheOperationChain.catch(() => {}).then(operation);
    cacheOperationChain = result.catch(() => {});
    return result;
  }

  let removedStaleResults = false;
  for (const [key, result] of Object.entries(state.screamer)) {
    if (result?.analysisVersion === ANALYSIS_VERSION) continue;
    delete state.screamer[key];
    removedStaleResults = true;
  }
  if (removedStaleResults) saveTabSoon();

  const filenameFromUrl = (url) => {
    try {
      return decodeURIComponent(new URL(url, location.href).pathname.split('/').pop());
    } catch {
      return 'media';
    }
  };

  const mediaUrl = (a) => {
    try {
      const u = new URL(a.href, location.href);
      if (!MIRRORS.includes(u.hostname) || !u.pathname.includes('/src/') || !MEDIA_EXT.test(u.pathname)) return null;
      u.hash = '';
      return u.href;
    } catch {
      return null;
    }
  };

  const formatRate = (b) =>
    !Number.isFinite(b) || b <= 0
      ? '0 KB/s'
      : b >= 1073741824
        ? `${(b / 1073741824).toFixed(2)} GB/s`
        : b >= 1048576
          ? `${(b / 1048576).toFixed(1)} MB/s`
          : `${(b / 1024).toFixed(0)} KB/s`;

  const formatBytes = (b) =>
    !Number.isFinite(b) || b <= 0
      ? '0 B'
      : b >= 1073741824
        ? `${(b / 1073741824).toFixed(2)} GB`
        : b >= 1048576
          ? `${(b / 1048576).toFixed(1)} MB`
          : `${(b / 1024).toFixed(0)} KB`;

  const shortName = (n) => (n.length <= 25 ? n : n.slice(0, 11) + '…' + n.slice(-11));

  const formatTime = (s) => {
    const m = Math.floor(s / 60),
      x = s - m * 60;
    return m ? `${m}:${x.toFixed(1).padStart(4, '0')}` : `${x.toFixed(1)}s`;
  };

  const formatRiskPoints = (score) => {
    const points = Math.max(0, Math.min(1, Number.isFinite(score) ? score : 0)) * 100;
    if (points === 0) return '0';
    if (points < 0.01) return '<0.01';
    if (points < 1) return points.toFixed(2);
    if (points < 10) return points.toFixed(1);
    return points.toFixed(0);
  };

  async function readCacheMeta(target, name) {
    try {
      const response = await target.match(CACHE_META_URL);
      if (!response) return null;
      const meta = await response.json();
      return meta && typeof meta === 'object' ? meta : null;
    } catch (e) {
      console.warn('[spokoyno] cache metadata read failed:', name, e);
      return null;
    }
  }

  async function writeCacheMeta(target, meta) {
    const response = new Response(JSON.stringify(meta), {
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }
    });
    await target.put(CACHE_META_URL, response);
  }

  async function touchCurrentCache(force = false) {
    const now = Date.now();
    if (!force && now - lastCacheTouchAt < 60_000) return;
    const previous = await readCacheMeta(cache, cacheName);
    await writeCacheMeta(cache, {
      schema: 1,
      cacheName,
      token: state.token,
      origin: location.origin,
      createdAt: previous?.createdAt || now,
      lastSeenAt: now,
      orphanedAt: null
    });
    lastCacheTouchAt = now;
  }

  async function updateStoragePersistence(request = false, force = false) {
    const storage = navigator.storage;
    if (!storage?.persisted) {
      storagePersistence = 'unavailable';
      return false;
    }
    try {
      let persistent = await storage.persisted();
      if (!persistent && request && storage.persist) {
        let shouldRequest = force;
        try {
          const key = 'spokoyno-persist-attempt-v1';
          const lastAttempt = Number(localStorage.getItem(key)) || 0;
          shouldRequest ||= Date.now() - lastAttempt >= PERSIST_RETRY_INTERVAL;
        } catch {
          shouldRequest = true;
        }
        if (shouldRequest) {
          persistent = await storage.persist();
          try {
            localStorage.setItem('spokoyno-persist-attempt-v1', String(Date.now()));
          } catch {}
        }
      }
      storagePersistence = persistent ? 'persistent' : 'best-effort';
      return persistent;
    } catch (e) {
      storagePersistence = 'error';
      console.warn('[spokoyno] persistent storage check failed:', e);
      return false;
    }
  }

  function enqueueMedia(url, key, front = false, attempt = 0) {
    if (cached.has(key) || queued.has(key)) return false;
    const sk = screamerKey(url),
      previous = analysisResults.get(sk);
    if (previous?.status === 'media-error' || previous?.status === 'cache-error') {
      analysisResults.delete(sk);
      delete state.screamer[sk];
      failedMedia.delete(key);
      for (const badge of screamerBadges.get(sk) || []) if (badge.isConnected) renderScreamerResult(badge, null);
      saveTabSoon();
    }
    queued.add(key);
    const item = { url, key, attempt, generation: downloadGeneration };
    if (front) queue.unshift(item);
    else queue.push(item);
    return true;
  }

  function cancelDownloadRetries() {
    downloadGeneration++;
    for (const timer of retryTimers.values()) clearTimeout(timer);
    retryTimers.clear();
    for (const key of activeRequests.keys()) abortDownloadsForKey(key);
  }

  function trackDownloadRequest(key, request) {
    if (!activeRequests.has(key)) activeRequests.set(key, new Set());
    activeRequests.get(key).add(request);
  }

  function untrackDownloadRequest(key, request) {
    const requests = activeRequests.get(key);
    if (!requests) return;
    requests.delete(request);
    if (!requests.size) activeRequests.delete(key);
  }

  function abortDownloadsForKey(key) {
    const requests = activeRequests.get(key);
    if (!requests) return;
    activeRequests.delete(key);
    for (const request of requests) {
      try {
        request.abort();
      } catch {}
    }
    cancelSpeedTracking(key);
  }

  async function reconcileCache(reason = 'manual', force = false) {
    if (cacheReconcilePromise) return cacheReconcilePromise;
    const now = Date.now();
    if (!force && now - lastCacheReconcileAt < CACHE_RECONCILE_INTERVAL) return null;
    lastCacheReconcileAt = now;
    const generation = downloadGeneration;
    cacheReconcilePromise = withCacheLock(async () => {
      const names = await caches.keys();
      const existed = names.includes(cacheName);
      cache = await caches.open(cacheName);
      const requests = await cache.keys();
      const actual = new Set(requests.map((r) => r.url).filter((url) => url !== CACHE_META_URL));
      const missingSinceLastCheck = [...cached].filter((key) => !actual.has(key)).length;
      cached.clear();
      for (const key of actual) cached.add(key);
      await touchCurrentCache(true);

      let requeued = 0;
      if (generation === downloadGeneration) {
        for (const [key, url] of seenMedia) if (!actual.has(key) && enqueueMedia(url, key)) requeued++;
        scheduleUI();
        pump();
      }
      if (!existed || missingSinceLastCheck || requeued) {
        console.warn('[spokoyno] cache reconciled:', {
          reason,
          cacheRecreated: !existed,
          missingSinceLastCheck,
          requeued,
          entries: actual.size,
          origin: location.origin
        });
      }
      return { existed, entries: actual.size, missingSinceLastCheck, requeued };
    })
      .catch((e) => {
        console.warn('[spokoyno] cache reconciliation failed:', reason, e);
        return null;
      })
      .finally(() => {
        cacheReconcilePromise = null;
      });
    return cacheReconcilePromise;
  }

  async function getCacheDiagnostics() {
    const names = await caches.keys();
    const exists = names.includes(cacheName);
    let entries = 0;
    if (exists) {
      const target = await caches.open(cacheName);
      entries = (await target.keys()).filter((r) => r.url !== CACHE_META_URL).length;
    }
    await updateStoragePersistence(false);
    let estimate = {};
    try {
      estimate = (await navigator.storage?.estimate?.()) || {};
    } catch (e) {
      console.warn('[spokoyno] storage estimate failed:', e);
    }
    return {
      origin: location.origin,
      previousOrigin: originChangedFrom,
      cacheName,
      exists,
      entries,
      indexedEntries: cached.size,
      seenEntries: seen.size,
      persistence: storagePersistence,
      usage: estimate.usage,
      quota: estimate.quota,
      queued: queued.size,
      active: running,
      knownOrigins: Object.keys(state.cacheOrigins)
    };
  }

  const cacheDiagnosticText = (d) =>
    [
      `Origin: ${d.origin}`,
      d.previousOrigin ? `Previous origin in this tab: ${d.previousOrigin}` : '',
      `Cache: ${d.cacheName}`,
      `Physical cache exists: ${d.exists ? 'yes' : 'NO'}`,
      `Stored media entries: ${d.entries}`,
      `In-memory index: ${d.indexedEntries}`,
      `Media found on page: ${d.seenEntries}`,
      `Storage mode: ${d.persistence}`,
      `Origin usage / quota: ${Number.isFinite(d.usage) ? formatBytes(d.usage) : 'unknown'} / ${Number.isFinite(d.quota) ? formatBytes(d.quota) : 'unknown'}`,
      `Downloads queued / active: ${d.queued} / ${d.active}`,
      `Origins seen by this tab: ${d.knownOrigins.join(', ')}`
    ]
      .filter(Boolean)
      .join('\n');

  function probeMirror(domain, path) {
    return new Promise((resolve) => {
      const started = performance.now();
      let done = false,
        lastLoaded = 0,
        req;
      const finish = (x) => {
        if (!done) {
          done = true;
          resolve(x);
        }
      };

      try {
        req = GM_xmlhttpRequest({
          method: 'GET',
          url: mirrorUrl(domain, path),
          headers: { Range: `bytes=0-${PROBE_BYTES - 1}` },
          responseType: 'arraybuffer',
          timeout: PROBE_TIMEOUT,
          onprogress: (e) => {
            if (done) return;
            lastLoaded = e.loaded || 0;
            if (lastLoaded >= PROBE_BYTES) {
              const elapsed = performance.now() - started;
              finish({
                domain,
                ok: true,
                elapsed,
                bytes: lastLoaded,
                score: elapsed / lastLoaded
              });
              try {
                req.abort();
              } catch {}
            }
          },
          onload: (r) => {
            if (done) return;
            if (r.status < 200 || r.status >= 400) return finish({ domain, ok: false, score: Infinity });
            const bytes = r.response?.byteLength || lastLoaded || 1,
              elapsed = performance.now() - started;
            finish({
              domain,
              ok: true,
              elapsed,
              bytes,
              score: elapsed / bytes
            });
          },
          onerror: () => finish({ domain, ok: false, score: Infinity }),
          ontimeout: () => finish({ domain, ok: false, score: Infinity })
        });
      } catch {
        finish({ domain, ok: false, score: Infinity });
      }
    });
  }

  async function benchmarkMirrors(sampleUrl) {
    const results = await Promise.all(MIRRORS.map((d) => probeMirror(d, requestPath(sampleUrl))));
    const ok = results.filter((x) => x.ok).sort((a, b) => a.score - b.score);
    const winner = ok[0]?.domain || (MIRRORS.includes(location.hostname) ? location.hostname : MIRRORS[0]);
    state.mirror = winner;
    state.mirrorCheckedAt = Date.now();
    saveTabSoon();
    await flushTabState();
    console.table(
      results.map((x) => ({
        mirror: x.domain,
        ok: x.ok,
        speed: x.ok ? formatRate(1000 / x.score) : '-'
      }))
    );
    scheduleUI();
    return winner;
  }

  async function getPreferredMirror(url) {
    if (state.mirror && MIRRORS.includes(state.mirror) && Date.now() - state.mirrorCheckedAt < MIRROR_TTL)
      return state.mirror;
    if (!mirrorPromise) mirrorPromise = benchmarkMirrors(url).finally(() => (mirrorPromise = null));
    return mirrorPromise;
  }

  function beginSpeedTracking(key, url, domain) {
    const now = performance.now();
    const item = {
      key,
      name: filenameFromUrl(url),
      domain,
      loaded: 0,
      total: 0,
      bps: 0,
      started: now,
      lastBytes: 0,
      lastTime: now
    };
    activeDownloads.set(key, item);
    scheduleUI();
    return item;
  }

  function updateSpeedTracking(item, e) {
    const now = performance.now(),
      loaded = e.loaded || 0;
    item.loaded = loaded;
    if (e.lengthComputable && e.total) item.total = e.total;
    const dt = now - item.lastTime;
    if (dt >= 150) {
      const instant = (loaded - item.lastBytes) / (dt / 1000);
      if (Number.isFinite(instant) && instant >= 0)
        item.bps = item.bps <= 0 ? instant : item.bps * 0.65 + instant * 0.35;
      item.lastBytes = loaded;
      item.lastTime = now;
    }
    scheduleUI();
  }

  function finishSpeedTracking(key, bytes, domain) {
    const x = activeDownloads.get(key);
    if (!x) return;
    const duration = (performance.now() - x.started) / 1000,
      finalBytes = bytes || x.loaded;
    recentDownloads.unshift({
      name: x.name,
      domain,
      bytes: finalBytes,
      avgBps: duration > 0 ? finalBytes / duration : 0,
      duration
    });
    recentDownloads.length = Math.min(recentDownloads.length, RECENT_DOWNLOADS);
    activeDownloads.delete(key);
    scheduleUI();
  }

  function cancelSpeedTracking(key) {
    activeDownloads.delete(key);
    scheduleUI();
  }

  const parseContentType = (h) => h?.match(/^content-type:\s*([^\r\n]+)/im)?.[1]?.trim() || '';

  async function identifyMediaBlob(blob, contentType, path) {
    if (!blob || blob.size < 16) throw new Error('empty or truncated media response');
    const declared = String(contentType || '').toLowerCase();
    if (/\b(?:text\/html|application\/(?:json|xml)|text\/plain)\b/.test(declared)) {
      throw new Error(`unexpected content type ${contentType}`);
    }
    const bytes = new Uint8Array(await blob.slice(0, 64).arrayBuffer());
    const isWebm = bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3;
    const firstBox = String.fromCharCode(bytes[4] || 0, bytes[5] || 0, bytes[6] || 0, bytes[7] || 0);
    const isMp4 = ['ftyp', 'moov', 'mdat', 'wide', 'free', 'skip'].includes(firstBox);
    const isOgg = bytes[0] === 0x4f && bytes[1] === 0x67 && bytes[2] === 0x67 && bytes[3] === 0x53;
    const extension = new URL(path, location.origin).pathname.split('.').pop()?.toLowerCase();
    const valid = extension === 'webm' ? isWebm : extension === 'ogv' ? isOgg : isMp4;
    if (!valid) throw new Error(`response is not a valid ${extension || 'video'} container`);
    return isWebm ? 'video/webm' : isOgg ? 'video/ogg' : 'video/mp4';
  }

  function downloadBlob(domain, path, key, originalUrl, generation) {
    return new Promise((resolve, reject) => {
      const speed = beginSpeedTracking(key, originalUrl, domain);
      let request,
        settled = false;
      const fail = (error) => {
        if (settled) return;
        settled = true;
        if (request) untrackDownloadRequest(key, request);
        cancelSpeedTracking(key);
        reject(error);
      };
      const succeed = (value) => {
        if (settled) return;
        settled = true;
        if (request) untrackDownloadRequest(key, request);
        resolve(value);
      };
      try {
        request = GM_xmlhttpRequest({
          method: 'GET',
          url: mirrorUrl(domain, path),
          responseType: 'blob',
          timeout: DOWNLOAD_TIMEOUT,
          onprogress: (e) => updateSpeedTracking(speed, e),
          onload: async (r) => {
            if (r.status !== 200) return fail(new Error(`${domain}: HTTP ${r.status}`));
            const blob = r.response,
              declaredType = parseContentType(r.responseHeaders) || blob?.type,
              declaredLength = Number(r.responseHeaders?.match(/^content-length:\s*(\d+)/im)?.[1]) || 0;
            try {
              if (declaredLength && blob?.size !== declaredLength) {
                throw new Error(`truncated response (${blob?.size || 0}/${declaredLength} bytes)`);
              }
              const contentType = await identifyMediaBlob(blob, declaredType, path);
              if (generation !== downloadGeneration) return fail(new Error(`${domain}: download cancelled`));
              finishSpeedTracking(key, blob.size, domain);
              succeed({ blob, contentType, domain });
            } catch (e) {
              fail(new Error(`${domain}: ${e.message || e}`));
            }
          },
          onerror: () => fail(new Error(`${domain}: network error`)),
          ontimeout: () => fail(new Error(`${domain}: timeout`)),
          onabort: () => fail(new Error(`${domain}: download cancelled`))
        });
        trackDownloadRequest(key, request);
        if (settled) untrackDownloadRequest(key, request);
      } catch (e) {
        fail(e);
      }
    });
  }

  async function downloadFromBestMirror(url, key, generation) {
    const path = requestPath(url),
      preferred = await getPreferredMirror(url);
    const order = [preferred, ...MIRRORS.filter((x) => x !== preferred)];
    let lastError;
    for (const domain of order) {
      if (generation !== downloadGeneration || !seenMedia.has(key)) throw new Error('download cancelled');
      try {
        const r = await downloadBlob(domain, path, key, url, generation);
        if (domain !== preferred) {
          state.mirror = domain;
          state.mirrorCheckedAt = 0;
          saveTabSoon();
        }
        return r;
      } catch (e) {
        lastError = e;
        console.warn('[spokoyno] mirror failed:', domain, path, e);
      }
    }
    throw lastError || new Error('all mirrors failed');
  }

  function installPageStyles() {
    if (document.getElementById('tm2ch-extra-style')) return;
    const s = document.createElement('style');
    s.id = 'tm2ch-extra-style';
    s.textContent = `
    .tm2ch-screamer{display:inline-block;margin-left:6px;padding:1px 5px;border-radius:4px;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;vertical-align:middle;background:rgba(0,0,0,.10);cursor:default;user-select:none}
    .tm2ch-screamer[data-state="bad"]{font-weight:700;background:rgba(220,35,35,.22);color:#c22}
    .tm2ch-screamer[data-state="ok"]{background:rgba(40,150,60,.12)}
    .tm2ch-screamer[data-state="error"]{opacity:.65}
    .tm2ch-report{text-decoration:none!important}
    .tm2ch-report[data-state="bad"]{background:rgba(220,35,35,.30);color:#c22}
    .post__image.tm2ch-risk .post__file-preview,
    .post__image.tm2ch-risk video{outline:3px solid rgba(210,30,30,.88)!important;box-shadow:0 0 12px rgba(220,25,25,.58)!important}
  `;
    document.documentElement.append(s);
  }

  function renderScreamerResult(badge, r) {
    const expose = () => badge.setAttribute('aria-label', `${badge.textContent}. ${badge.title}`);
    badge.dataset.state = '';
    if (!r) {
      badge.textContent = '🔊 analyzing…';
      badge.title = 'Audio analysis is queued';
      expose();
      return;
    }
    if (r.status === 'no-audio') {
      badge.textContent = '🔇 no audio';
      badge.title = 'No audio track was detected';
      badge.dataset.state = 'ok';
      expose();
      return;
    }
    if (r.status === 'unsupported') {
      badge.textContent = '❔ audio analysis unavailable';
      badge.title = 'Web Audio decoding is unavailable in this browser';
      badge.dataset.state = 'error';
      expose();
      return;
    }
    if (r.status === 'decode-error') {
      badge.textContent = '❔ audio analysis failed';
      badge.title =
        'The browser could play the video container but could not expose its complete audio track for offline analysis. This is unknown, not safe.';
      badge.dataset.state = 'error';
      expose();
      return;
    }
    if (r.status === 'media-error' || r.status === 'cache-error') {
      badge.textContent = r.status === 'cache-error' ? '❔ cache failed' : '❔ media unavailable';
      badge.title = `${r.message || 'The media could not be prepared for analysis'}. This is unknown, not safe.`;
      badge.dataset.state = 'error';
      expose();
      return;
    }

    const displayRisk = Number.isFinite(r.displayRisk)
      ? r.displayRisk
      : Math.max(r.transitionConfidence ?? 0, r.startConfidence ?? 0, r.rescueConfidence ?? 0, r.confidence ?? 0);
    const risk = formatRiskPoints(displayRisk);
    const detectorName =
      {
        'loud-start': 'dangerous loud start',
        'short-spectral-burst': 'short edited spectral burst',
        'short-clipped-burst': 'short heavily clipped burst',
        transition: 'quiet-to-loud transition'
      }[r.detectionMode] || 'quiet-to-loud transition';
    if (r.suspicious) {
      badge.textContent =
        r.detectionMode === 'loud-start'
          ? `⚠ SCREAMER · risk ${risk}/100 · loud start @ ${formatTime(r.eventAt || 0)}`
          : `⚠ SCREAMER · risk ${risk}/100 · +${r.jumpDb.toFixed(0)} dB @ ${formatTime(r.eventAt || 0)}`;
      badge.dataset.state = 'bad';
    } else {
      badge.textContent = `🔊 normal · risk ${risk}/100`;
      badge.dataset.state = 'ok';
    }

    badge.title = [
      `Risk score: ${risk}/100 (heuristic, not a probability)`,
      `Merged decision score: ${formatRiskPoints(r.decisionScore ?? 0)}/100`,
      `Strongest detector: ${detectorName}`,
      `Event: ${formatTime(r.eventAt || 0)}`,
      Number.isFinite(r.baselineDb)
        ? `Baseline (previous 1–3 s): ${r.baselineDb.toFixed(1)} dB`
        : 'Baseline: unavailable near start',
      `Sustained event level: ${r.eventDb.toFixed(1)} dB`,
      Number.isFinite(r.jumpDb) ? `Jump: +${r.jumpDb.toFixed(1)} dB` : '',
      Number.isFinite(r.baselineMadDb) ? `Baseline MAD: ${r.baselineMadDb.toFixed(1)} dB` : '',
      Number.isFinite(r.robustZ) ? `Robust normalized jump: ${r.robustZ.toFixed(2)}` : '',
      Number.isFinite(r.attackMs)
        ? `10 ms attack estimate: ${r.attackMs.toFixed(0)} ms`
        : '10 ms attack estimate: unavailable',
      `Event duration: ${r.eventDuration.toFixed(2)} s`,
      `Event peak: ${r.eventPeakDb.toFixed(1)} dBFS`,
      `Event near-clipping: ${r.eventNearClipPct.toFixed(2)}%`,
      `Onset spectral flux: ${r.spectralFlux.toFixed(3)}`,
      Number.isFinite(r.spectralFluxNear) ? `Nearby maximum spectral flux: ${r.spectralFluxNear.toFixed(3)}` : '',
      Number.isFinite(r.spectralShapeDistance)
        ? `Baseline → event spectral distance: ${r.spectralShapeDistance.toFixed(3)}`
        : '',
      `Transition path: ${formatRiskPoints(r.transitionConfidence)}/100`,
      `Loud-start path: ${formatRiskPoints(r.startConfidence)}/100`,
      Number.isFinite(r.rescueConfidence) ? `Short-burst rescue path: ${formatRiskPoints(r.rescueConfidence)}/100` : '',
      `Start spectral flatness: ${r.startSpectralFlatness.toFixed(3)}`,
      `Start high-frequency brightness: ${r.startBrightnessDb.toFixed(1)} dB`,
      `Track median: ${r.medianDb.toFixed(1)} dB`,
      `Track peak: ${r.peakDb.toFixed(1)} dBFS`,
      r.reason
    ]
      .filter(Boolean)
      .join('\n');
    expose();
  }

  function refreshAttachmentRisk(sk) {
    const modelRisk = analysisResults.get(sk)?.suspicious === true;
    const reportedRisk = communityReports.has(sk);
    for (const figure of attachmentFigures.get(sk) || []) {
      if (!figure.isConnected) continue;
      figure.classList.toggle('tm2ch-risk', modelRisk || reportedRisk);
      figure.dataset.tm2chModelRisk = String(modelRisk);
      figure.dataset.tm2chReportedRisk = String(reportedRisk);
    }
  }

  function rememberAttachment(anchor, url) {
    const figure = anchor.closest?.('figure.post__image');
    if (!figure) return;
    const sk = screamerKey(url);
    if (!attachmentFigures.has(sk)) attachmentFigures.set(sk, new Set());
    attachmentFigures.get(sk).add(figure);
    refreshAttachmentRisk(sk);
  }

  function addMappedElement(map, key, element) {
    if (!map.has(key)) map.set(key, new Set());
    map.get(key).add(element);
  }

  function renderCommunityReport(sk, reports) {
    const count = reports.length;
    const label = `🚩 reported${count > 1 ? ` ×${count}` : ''}`;
    const title = [
      `Unverified community screamer report${count === 1 ? '' : 's'}: ${count}`,
      ...reports.map((r) => `Reply #${r.reportPost} → post #${r.targetPost}: “${r.text}”`),
      'Independent of the audio-analysis percentage; click to open the first reporting reply.'
    ].join('\n');
    for (const modelBadge of screamerBadges.get(sk) || []) {
      if (!modelBadge.isConnected) continue;
      let badge = modelBadge._tm2chCommunityBadge;
      if (!badge?.isConnected) {
        badge = document.createElement('a');
        badge.className = 'tm2ch-screamer tm2ch-report';
        badge.dataset.state = 'bad';
        modelBadge.insertAdjacentElement('afterend', badge);
        modelBadge._tm2chCommunityBadge = badge;
        addMappedElement(communityBadges, sk, badge);
      }
      if (badge.textContent !== label) badge.textContent = label;
      badge.href = `#${reports[0].reportPost}`;
      badge.setAttribute('aria-label', `${count} unverified community screamer report${count === 1 ? '' : 's'}`);
      badge.title = title;
    }
  }

  function messageReportTargets(message) {
    let text = '';
    const replies = [];
    const walk = (node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        text += node.nodeValue || '';
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      if (node.matches('.post-reply-link[data-num]')) {
        const start = text.length;
        text += node.textContent || '';
        replies.push({ targetPost: node.dataset.num, start, end: text.length });
        return;
      }
      if (node.matches('br')) text += '\n';
      for (const child of node.childNodes) walk(child);
    };
    walk(message);
    const targets = new Set(),
      keyword = /scream\w*|скрим\w*/giu;
    for (const match of text.matchAll(keyword)) {
      const before = text.slice(Math.max(0, match.index - 32), match.index).toLowerCase();
      if (/(?:\b(?:not|no)\b|(?:^|\s)не|(?:^|\s)без)\s+(?:\S+\s+){0,2}$/.test(before)) continue;
      const at = match.index;
      const preceding = replies.filter((r) => r.end <= at).sort((a, b) => b.end - a.end)[0];
      const following = replies.filter((r) => r.start > at).sort((a, b) => a.start - b.start)[0];
      const chosen =
        preceding && at - preceding.end <= 200 ? preceding : following && following.start - at <= 80 ? following : null;
      if (chosen && /^\d+$/.test(chosen.targetPost)) targets.add(chosen.targetPost);
    }
    return { text, targets };
  }

  function scanCommunityReports() {
    const next = new Map();
    for (const post of document.querySelectorAll('.post[data-num]')) {
      const message = post.querySelector('.post__message');
      if (!message || !SCREAMER_REPORT_RE.test(message.textContent || '')) continue;
      const reportPost = post.dataset.num,
        parsed = messageReportTargets(message),
        text = parsed.text.replace(/\s+/g, ' ').trim().slice(0, 180);
      for (const targetPost of parsed.targets) {
        const target = document.getElementById(`post-${targetPost}`);
        if (!target) continue;
        const found = new Set();
        for (const anchor of target.querySelectorAll('.post__images a[href]')) {
          const url = mediaUrl(anchor);
          if (!url) continue;
          const sk = screamerKey(url);
          if (found.has(sk)) continue;
          found.add(sk);
          if (!next.has(sk)) next.set(sk, []);
          const reports = next.get(sk);
          if (!reports.some((r) => r.reportPost === reportPost)) reports.push({ reportPost, targetPost, text });
        }
      }
    }

    for (const [sk, badges] of communityBadges) {
      if (next.has(sk)) continue;
      for (const badge of badges) badge.remove();
      communityBadges.delete(sk);
    }
    communityReports.clear();
    for (const [sk, reports] of next) {
      communityReports.set(sk, reports);
      renderCommunityReport(sk, reports);
    }
    for (const sk of attachmentFigures.keys()) refreshAttachmentRisk(sk);
  }

  function scheduleCommunityScan() {
    if (communityScanScheduled) return;
    communityScanScheduled = true;
    setTimeout(() => {
      communityScanScheduled = false;
      scanCommunityReports();
    }, 50);
  }

  function attachScreamerBadge(anchor, url) {
    const sk = screamerKey(url),
      figure = anchor.closest?.('figure.post__image'),
      existing = figure?._tm2chScreamerBadge;
    if (existing?.isConnected) return;
    const b = document.createElement('span');
    b.className = 'tm2ch-screamer';
    b.dataset.sk = sk;
    b.tabIndex = 0;
    const host = figure?.querySelector('.post__file-link') || anchor;
    host.insertAdjacentElement('afterend', b);
    if (figure) figure._tm2chScreamerBadge = b;
    addMappedElement(screamerBadges, sk, b);
    renderScreamerResult(b, analysisResults.get(sk));
    if (communityReports.has(sk)) renderCommunityReport(sk, communityReports.get(sk));
    refreshAttachmentRisk(sk);
  }

  function publishScreamerResult(sk, r) {
    r.analysisVersion = ANALYSIS_VERSION;
    analysisResults.set(sk, r);
    state.screamer[sk] = r;
    saveTabSoon();
    for (const b of screamerBadges.get(sk) || []) if (b.isConnected) renderScreamerResult(b, r);
    refreshAttachmentRisk(sk);
  }

  const dbPower = (x) => 10 * Math.log10(Math.max(x, 1e-9));
  const dbAmp = (x) => 20 * Math.log10(Math.max(x, 10 ** (-90 / 20)));
  const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));

  function percentile(values, p) {
    if (!values.length) return -90;
    const a = Array.from(values).sort((x, y) => x - y),
      at = (a.length - 1) * p;
    const lo = Math.floor(at),
      hi = Math.ceil(at),
      f = at - lo;
    return a[lo] * (1 - f) + a[hi] * f;
  }

  function percentileSorted(values, p) {
    if (!values.length) return -90;
    const at = (values.length - 1) * p,
      lo = Math.floor(at),
      hi = Math.ceil(at),
      f = at - lo;
    return values[lo] * (1 - f) + values[hi] * f;
  }

  function sortedIndex(values, value) {
    let lo = 0,
      hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] < value) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function insertSorted(values, value) {
    values.splice(sortedIndex(values, value), 0, value);
  }

  function removeSorted(values, value) {
    const at = sortedIndex(values, value);
    if (at < values.length) values.splice(at, 1);
  }

  function percentileRange(values, start, end, p, scratch) {
    scratch.length = 0;
    for (let i = start; i < end; i++) scratch.push(values[i]);
    scratch.sort((a, b) => a - b);
    return percentileSorted(scratch, p);
  }

  function makeHighPass(fs, f0 = 70, q = Math.SQRT1_2) {
    const w = (2 * Math.PI * f0) / fs,
      c = Math.cos(w),
      alpha = Math.sin(w) / (2 * q),
      a0 = 1 + alpha;
    return {
      b0: (1 + c) / 2 / a0,
      b1: -(1 + c) / a0,
      b2: (1 + c) / 2 / a0,
      a1: (-2 * c) / a0,
      a2: (1 - alpha) / a0,
      z1: 0,
      z2: 0
    };
  }

  function makeHighShelf(fs, f0 = 1500, gainDb = 4, q = Math.SQRT1_2) {
    const a = 10 ** (gainDb / 40),
      w = (2 * Math.PI * f0) / fs,
      c = Math.cos(w),
      alpha = Math.sin(w) / (2 * q);
    const beta = 2 * Math.sqrt(a) * alpha,
      a0 = a + 1 - (a - 1) * c + beta;
    return {
      b0: (a * (a + 1 + (a - 1) * c + beta)) / a0,
      b1: (-2 * a * (a - 1 + (a + 1) * c)) / a0,
      b2: (a * (a + 1 + (a - 1) * c - beta)) / a0,
      a1: (2 * (a - 1 - (a + 1) * c)) / a0,
      a2: (a + 1 - (a - 1) * c - beta) / a0,
      z1: 0,
      z2: 0
    };
  }

  function biquad(x, s) {
    const y = s.b0 * x + s.z1;
    s.z1 = s.b1 * x - s.a1 * y + s.z2;
    s.z2 = s.b2 * x - s.a2 * y;
    return y;
  }

  async function extractTimeline(buf) {
    const stride = 1,
      effectiveRate = buf.sampleRate;
    const windowFrames = Math.max(1, Math.round(buf.sampleRate * ANALYSIS_WINDOW));
    const windowSeconds = windowFrames / buf.sampleRate,
      count = Math.max(1, Math.floor(buf.length / windowFrames));
    const levels = new Float32Array(count),
      peaks = new Float32Array(count);
    const clipCounts = new Uint32Array(count),
      sampleCounts = new Uint32Array(count);
    const channels = Array.from({ length: buf.numberOfChannels }, (_, ch) => buf.getChannelData(ch));
    const filters = channels.map(() => [makeHighPass(effectiveRate), makeHighShelf(effectiveRate)]);
    let trackPeak = 0,
      totalSq = 0,
      totalN = 0,
      totalClip = 0;

    for (let wi = 0; wi < count; wi++) {
      const start = wi * windowFrames,
        end = wi === count - 1 && buf.length < windowFrames ? buf.length : start + windowFrames;
      let rawSq = 0,
        weightedSq = 0,
        n = 0,
        peak = 0,
        clips = 0;
      for (let i = start; i < end; i++) {
        for (let ch = 0; ch < channels.length; ch++) {
          const v = channels[ch][i] || 0,
            av = Math.abs(v);
          rawSq += v * v;
          if (av > peak) peak = av;
          if (av >= 10 ** (-1 / 20)) clips++;
          const weighted = biquad(biquad(v, filters[ch][0]), filters[ch][1]);
          weightedSq += weighted * weighted;
          n++;
        }
      }
      levels[wi] = Math.max(-90, -0.691 + dbPower(weightedSq / Math.max(1, n)));
      peaks[wi] = dbAmp(peak);
      clipCounts[wi] = clips;
      sampleCounts[wi] = n;
      if (peak > trackPeak) trackPeak = peak;
      totalSq += rawSq;
      totalN += n;
      totalClip += clips;
      if (wi % 100 === 99) await yieldMain();
    }
    return {
      levels,
      peaks,
      clipCounts,
      sampleCounts,
      stride,
      windowFrames,
      windowSeconds,
      peakDb: dbAmp(trackPeak),
      rmsDb: dbPower(totalSq / Math.max(1, totalN)),
      nearClipPct: (totalClip / Math.max(1, totalN)) * 100
    };
  }

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const angle = (-2 * Math.PI) / len,
        wlenR = Math.cos(angle),
        wlenI = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let wr = 1,
          wi = 0;
        for (let j = 0; j < len / 2; j++) {
          const uR = re[i + j],
            uI = im[i + j],
            k = i + j + len / 2;
          const vR = re[k] * wr - im[k] * wi,
            vI = re[k] * wi + im[k] * wr;
          re[i + j] = uR + vR;
          im[i + j] = uI + vI;
          re[k] = uR - vR;
          im[k] = uI - vI;
          const nextR = wr * wlenR - wi * wlenI;
          wi = wr * wlenI + wi * wlenR;
          wr = nextR;
        }
      }
    }
  }

  function spectrumProfile(buf, windowIndex, windowFrames) {
    const start = windowIndex * windowFrames,
      end = Math.min(buf.length, start + windowFrames);
    if (start < 0 || start >= end) return null;
    let n = 1;
    while (n < end - start) n <<= 1;
    const bins = Math.min(n / 2 + 1, Math.floor((7800 * n) / buf.sampleRate) + 1),
      out = new Float64Array(bins);
    for (let ch = 0; ch < buf.numberOfChannels; ch++) {
      const data = buf.getChannelData(ch),
        re = new Float64Array(n),
        im = new Float64Array(n);
      for (let i = start; i < end; i++) {
        const j = i - start,
          hann = end - start > 1 ? 0.5 - 0.5 * Math.cos((2 * Math.PI * j) / (end - start - 1)) : 1;
        re[j] = data[i] * hann;
      }
      fft(re, im);
      for (let i = 0; i < bins; i++) out[i] += re[i] * re[i] + im[i] * im[i] + 1e-12;
    }
    let total = 0;
    let logTotal = 0;
    for (let i = 0; i < bins; i++) {
      const p = out[i] / buf.numberOfChannels;
      out[i] = p;
      total += p;
      logTotal += Math.log(p);
    }
    const flatness = Math.exp(logTotal / bins) / (total / bins);
    for (let i = 0; i < bins; i++) out[i] /= total;
    return { bins: out, flatness };
  }

  function onsetSpectralFlux(buf, index, windowFrames) {
    if (index < 1) return 0;
    const before = spectrumProfile(buf, index - 1, windowFrames),
      current = spectrumProfile(buf, index, windowFrames);
    if (!before || !current) return 0;
    return profileFlux(before, current);
  }

  function profileFlux(before, current) {
    const n = Math.min(before.bins.length, current.bins.length);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = current.bins[i] - before.bins[i];
      if (d > 0) sum += d * d;
    }
    return Math.sqrt(sum);
  }

  async function transitionSpectralFeatures(buf, index, timeline) {
    const cache = new Map();
    const getProfile = (i) => {
      if (i < 0 || i >= timeline.levels.length) return null;
      if (!cache.has(i)) cache.set(i, spectrumProfile(buf, i, timeline.windowFrames));
      return cache.get(i);
    };
    const onsetBefore = getProfile(index - 1),
      onsetCurrent = getProfile(index);
    const onsetFlux = onsetBefore && onsetCurrent ? profileFlux(onsetBefore, onsetCurrent) : 0;
    let nearbyFlux = onsetFlux;
    for (let i = Math.max(1, index - 2); i <= Math.min(timeline.levels.length - 1, index + 2); i++) {
      const before = getProfile(i - 1),
        current = getProfile(i);
      if (before && current) nearbyFlux = Math.max(nearbyFlux, profileFlux(before, current));
    }

    const baselineProfiles = [],
      eventProfiles = [];
    for (let i = Math.max(0, index - 40); i < Math.max(0, index - BASELINE_GAP); i++) {
      const profile = getProfile(i);
      if (profile) baselineProfiles.push(profile);
      if (i % 8 === 7) await yieldMain();
    }
    for (let i = index; i < Math.min(timeline.levels.length, index + EVENT_WINDOWS); i++) {
      const profile = getProfile(i);
      if (profile) eventProfiles.push(profile);
    }

    let shapeDistance = 0;
    if (baselineProfiles.length && eventProfiles.length) {
      const bins = Math.min(baselineProfiles[0].bins.length, eventProfiles[0].bins.length),
        baselineMean = new Float64Array(bins),
        eventMean = new Float64Array(bins);
      for (const profile of baselineProfiles) {
        for (let i = 0; i < bins; i++) baselineMean[i] += profile.bins[i] / baselineProfiles.length;
      }
      for (const profile of eventProfiles) {
        for (let i = 0; i < bins; i++) eventMean[i] += profile.bins[i] / eventProfiles.length;
      }
      let distance = 0;
      for (let i = 0; i < bins; i++) {
        const d = Math.sqrt(eventMean[i]) - Math.sqrt(baselineMean[i]);
        distance += d * d;
      }
      shapeDistance = Math.sqrt(distance) / Math.SQRT2;
    }
    return { onsetFlux, nearbyFlux, shapeDistance };
  }

  function windowBrightness(buf, windowIndex, windowFrames, stride) {
    const start = windowIndex * windowFrames,
      end = Math.min(buf.length, start + windowFrames);
    let energy = 0,
      differenceEnergy = 0,
      count = 0;
    for (let ch = 0; ch < buf.numberOfChannels; ch++) {
      const data = buf.getChannelData(ch);
      let previous = null;
      for (let i = start; i < end; i += stride) {
        const value = data[i];
        energy += value * value;
        if (previous !== null) {
          const difference = value - previous;
          differenceEnergy += difference * difference;
        }
        previous = value;
        count++;
      }
    }
    const differences = count - buf.numberOfChannels;
    if (differences < 1 || energy <= 0) return -90;
    return dbPower(differenceEnergy / differences / (energy / count));
  }

  function eventDuration(levels, start, threshold, windowSeconds) {
    let lastAbove = start - 1,
      misses = 0;
    for (let i = start; i < levels.length && i < start + 60; i++) {
      if (levels[i] >= threshold) {
        lastAbove = i;
        misses = 0;
      } else if (++misses >= 2) break;
    }
    return Math.max(0, (lastAbove - start + 1) * windowSeconds);
  }

  function maxChange(levels, seconds, windowSeconds) {
    const d = Math.max(1, Math.round(seconds / windowSeconds));
    let max = 0;
    for (let i = 0; i + d < levels.length; i++) {
      const before = (levels[i] + (levels[i + 1] ?? levels[i])) / 2;
      const after = (levels[i + d] + (levels[i + d + 1] ?? levels[i + d])) / 2;
      max = Math.max(max, after - before);
    }
    return max;
  }

  function measureAttack(buf, index, timeline) {
    const { stride, windowFrames } = timeline;
    const eventFrame = index * windowFrames;
    const visibleStart = Math.max(0, eventFrame - Math.round(buf.sampleRate * 0.5));
    const warmupStart = Math.max(0, visibleStart - Math.round(buf.sampleRate * 3));
    const end = Math.min(buf.length, eventFrame + Math.round(buf.sampleRate * 0.8));
    const fineFrames = Math.max(1, Math.round(buf.sampleRate * 0.01));
    const channels = Array.from({ length: buf.numberOfChannels }, (_, ch) => buf.getChannelData(ch));
    const filters = channels.map(() => [makeHighPass(buf.sampleRate), makeHighShelf(buf.sampleRate)]);
    const levels = [];
    for (let frameStart = warmupStart; frameStart < end; frameStart += fineFrames) {
      const frameEnd = Math.min(end, frameStart + fineFrames);
      let square = 0,
        count = 0;
      for (let i = frameStart; i < frameEnd; i += stride) {
        for (let ch = 0; ch < channels.length; ch++) {
          const weighted = biquad(biquad(channels[ch][i] || 0, filters[ch][0]), filters[ch][1]);
          square += weighted * weighted;
          count++;
        }
      }
      if (frameStart >= visibleStart) levels.push(Math.max(-90, -0.691 + dbPower(square / Math.max(1, count))));
    }
    const eventAt = Math.floor((eventFrame - visibleStart) / fineFrames);
    const peakEnd = Math.min(levels.length, eventAt + 50);
    if (eventAt < 0 || peakEnd <= eventAt) return null;
    let peak = eventAt;
    for (let i = eventAt + 1; i < peakEnd; i++) if (levels[i] > levels[peak]) peak = i;
    let lastBelow = -1;
    for (let i = 0; i < peak; i++) if (levels[i] <= -30) lastBelow = i;
    if (lastBelow < 0) return null;
    const target = levels[peak] - 3;
    for (let i = lastBelow + 1; i <= peak; i++) {
      if (levels[i] >= target) return ((i - lastBelow) * fineFrames * 1000) / buf.sampleRate;
    }
    return null;
  }

  async function nativeNoAudioProbe(blob) {
    return new Promise((resolve) => {
      const video = document.createElement('video'),
        objectUrl = URL.createObjectURL(blob);
      let done = false;
      const finish = (value) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        video.removeAttribute('src');
        video.load();
        URL.revokeObjectURL(objectUrl);
        resolve(value);
      };
      const timer = setTimeout(() => finish(null), 5000);
      video.preload = 'metadata';
      video.muted = true;
      video.onloadedmetadata = () => {
        if (typeof video.mozHasAudio === 'boolean') return finish(!video.mozHasAudio);
        if ('audioTracks' in video && video.audioTracks) return finish(video.audioTracks.length === 0);
        finish(null);
      };
      video.onerror = () => finish(null);
      video.src = objectUrl;
    });
  }

  function getDecoderContext() {
    if (decoderContext) return decoderContext;
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (OfflineCtx) {
      try {
        decoderContext = new OfflineCtx(1, 1, ANALYSIS_TARGET_RATE);
        return decoderContext;
      } catch (e) {
        console.warn('[spokoyno screamer] target-rate offline decoder unavailable:', e);
      }
    }
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    try {
      decoderContext = new AudioCtx({ sampleRate: ANALYSIS_TARGET_RATE });
    } catch {
      try {
        decoderContext = new AudioCtx();
      } catch (e) {
        console.warn('[spokoyno screamer] audio decoder unavailable:', e);
        return null;
      }
    }
    return decoderContext;
  }

  async function resampleForAnalysis(buf) {
    if (buf.sampleRate === ANALYSIS_TARGET_RATE) return buf;
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!OfflineCtx) throw new Error('OfflineAudioContext is required for anti-aliased 16 kHz resampling');
    const channels = Math.max(1, Math.min(2, buf.numberOfChannels));
    const length = Math.max(1, Math.ceil(buf.duration * ANALYSIS_TARGET_RATE));
    const context = new OfflineCtx(channels, length, ANALYSIS_TARGET_RATE);
    const source = context.createBufferSource();
    source.buffer = buf;
    source.connect(context.destination);
    source.start();
    return context.startRendering();
  }

  async function analyzeScreamer(blob) {
    const context = getDecoderContext();
    if (!context) return { status: 'unsupported' };

    let buf, encoded;
    try {
      encoded = await blob.arrayBuffer();
      buf = await context.decodeAudioData(encoded);
      encoded = null;
      buf = await resampleForAnalysis(buf);
    } catch (e) {
      console.warn('[spokoyno screamer] decode failed:', e);
      return {
        status: (await nativeNoAudioProbe(blob)) === true ? 'no-audio' : 'decode-error'
      };
    }
    if (!buf?.numberOfChannels || !buf.length) return { status: 'no-audio' };

    const timeline = await extractTimeline(buf),
      { levels, peaks, clipCounts, sampleCounts, stride, windowSeconds } = timeline;
    if (!levels.length) return { status: 'no-audio' };

    // A separate path covers near-full-scale audio before a 1 s baseline exists. It deliberately
    // requires sustained loudness plus clipping, noise-like flatness, or high-frequency energy so
    // ordinary loud intros and music do not inherit the transition detector's assumptions.
    const eventScratch = [],
      startLimit = Math.min(Math.max(0, levels.length - EVENT_WINDOWS), Math.round(1 / windowSeconds));
    let startBest = 0,
      startLevel = -90;
    for (let i = 0; i <= startLimit; i++) {
      const level = percentileRange(levels, i, Math.min(levels.length, i + EVENT_WINDOWS), 0.25, eventScratch);
      if (level > startLevel) {
        startBest = i;
        startLevel = level;
      }
    }
    const startTarget = Math.max(-6, startLevel - 3);
    let startPreview = startBest;
    for (let i = 0; i <= startLimit; i++) {
      if (percentileRange(levels, i, Math.min(levels.length, i + EVENT_WINDOWS), 0.25, eventScratch) >= startTarget) {
        startPreview = i;
        break;
      }
    }
    const onsetThreshold = Math.max(-9, startLevel - 6);
    let start = startPreview;
    for (let i = startPreview; i < Math.min(levels.length, startPreview + EVENT_WINDOWS); i++) {
      if (levels[i] >= onsetThreshold) {
        start = i;
        break;
      }
    }
    const startDuration = eventDuration(levels, start, -9, windowSeconds),
      startLookEnd = Math.min(levels.length, start + 20);
    let startPeak = -90,
      startClips = 0,
      startSamples = 0;
    for (let i = start; i < startLookEnd; i++) {
      startPeak = Math.max(startPeak, peaks[i]);
      startClips += clipCounts[i];
      startSamples += sampleCounts[i];
    }
    const startNearClip = startClips / Math.max(1, startSamples);
    const startFlatnessValues = [],
      startBrightnessValues = [];
    for (let i = start; i < Math.min(levels.length, start + EVENT_WINDOWS); i++) {
      const profile = spectrumProfile(buf, i, timeline.windowFrames);
      if (profile) startFlatnessValues.push(profile.flatness);
      startBrightnessValues.push(windowBrightness(buf, i, timeline.windowFrames, stride));
    }
    const startSpectralFlatness = startFlatnessValues.length ? percentile(startFlatnessValues, 0.5) : 0;
    const startBrightnessDb = startBrightnessValues.length ? percentile(startBrightnessValues, 0.5) : -90;
    const startLoudComponent = sigmoid((startLevel + 6) / 2);
    const startDurationComponent = sigmoid((startDuration - 0.35) / 0.15);
    const startClipComponent = sigmoid((startNearClip - 0.01) / 0.03);
    const startNoiseComponent = sigmoid((startSpectralFlatness - 0.025) / 0.025);
    const startBrightnessComponent = sigmoid((startBrightnessDb + 8) / 2.5);
    let startConfidence = startLoudComponent ** 1.1 * startDurationComponent ** 0.7;
    startConfidence *= 0.68 + 0.12 * startClipComponent + 0.14 * startNoiseComponent + 0.06 * startBrightnessComponent;
    startConfidence = Math.max(0, Math.min(1, startConfidence));
    const startSpectralEvidence = startSpectralFlatness >= 0.04 || startBrightnessDb >= -5 || startNearClip >= 0.08;
    const startEligible = startLevel >= -3 && startDuration >= 0.5 && startSpectralEvidence;

    let best = -1,
      bestEnvelope = -1,
      bestBaseline = -90,
      bestEvent = -90,
      bestJump = 0,
      bestBaselineMad = 0;
    const historySorted = [],
      madScratch = [];
    let historyLo = 0,
      historyHi = 0;
    for (let i = 0; i + EVENT_WINDOWS <= levels.length; i++) {
      const lo = Math.max(0, i - BASELINE_WINDOWS),
        hi = i - BASELINE_GAP;
      while (historyHi < Math.max(0, hi)) insertSorted(historySorted, levels[historyHi++]);
      while (historyLo < lo) removeSorted(historySorted, levels[historyLo++]);
      if (hi - lo < MIN_BASELINE_WINDOWS) continue;
      const baseline = percentileSorted(historySorted, 0.5);
      const event = percentileRange(levels, i, i + EVENT_WINDOWS, 0.25, eventScratch);
      const jump = event - baseline;
      const envelope = sigmoid((event + 10.5) / 2.5) ** 0.9 * sigmoid((jump - 13) / 3.5) ** 1.25;
      if (envelope > bestEnvelope) {
        best = i;
        bestEnvelope = envelope;
        bestBaseline = baseline;
        bestEvent = event;
        bestJump = jump;
        madScratch.length = historySorted.length;
        for (let j = 0; j < historySorted.length; j++) madScratch[j] = Math.abs(historySorted[j] - baseline);
        madScratch.sort((a, b) => a - b);
        bestBaselineMad = percentileSorted(madScratch, 0.5);
      }
      if (i % 250 === 249) await yieldMain();
    }

    const hasTransition = best >= 0;
    if (best < 0) {
      best = 0;
      bestBaseline = percentile(levels, 0.5);
      bestEvent = levels[0];
      bestJump = bestEvent - bestBaseline;
      bestBaselineMad = percentile(
        Array.from(levels, (level) => Math.abs(level - bestBaseline)),
        0.5
      );
    }

    const threshold = Math.max(-13, bestBaseline + 9);
    const duration = eventDuration(levels, best, threshold, windowSeconds);
    const lookEnd = Math.min(levels.length, best + EVENT_LOOKAHEAD);
    let eventPeak = -90,
      eventClips = 0,
      eventSamples = 0;
    for (let i = best; i < lookEnd; i++) {
      eventPeak = Math.max(eventPeak, peaks[i]);
      eventClips += clipCounts[i];
      eventSamples += sampleCounts[i];
    }
    const eventNearClip = eventClips / Math.max(1, eventSamples);
    const transitionSpectrum = hasTransition
      ? await transitionSpectralFeatures(buf, best, timeline)
      : { onsetFlux: 0, nearbyFlux: 0, shapeDistance: 0 };
    const spectralFlux = transitionSpectrum.onsetFlux,
      spectralFluxNear = transitionSpectrum.nearbyFlux,
      spectralShapeDistance = transitionSpectrum.shapeDistance;
    const loudComponent = sigmoid((bestEvent + 10.5) / 2.5);
    const jumpComponent = sigmoid((bestJump - 13) / 3.5);
    const durationComponent = sigmoid((duration - 0.18) / 0.09);
    const quietComponent = sigmoid((-bestBaseline - 15) / 5);
    const clipComponent = sigmoid((eventNearClip - 0.005) / 0.012);
    const fluxComponent = sigmoid((spectralFlux - 0.22) / 0.08);
    let transitionConfidence = hasTransition
      ? loudComponent ** 0.9 * jumpComponent ** 1.25 * durationComponent ** 0.65
      : 0;
    transitionConfidence *= 0.87 + 0.1 * quietComponent + 0.03 * clipComponent;
    transitionConfidence *= 0.65 + 0.35 * fluxComponent;
    transitionConfidence = Math.max(0, Math.min(1, transitionConfidence));
    const transitionEligible = hasTransition && bestEvent >= -6 && bestJump >= 14 && duration >= 0.15;
    // Conservative rescue paths cover short edited bursts that the general score intentionally
    // suppresses. One requires a large spectral-distribution change; the other requires severe
    // clipping. Upper duration bounds keep ordinary sustained screams, speech, music drops, and
    // movie effects on the main transition path. These rules were tuned after positives #5/#6,
    // so their risk values are evidence scores, not calibrated probabilities.
    const spectralBurstEligible =
      hasTransition &&
      bestEvent >= -4 &&
      bestJump >= 16 &&
      duration >= 0.3 &&
      duration <= 1.05 &&
      spectralFluxNear >= 0.3 &&
      spectralShapeDistance >= 0.8;
    const clippedBurstEligible =
      hasTransition &&
      bestEvent >= -3 &&
      bestJump >= 10 &&
      duration >= 0.25 &&
      duration <= 0.55 &&
      eventNearClip >= 0.35 &&
      spectralFluxNear >= 0.3;
    const spectralBurstConfidence = spectralBurstEligible
      ? Math.min(
          1,
          0.8 +
            0.04 * sigmoid((spectralShapeDistance - 0.85) / 0.08) +
            0.04 * sigmoid((bestJump - 18) / 3) +
            0.04 * sigmoid((bestEvent + 3) / 1.5)
        )
      : 0;
    const clippedBurstConfidence = clippedBurstEligible
      ? Math.min(
          1,
          0.8 +
            0.04 * sigmoid((eventNearClip - 0.4) / 0.08) +
            0.04 * sigmoid((bestJump - 12) / 2) +
            0.04 * sigmoid((bestEvent + 2) / 1.2)
        )
      : 0;
    const rescueMode =
      clippedBurstConfidence > spectralBurstConfidence ? 'short-clipped-burst' : 'short-spectral-burst';
    const rescueConfidence = Math.max(spectralBurstConfidence, clippedBurstConfidence);
    const transitionDecisionScore = transitionEligible ? transitionConfidence : 0;
    const startDecisionScore = startEligible ? startConfidence : 0;
    const rescueDecisionScore = rescueConfidence;
    const decisionScore = Math.max(transitionDecisionScore, startDecisionScore, rescueDecisionScore);
    const suspicious = decisionScore >= SCREAMER_CONFIDENCE;
    const decisionMode =
      rescueDecisionScore === decisionScore && rescueDecisionScore > 0
        ? rescueMode
        : startDecisionScore === decisionScore && startDecisionScore > transitionDecisionScore
          ? 'loud-start'
          : 'transition';
    const riskMode = startConfidence > transitionConfidence ? 'loud-start' : 'transition';
    const detectionMode = suspicious ? decisionMode : riskMode;
    const displayRisk =
      detectionMode === 'loud-start'
        ? startConfidence
        : detectionMode === 'short-spectral-burst' || detectionMode === 'short-clipped-burst'
          ? rescueConfidence
          : transitionConfidence;
    const confidence = displayRisk;
    const startSuspicious = suspicious && detectionMode === 'loud-start';
    const transitionSuspicious = suspicious && detectionMode !== 'loud-start';
    const sortedLevels = Array.from(levels).sort((a, b) => a - b),
      median = percentileSorted(sortedLevels, 0.5);
    const reason =
      detectionMode === 'loud-start'
        ? startSuspicious
          ? 'Sustained near-full-scale broadband or clipped audio begins before a reliable baseline exists'
          : 'Opening audio did not combine enough loudness, persistence, and broadband evidence'
        : transitionSuspicious
          ? detectionMode === 'short-spectral-burst'
            ? 'Short near-full-scale burst with a large local spectral-distribution change'
            : detectionMode === 'short-clipped-burst'
              ? 'Short near-full-scale transition with severe clipping'
              : 'Sustained near-full-scale event after a quieter baseline, with a changed onset spectrum'
          : !hasTransition
            ? 'No quiet-to-loud transition had enough preceding audio for a reliable baseline'
            : bestJump < 14
              ? 'No sufficiently large sustained local transition'
              : bestEvent < -6
                ? 'The strongest transition did not become near-full-scale'
                : spectralFlux < 0.22
                  ? 'The loudness changed, but the onset spectrum remained similar'
                  : 'Combined evidence stayed below the warning threshold';
    const useStartEvent = detectionMode === 'loud-start';
    const selectedSpectralFlux = useStartEvent ? onsetSpectralFlux(buf, start, timeline.windowFrames) : spectralFlux;
    const attackMs = measureAttack(buf, useStartEvent ? start : best, timeline);
    const robustScale = Math.max(2, 1.4826 * bestBaselineMad);

    return {
      status: 'ok',
      suspicious,
      confidence,
      displayRisk,
      decisionScore,
      detectionMode,
      decisionMode,
      riskMode,
      transitionConfidence,
      startConfidence,
      rescueConfidence,
      rescueMode: rescueConfidence ? rescueMode : null,
      eventAt: (useStartEvent ? start : best) * windowSeconds,
      jumpDb: useStartEvent ? null : bestJump,
      eventDb: useStartEvent ? startLevel : bestEvent,
      baselineDb: useStartEvent ? null : bestBaseline,
      baselineMadDb: useStartEvent ? null : bestBaselineMad,
      robustZ: useStartEvent ? null : bestJump / robustScale,
      attackMs,
      eventDuration: useStartEvent ? startDuration : duration,
      eventPeakDb: useStartEvent ? startPeak : eventPeak,
      eventNearClipPct: (useStartEvent ? startNearClip : eventNearClip) * 100,
      spectralFlux: selectedSpectralFlux,
      spectralFluxNear,
      spectralShapeDistance,
      startAt: start * windowSeconds,
      startEventDb: startLevel,
      startDuration,
      startNearClipPct: startNearClip * 100,
      startSpectralFlatness,
      startBrightnessDb,
      medianDb: median,
      p10Db: percentileSorted(sortedLevels, 0.1),
      p25Db: percentileSorted(sortedLevels, 0.25),
      p75Db: percentileSorted(sortedLevels, 0.75),
      p90Db: percentileSorted(sortedLevels, 0.9),
      p95Db: percentileSorted(sortedLevels, 0.95),
      p99Db: percentileSorted(sortedLevels, 0.99),
      peakDb: timeline.peakDb,
      rmsDb: timeline.rmsDb,
      crestDb: timeline.peakDb - timeline.rmsDb,
      nearClipPct: timeline.nearClipPct,
      maxChange50Db: maxChange(levels, 0.05, windowSeconds),
      maxChange100Db: maxChange(levels, 0.1, windowSeconds),
      maxChange250Db: maxChange(levels, 0.25, windowSeconds),
      maxChange500Db: maxChange(levels, 0.5, windowSeconds),
      maxChange1000Db: maxChange(levels, 1, windowSeconds),
      duration: buf.duration,
      reason
    };
  }

  function queueScreamerAnalysis(url, key) {
    const sk = screamerKey(url);
    if (analysisResults.has(sk) || analysisQueued.has(sk)) return;
    analysisQueued.add(sk);
    analysisQueue.push({ url, key, sk, generation: analysisGeneration });
    runAnalysisQueue();
  }

  async function runAnalysisQueue() {
    if (analysisRunning) return;
    analysisRunning = true;
    try {
      while (analysisQueue.length) {
        const item = analysisQueue.shift();
        analysisQueued.delete(item.sk);
        if (item.generation !== analysisGeneration || analysisResults.has(item.sk)) continue;
        try {
          const analysisCache = cache,
            response = await analysisCache.match(item.key);
          if (item.generation !== analysisGeneration) continue;
          if (!response) {
            cached.delete(item.key);
            if (seenMedia.has(item.key)) {
              enqueueMedia(item.url, item.key, true);
              pump();
            }
            continue;
          }
          const blob = await response.blob();
          if (item.generation !== analysisGeneration) continue;
          try {
            await identifyMediaBlob(blob, response.headers.get('Content-Type'), item.url);
          } catch (e) {
            await withCacheLock(() => analysisCache.delete(item.key));
            if (item.generation !== analysisGeneration) continue;
            cached.delete(item.key);
            if (seenMedia.has(item.key)) {
              enqueueMedia(item.url, item.key, true);
              pump();
            } else publishScreamerResult(item.sk, { status: 'media-error', message: String(e.message || e) });
            continue;
          }
          const result = await analyzeScreamer(blob);
          if (item.generation === analysisGeneration) publishScreamerResult(item.sk, result);
        } catch (e) {
          console.warn('[spokoyno screamer] analysis failed:', item.url, e);
          if (item.generation === analysisGeneration) publishScreamerResult(item.sk, { status: 'decode-error' });
        }
        await yieldMain();
      }
    } finally {
      analysisRunning = false;
      if (tabStateDirty) await flushTabState();
    }
  }

  function discover(root) {
    const links = [];
    if (root.matches?.('figure.post__image a[href]')) links.push(root);
    if (root.querySelectorAll) links.push(...root.querySelectorAll('figure.post__image a[href]'));
    for (const a of links) {
      const url = mediaUrl(a);
      if (!url) continue;
      a.dataset.tm2chMediaUrl = url;
      const key = canonicalKey(url);
      seen.add(key);
      seenMedia.set(key, url);
      rememberAttachment(a, url);
      attachScreamerBadge(a, url);
      if (cached.has(key)) {
        queueScreamerAnalysis(url, key);
        continue;
      }
      enqueueMedia(url, key);
    }
    scheduleUI();
    pump();
  }

  function scheduleDomPrune() {
    if (domPruneScheduled) return;
    domPruneScheduled = true;
    setTimeout(() => {
      domPruneScheduled = false;
      pruneDisconnectedMedia();
    }, 50);
  }

  function pruneDisconnectedMedia() {
    const wantedKeys = new Set();
    for (const anchor of document.querySelectorAll('figure.post__image a[data-tm2ch-media-url]')) {
      wantedKeys.add(canonicalKey(anchor.dataset.tm2chMediaUrl));
    }
    for (const [sk, figures] of attachmentFigures) {
      for (const figure of figures) if (!figure.isConnected) figures.delete(figure);
      if (!figures.size) attachmentFigures.delete(sk);
    }
    for (const map of [screamerBadges, communityBadges]) {
      for (const [sk, badges] of map) {
        for (const badge of badges) if (!badge.isConnected) badges.delete(badge);
        if (!badges.size) map.delete(sk);
      }
    }
    for (const [key] of seenMedia) {
      if (wantedKeys.has(key)) continue;
      seenMedia.delete(key);
      seen.delete(key);
      queued.delete(key);
      failedMedia.delete(key);
      abortDownloadsForKey(key);
      const timer = retryTimers.get(key);
      if (timer) clearTimeout(timer);
      retryTimers.delete(key);
      for (let i = queue.length - 1; i >= 0; i--) if (queue[i].key === key) queue.splice(i, 1);
      for (let i = analysisQueue.length - 1; i >= 0; i--) {
        if (analysisQueue[i].key !== key) continue;
        analysisQueued.delete(analysisQueue[i].sk);
        analysisQueue.splice(i, 1);
      }
    }
    scheduleUI();
  }

  function scheduleDownloadRetry(item, error) {
    const nextAttempt = item.attempt + 1;
    const message = String(error?.message || error);
    const permanent =
      /quota|QuotaExceeded/i.test(message) || (/HTTP 4\d\d/.test(message) && !/HTTP (?:408|429)/.test(message));
    if (
      permanent ||
      nextAttempt >= MAX_DOWNLOAD_ATTEMPTS ||
      item.generation !== downloadGeneration ||
      !seenMedia.has(item.key)
    )
      return false;
    const delay = RETRY_BASE_DELAY * 2 ** item.attempt;
    console.warn(
      `[spokoyno] retrying ${filenameFromUrl(item.url)} in ${(delay / 1000).toFixed(0)} s ` +
        `(attempt ${nextAttempt + 1}/${MAX_DOWNLOAD_ATTEMPTS})`
    );
    const timer = setTimeout(() => {
      retryTimers.delete(item.key);
      if (item.generation !== downloadGeneration || cached.has(item.key) || !seenMedia.has(item.key)) {
        if (item.generation === downloadGeneration) queued.delete(item.key);
        scheduleUI();
        return;
      }
      queue.push({ ...item, attempt: nextAttempt });
      scheduleUI();
      pump();
    }, delay);
    retryTimers.set(item.key, timer);
    return true;
  }

  function pump() {
    while (running < CONCURRENCY && queue.length) {
      const item = queue.shift();
      if (item.generation !== downloadGeneration || !seenMedia.has(item.key)) {
        queued.delete(item.key);
        continue;
      }
      let retryScheduled = false;
      running++;
      const operation = preload(item);
      activePreloads.add(operation);
      operation
        .catch((e) => {
          if (item.generation !== downloadGeneration || !seenMedia.has(item.key)) return;
          console.warn('[spokoyno] preload failed:', item.url, e);
          retryScheduled = scheduleDownloadRetry(item, e);
          if (!retryScheduled) {
            failedMedia.add(item.key);
            const message = String(e?.message || e);
            publishScreamerResult(screamerKey(item.url), {
              status: /quota|cache|storage/i.test(message) ? 'cache-error' : 'media-error',
              message
            });
          }
        })
        .finally(() => {
          activePreloads.delete(operation);
          if (item.generation === downloadGeneration && !retryScheduled) queued.delete(item.key);
          running--;
          scheduleUI();
          pump();
        });
    }
  }

  async function preload({ url, key, generation }) {
    if (generation !== downloadGeneration || !seenMedia.has(key)) return;
    const existing = await cache.match(key);
    if (generation !== downloadGeneration || !seenMedia.has(key)) return;
    if (existing) {
      failedMedia.delete(key);
      cached.add(key);
      queueScreamerAnalysis(url, key);
      scheduleUI();
      return;
    }
    const { blob, contentType, domain } = await downloadFromBestMirror(url, key, generation);
    if (generation !== downloadGeneration || !seenMedia.has(key)) return;
    const headers = new Headers();
    if (contentType) headers.set('Content-Type', contentType);
    headers.set('X-TM-2ch-Mirror', domain);
    headers.set('X-TM-2ch-Original-Path', mediaPath(url));
    const stored = await withCacheLock(async () => {
      if (generation !== downloadGeneration || !seenMedia.has(key)) return false;
      await cache.put(key, new Response(blob, { status: 200, headers }));
      return true;
    });
    if (!stored || generation !== downloadGeneration || !seenMedia.has(key)) return;
    failedMedia.delete(key);
    cached.add(key);
    touchCurrentCache().catch((e) => console.warn('[spokoyno] cache metadata touch failed:', e));
    queueScreamerAnalysis(url, key);
    scheduleUI();
  }

  function installClickHandler() {
    window.addEventListener(
      'click',
      async (e) => {
        if (e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
        const a = e.target?.closest?.('a[href]');
        if (!a?.closest?.('figure.post__image')) return;
        const url = a.dataset.tm2chMediaUrl || mediaUrl(a);
        if (!url) return;
        const key = canonicalKey(url);
        if (!cached.has(key)) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        try {
          await showFromCache(url, a.target || '_self');
        } catch (err) {
          console.warn('[spokoyno] viewer:', err);
          followOriginalLink(url, a.target || '_self');
        }
      },
      true
    );
  }

  function followOriginalLink(url, target = '_self') {
    if (!target || target === '_self') location.href = url;
    else GM_openInTab(url, { active: true, insert: true, setParent: true });
  }

  async function showFromCache(url, originalTarget = '_self') {
    const key = canonicalKey(url),
      r = await cache.match(key);
    if (!r) {
      cached.delete(key);
      enqueueMedia(url, key, true);
      pump();
      throw new Error('cache miss');
    }
    const blob = await r.blob();
    try {
      await identifyMediaBlob(blob, r.headers.get('Content-Type'), url);
    } catch (e) {
      await withCacheLock(() => cache.delete(key));
      cached.delete(key);
      enqueueMedia(url, key, true);
      pump();
      throw e;
    }
    const objectUrl = URL.createObjectURL(blob);
    const old = document.getElementById('tm2ch-cache-viewer');
    if (old?._tmClose) old._tmClose();
    else old?.remove();
    const overlay = document.createElement('div');
    overlay.id = 'tm2ch-cache-viewer';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', `Cached video: ${filenameFromUrl(url)}`);
    overlay.style.cssText =
      'position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;background:rgba(0,0,0,.94);cursor:zoom-out';
    const media = document.createElement('video');
    media.controls = true;
    media.autoplay = true;
    media.loop = true;
    media.src = objectUrl;
    media.tabIndex = 0;
    media.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;cursor:default';
    media.addEventListener('click', (e) => e.stopPropagation());
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.textContent = '×';
    closeButton.setAttribute('aria-label', 'Close cached video');
    closeButton.style.cssText =
      'position:absolute;right:16px;top:12px;border:0;background:rgba(0,0,0,.55);color:#fff;font:32px/1 sans-serif;cursor:pointer;border-radius:4px;padding:2px 9px';
    closeButton.addEventListener('click', (e) => {
      e.stopPropagation();
      close();
    });
    const errorLink = document.createElement('a');
    errorLink.href = url;
    errorLink.target = originalTarget;
    errorLink.rel = originalTarget === '_blank' ? 'noopener' : '';
    errorLink.textContent = 'Cached playback failed — open the original video';
    errorLink.style.cssText =
      'display:none;padding:12px 16px;border-radius:6px;background:#fff;color:#b00;font:16px/1.4 sans-serif;cursor:pointer';
    errorLink.addEventListener('click', (e) => e.stopPropagation());
    const previousFocus = document.activeElement;
    let closed = false;
    const close = () => {
      if (closed) return;
      closed = true;
      document.removeEventListener('keydown', keydown, true);
      try {
        media.pause();
      } catch {}
      media.removeAttribute('src');
      media.load();
      overlay.remove();
      URL.revokeObjectURL(objectUrl);
      if (previousFocus?.isConnected) previousFocus.focus?.();
    };
    const keydown = (e) => {
      if (e.key === 'Escape') {
        close();
        return;
      }
      if (e.key !== 'Tab') return;
      const focusable = [media, errorLink, closeButton].filter(
        (element) => element.isConnected && getComputedStyle(element).display !== 'none'
      );
      if (!focusable.length) return;
      const current = focusable.indexOf(document.activeElement);
      const next = e.shiftKey
        ? current <= 0
          ? focusable.length - 1
          : current - 1
        : current < 0 || current === focusable.length - 1
          ? 0
          : current + 1;
      e.preventDefault();
      focusable[next].focus();
    };
    overlay._tmClose = close;
    overlay.addEventListener('click', close);
    document.addEventListener('keydown', keydown, true);
    media.addEventListener('error', () => {
      if (closed || !media.src) return;
      media.style.display = 'none';
      errorLink.style.display = 'inline-block';
      errorLink.focus();
    });
    overlay.append(media, errorLink, closeButton);
    document.documentElement.append(overlay);
    media.focus();
  }

  function installUI() {
    if (uiRoot) return;
    uiRoot = document.createElement('div');
    uiRoot.style.cssText =
      'position:fixed;right:10px;bottom:10px;z-index:2147483646;display:flex;gap:8px;align-items:flex-end;font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:white;pointer-events:none;user-select:none';
    speedBox = document.createElement('div');
    speedBox.style.cssText =
      'width:330px;padding:8px;box-sizing:border-box;border-radius:7px;background:rgba(0,0,0,.86);backdrop-filter:blur(4px)';
    summaryBox = document.createElement('div');
    summaryBox.style.cssText = 'padding:6px 8px;border-radius:7px;white-space:nowrap;background:rgba(0,0,0,.86)';
    uiRoot.append(speedBox, summaryBox);
    document.documentElement.append(uiRoot);
    renderUI();
  }

  function scheduleUI() {
    if (uiScheduled) return;
    uiScheduled = true;
    setTimeout(() => {
      uiScheduled = false;
      renderUI();
    }, 100);
  }

  function renderUI() {
    if (!summaryBox || !speedBox) return;
    let threadDone = 0;
    for (const key of seen) if (cached.has(key)) threadDone++;
    const mirror = state.mirror ? state.mirror.replace('2ch.', '') : '?';
    summaryBox.textContent =
      `thread ${threadDone}/${seen.size} · tab ${cached.size} · via ${mirror}` +
      `${queue.length ? ` · q${queue.length}` : ''}` +
      `${failedMedia.size ? ` · err ${failedMedia.size}` : ''}`;
    const active = [...activeDownloads.values()];
    const totalBps = active.reduce((s, x) => s + (Number.isFinite(x.bps) ? x.bps : 0), 0);
    speedBox.replaceChildren();
    const header = document.createElement('div');
    header.style.cssText =
      'display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:6px;font-weight:700';
    const total = document.createElement('span'),
      count = document.createElement('span');
    total.textContent = `Σ ${formatRate(totalBps)}`;
    count.style.opacity = '.65';
    count.textContent = `${active.length} active`;
    header.append(total, count);
    speedBox.append(header);

    if (active.length) {
      const max = Math.max(1, ...active.map((x) => x.bps || 0));
      for (const x of active) {
        const row = document.createElement('div');
        row.style.cssText =
          'position:relative;margin-top:4px;height:29px;overflow:hidden;border-radius:4px;background:rgba(255,255,255,.075)';
        const bar = document.createElement('div');
        bar.style.cssText = `position:absolute;left:0;top:0;bottom:0;width:${Math.max(0, Math.min(100, (x.bps / max) * 100))}%;background:rgba(255,255,255,.13);transition:width .15s linear`;
        const c = document.createElement('div');
        c.style.cssText =
          'position:relative;z-index:1;height:100%;display:flex;align-items:center;gap:7px;padding:0 6px;box-sizing:border-box';
        const n = document.createElement('span'),
          p = document.createElement('span'),
          s = document.createElement('span');
        n.style.cssText = 'flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;opacity:.82';
        n.textContent = shortName(x.name);
        n.title = x.name;
        p.style.cssText = 'opacity:.55;white-space:nowrap';
        p.textContent = x.total > 0 ? `${Math.floor((x.loaded / x.total) * 100)}%` : formatBytes(x.loaded);
        s.style.cssText = 'min-width:78px;text-align:right;white-space:nowrap;font-weight:700';
        s.textContent = formatRate(x.bps);
        c.append(n, p, s);
        row.append(bar, c);
        speedBox.append(row);
      }
    } else {
      const idle = document.createElement('div');
      idle.style.cssText = 'padding:4px 0;opacity:.5';
      idle.textContent = queue.length ? 'waiting for downloads…' : 'idle';
      speedBox.append(idle);
    }

    if (recentDownloads.length) {
      const sep = document.createElement('div');
      sep.style.cssText = 'margin:7px 0 4px;padding-top:5px;border-top:1px solid rgba(255,255,255,.12);opacity:.55';
      sep.textContent = 'recent avg';
      speedBox.append(sep);
      for (const x of recentDownloads.slice(0, 4)) {
        const row = document.createElement('div'),
          n = document.createElement('span');
        const d = document.createElement('span'),
          s = document.createElement('span');
        row.style.cssText = 'display:flex;gap:6px;padding:1px 0;opacity:.68';
        n.style.cssText = 'flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis';
        n.textContent = shortName(x.name);
        d.style.opacity = '.6';
        d.textContent = x.domain.replace('2ch.', '');
        s.style.cssText = 'min-width:78px;text-align:right';
        s.textContent = formatRate(x.avgBps);
        row.append(n, d, s);
        speedBox.append(row);
      }
    }
  }

  async function cleanupOrphanCaches() {
    try {
      const tabs = await gmGetTabs(),
        states = Object.values(tabs);
      const alive = new Set(
        states
          .filter((x) => x?.tm2chMediaV4?.lastOrigin === location.origin)
          .map((x) => x.tm2chMediaV4.cacheName)
          .filter((x) => typeof x === 'string' && x.startsWith(PREFIX))
      );
      const now = Date.now();
      await touchCurrentCache();
      for (const name of await caches.keys()) {
        if (!name.startsWith(PREFIX) || name === cacheName) continue;
        const target = await caches.open(name);
        const previous = await readCacheMeta(target, name);
        if (alive.has(name)) {
          if (previous?.orphanedAt) await writeCacheMeta(target, { ...previous, orphanedAt: null });
          else if (!previous) {
            await writeCacheMeta(target, {
              schema: 1,
              cacheName: name,
              origin: location.origin,
              createdAt: now,
              lastSeenAt: now,
              orphanedAt: null
            });
          }
          continue;
        }
        if (!previous?.orphanedAt) {
          await writeCacheMeta(target, {
            schema: 1,
            cacheName: name,
            origin: previous?.origin || location.origin,
            createdAt: previous?.createdAt || now,
            lastSeenAt: previous?.lastSeenAt || 0,
            orphanedAt: now
          });
          continue;
        }
        if (now - previous.orphanedAt >= ORPHAN_GRACE) {
          await caches.delete(name);
          console.info('[spokoyno] deleted cache after one-hour orphan grace:', name);
        }
      }
    } catch (e) {
      console.warn('[spokoyno] cleanup skipped:', e);
    }
  }

  GM_registerMenuCommand('Проверить media-cache текущего зеркала', async () => {
    await reconcileCache('menu', true);
    const diagnostics = await getCacheDiagnostics();
    console.table(diagnostics);
    window.alert(`Spokoyno cache diagnostics\n\n${cacheDiagnosticText(diagnostics)}`);
  });

  GM_registerMenuCommand('Запросить постоянное хранение', async () => {
    const persistent = await updateStoragePersistence(true, true);
    const diagnostics = await getCacheDiagnostics();
    window.alert(
      `${persistent ? 'Persistent storage granted.' : 'Persistent storage was not granted.'}\n\n${cacheDiagnosticText(diagnostics)}`
    );
  });

  GM_registerMenuCommand('Перетестировать зеркала', async () => {
    state.mirror = null;
    state.mirrorCheckedAt = 0;
    saveTabSoon();
    await flushTabState();
    mirrorPromise = null;
    const a = [...document.querySelectorAll('figure.post__image a[href]')].find((x) => mediaUrl(x));
    if (a) await getPreferredMirror(mediaUrl(a));
    scheduleUI();
  });

  GM_registerMenuCommand('Очистить media-cache вкладки на этом зеркале', async () => {
    cancelDownloadRetries();
    queue.length = 0;
    queued.clear();
    analysisQueue.length = 0;
    analysisQueued.clear();
    analysisGeneration++;
    await Promise.allSettled([...activePreloads]);
    await withCacheLock(async () => {
      await caches.delete(cacheName);
      cache = await caches.open(cacheName);
    });
    lastCacheTouchAt = 0;
    await touchCurrentCache(true);
    cached.clear();
    recentDownloads.length = 0;
    failedMedia.clear();
    state.screamer = {};
    analysisResults.clear();
    saveTabSoon();
    await flushTabState();
    for (const [sk, badges] of screamerBadges) {
      for (const b of badges) if (b.isConnected) renderScreamerResult(b, null);
      refreshAttachmentRisk(sk);
    }
    discover(document);
  });

  GM_registerMenuCommand('Сбросить результаты screamer detector', async () => {
    analysisQueue.length = 0;
    analysisQueued.clear();
    analysisGeneration++;
    state.screamer = {};
    analysisResults.clear();
    saveTabSoon();
    await flushTabState();
    for (const [sk, badges] of screamerBadges) {
      for (const b of badges) if (b.isConnected) renderScreamerResult(b, null);
      refreshAttachmentRisk(sk);
      const a = [...document.querySelectorAll('figure.post__image a[href]')].find(
        (x) => x.dataset.tm2chMediaUrl && screamerKey(x.dataset.tm2chMediaUrl) === sk
      );
      if (a) {
        const url = a.dataset.tm2chMediaUrl;
        queueScreamerAnalysis(url, canonicalKey(url));
      }
    }
  });

  function start() {
    installPageStyles();
    installUI();
    discover(document);
    updateStoragePersistence(false).then((persistent) => {
      console.info(`[spokoyno] origin storage: ${persistent ? 'persistent' : storagePersistence}`);
    });
    reconcileCache('start', true).then(() => cleanupOrphanCaches());
    scheduleCommunityScan();
    new MutationObserver((rs) => {
      let communityChanged = false,
        attachmentsRemoved = false;
      for (const r of rs) {
        if (uiRoot?.contains(r.target) || r.target.closest?.('.tm2ch-screamer,#tm2ch-cache-viewer')) continue;
        if (r.target.closest?.('.post__message')) communityChanged = true;
        for (const n of r.addedNodes) {
          if (n.nodeType !== Node.ELEMENT_NODE) continue;
          discover(n);
          if (
            n.matches?.('.post,.post__message,.post-reply-link') ||
            n.querySelector?.('.post,.post__message,.post-reply-link')
          )
            communityChanged = true;
        }
        for (const n of r.removedNodes) {
          if (n.nodeType !== Node.ELEMENT_NODE) continue;
          if (
            n.matches?.('.post,.post__message,.post-reply-link') ||
            n.querySelector?.('.post,.post__message,.post-reply-link')
          )
            communityChanged = true;
          if (
            n.matches?.('figure.post__image,.post,a[data-tm2ch-media-url]') ||
            n.querySelector?.('figure.post__image,a[data-tm2ch-media-url]')
          )
            attachmentsRemoved = true;
        }
      }
      if (communityChanged) scheduleCommunityScan();
      if (attachmentsRemoved) scheduleDomPrune();
    }).observe(document.documentElement, { childList: true, subtree: true });
    window.addEventListener('pageshow', () => reconcileCache('pageshow', true));
    window.addEventListener('focus', () => reconcileCache('focus'));
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') reconcileCache('visible');
      else flushTabState();
    });
    window.addEventListener('pagehide', () => flushTabState());
    setInterval(cleanupOrphanCaches, CLEANUP_INTERVAL);
  }

  installClickHandler();
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
