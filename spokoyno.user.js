// ==UserScript==
// @name         Spokoyno — 2ch WebM Companion
// @namespace    local.spokoyno
// @version      6.0.0
// @description  Tab-local video cache, fastest mirror, speed monitor and local ML screamer scores
// @updateURL    https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js
// @downloadURL  https://raw.githubusercontent.com/godlikedh/spokoyno/main/spokoyno.user.js
// @match        https://2ch.org/*/res/*.html*
// @match        https://2ch.su/*/res/*.html*
// @match        https://2ch.life/*/res/*.html*
// @connect      2ch.org
// @connect      2ch.su
// @connect      2ch.life
// @require      https://raw.githubusercontent.com/godlikedh/spokoyno/9e292bbfbf5858b17bfab9a95a3d5ba627183489/ml/assets/runtime.js#sha256=92a8808cc74884af032f8102146f8e104d5e111def6db41cffc86e42a5cf6e03
// @resource     spokoynoMLRuntime https://raw.githubusercontent.com/godlikedh/spokoyno/9e292bbfbf5858b17bfab9a95a3d5ba627183489/ml/assets/runtime.js#sha256=92a8808cc74884af032f8102146f8e104d5e111def6db41cffc86e42a5cf6e03
// @resource     spokoynoMLModel https://raw.githubusercontent.com/godlikedh/spokoyno/9e292bbfbf5858b17bfab9a95a3d5ba627183489/ml/assets/model.json#sha256=8620f9905d0f97477f1bc0c5a29ac59711765c3b73d3ef21c57de94739f116cd
// @resource     spokoynoMLWeights https://raw.githubusercontent.com/godlikedh/spokoyno/9e292bbfbf5858b17bfab9a95a3d5ba627183489/ml/assets/weights.bin.b64#sha256=8feb55dc5864c60f83d5b3a6382af6b31052e2b2879d2a4aba7c1a3e280f4ca5
// @resource     spokoynoMLWasm https://raw.githubusercontent.com/godlikedh/spokoyno/9e292bbfbf5858b17bfab9a95a3d5ba627183489/ml/assets/tf.wasm.b64#sha256=d58d377303468920672e747d43487cde8180e15e9bc2121808e211d7d0b42a72
// @resource     spokoynoMLSimd https://raw.githubusercontent.com/godlikedh/spokoyno/9e292bbfbf5858b17bfab9a95a3d5ba627183489/ml/assets/tf-simd.wasm.b64#sha256=630134cad54a86cc834cbf6a2866577ed7e2dc0ffa4df44d464fd5aa5ae843b9
// @run-at       document-start
// @noframes
// @grant        GM_xmlhttpRequest
// @grant        GM_getTab
// @grant        GM_saveTab
// @grant        GM_getTabs
// @grant        GM_registerMenuCommand
// @grant        GM_getResourceText
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
  const ANALYSIS_VERSION = 11,
    ANALYSIS_TARGET_RATE = 16_000,
    ANALYSIS_WORKER_TIMEOUT = 120_000,
    SCREAMER_MAYBE_CONFIDENCE = 0.6,
    SCREAMER_CONFIDENCE = 0.8;
  const ML_RESOURCE_HASHES = {
    spokoynoMLRuntime: '92a8808cc74884af032f8102146f8e104d5e111def6db41cffc86e42a5cf6e03',
    spokoynoMLModel: '8620f9905d0f97477f1bc0c5a29ac59711765c3b73d3ef21c57de94739f116cd',
    spokoynoMLWeights: '8feb55dc5864c60f83d5b3a6382af6b31052e2b2879d2a4aba7c1a3e280f4ca5',
    spokoynoMLWasm: 'd58d377303468920672e747d43487cde8180e15e9bc2121808e211d7d0b42a72',
    spokoynoMLSimd: '630134cad54a86cc834cbf6a2866577ed7e2dc0ffa4df44d464fd5aa5ae843b9'
  };

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
  const analysisQueued = new Map(),
    analysisQueue = [],
    activeAnalyses = new Map(),
    analysisWorkers = new Map();
  let analysisWorkerUrl = null,
    analysisWorkersUnavailable = false,
    mlResourcesPromise = null,
    mlRuntimeSource = '',
    mainMLPromise = null,
    mainMLChain = Promise.resolve();
  const screamerBadges = new Map(),
    communityBadges = new Map(),
    communityReports = new Map(),
    attachmentFigures = new Map();

  let analysisGeneration = 0,
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
      analysisQueued: analysisQueue.length,
      analysisActive: activeAnalyses.size,
      analysisConcurrency: CONCURRENCY,
      analysisWorkers: analysisWorkers.size,
      analysisMode: analysisWorkersUnavailable || typeof Worker !== 'function' ? 'main-thread fallback' : 'worker pool',
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
      `Audio analyses queued / active / limit: ${d.analysisQueued} / ${d.analysisActive} / ${d.analysisConcurrency}`,
      `Audio computation: ${d.analysisMode}`,
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
    .tm2ch-screamer[data-state="maybe"]{font-weight:700;background:rgba(245,185,25,.28);color:#805600}
    .tm2ch-screamer[data-state="ok"]{background:rgba(40,150,60,.12)}
    .tm2ch-screamer[data-state="error"]{opacity:.65}
    .tm2ch-report{text-decoration:none!important}
    .tm2ch-report[data-state="bad"]{background:rgba(220,35,35,.30);color:#c22}
    .post__image.tm2ch-risk .post__file-preview,
    .post__image.tm2ch-risk video{outline:3px solid rgba(210,30,30,.88)!important;box-shadow:0 0 12px rgba(220,25,25,.58)!important}
    .post__image.tm2ch-risk-maybe .post__file-preview,
    .post__image.tm2ch-risk-maybe video{outline:3px solid rgba(225,165,15,.95)!important;box-shadow:0 0 12px rgba(225,165,15,.38)!important}
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
    if (r.status === 'model-error' || r.status === 'too-short') {
      badge.textContent = r.status === 'too-short' ? '❔ audio too short' : '❔ ML analysis failed';
      badge.title = `${r.message || 'The ML model could not analyze this audio'}. This is unknown, not safe. Use the screamer-results reset menu to retry.`;
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

    const score = screamerScore(r);
    const tier = screamerTier(score);
    if (tier === 'unknown') {
      badge.textContent = '❔ audio score unavailable';
      badge.title = 'No valid analysis score is available. This is unknown, not safe.';
      badge.dataset.state = 'error';
      expose();
      return;
    }
    const risk = formatScreamerScore(score);
    if (tier === 'alert') {
      badge.textContent = `⚠ SCREAMER · score ${risk} · @ ${formatTime(r.eventAt || 0)}`;
      badge.dataset.state = 'bad';
    } else if (tier === 'maybe') {
      badge.textContent = `⚠ MAYBE · score ${risk} · @ ${formatTime(r.eventAt || 0)}`;
      badge.dataset.state = 'maybe';
    } else {
      badge.textContent = `🔊 low risk · score ${risk}`;
      badge.dataset.state = 'ok';
    }
    badge.title = [
      `Screamer score: ${score} (ML evidence, not a probability)`,
      `Model: ${r.model || 'synthetic-mil-20pct'}`,
      `Yellow: ${SCREAMER_MAYBE_CONFIDENCE} ≤ score < ${SCREAMER_CONFIDENCE}; red: score ≥ ${SCREAMER_CONFIDENCE}`,
      ...(r.events || [])
        .slice(0, 3)
        .map((event) => `Possible event: ${formatTime(event.start_s)} · ${formatScreamerScore(event.score)}`),
      'Audio is analyzed locally. Event times are model estimates.'
    ].join('\n');
    expose();
  }

  function refreshAttachmentRisk(sk) {
    const score = screamerScore(analysisResults.get(sk));
    const tier = screamerTier(score);
    const modelRisk = tier === 'alert';
    const reportedRisk = communityReports.has(sk);
    for (const figure of attachmentFigures.get(sk) || []) {
      if (!figure.isConnected) continue;
      figure.classList.toggle('tm2ch-risk', modelRisk || reportedRisk);
      figure.classList.toggle('tm2ch-risk-maybe', !reportedRisk && tier === 'maybe');
      figure.dataset.tm2chModelRisk = String(modelRisk);
      figure.dataset.tm2chRiskTier = tier;
      figure.dataset.tm2chScore = score === null ? '' : String(score);
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

  // ML_INFERENCE_START
  function screamerScore(result) {
    if (result?.status !== 'ok') return null;
    const score = result.score ?? result.decisionScore;
    return Number.isFinite(score) && score >= 0 && score <= 1 ? score : null;
  }

  function screamerTier(score) {
    if (!Number.isFinite(score) || score < 0 || score > 1) return 'unknown';
    return score >= SCREAMER_CONFIDENCE ? 'alert' : score >= SCREAMER_MAYBE_CONFIDENCE ? 'maybe' : 'low';
  }

  function formatScreamerScore(score) {
    if (score > 0 && score < 0.001) return score.toExponential(2);
    // Keep threshold-adjacent values in their actual displayed tier.
    return (Math.floor(score * 1000) / 1000).toFixed(3);
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

  async function analyzeScreamer(blob, slot = null, isCurrent = () => true) {
    if (!isCurrent()) return null;
    const context = getDecoderContext();
    if (!context) return { status: 'unsupported' };

    let buf, encoded;
    try {
      encoded = await blob.arrayBuffer();
      if (!isCurrent()) return null;
      buf = await context.decodeAudioData(encoded);
      encoded = null;
      if (!isCurrent()) return null;
      buf = await resampleForAnalysis(buf);
    } catch (e) {
      if (!isCurrent()) return null;
      console.warn('[spokoyno screamer] decode failed:', e);
      return {
        status: (await nativeNoAudioProbe(blob)) === true ? 'no-audio' : 'decode-error'
      };
    }
    if (!isCurrent()) return null;
    return analyzeDecodedAudio(buf, slot, isCurrent);
  }

  async function loadMLResources() {
    if (!mlResourcesPromise) {
      mlResourcesPromise = (async () => {
        const values = {};
        for (const [name, expected] of Object.entries(ML_RESOURCE_HASHES)) {
          const value = GM_getResourceText(name);
          if (!value || (await SpokoynoML.digest(new TextEncoder().encode(value))) !== expected)
            throw new Error(`Missing or damaged model resource: ${name}; reinstall the userscript`);
          values[name] = value;
        }
        mlRuntimeSource = values.spokoynoMLRuntime;
        if (!mlRuntimeSource || !values.spokoynoMLModel) throw new Error('ML release assets are not configured');
        return {
          model: values.spokoynoMLModel,
          weights: values.spokoynoMLWeights,
          wasm: values.spokoynoMLWasm,
          simd: values.spokoynoMLSimd
        };
      })().catch((error) => {
        mlResourcesPromise = null;
        throw error;
      });
    }
    return mlResourcesPromise;
  }

  function makeAnalysisWorkerSource() {
    if (!mlRuntimeSource) throw new Error('ML assets have not loaded');
    return (
      mlRuntimeSource +
      `\n;let modelPromise;
      self.onmessage = async ({data}) => {
        try {
          if (!modelPromise) modelPromise = SpokoynoML.initialize(tf, data.resources);
          const detector = await modelPromise;
          const result = await detector.score(data.channels, data.sampleRate);
          self.postMessage({result});
        } catch (error) {
          self.postMessage({result:{status:'model-error',message:String(error?.message || error)}});
        }
      };`
    );
  }

  function getAnalysisWorker(slot) {
    if (slot === null || analysisWorkersUnavailable || typeof Worker !== 'function') return null;
    if (!analysisWorkers.has(slot)) {
      if (!analysisWorkerUrl)
        analysisWorkerUrl = URL.createObjectURL(new Blob([makeAnalysisWorkerSource()], { type: 'text/javascript' }));
      const worker = new Worker(analysisWorkerUrl, { name: `spokoyno-ml-${slot + 1}` });
      analysisWorkers.set(slot, { worker, cancel: null, initialized: false });
    }
    return analysisWorkers.get(slot);
  }

  function stopAnalysisWorker(slot) {
    const entry = analysisWorkers.get(slot);
    if (entry) {
      analysisWorkers.delete(slot);
      entry.worker.terminate();
      entry.cancel?.();
    }
    if (!analysisWorkers.size && analysisWorkerUrl) {
      URL.revokeObjectURL(analysisWorkerUrl);
      analysisWorkerUrl = null;
    }
  }

  async function analyzeAudioBuffer(buf, resources, isCurrent = () => true) {
    // A single fallback model is shared; serialize its work to bound memory on
    // browsers whose CSP blocks workers. It runs the same frozen ML pipeline.
    const run = mainMLChain.then(async () => {
      if (!isCurrent()) return null;
      mainMLPromise ||= SpokoynoML.initialize(tf, resources).catch((error) => {
        mainMLPromise = null;
        throw error;
      });
      const model = await mainMLPromise;
      if (!isCurrent()) return null;
      await yieldMain();
      const result = await model.score(
        Array.from({ length: Math.min(2, buf.numberOfChannels) }, (_, i) => buf.getChannelData(i)),
        buf.sampleRate
      );
      return isCurrent() ? result : null;
    });
    mainMLChain = run.catch(() => {});
    return run;
  }

  async function analyzeDecodedAudio(buf, slot, isCurrent = () => true) {
    if (!isCurrent()) return null;
    if (!buf?.numberOfChannels || !buf.length) return { status: 'no-audio' };
    let resources;
    try {
      resources = await loadMLResources();
    } catch (error) {
      return isCurrent() ? { status: 'model-error', message: String(error.message || error) } : null;
    }
    if (!isCurrent()) return null;
    try {
      const entry = getAnalysisWorker(slot);
      if (entry) {
        const channels = Array.from({ length: Math.min(2, buf.numberOfChannels) }, (_, i) =>
          buf.getChannelData(i).slice()
        );
        const result = await new Promise((resolve, reject) => {
          const worker = entry.worker;
          let settled = false;
          const finish = (value, error) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            worker.onmessage = worker.onerror = worker.onmessageerror = null;
            entry.cancel = null;
            if (error) reject(error);
            else resolve(value);
          };
          const timer = setTimeout(() => finish(null, new Error('ML worker timed out')), ANALYSIS_WORKER_TIMEOUT);
          entry.cancel = () => finish(null);
          worker.onmessage = ({ data }) =>
            data?.result ? finish(data.result) : finish(null, new Error('invalid ML worker response'));
          worker.onerror = (event) => {
            event.preventDefault?.();
            finish(null, new Error(event.message || 'ML worker unavailable'));
          };
          worker.onmessageerror = () => finish(null, new Error('ML worker message failed'));
          try {
            worker.postMessage(
              { sampleRate: buf.sampleRate, channels, resources: entry.initialized ? null : resources },
              channels.map((c) => c.buffer)
            );
            entry.initialized = true;
          } catch (error) {
            finish(null, error);
          }
        });
        if (result?.status === 'model-error') stopAnalysisWorker(slot);
        return isCurrent() ? result : null;
      }
    } catch (error) {
      stopAnalysisWorker(slot);
      analysisWorkersUnavailable = true;
      console.warn('[spokoyno ML] worker unavailable; using local main-thread ML:', error);
    }
    try {
      return await analyzeAudioBuffer(buf, resources, isCurrent);
    } catch (error) {
      return isCurrent() ? { status: 'model-error', message: String(error.message || error) } : null;
    }
  }

  function queueScreamerAnalysis(url, key) {
    const sk = screamerKey(url);
    if (analysisResults.has(sk) || analysisQueued.has(sk)) return;
    const item = { url, key, sk, generation: analysisGeneration };
    analysisQueued.set(sk, item);
    analysisQueue.push(item);
    runAnalysisQueue();
  }

  function isCurrentAnalysis(item) {
    return item.generation === analysisGeneration && analysisQueued.get(item.sk) === item;
  }

  async function runAnalysisItem(item, slot) {
    const isCurrent = () => isCurrentAnalysis(item);
    try {
      const analysisCache = cache,
        response = await analysisCache.match(item.key);
      if (!isCurrent()) return;
      if (!response) {
        cached.delete(item.key);
        if (seenMedia.has(item.key)) {
          enqueueMedia(item.url, item.key, true);
          pump();
        }
        return;
      }
      const blob = await response.blob();
      if (!isCurrent()) return;
      try {
        await identifyMediaBlob(blob, response.headers.get('Content-Type'), item.url);
      } catch (e) {
        // A reset may happen while waiting for the cache lock. Never let an old
        // analysis delete media belonging to the new generation.
        await withCacheLock(() => (isCurrent() ? analysisCache.delete(item.key) : undefined));
        if (!isCurrent()) return;
        cached.delete(item.key);
        if (seenMedia.has(item.key)) {
          enqueueMedia(item.url, item.key, true);
          pump();
        } else publishScreamerResult(item.sk, { status: 'media-error', message: String(e.message || e) });
        return;
      }
      if (!isCurrent()) return;
      const result = await analyzeScreamer(blob, slot, isCurrent);
      if (isCurrent() && result) publishScreamerResult(item.sk, result);
    } catch (e) {
      console.warn('[spokoyno screamer] analysis failed:', item.url, e);
      if (isCurrent()) publishScreamerResult(item.sk, { status: 'model-error', message: String(e.message || e) });
    } finally {
      if (analysisQueued.get(item.sk) === item) analysisQueued.delete(item.sk);
    }
  }

  function runAnalysisQueue() {
    while (activeAnalyses.size < CONCURRENCY && analysisQueue.length) {
      const item = analysisQueue.shift();
      if (!isCurrentAnalysis(item) || analysisResults.has(item.sk)) {
        if (analysisQueued.get(item.sk) === item) analysisQueued.delete(item.sk);
        continue;
      }
      let slot = 0;
      while (activeAnalyses.has(slot)) slot++;
      activeAnalyses.set(slot, item);
      runAnalysisItem(item, slot)
        .finally(() => {
          activeAnalyses.delete(slot);
          runAnalysisQueue();
          if (!activeAnalyses.size && !analysisQueue.length && tabStateDirty) void flushTabState();
          scheduleUI();
        })
        .catch((error) => console.warn('[spokoyno screamer] analysis queue completion failed:', error));
    }
    scheduleUI();
  }

  function resetAnalysisQueue() {
    analysisGeneration++;
    mlResourcesPromise = null;
    analysisQueue.length = 0;
    analysisQueued.clear();
    for (const slot of analysisWorkers.keys()) stopAnalysisWorker(slot);
    // Native decodes cannot be aborted. Keep their slots occupied until they
    // settle; their generation checks prevent work or publication afterward.
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
      for (const [slot, item] of activeAnalyses) {
        if (item.key !== key) continue;
        if (analysisQueued.get(item.sk) === item) analysisQueued.delete(item.sk);
        stopAnalysisWorker(slot);
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
      `${activeAnalyses.size || analysisQueue.length ? ` · audio ${activeAnalyses.size}/${CONCURRENCY} +${analysisQueue.length} queued` : ''}` +
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
    resetAnalysisQueue();
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
    resetAnalysisQueue();
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
