// ==UserScript==
// @name         Spokoyno — 2ch WebM Companion
// @namespace    local.spokoyno
// @version      5.4.0
// @description  Tab-local video cache, fastest mirror, speed monitor and event-based screamer warning
// @match        https://2ch.org/*
// @match        https://2ch.su/*
// @match        https://2ch.life/*
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
// ==/UserScript==

(async () => {
  'use strict';

  const MIRRORS = ['2ch.org', '2ch.su', '2ch.life'];
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
    PERSIST_RETRY_INTERVAL = 30 * 24 * 60 * 60_000,
    MAX_DOWNLOAD_ATTEMPTS = 4,
    RETRY_BASE_DELAY = 2_000,
    RECENT_DOWNLOADS = 8;
  const CACHE_META_URL = `${location.origin}/__tm2ch_cache_meta_v1__`;
  const MEDIA_EXT = /\.(?:mp4|webm|m4v|mov|ogv)$/i;
  const SCREAMER_REPORT_RE = /scream|скрим/i;
  const ANALYSIS_VERSION = 3,
    ANALYSIS_WINDOW = 0.05,
    ANALYSIS_TARGET_RATE = 16_000;
  const BASELINE_WINDOWS = 60,
    BASELINE_GAP = 2,
    MIN_BASELINE_WINDOWS = 20,
    EVENT_WINDOWS = 6;
  const EVENT_LOOKAHEAD = 10,
    SCREAMER_CONFIDENCE = 0.8;

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
    running = 0,
    errors = 0;
  let decoderContext = null;

  const cached = new Set(),
    seen = new Set(),
    seenMedia = new Map(),
    queued = new Set(),
    queue = [];
  const activeDownloads = new Map(),
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
    storagePersistence = 'unknown';

  try {
    for (const r of await cache.keys()) if (r.url !== CACHE_META_URL) cached.add(r.url);
  } catch (e) {
    console.warn('[spokoyno] cache restore failed:', e);
  }

  const mediaPath = (url) => {
    const u = new URL(url, location.href);
    return u.pathname + u.search;
  };

  // This canonicalizes mirrors only inside the current page origin's CacheStorage.
  const canonicalKey = (url) => location.origin + '/__tm2ch_cache_v4__' + mediaPath(url);
  const screamerKey = (url) => mediaPath(url);
  const mirrorUrl = (domain, path) => `https://${domain}${path}`;

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
          if (shouldRequest) localStorage.setItem(key, String(Date.now()));
        } catch {
          shouldRequest = true;
        }
        if (shouldRequest) persistent = await storage.persist();
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
  }

  async function reconcileCache(reason = 'manual', force = false) {
    if (cacheReconcilePromise) return cacheReconcilePromise;
    const now = Date.now();
    if (!force && now - lastCacheReconcileAt < CACHE_RECONCILE_INTERVAL) return null;
    lastCacheReconcileAt = now;
    const generation = downloadGeneration;
    cacheReconcilePromise = (async () => {
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
    })()
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
    const results = await Promise.all(MIRRORS.map((d) => probeMirror(d, mediaPath(sampleUrl))));
    const ok = results.filter((x) => x.ok).sort((a, b) => a.score - b.score);
    const winner = ok[0]?.domain || (MIRRORS.includes(location.hostname) ? location.hostname : MIRRORS[0]);
    state.mirror = winner;
    state.mirrorCheckedAt = Date.now();
    await gmSaveTab(tab);
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

  function downloadBlob(domain, path, key, originalUrl) {
    return new Promise((resolve, reject) => {
      const speed = beginSpeedTracking(key, originalUrl, domain);
      GM_xmlhttpRequest({
        method: 'GET',
        url: mirrorUrl(domain, path),
        responseType: 'blob',
        timeout: DOWNLOAD_TIMEOUT,
        onprogress: (e) => updateSpeedTracking(speed, e),
        onload: (r) => {
          if (r.status < 200 || r.status >= 400) {
            cancelSpeedTracking(key);
            return reject(new Error(`${domain}: HTTP ${r.status}`));
          }
          const blob = r.response;
          if (!blob) {
            cancelSpeedTracking(key);
            return reject(new Error(`${domain}: empty response`));
          }
          finishSpeedTracking(key, blob.size, domain);
          resolve({
            blob,
            contentType: blob.type || parseContentType(r.responseHeaders),
            domain
          });
        },
        onerror: () => {
          cancelSpeedTracking(key);
          reject(new Error(`${domain}: network error`));
        },
        ontimeout: () => {
          cancelSpeedTracking(key);
          reject(new Error(`${domain}: timeout`));
        }
      });
    });
  }

  async function downloadFromBestMirror(url, key) {
    const path = mediaPath(url),
      preferred = await getPreferredMirror(url);
    const order = [preferred, ...MIRRORS.filter((x) => x !== preferred)];
    let lastError;
    for (const domain of order) {
      try {
        const r = await downloadBlob(domain, path, key, url);
        if (domain !== preferred) {
          state.mirror = domain;
          state.mirrorCheckedAt = 0;
          gmSaveTab(tab);
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
    badge.dataset.state = '';
    if (!r) {
      badge.textContent = '🔊 analyzing…';
      badge.title = 'Audio analysis is queued';
      return;
    }
    if (r.status === 'no-audio') {
      badge.textContent = '🔇 no audio';
      badge.title = 'No audio track was detected';
      badge.dataset.state = 'ok';
      return;
    }
    if (r.status === 'unsupported') {
      badge.textContent = '❔ audio analysis unavailable';
      badge.title = 'Web Audio decoding is unavailable in this browser';
      badge.dataset.state = 'error';
      return;
    }
    if (r.status === 'decode-error') {
      badge.textContent = '❔ audio analysis failed';
      badge.title =
        'The browser could play the video container but could not expose its complete audio track for offline analysis. This is unknown, not safe.';
      badge.dataset.state = 'error';
      return;
    }

    const risk = Math.round(r.confidence * 100);
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
      `Merged decision score: ${Math.round((r.decisionScore ?? 0) * 100)}/100`,
      `Strongest detector: ${r.detectionMode === 'loud-start' ? 'dangerous loud start' : 'quiet-to-loud transition'}`,
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
      `Transition path: ${Math.round(r.transitionConfidence * 100)}%`,
      `Loud-start path: ${Math.round(r.startConfidence * 100)}%`,
      `Start spectral flatness: ${r.startSpectralFlatness.toFixed(3)}`,
      `Start high-frequency brightness: ${r.startBrightnessDb.toFixed(1)} dB`,
      `Track median: ${r.medianDb.toFixed(1)} dB`,
      `Track peak: ${r.peakDb.toFixed(1)} dBFS`,
      r.reason
    ]
      .filter(Boolean)
      .join('\n');
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

  function renderCommunityReport(sk, reports) {
    let badge = communityBadges.get(sk);
    if (!badge?.isConnected) {
      const modelBadge = screamerBadges.get(sk);
      if (!modelBadge?.isConnected) return;
      badge = document.createElement('a');
      badge.className = 'tm2ch-screamer tm2ch-report';
      badge.dataset.state = 'bad';
      modelBadge.insertAdjacentElement('afterend', badge);
      communityBadges.set(sk, badge);
    }
    const count = reports.length;
    const label = `🚩 reported${count > 1 ? ` ×${count}` : ''}`;
    if (badge.textContent !== label) badge.textContent = label;
    badge.href = `#${reports[0].reportPost}`;
    badge.setAttribute('aria-label', `${count} unverified community screamer report${count === 1 ? '' : 's'}`);
    badge.title = [
      `Unverified community screamer report${count === 1 ? '' : 's'}: ${count}`,
      ...reports.map((r) => `Reply #${r.reportPost} → post #${r.targetPost}: “${r.text}”`),
      'Independent of the audio-analysis percentage; click to open the first reporting reply.'
    ].join('\n');
  }

  function scanCommunityReports() {
    const next = new Map();
    for (const post of document.querySelectorAll('.post[data-num]')) {
      const message = post.querySelector('.post__message');
      if (!message || !SCREAMER_REPORT_RE.test(message.textContent || '')) continue;
      const reportPost = post.dataset.num;
      const text = (message.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 180);
      for (const replyLink of message.querySelectorAll('.post-reply-link[data-num]')) {
        const targetPost = replyLink.dataset.num;
        if (!/^\d+$/.test(targetPost)) continue;
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

    for (const [sk, badge] of communityBadges) {
      if (next.has(sk)) continue;
      badge.remove();
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
      existing = screamerBadges.get(sk);
    if (existing?.isConnected) return;
    const b = document.createElement('span');
    b.className = 'tm2ch-screamer';
    b.dataset.sk = sk;
    anchor.insertAdjacentElement('afterend', b);
    screamerBadges.set(sk, b);
    renderScreamerResult(b, analysisResults.get(sk));
    if (communityReports.has(sk)) renderCommunityReport(sk, communityReports.get(sk));
    refreshAttachmentRisk(sk);
  }

  function publishScreamerResult(sk, r) {
    r.analysisVersion = ANALYSIS_VERSION;
    analysisResults.set(sk, r);
    state.screamer[sk] = r;
    gmSaveTab(tab).catch?.(() => {});
    const b = screamerBadges.get(sk);
    if (b?.isConnected) renderScreamerResult(b, r);
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
    const stride = Math.max(1, Math.floor(buf.sampleRate / ANALYSIS_TARGET_RATE));
    const effectiveRate = buf.sampleRate / stride;
    const windowFrames = Math.max(stride, Math.round((buf.sampleRate * ANALYSIS_WINDOW) / stride) * stride);
    const windowSeconds = windowFrames / buf.sampleRate,
      count = Math.ceil(buf.length / windowFrames);
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
        end = Math.min(buf.length, start + windowFrames);
      let rawSq = 0,
        weightedSq = 0,
        n = 0,
        peak = 0,
        clips = 0;
      for (let i = start; i < end; i += stride) {
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
    const re = new Float64Array(n),
      im = new Float64Array(n);
    for (let i = start; i < end; i++) {
      let v = 0;
      for (let ch = 0; ch < buf.numberOfChannels; ch++) v += buf.getChannelData(ch)[i];
      const j = i - start,
        hann = end - start > 1 ? 0.5 - 0.5 * Math.cos((2 * Math.PI * j) / (end - start - 1)) : 1;
      re[j] = (v / buf.numberOfChannels) * hann;
    }
    fft(re, im);
    const bins = Math.min(n / 2 + 1, Math.floor((7800 * n) / buf.sampleRate) + 1),
      out = new Float64Array(bins);
    let total = 0;
    let logTotal = 0;
    for (let i = 0; i < bins; i++) {
      const p = re[i] * re[i] + im[i] * im[i] + 1e-12;
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
    const n = Math.min(before.bins.length, current.bins.length);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = current.bins[i] - before.bins[i];
      if (d > 0) sum += d * d;
    }
    return Math.sqrt(sum);
  }

  function windowBrightness(buf, windowIndex, windowFrames, stride) {
    const start = windowIndex * windowFrames,
      end = Math.min(buf.length, start + windowFrames);
    let energy = 0,
      differenceEnergy = 0,
      previous = null,
      count = 0;
    for (let i = start; i < end; i += stride) {
      let value = 0;
      for (let ch = 0; ch < buf.numberOfChannels; ch++) value += buf.getChannelData(ch)[i];
      value /= buf.numberOfChannels;
      energy += value * value;
      if (previous !== null) {
        const difference = value - previous;
        differenceEnergy += difference * difference;
      }
      previous = value;
      count++;
    }
    if (count < 2 || energy <= 0) return -90;
    return dbPower(differenceEnergy / (count - 1) / (energy / count));
  }

  function eventDuration(levels, start, threshold, windowSeconds) {
    let end = start,
      misses = 0;
    while (end < levels.length && end < start + 60) {
      if (levels[end] >= threshold) misses = 0;
      else if (++misses >= 2) break;
      end++;
    }
    return Math.max(0, (end - start - Math.max(0, misses - 1)) * windowSeconds);
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
    const start = Math.max(0, eventFrame - Math.round(buf.sampleRate * 0.5));
    const end = Math.min(buf.length, eventFrame + Math.round(buf.sampleRate * 0.8));
    const fineFrames = Math.max(stride, Math.round((buf.sampleRate * 0.01) / stride) * stride);
    const channels = Array.from({ length: buf.numberOfChannels }, (_, ch) => buf.getChannelData(ch));
    const effectiveRate = buf.sampleRate / stride;
    const filters = channels.map(() => [makeHighPass(effectiveRate), makeHighShelf(effectiveRate)]);
    const levels = [];
    for (let frameStart = start; frameStart < end; frameStart += fineFrames) {
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
      levels.push(Math.max(-90, -0.691 + dbPower(square / Math.max(1, count))));
    }
    const eventAt = Math.floor((eventFrame - start) / fineFrames);
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

  function bytesContain(bytes, text, start = 0, limit = bytes.length) {
    const needle = Array.from(text, (c) => c.charCodeAt(0)),
      end = Math.min(bytes.length, limit) - needle.length;
    outer: for (let i = start; i <= end; i++) {
      for (let j = 0; j < needle.length; j++) if (bytes[i + j] !== needle[j]) continue outer;
      return true;
    }
    return false;
  }

  function ebmlVint(bytes, at, keepMarker = false, maxBytes = 8) {
    if (at >= bytes.length) return null;
    let mask = 0x80,
      length = 1;
    while (length <= maxBytes && !(bytes[at] & mask)) {
      mask >>= 1;
      length++;
    }
    if (length > maxBytes || at + length > bytes.length) return null;
    let value = keepMarker ? bytes[at] : bytes[at] & (mask - 1);
    for (let i = 1; i < length; i++) value = value * 256 + bytes[at + i];
    return { length, value };
  }

  function webmHasAudio(bytes) {
    const limit = Math.min(bytes.length, 2 * 1024 * 1024);
    for (let i = 0; i + 5 < limit; i++) {
      if (bytes[i] !== 0x16 || bytes[i + 1] !== 0x54 || bytes[i + 2] !== 0xae || bytes[i + 3] !== 0x6b) continue;
      const tracksSize = ebmlVint(bytes, i + 4);
      if (!tracksSize) continue;
      let at = i + 4 + tracksSize.length,
        end = Math.min(limit, at + tracksSize.value),
        sawTrack = false;
      while (at < end) {
        const id = ebmlVint(bytes, at, true, 4);
        if (!id) break;
        const size = ebmlVint(bytes, at + id.length);
        if (!size) break;
        const data = at + id.length + size.length,
          next = data + size.value;
        if (next > end || next <= data) break;
        if (id.value === 0xae) {
          sawTrack = true;
          for (let p = data; p < next; ) {
            const childId = ebmlVint(bytes, p, true, 4);
            if (!childId) break;
            const childSize = ebmlVint(bytes, p + childId.length);
            if (!childSize) break;
            const childData = p + childId.length + childSize.length,
              childEnd = childData + childSize.value;
            if (childEnd > next || childEnd <= childData) break;
            if (childId.value === 0x83) {
              let type = 0;
              for (let q = childData; q < childEnd; q++) type = type * 256 + bytes[q];
              if (type === 2) return true;
            }
            p = childEnd;
          }
        }
        at = next;
      }
      if (sawTrack) return false;
    }
    return null;
  }

  function sniffContainerAudio(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer),
      isMp4 = bytes.length >= 12 && bytesContain(bytes, 'ftyp', 0, 16);
    if (isMp4) {
      const view = new DataView(arrayBuffer);
      for (let at = 0; at + 8 <= bytes.length; ) {
        let size = view.getUint32(at),
          header = 8;
        const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
        if (size === 1 && at + 16 <= bytes.length) {
          size = Number(view.getBigUint64(at + 8));
          header = 16;
        } else if (size === 0) size = bytes.length - at;
        if (size < header || at + size > bytes.length) break;
        if (type === 'moov') return bytesContain(bytes, 'soun', at + header, at + size);
        at += size;
      }
      return null;
    }
    const isWebm =
      bytes.length >= 4 && bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3;
    if (isWebm) return webmHasAudio(bytes);
    return null;
  }

  async function analyzeScreamer(blob) {
    const DecoderCtx =
      window.AudioContext ||
      window.webkitAudioContext ||
      window.OfflineAudioContext ||
      window.webkitOfflineAudioContext;
    if (!DecoderCtx) return { status: 'unsupported' };

    let buf, encoded;
    try {
      encoded = await blob.arrayBuffer();
      if (sniffContainerAudio(encoded) === false) return { status: 'no-audio' };
      if (!decoderContext) {
        try {
          decoderContext = new DecoderCtx();
        } catch {
          decoderContext = new DecoderCtx(1, 1, 44100);
        }
      }
      buf = await decoderContext.decodeAudioData(encoded);
      encoded = null;
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
    const startLimit = Math.min(Math.max(0, levels.length - EVENT_WINDOWS), Math.round(2 / windowSeconds));
    let startBest = 0,
      startLevel = -90;
    for (let i = 0; i <= startLimit; i++) {
      const level = percentile(levels.slice(i, i + EVENT_WINDOWS), 0.25);
      if (level > startLevel) {
        startBest = i;
        startLevel = level;
      }
    }
    const startTarget = Math.max(-6, startLevel - 3);
    let startPreview = startBest;
    for (let i = 0; i <= startLimit; i++) {
      if (percentile(levels.slice(i, i + EVENT_WINDOWS), 0.25) >= startTarget) {
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
    for (let i = 0; i + EVENT_WINDOWS <= levels.length; i++) {
      const lo = Math.max(0, i - BASELINE_WINDOWS),
        hi = i - BASELINE_GAP;
      if (hi - lo < MIN_BASELINE_WINDOWS) continue;
      const history = levels.slice(lo, hi);
      const baseline = percentile(history, 0.5);
      const baselineMad = percentile(
        Array.from(history, (level) => Math.abs(level - baseline)),
        0.5
      );
      const event = percentile(levels.slice(i, i + EVENT_WINDOWS), 0.25);
      const jump = event - baseline;
      const envelope = sigmoid((event + 10.5) / 2.5) ** 0.9 * sigmoid((jump - 13) / 3.5) ** 1.25;
      if (envelope > bestEnvelope) {
        best = i;
        bestEnvelope = envelope;
        bestBaseline = baseline;
        bestEvent = event;
        bestJump = jump;
        bestBaselineMad = baselineMad;
      }
    }

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
    const eventNearClip = eventClips / Math.max(1, eventSamples),
      spectralFlux = onsetSpectralFlux(buf, best, timeline.windowFrames);
    const loudComponent = sigmoid((bestEvent + 10.5) / 2.5);
    const jumpComponent = sigmoid((bestJump - 13) / 3.5);
    const durationComponent = sigmoid((duration - 0.18) / 0.09);
    const quietComponent = sigmoid((-bestBaseline - 15) / 5);
    const clipComponent = sigmoid((eventNearClip - 0.005) / 0.012);
    const fluxComponent = sigmoid((spectralFlux - 0.22) / 0.08);
    let transitionConfidence = loudComponent ** 0.9 * jumpComponent ** 1.25 * durationComponent ** 0.65;
    transitionConfidence *= 0.87 + 0.1 * quietComponent + 0.03 * clipComponent;
    transitionConfidence *= 0.65 + 0.35 * fluxComponent;
    transitionConfidence = Math.max(0, Math.min(1, transitionConfidence));
    const transitionEligible = bestEvent >= -6 && bestJump >= 14 && duration >= 0.15;
    const transitionDecisionScore = transitionEligible ? transitionConfidence : 0;
    const startDecisionScore = startEligible ? startConfidence : 0;
    const decisionScore = Math.max(transitionDecisionScore, startDecisionScore);
    const suspicious = decisionScore >= SCREAMER_CONFIDENCE;
    const confidence = decisionScore;
    const detectionMode =
      startDecisionScore !== transitionDecisionScore
        ? startDecisionScore > transitionDecisionScore
          ? 'loud-start'
          : 'transition'
        : startConfidence > transitionConfidence
          ? 'loud-start'
          : 'transition';
    const startSuspicious = suspicious && detectionMode === 'loud-start';
    const transitionSuspicious = suspicious && detectionMode === 'transition';
    const median = percentile(levels, 0.5);
    const reason =
      detectionMode === 'loud-start'
        ? startSuspicious
          ? 'Sustained near-full-scale broadband or clipped audio begins before a reliable baseline exists'
          : 'Opening audio did not combine enough loudness, persistence, and broadband evidence'
        : transitionSuspicious
          ? 'Sustained near-full-scale event after a quieter baseline, with a changed onset spectrum'
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
      decisionScore,
      detectionMode,
      transitionConfidence,
      startConfidence,
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
      startAt: start * windowSeconds,
      startEventDb: startLevel,
      startDuration,
      startNearClipPct: startNearClip * 100,
      startSpectralFlatness,
      startBrightnessDb,
      medianDb: median,
      p10Db: percentile(levels, 0.1),
      p25Db: percentile(levels, 0.25),
      p75Db: percentile(levels, 0.75),
      p90Db: percentile(levels, 0.9),
      p95Db: percentile(levels, 0.95),
      p99Db: percentile(levels, 0.99),
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

  function queueScreamerAnalysis(url, key, blob = null) {
    const sk = screamerKey(url);
    if (analysisResults.has(sk) || analysisQueued.has(sk)) return;
    analysisQueued.add(sk);
    analysisQueue.push({ url, key, sk, blob, generation: analysisGeneration });
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
          let blob = item.blob;
          if (!blob) {
            const r = await cache.match(item.key);
            if (!r) continue;
            blob = await r.blob();
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
    }
  }

  function discover(root) {
    const links = [];
    if (root.matches?.('a[href]')) links.push(root);
    if (root.querySelectorAll) links.push(...root.querySelectorAll('a[href]'));
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

  function scheduleDownloadRetry(item, error) {
    const nextAttempt = item.attempt + 1;
    const message = String(error?.message || error);
    const permanent =
      /quota|QuotaExceeded/i.test(message) || (/HTTP 4\d\d/.test(message) && !/HTTP (?:408|429)/.test(message));
    if (permanent || nextAttempt >= MAX_DOWNLOAD_ATTEMPTS || item.generation !== downloadGeneration) return false;
    const delay = RETRY_BASE_DELAY * 2 ** item.attempt;
    console.warn(
      `[spokoyno] retrying ${filenameFromUrl(item.url)} in ${(delay / 1000).toFixed(0)} s ` +
        `(attempt ${nextAttempt + 1}/${MAX_DOWNLOAD_ATTEMPTS})`
    );
    const timer = setTimeout(() => {
      retryTimers.delete(item.key);
      if (item.generation !== downloadGeneration || cached.has(item.key)) {
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
      let retryScheduled = false;
      running++;
      preload(item)
        .catch((e) => {
          if (item.generation !== downloadGeneration) return;
          errors++;
          console.warn('[spokoyno] preload failed:', item.url, e);
          retryScheduled = scheduleDownloadRetry(item, e);
        })
        .finally(() => {
          if (item.generation === downloadGeneration && !retryScheduled) queued.delete(item.key);
          running--;
          scheduleUI();
          pump();
        });
    }
  }

  async function preload({ url, key, generation }) {
    if (generation !== downloadGeneration) return;
    const existing = await cache.match(key);
    if (existing) {
      cached.add(key);
      queueScreamerAnalysis(url, key);
      scheduleUI();
      return;
    }
    const { blob, contentType, domain } = await downloadFromBestMirror(url, key);
    if (generation !== downloadGeneration) return;
    const headers = new Headers();
    if (contentType) headers.set('Content-Type', contentType);
    headers.set('X-TM-2ch-Mirror', domain);
    headers.set('X-TM-2ch-Original-Path', mediaPath(url));
    await cache.put(key, new Response(blob, { status: 200, headers }));
    cached.add(key);
    touchCurrentCache().catch((e) => console.warn('[spokoyno] cache metadata touch failed:', e));
    queueScreamerAnalysis(url, key, blob);
    scheduleUI();
  }

  function installClickHandler() {
    window.addEventListener(
      'click',
      async (e) => {
        if (e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
        const a = e.target?.closest?.('a[href]');
        if (!a) return;
        const url = a.dataset.tm2chMediaUrl || mediaUrl(a);
        if (!url) return;
        const key = canonicalKey(url);
        if (!cached.has(key)) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        try {
          await showFromCache(url);
        } catch (err) {
          console.warn('[spokoyno] viewer:', err);
          location.href = url;
        }
      },
      true
    );
  }

  async function showFromCache(url) {
    const key = canonicalKey(url),
      r = await cache.match(key);
    if (!r) {
      cached.delete(key);
      enqueueMedia(url, key, true);
      pump();
      throw new Error('cache miss');
    }
    const blob = await r.blob(),
      objectUrl = URL.createObjectURL(blob);
    const old = document.getElementById('tm2ch-cache-viewer');
    if (old?._tmClose) old._tmClose();
    else old?.remove();
    const overlay = document.createElement('div');
    overlay.id = 'tm2ch-cache-viewer';
    overlay.style.cssText =
      'position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;background:rgba(0,0,0,.94);cursor:zoom-out';
    const media = document.createElement('video');
    media.controls = true;
    media.autoplay = true;
    media.loop = true;
    media.src = objectUrl;
    media.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;cursor:default';
    media.addEventListener('click', (e) => e.stopPropagation());
    let closed = false;
    const close = () => {
      if (closed) return;
      closed = true;
      document.removeEventListener('keydown', keydown, true);
      try {
        media.pause();
      } catch {}
      overlay.remove();
      URL.revokeObjectURL(objectUrl);
    };
    const keydown = (e) => {
      if (e.key === 'Escape') close();
    };
    overlay._tmClose = close;
    overlay.addEventListener('click', close);
    document.addEventListener('keydown', keydown, true);
    overlay.append(media);
    document.documentElement.append(overlay);
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
      `${errors ? ` · err ${errors}` : ''}`;
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
        states.map((x) => x?.tm2chMediaV4?.cacheName).filter((x) => typeof x === 'string' && x.startsWith(PREFIX))
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

  GM_registerMenuCommand('Проверить и восстановить media-cache', async () => {
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
    await gmSaveTab(tab);
    mirrorPromise = null;
    const a = [...document.querySelectorAll('a[href]')].find((x) => mediaUrl(x));
    if (a) await getPreferredMirror(mediaUrl(a));
    scheduleUI();
  });

  GM_registerMenuCommand('Очистить media-cache этой вкладки', async () => {
    cancelDownloadRetries();
    queue.length = 0;
    queued.clear();
    analysisQueue.length = 0;
    analysisQueued.clear();
    analysisGeneration++;
    await caches.delete(cacheName);
    cache = await caches.open(cacheName);
    lastCacheTouchAt = 0;
    await touchCurrentCache(true);
    cached.clear();
    recentDownloads.length = 0;
    errors = 0;
    state.screamer = {};
    analysisResults.clear();
    await gmSaveTab(tab);
    for (const [sk, b] of screamerBadges) {
      if (b.isConnected) renderScreamerResult(b, null);
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
    await gmSaveTab(tab);
    for (const [sk, b] of screamerBadges) {
      if (!b.isConnected) continue;
      renderScreamerResult(b, null);
      refreshAttachmentRisk(sk);
      const a = [...document.querySelectorAll('a[href]')].find(
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
    updateStoragePersistence(true).then((persistent) => {
      console.info(`[spokoyno] origin storage: ${persistent ? 'persistent' : storagePersistence}`);
    });
    reconcileCache('start', true).then(() => cleanupOrphanCaches());
    scheduleCommunityScan();
    new MutationObserver((rs) => {
      let communityChanged = false;
      for (const r of rs) {
        if (r.target.closest?.('.post')) communityChanged = true;
        for (const n of r.addedNodes) {
          if (n.nodeType !== Node.ELEMENT_NODE) continue;
          discover(n);
          if (n.matches?.('.post') || n.querySelector?.('.post')) communityChanged = true;
        }
        for (const n of r.removedNodes) {
          if (n.nodeType === Node.ELEMENT_NODE && (n.matches?.('.post') || n.querySelector?.('.post'))) {
            communityChanged = true;
          }
        }
      }
      if (communityChanged) scheduleCommunityScan();
    }).observe(document.documentElement, { childList: true, subtree: true });
    window.addEventListener('pageshow', () => reconcileCache('pageshow', true));
    window.addEventListener('focus', () => reconcileCache('focus'));
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') reconcileCache('visible');
    });
    setInterval(cleanupOrphanCaches, CLEANUP_INTERVAL);
  }

  installClickHandler();
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
