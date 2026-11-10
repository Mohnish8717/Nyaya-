/* Nyaya service worker.
 *
 * The goal is narrow and deliberate: an installed app must open instantly and
 * open at all when the network is gone, and it must never serve a stale legal
 * answer as if it were current.
 *
 *   - the shell (HTML, CSS, JS, icons) is cached and served cache-first, so the
 *     second launch does not wait for the network;
 *   - navigations are network-first with a cached shell fallback, because a
 *     deploy must reach the user on the next load, not the one after;
 *   - anything under /api/ is never cached or intercepted. A cached turn would
 *     be a cached legal answer, and this app's whole claim is that answers come
 *     from the sources retrieved for *this* question.
 *
 * Bump VERSION when the shell changes; the old caches are dropped on activate.
 */
const VERSION = 'nyaya-shell-v7';

const SHELL = [
  '/',
  '/manifest.webmanifest',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION)
      // addAll is all-or-nothing; one missing file would leave no shell at all.
      .then((cache) => Promise.all(SHELL.map((url) => cache.add(url).catch(() => null))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== VERSION).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

function isApi(url) {
  return url.pathname.startsWith('/api/');
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  // Voice and turns are live by definition. Not cached, not intercepted.
  if (isApi(url)) return;

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put('/', copy)).catch(() => {});
          return response;
        })
        .catch(async () => {
          const cached = await caches.match('/', { ignoreSearch: true });
          return cached || new Response(
            '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">'
            // Same three materials as the app: warm ink, paper, one amber lamp.
            + '<body style="margin:0;background:#14100d;color:#f2eadc;font:16px/1.6 ui-sans-serif,system-ui;padding:32px">'
            + '<h1 style="font:500 24px/1.2 Georgia,serif;letter-spacing:-0.02em">Nyaya is offline</h1>'
            + '<p style="color:#c6b8a5;max-width:46ch">The conversation needs a connection. '
            + 'Your saved session is still on this device and will be here when you reconnect.</p></body>',
            { headers: { 'Content-Type': 'text/html; charset=utf-8' }, status: 503 },
          );
        }),
    );
    return;
  }

  // Match the full URL, query included. The page requests `?v=N`, so bumping N
  // in index.html is what makes a deploy reach a returning user; ignoring the
  // query here would pin them to an old stylesheet until the cache version was
  // bumped by hand.
  event.respondWith(
    caches.match(request).then((cached) => {
      const network = fetch(request)
        .then((response) => {
          if (response && response.ok) {
            const copy = response.clone();
            caches.open(VERSION).then((cache) => cache.put(request, copy)).catch(() => {});
          }
          return response;
        })
        .catch(() => cached);
      return cached || network;
    }),
  );
});
