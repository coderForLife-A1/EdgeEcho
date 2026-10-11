const CACHE_NAME = 'EdgeEcho-v3';
const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './app.js',
    './manifest.json',
    './models/yamnet.tflite',
    './models/yamnet_class_map.csv',
    './tflite/tflite_web_api_cc.js',
    './tflite/tflite_web_api_cc.wasm',
    './tflite/tflite_web_api_cc_simd.js',
    './tflite/tflite_web_api_cc_simd.wasm',
    // You must cache the CDN links or it won't load offline
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js'
];

self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(ASSETS_TO_CACHE))
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.map((key) => {
                    if (key !== CACHE_NAME) {
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // Return cached version if found, else fetch from network
                return response || fetch(event.request);
            })
    );
});