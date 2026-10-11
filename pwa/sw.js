const CACHE_NAME = 'EdgeEcho-v1';
const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './app.js',
    './manifest.json',
    './models/yamnet.tflite',
    './models/yamnet_class_map.csv',
    // You must cache the CDN links or it won't load offline
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(ASSETS_TO_CACHE))
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