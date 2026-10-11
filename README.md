# EdgeEcho 🌿🔊

> **An air-gapped, zero-screen audio companion that identifies wildlife and nature sounds completely offline.**

[![Open Source](https://img.shields.io/badge/Open%20Source-MIT-brightgreen.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-YAMNet%20(TFLite)-blue.svg)](https://www.kaggle.com/models/google/yamnet)
[![PWA](https://img.shields.io/badge/PWA-Offline%20Ready-success.svg)](sw.js)

---

## 💡 What is EdgeEcho?

**EdgeEcho** is a Progressive Web Application (PWA) built for the **Touch Grass** challenge. It turns your mobile device into a screen-free, intelligent nature companion.

Most outdoor apps force you to constantly pull out your phone, stare at a camera screen, or wait for cloud APIs to process images or audio—completely defeating the point of being outside. **EdgeEcho flips this paradigm:**

1. Tap **"Start Listening"**.
2. Lock your phone and put it in your pocket.
3. Walk, hike, and enjoy nature.

When EdgeEcho hears a bird, flowing water, wind, or wildlife, it runs local edge AI inference and uses offline Text-to-Speech to quietly tell you what you heard directly in your earbud. **The screen is the shortest part of the experience.**

---

## 🌲 Why Open Source & Open Innovation Matter Here

In the backcountry, closed-source cloud AI completely fails. EdgeEcho relies on open-weight models and open web standards for three non-negotiable reasons:

* **Zero Connectivity / Air-Gapped:** Cloud APIs require active LTE/5G. Deep in a trail or national park, there is no signal. EdgeEcho runs an open-weight **YAMNet (TFLite)** model directly inside the browser using WebAssembly and TensorFlow.js. It operates 100% offline in airplane mode.
* **Absolute Privacy:** Continuous microphone streaming to a corporate server is a massive privacy hazard. Because EdgeEcho processes audio entirely on-device, your location and audio buffers never leave your local device RAM.
* **Zero Cost & Free Access:** Closed APIs charge per minute of streaming audio. Open-weight models running on local hardware cost $0.00 to operate indefinitely.

---

## 🛠️ How It Works (System Architecture)

EdgeEcho leverages an offline-first PWA architecture powered by TensorFlow.js and Service Workers:

    [ Ambient Sound ] ──> [ Mobile Mic (16kHz) ] ──> [ Service Worker (Offline Cache) ]
                                                                │
                                                                ▼
    [ Offline Speech Output ] <── [ YAMNet TFLite Engine ] <── [ Audio Spectrogram Array ]

1. **Audio Ingestion:** Captures 1-second audio frames via the Web Audio API downsampled strictly to **16kHz mono**.
2. **On-Device Inference:** Passes the array into a local, quantized **YAMNet model** (3.7 MB MobileNetV1 architecture pre-trained on Google's AudioSet).
3. **Label Matching:** Cross-references the top score against a local `yamnet_class_map.csv` index covering **521 acoustic classes**.
4. **Non-Visual Feedback:** Filters out silence and background noise. If confidence exceeds 50%, it invokes the Web Speech API (`speechSynthesis`) to deliver hands-free audio alerts.

---

## 🚀 How to Run EdgeEcho Locally

Because modern browsers enforce strict security on microphone permissions (`getUserMedia`), you must serve the files via `HTTPS` or `localhost`.

### Prerequisites
* Python 3.x installed (or Node.js)
* A modern browser (Chrome, Safari, Firefox, Edge)

### 1. Clone the Repository
    git clone https://github.com/coderForLife-A1/edgeecho.git
    cd edgeecho

### 2. File Verification
Ensure the open-weight model and label map are present in the `/models` directory:
    edgeecho/
    ├── index.html
    ├── app.js
    ├── sw.js
    ├── manifest.json
    └── models/
        ├── yamnet.tflite
        └── yamnet_class_map.csv

### 3. Start a Local Server
    # Using Python
    python3 -m http.server 8000
    
    # OR using Node
    npx http-server -p 8000

### 4. Test the App
1. Open `http://localhost:8000` in your browser.
2. Open DevTools (`F12`) -> **Console** to monitor local logs.
3. Click **"Start Listening"** and grant microphone permissions.
4. Play a clip of bird songs or wildlife—EdgeEcho will announce the sound out loud.

---

## ✈️ Testing Full Offline Mode (Airplane Mode)

1. Deploy the code to **GitHub Pages** or access your local HTTPS build on your mobile device.
2. Open the page once while connected to the internet so the **Service Worker (`sw.js`)** caches the HTML, JS, and the 3.7MB model binary.
3. Turn on **Airplane Mode** (disable Wi-Fi and Cellular Data).
4. Refresh or reopen the page—EdgeEcho will load and run inference with zero network connectivity.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.