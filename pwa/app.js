let tfliteModel;
let audioContext;
let labels = [];
let isListening = false;
let activeUtterance = null;

const startBtn = document.getElementById('start-btn');
const statusText = document.getElementById('status');

// Load the CSV labels
async function loadLabels() {
    const response = await fetch('models/yamnet_class_map.csv');
    const text = await response.text();
    const lines = text.trim().split(/\r?\n/);
    labels = [];

    // Skip the header row (index,mid,display_name)
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        // Display name is after the second comma and may be wrapped in quotes
        const firstComma = line.indexOf(',');
        const secondComma = line.indexOf(',', firstComma + 1);
        if (firstComma !== -1 && secondComma !== -1) {
            let displayName = line.substring(secondComma + 1).trim();
            if (displayName.startsWith('"') && displayName.endsWith('"')) {
                displayName = displayName.substring(1, displayName.length - 1).trim();
            }
            labels.push(displayName);
        }
    }
}

// Initialize AI and Audio
async function initSystem() {
    statusText.innerText = "Loading Model...";
    startBtn.disabled = true;

    try {
        // Point the TFLite runtime to the absolute URL of local WASM binaries
        if (typeof tflite !== 'undefined' && tflite.setWasmPath) {
            const wasmBase = new URL('tflite/', window.location.href).href;
            tflite.setWasmPath(wasmBase);
        }

        // Load the TFLite model from the local models folder
        tfliteModel = await tflite.loadTFLiteModel('models/yamnet.tflite');
        await loadLabels();

        statusText.innerText = "Model Loaded. Ready to hike.";
        startBtn.disabled = false;
    } catch (e) {
        statusText.innerText = "Error loading model: " + (e.message || e);
        console.error("Failed to load model:", e);
    }
}

// Speak the classification
function speak(text) {
    if (!('speechSynthesis' in window)) return;

    try {
        window.speechSynthesis.cancel();
        activeUtterance = new SpeechSynthesisUtterance(text);
        activeUtterance.rate = 1.05;
        activeUtterance.onend = () => { activeUtterance = null; };
        activeUtterance.onerror = (e) => {
            console.warn("TTS warning:", e);
            activeUtterance = null;
        };
        window.speechSynthesis.speak(activeUtterance);
    } catch (e) {
        console.error("TTS execution error:", e);
    }
}

// Resample arbitrary audio stream to strictly 16,000 Hz for YAMNet
function resampleTo16k(audioBuffer, fromSampleRate) {
    if (fromSampleRate === 16000) return audioBuffer;
    const ratio = fromSampleRate / 16000;
    const targetLength = Math.round(audioBuffer.length / ratio);
    const result = new Float32Array(targetLength);
    for (let i = 0; i < targetLength; i++) {
        const srcPos = i * ratio;
        const low = Math.floor(srcPos);
        const high = Math.min(low + 1, audioBuffer.length - 1);
        const weight = srcPos - low;
        result[i] = audioBuffer[low] * (1 - weight) + audioBuffer[high] * weight;
    }
    return result;
}

// Microphone capture and inference loop
async function startListening() {
    isListening = true;
    startBtn.classList.add('hidden');
    statusText.innerText = "Listening... Screen off, phone in pocket.";

    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: true
            },
            video: false
        });

        const source = audioContext.createMediaStreamSource(stream);

        // Buffer size 4096 gives low-latency slicing
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        // Zero-gain node keeps ScriptProcessor alive without acoustic feedback
        const muteNode = audioContext.createGain();
        muteNode.gain.setValueAtTime(0, audioContext.currentTime);

        source.connect(processor);
        processor.connect(muteNode);
        muteNode.connect(audioContext.destination);

        let audio16kBuffer = new Float32Array(0);
        let isPredicting = false;
        let lastSpokenClass = "";
        let lastSpokenTime = 0;

        processor.onaudioprocess = (e) => {
            if (!isListening) return;

            // Get channel audio data and resample to strict 16kHz
            const rawData = e.inputBuffer.getChannelData(0);
            const resampled16k = resampleTo16k(rawData, audioContext.sampleRate);

            // Accumulate 16kHz samples
            const updated = new Float32Array(audio16kBuffer.length + resampled16k.length);
            updated.set(audio16kBuffer);
            updated.set(resampled16k, audio16kBuffer.length);
            audio16kBuffer = updated;

            // Prevent buffer blowout if backgrounded
            if (audio16kBuffer.length > 48000) {
                audio16kBuffer = audio16kBuffer.slice(audio16kBuffer.length - 24000);
            }

            // YAMNet requires exactly 15,600 samples (0.975s at 16kHz)
            if (audio16kBuffer.length < 15600 || isPredicting) return;

            const waveform = audio16kBuffer.slice(0, 15600);
            // Slide forward by 8000 samples (0.5s hop for 50% overlap)
            audio16kBuffer = audio16kBuffer.slice(8000);

            isPredicting = true;
            try {
                const inputTensor = tf.tensor1d(waveform);
                const rawOutput = tfliteModel.predict(inputTensor);

                let scoresTensor;
                if (rawOutput instanceof tf.Tensor) {
                    scoresTensor = rawOutput;
                } else if (rawOutput && typeof rawOutput === 'object') {
                    scoresTensor = rawOutput['tower0/network/layer32/final_output']
                        || rawOutput['Identity']
                        || Object.values(rawOutput)[0];
                }

                if (!scoresTensor) {
                    tf.dispose([inputTensor, rawOutput]);
                    return;
                }

                const scores = scoresTensor.dataSync();

                // Find highest probability
                let maxScore = -1;
                let maxIndex = -1;
                for (let i = 0; i < scores.length; i++) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i];
                        maxIndex = i;
                    }
                }

                if (maxIndex >= 0 && maxIndex < labels.length) {
                    const className = labels[maxIndex];
                    const percent = Math.round(maxScore * 100);

                    // Update minimal status
                    statusText.innerText = `Listening... Heard: ${className} (${percent}%)`;
                    console.log(`Detected: ${className} (${maxScore.toFixed(2)})`);

                    // Filter out background silence/noise
                    const IGNORE = ["Silence", "Background noise", "Inside, small room", "Sound", "Noise"];
                    const isIgnored = IGNORE.some(term => className.toLowerCase().includes(term.toLowerCase()));

                    // YAMNet 521-class detection confidence threshold
                    const now = Date.now();
                    if (maxScore >= 0.25 && !isIgnored) {
                        // Prevent rapid repetition of the same sound within 3 seconds
                        if (className !== lastSpokenClass || now - lastSpokenTime > 3000) {
                            lastSpokenClass = className;
                            lastSpokenTime = now;
                            speak(`I heard ${className}`);
                        }
                    }
                }

                // Cleanup memory
                inputTensor.dispose();
                if (rawOutput instanceof tf.Tensor) {
                    rawOutput.dispose();
                } else if (rawOutput && typeof rawOutput === 'object') {
                    Object.values(rawOutput).forEach(t => t?.dispose?.());
                }
            } catch (err) {
                console.error("Inference execution error:", err);
            } finally {
                isPredicting = false;
            }
        };
    } catch (err) {
        statusText.innerText = "Audio Error: " + (err.message || err);
        startBtn.classList.remove('hidden');
        console.error("Audio initialization failed:", err);
    }
}

startBtn.addEventListener('click', async () => {
    // Immediate audio feedback to unlock speech synthesis on user interaction
    speak("EdgeEcho active. Listening.");
    await startListening();
});

// Boot sequence
initSystem();