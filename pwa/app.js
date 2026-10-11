let tfliteModel;
let audioContext;
let labels = [];
let isListening = false;

const startBtn = document.getElementById('start-btn');
const statusText = document.getElementById('status');

// Load the CSV labels
async function loadLabels() {
    const response = await fetch('models/yamnet_class_map.csv');
    const text = await response.text();
    const rows = text.split('\n');
    labels = rows.slice(1).map(row => {
        const cols = row.split(',');
        return cols[2]; // The string name of the class
    });
}

// Initialize AI and Audio
async function initSystem() {
    statusText.innerText = "Loading Model...";
    startBtn.disabled = true;

    try {
        // Load the TFLite model from the local models folder
        tfliteModel = await tflite.loadTFLiteModel('models/yamnet.tflite');
        await loadLabels();

        statusText.innerText = "Model Loaded. Ready to hike.";
        startBtn.disabled = false;
    } catch (e) {
        statusText.innerText = "Error loading model. Check console.";
        console.error(e);
    }
}

// Speak the classification
function speak(text) {
    // Prevent it from constantly talking over itself
    if (window.speechSynthesis.speaking) return;

    const utterance = new SpeechSynthesisUtterance(`I heard ${text}`);
    utterance.rate = 1.1;
    window.speechSynthesis.speak(utterance);
}

// Microphone capture and inference loop
async function startListening() {
    isListening = true;
    startBtn.classList.add('hidden');
    statusText.innerText = "Listening... Screen off, phone in pocket.";

    // YAMNet requires exactly 16kHz
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const source = audioContext.createMediaStreamSource(stream);

    // Buffer size of 16384 (roughly 1 second of audio at 16kHz)
    const processor = audioContext.createScriptProcessor(16384, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (e) => {
        if (!isListening) return;

        // Get the raw float32 array of audio data
        const channelData = e.inputBuffer.getChannelData(0);

        // YAMNet expects a 1D tensor
        const inputTensor = tf.tensor1d(channelData);

        // Run inference
        const outputTensor = tfliteModel.predict(inputTensor);
        const scores = outputTensor.dataSync(); // Extract probabilities

        // Find highest probability
        let maxScore = -1;
        let maxIndex = -1;
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }

        const className = labels[maxIndex];

        // 0.5 is 50% confidence. Adjust as needed.
        if (maxScore > 0.50) {
            console.log(`Detected: ${className} (${maxScore.toFixed(2)})`);
            // Filter out boring stuff
            if (!className.includes("Silence") && !className.includes("Background")) {
                speak(className);
            }
        }

        // Cleanup tensors to prevent memory leak
        tf.dispose([inputTensor, outputTensor]);
    };
}

startBtn.addEventListener('click', async () => {
    // Audio contexts require a user gesture to start in modern browsers
    await startListening();
});

// Boot sequence
initSystem();