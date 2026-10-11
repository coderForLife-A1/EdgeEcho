import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import csv
import subprocess
import time
# Use RPi.GPIO if using standard Debian, or gpiod for newer kernels
import RPi.GPIO as GPIO 

# --- CONFIGURATION ---
MODEL_PATH = "models/yamnet.tflite"
LABELS_PATH = "models/yamnet_class_map.csv"
SAMPLE_RATE = 16000 # YAMNet strict requirement
DURATION = 1.0 # Record for 1 second
CONFIDENCE_THRESHOLD = 0.50 # Ignore sounds below this certainty
STM32_INTERRUPT_PIN = 17 # The GPIO pin receiving the HIGH signal from STM32

def load_labels(file_path):
    """Parses the CSV into a dictionary: {index: 'Class Name'}"""
    labels = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            labels[int(row[0])] = row[2]
    return labels

def record_audio():
    """Records 1 sec of 16kHz mono audio from the I2S mic."""
    print("Recording audio buffer...")
    # dtype must be float32, channels=1 (mono)
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait() # Block until recording finishes
    
    # YAMNet expects a 1D array, so flatten the 2D output from sounddevice
    return np.squeeze(audio_data)

def run_inference(interpreter, waveform):
    """Passes the waveform to YAMNet and returns the top class index and score."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # The input tensor expects a 1D float32 array
    interpreter.set_tensor(input_details[0]['index'], waveform)
    
    print("Running TFLite inference...")
    interpreter.invoke()
    
    # YAMNet returns an array of 521 probabilities
    scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Average the scores across the timeframes (YAMNet returns a 2D array of frames x classes)
    mean_scores = np.mean(scores, axis=0)
    
    top_class_index = np.argmax(mean_scores)
    top_score = mean_scores[top_class_index]
    
    return top_class_index, top_score

def speak(text):
    """Uses offline espeak-ng to output the classification."""
    print(f"Speaking: {text}")
    # -v en-us adjusts voice, -s 150 adjusts speed
    subprocess.run(['espeak-ng', '-v', 'en-us', '-s', '150', text])

def main():
    print("Loading TFLite Model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    labels = load_labels(LABELS_PATH)
    
    # Setup GPIO for the STM32 interrupt
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(STM32_INTERRUPT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    print(f"System Ready. Waiting for STM32 trigger on GPIO {STM32_INTERRUPT_PIN}...")
    
    try:
        while True:
            # Block and wait for a rising edge (voltage going HIGH) from the STM32
            GPIO.wait_for_edge(STM32_INTERRUPT_PIN, GPIO.RISING)
            
            print("Interrupt received from STM32! Waking up...")
            
            # Step 1: Capture audio
            waveform = record_audio()
            
            # Step 2: Classify
            class_idx, score = run_inference(interpreter, waveform)
            class_name = labels.get(class_idx, "Unknown")
            
            print(f"Classification: {class_name} (Confidence: {score:.2f})")
            
            # Step 3: Audio Feedback (Only if confidence is high enough)
            if score >= CONFIDENCE_THRESHOLD:
                # Filter out generic noise classes so it only speaks for interesting events
                if class_name not in ["Silence", "Background noise", "Inside, small room"]:
                     speak(f"I heard a {class_name}")
            
            # Script loops back and waits for the next STM32 interrupt to conserve power
            print("Task complete. Yielding back to STM32...")
            time.sleep(1) # Debounce delay
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()