/*
 * uno_q_sound_trigger.ino
 * ---------------------------------------------------------------------------
 * Runs on the STM32U585 MCU of the Arduino UNO Q.
 *
 * Job: be the always-on, ultra-cheap "ear". Continuously watch a MAX4466
 * electret mic amplifier and, when a sustained loud sound is detected, pulse a
 * GPIO line HIGH so the Linux side (Qualcomm QRB2210) knows it should record
 * and classify.
 *
 * Signal chain
 *   MAX4466 OUT -> A0 (ADC)
 *   -> sample at 4 kHz
 *   -> per 20 ms frame: peak-to-peak amplitude (DC bias of the MAX4466 is
 *      ~VCC/2, so raw ADC values are useless; the swing around the bias is
 *      what represents loudness)
 *   -> moving average over the last 8 frames (160 ms) to reject clicks/pops
 *   -> Schmitt trigger (separate ON/OFF thresholds) + cooldown
 *   -> TRIGGER_PIN pulse HIGH for PULSE_MS
 *
 * Wiring
 *   MAX4466 VCC -> 3V3, GND -> GND, OUT -> A0
 *   TRIGGER_PIN (D2) -> the MPU-side GPIO line that python script watches
 *   (common ground between both ends is mandatory)
 *
 * IMPORTANT: on the UNO Q the header pins belong to the STM32. Check the UNO Q
 * pinout/schematic for a line that actually reaches an MPU GPIO. If none is
 * reachable, set USE_BRIDGE to 1 to notify the Linux side over the internal
 * Arduino Bridge (RPC) instead of a wire. See the notes in the README/reply.
 * ---------------------------------------------------------------------------
 */

#include <Arduino.h>

// Set to 1 to ALSO send an RPC notification over the internal MCU<->MPU link.
// Requires the Arduino_RouterBridge library (ships with Arduino App Lab).
#define USE_BRIDGE 0
// Set to 1 for tuning: prints the averaged level so you can pick thresholds.
#define DEBUG_SERIAL 0

#if USE_BRIDGE
  #include <Arduino_RouterBridge.h>
#endif

// ============================== CONFIGURATION ==============================
static const uint8_t  MIC_PIN          = A0;   // MAX4466 output
static const uint8_t  TRIGGER_PIN      = 2;    // wake line to the MPU

static const uint8_t  ADC_BITS         = 12;   // 0..4095
static const uint16_t ADC_MAX          = (1u << ADC_BITS) - 1u;

static const uint32_t SAMPLE_PERIOD_US = 250;  // 4 kHz sampling
static const uint16_t FRAME_SAMPLES    = 80;   // 80 samples = 20 ms per frame
static const uint8_t  AVG_FRAMES       = 8;    // moving average length (160 ms)

// Thresholds are the averaged peak-to-peak swing as a fraction of ADC full
// scale (0.0 - 1.0). Start here, then tune with DEBUG_SERIAL while making the
// sounds you actually want to detect. Also adjust the MAX4466 gain trimpot.
static const float    TRIGGER_ON       = 0.10f; // fire above this
static const float    TRIGGER_OFF      = 0.05f; // re-arm only below this

static const uint32_t PULSE_MS         = 50;    // HIGH pulse width
static const uint32_t COOLDOWN_MS      = 6000;  // > record(3 s) + inference time
static const uint32_t SETTLE_MS        = 500;   // let the mic bias settle at boot

// ================================= STATE ===================================
static float    ring[AVG_FRAMES];               // last N frame amplitudes
static uint8_t  ringIdx     = 0;
static uint8_t  ringFilled  = 0;                // avoid false trigger at boot

static uint16_t frameMin    = ADC_MAX;
static uint16_t frameMax    = 0;
static uint16_t frameCount  = 0;

static uint32_t nextSampleUs = 0;

static bool     armed        = true;            // Schmitt-trigger state
static uint32_t lastFireMs   = 0;
static bool     pulseActive  = false;
static uint32_t pulseStartMs = 0;

// ============================== HELPERS ====================================

// Average of the ring buffer (8 elements -> trivial cost, no drift unlike a
// running sum).
static float movingAverage() {
  float sum = 0.0f;
  for (uint8_t i = 0; i < AVG_FRAMES; i++) sum += ring[i];
  return sum / AVG_FRAMES;
}

// Raise the wake line (and optionally notify over the Bridge).
static void fireTrigger(float level) {
  digitalWrite(TRIGGER_PIN, HIGH);
  pulseActive  = true;
  pulseStartMs = millis();
  lastFireMs   = pulseStartMs;
  armed        = false;                         // must drop below OFF to re-arm
#if USE_BRIDGE
  Bridge.notify("sound_trigger", level);
#endif
#if DEBUG_SERIAL
  Serial.print("TRIGGER level=");
  Serial.println(level, 3);
#endif
  (void)level;
}

// Non-blocking pulse end: keeps the sampling loop running while HIGH.
static void servicePulse() {
  if (pulseActive && (millis() - pulseStartMs) >= PULSE_MS) {
    digitalWrite(TRIGGER_PIN, LOW);
    pulseActive = false;
  }
}

// Called once per completed 20 ms frame.
static void processFrame() {
  // Peak-to-peak swing, normalised 0..1. Immune to the DC bias.
  float amp = (float)(frameMax - frameMin) / (float)ADC_MAX;

  ring[ringIdx] = amp;
  ringIdx = (ringIdx + 1) % AVG_FRAMES;
  if (ringFilled < AVG_FRAMES) ringFilled++;

  const float avg = movingAverage();

#if DEBUG_SERIAL
  Serial.println(avg, 3);
#endif

  // Re-arm only when the sound has died away AND the cooldown has elapsed.
  // This gives the MPU time to finish recording/inference and prevents one
  // long sound from firing repeatedly.
  if (!armed && avg < TRIGGER_OFF && (millis() - lastFireMs) >= COOLDOWN_MS) {
    armed = true;
  }

  // Fire only once the filter window is full (no false trigger at power-up).
  if (armed && ringFilled >= AVG_FRAMES && avg > TRIGGER_ON) {
    fireTrigger(avg);
  }
}

// ================================ ARDUINO ==================================
void setup() {
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);

  analogReadResolution(ADC_BITS);

#if DEBUG_SERIAL
  Serial.begin(115200);
#endif
#if USE_BRIDGE
  Bridge.begin();
#endif

  for (uint8_t i = 0; i < AVG_FRAMES; i++) ring[i] = 0.0f;

  delay(SETTLE_MS);
  nextSampleUs = micros() + SAMPLE_PERIOD_US;
  lastFireMs   = millis() - COOLDOWN_MS;          // allow first trigger at once
}

void loop() {
  const uint32_t nowUs = micros();

  // If something stalled us for several sample periods, resync instead of
  // bursting to catch up (a burst would corrupt the frame statistics).
  if ((int32_t)(nowUs - nextSampleUs) > (int32_t)(4 * SAMPLE_PERIOD_US)) {
    nextSampleUs = nowUs;
  }

  // Wrap-safe fixed-rate sampling using signed subtraction of micros().
  if ((int32_t)(nowUs - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_PERIOD_US;

    const uint16_t s = (uint16_t)analogRead(MIC_PIN);
    if (s < frameMin) frameMin = s;
    if (s > frameMax) frameMax = s;

    if (++frameCount >= FRAME_SAMPLES) {
      processFrame();
      frameMin   = ADC_MAX;
      frameMax   = 0;
      frameCount = 0;
    }
  }

  servicePulse();
}
