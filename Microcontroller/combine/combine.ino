#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"            // untuk checkForBeat()
#include <LiquidCrystal_I2C.h>
#include <arduinoFFT.h>
#include <math.h>
#include "decisiontree_model.h" 

// ===== Perangkat =====
LiquidCrystal_I2C lcd(0x27, 16, 2);
MAX30105 particleSensor;

// ===== BPM avg =====
const byte RATE_SIZE = 4;
byte  rates[RATE_SIZE] = {0};
byte  rateSpot = 0;
long  lastBeat = 0;
float beatsPerMinute = 0;
int   beatAvg = 0;

// ===== RR buffer untuk HRV =====
#define MAX_RR_INTERVALS 512
float rrIntervals[MAX_RR_INTERVALS];
int   rrCount = 0;

// ===== Sesi 5 menit =====
const unsigned long MEAS_MS = 5UL * 60UL * 1000UL;
unsigned long startTime = 0;
bool collecting = false;

// ===== Deteksi jari =====
const long FINGER_THRESHOLD = 50000;

// ===== FFT untuk HF =====
float vReal[MAX_RR_INTERVALS];
float vImag[MAX_RR_INTERVALS];
ArduinoFFT<float> FFT(vReal, vImag, MAX_RR_INTERVALS, 1.0f);

// ==== Scaler fitur (samakan dgn training) ====
float meanVals[2]  = {7.7733f, 20.1531f};    // mean pNN50, HF
float scaleVals[2] = {9.4457f, 15.7324f};    // scale pNN50, HF

// ==== Model ====
Eloquent::ML::Port::DecisionTree classifier;

// ===== Deklarasi =====
struct HRVFeatures { float pnn50; float hf; };
HRVFeatures calculateHRV();
void printMMSS(uint32_t ms);
int  predictCancer(const HRVFeatures& hrv);
void displayResult(const HRVFeatures& hrv, int prediction);
// (opsional) tampil fitur saja:
// void showHRV(const HRVFeatures& hrv);

void setup() {
  Serial.begin(115200);
  // >>> penting untuk ESP32:
  Wire.begin(21, 22);

  lcd.init(); lcd.backlight();
  lcd.clear(); lcd.setCursor(0,0); lcd.print("HRV Cancer");
  lcd.setCursor(0,1); lcd.print("Detection");
  delay(1200);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    lcd.clear(); lcd.print("Sensor Error!");
    while (1);
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);

  lcd.clear(); lcd.print("Place finger...");
  Serial.println("Place your index finger on the sensor.");
}

void loop() {
  unsigned long now = millis();
  long irValue = particleSensor.getIR();

  // ==== deteksi beat + BPM avg ====
  if (checkForBeat(irValue)) {
    long delta = now - lastBeat;     // ms antar beat
    lastBeat = now;

    if (delta >= 300 && delta <= 2000) {
      beatsPerMinute = 60.0f / (delta / 1000.0f);

      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;

        int sum = 0, cnt = 0;
        for (byte x=0; x<RATE_SIZE; x++) { sum += rates[x]; if (rates[x] != 0) cnt++; }
        beatAvg = (cnt > 0) ? (sum / cnt) : (int)beatsPerMinute;
      }

      if (rrCount < MAX_RR_INTERVALS) rrIntervals[rrCount++] = (float)delta;
    }
  }

  // ==== tampil live (BPM avg) ====
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("BPM=");
  lcd.print(beatAvg);
  if (collecting) { lcd.setCursor(10,0); printMMSS(now - startTime); }

  lcd.setCursor(0,1);
  if (irValue < FINGER_THRESHOLD) {
    lcd.print("No finger       ");
  } else {
    lcd.print("RR:"); lcd.print(rrCount); lcd.print("/"); lcd.print(MAX_RR_INTERVALS); lcd.print("   ");
  }

  // ==== kendali sesi 5 menit ====
  if (irValue > FINGER_THRESHOLD && !collecting) {
    collecting = true;
    startTime  = now;
    rrCount    = 0;
    lastBeat   = 0;
    for (int i=0;i<RATE_SIZE;i++) rates[i] = 0;
    rateSpot = 0; beatAvg = 0; beatsPerMinute = 0;
    Serial.println("Mulai akuisisi 5 menit...");
  }

  if (collecting && (now - startTime >= MEAS_MS)) {
    collecting = false;

    lcd.clear(); lcd.setCursor(0,0); lcd.print("Analyzing...");
    HRVFeatures hrv = calculateHRV();

    Serial.println("\n=== HRV Features ===");
    Serial.print("pNN50: "); Serial.println(hrv.pnn50, 2);
    Serial.print("HF   : "); Serial.println(hrv.hf, 2);

    // >>> panggil KLASIFIKASI (ini yang tadi belum dipanggil)
    int prediction = predictCancer(hrv);
    displayResult(hrv, prediction);

    // siap sesi berikutnya
    rrCount = 0; lastBeat = 0;
    for (int i=0;i<RATE_SIZE;i++) rates[i] = 0;
    rateSpot = 0; beatAvg = 0; beatsPerMinute = 0;

    delay(5000);
    lcd.clear(); lcd.print("Place finger...");
  }

  // debug
  Serial.println(irValue);

  delay(10);
}

// ===== Hitung HRV: pNN50 + HF 0.15–0.40 Hz =====
HRVFeatures calculateHRV() {
  HRVFeatures hrv; hrv.pnn50 = 0; hrv.hf = 0;
  if (rrCount < 10) return hrv;

  // pNN50
  int nn50 = 0;
  for (int i = 1; i < rrCount; i++) {
    float d = fabsf(rrIntervals[i] - rrIntervals[i - 1]);
    if (d > 50.0f) nn50++;
  }
  hrv.pnn50 = 100.0f * ((float)nn50 / (float)(rrCount - 1));

  // fs dari RR
  float totalTimeMs = 0;
  for (int i = 0; i < rrCount; i++) totalTimeMs += rrIntervals[i];
  float totalTime = totalTimeMs / 1000.0f;
  float fs = (totalTime > 0) ? (rrCount / totalTime) : 1.0f;

  // Tachogram (RR-mean) + zero-pad
  float meanRR = 0.0f; 
  for (int i=0; i<rrCount; i++) meanRR += rrIntervals[i];
  meanRR /= rrCount;

  for (int i=0; i<rrCount; i++){ vReal[i] = rrIntervals[i] - meanRR; vImag[i] = 0.0f; }
  for (int i=rrCount; i<MAX_RR_INTERVALS; i++){ vReal[i] = 0.0f; vImag[i] = 0.0f; }

  FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();

  // HF 0.15–0.40 Hz
  float hfPower = 0.0f;             // (fixed)
  const int N = MAX_RR_INTERVALS;   // panjang FFT
  const float freqRes = fs / N;
  const int halfN = N / 2;

  for (int i=0; i<halfN; i++){
    float f = i * freqRes;
    if (f >= 0.15f && f <= 0.40f) {
      float mag = vReal[i];
      hfPower += mag * mag;         // proxy power (tanpa normalisasi rumit)
    }
  }
  hrv.hf = hfPower;

  Serial.print("fs(auto): "); Serial.print(fs,3); Serial.println(" Hz");
  return hrv;
}

// ===== Klasifikasi =====
int predictCancer(const HRVFeatures& hrv) {
  float raw[2] = {hrv.pnn50, hrv.hf};
  float input[2];
  for (int i = 0; i < 2; i++) input[i] = (raw[i] - meanVals[i]) / scaleVals[i];
  return (int)classifier.predict(input); // 0=Normal, 1=Cancer
}

// ===== Tampilkan hasil ke LCD & Serial =====
void displayResult(const HRVFeatures& hrv, int prediction) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(prediction == 0 ? "Result: NORMAL" : "Result: CANCER");
  lcd.setCursor(0, 1);
  lcd.print("p:"); lcd.print((int)(hrv.pnn50 + 0.5)); lcd.print("% ");
  lcd.print("HF:"); lcd.print((int)(hrv.hf + 0.5));

  Serial.println("\n=== PREDICTION ===");
  Serial.println(prediction == 0 ? "NORMAL" : "CANCER");
  delay(10000);
}

// ===== Helper waktu mm:ss =====
void printMMSS(uint32_t ms) {
  uint32_t s = ms / 1000UL;
  uint16_t m = s / 60U;
  uint8_t  r = s % 60U;
  char buf[6];
  snprintf(buf, sizeof(buf), "%u:%02u", m, r);
  lcd.print(buf);
}
