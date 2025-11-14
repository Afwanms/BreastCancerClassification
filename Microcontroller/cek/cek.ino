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

// ===== VAR LIVE NN INTERVAL =====
unsigned long lastNN = 0;   // NN interval (ms) terakhir yang valid

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
struct HRVFeatures {
  float pnn50;
  float hf;
};
// ===== Hitung HRV: pNN50 + HF 0.15–0.40 Hz =====
HRVFeatures calculateHRV() {
  HRVFeatures hrv;
  hrv.pnn50 = 0.0f;
  hrv.hf    = 0.0f;

  if (rrCount < 10) return hrv;

  hrv.pnn50 = calculatePNN50(rrIntervals, rrCount);
  hrv.hf    = calculateHF(rrIntervals, rrCount);

  return hrv;
}


// ===== Klasifikasi =====
int predictCancer(const HRVFeatures& hrv) {
  float raw[2] = {hrv.pnn50, hrv.hf};
  float input[2];
  for (int i = 0; i < 2; i++)
    input[i] = (raw[i] - meanVals[i]) / scaleVals[i];
  return (int)classifier.predict(input); // 0=Normal, 1=Cancer
}


float calculatePNN50(const float* rr, int count) {
  if (count < 2) return 0.0f;

  int nn50 = 0;
  for (int i = 1; i < count; i++) {
    float d = fabsf(rr[i] - rr[i - 1]);  // selisih dua RR bertetangga (ms)
    if (d > 50.0f) nn50++;
  }

  return 100.0f * ((float)nn50 / (float)(count - 1));  // dalam %
}

float calculateHF(const float* rr, int count) {
  if (count < 10) return 0.0f;

  // 1) Hitung fs "efektif" dari RR (Hz)
  float totalTimeMs = 0.0f;
  for (int i = 0; i < count; i++) totalTimeMs += rr[i];
  float totalTime_s = totalTimeMs / 1000.0f;
  float fs = (totalTime_s > 0.0f) ? (count / totalTime_s) : 1.0f;  // sampel per detik

  // 2) Buat tachogram RR (dalam detik, zero-mean)
  float meanRR_s = 0.0f;
  for (int i = 0; i < count; i++) {
    meanRR_s += rr[i] / 1000.0f;
  }
  meanRR_s /= count;

  for (int i = 0; i < count; i++) {
    float rr_s = rr[i] / 1000.0f;   // ms -> s
    vReal[i] = rr_s - meanRR_s;    // zero-mean (detik)
    vImag[i] = 0.0f;
  }
  // zero-pad sampai N FFT
  for (int i = count; i < MAX_RR_INTERVALS; i++) {
    vReal[i] = 0.0f;
    vImag[i] = 0.0f;
  }

  // 3) FFT
  FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();

  const int   N      = MAX_RR_INTERVALS;
  const float Nf     = (float)N;
  const float df     = fs / Nf;       // resolusi frekuensi
  const int   halfN  = N / 2;

  float hf_s2 = 0.0f;      // HF power dalam satuan s^2

  for (int k = 1; k < halfN; k++) {   // skip DC (k=0)
    float f = k * df;
    if (f >= 0.15f && f <= 0.40f) {   // band HF
      float mag = vReal[k];          // |X(k)| dari FFT

      // Periodogram approx:
      // PSD(k) ≈ |X(k)|^2 / (N * fs)   => s^2/Hz
      float psd = (mag * mag) / (Nf * fs);

      // Power band = PSD * delta_f  => s^2
      float bandPower_s2 = psd * df;

      hf_s2 += bandPower_s2;
    }
  }

  // s^2 -> ms^2   (1 s = 1000 ms => 1 s^2 = 1e6 ms^2)
  float hf_ms2 = hf_s2 * 1e6f;

  return hf_ms2;
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

  // ==== deteksi beat + BPM avg + RR (NN interval) ====
  if (checkForBeat(irValue)) {
    long delta = now - lastBeat;     // ms antar beat (INI RR / NN interval)
    lastBeat = now;

    if (delta >= 300 && delta <= 2000) {   // filter detak tidak wajar
      lastNN = delta;  // simpan NN interval terakhir yang valid

      // hitung BPM dari RR
      beatsPerMinute = 60000.0f / (float)delta;

      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;

        int sum = 0, cnt = 0;
        for (byte x=0; x<RATE_SIZE; x++) { 
          sum += rates[x]; 
          if (rates[x] != 0) cnt++; 
        }
        beatAvg = (cnt > 0) ? (sum / cnt) : (int)beatsPerMinute;
      }

      // simpan RR ke buffer HRV
      if (rrCount < MAX_RR_INTERVALS) {
        rrIntervals[rrCount++] = (float)delta;
      }
    }
  }

  // ==== tampil live (BPM avg + NN interval terakhir) ====
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("BPM=");
  lcd.print(beatAvg);
  if (collecting) { 
    lcd.setCursor(10,0); 
    printMMSS(now - startTime);  // timer 5 menit
  }

  // ==== deteksi jari lepas ====
  if (irValue < FINGER_THRESHOLD) {
    if (collecting) {
      collecting = false;
    }
    // reset semua state agar benar-benar “fresh”
    rrCount = 0;
    lastBeat = 0;
    lastNN = 0;
    beatsPerMinute = 0;
    beatAvg = 0;
    rateSpot = 0;
    for (int i=0; i<RATE_SIZE; i++) rates[i] = 0;

    lcd.clear();
    lcd.setCursor(0,0); lcd.print("No finger       ");
    lcd.setCursor(0,1); lcd.print("detected        ");
    Serial.println("No finger. Reset state.");
    delay(80);
    return;                 // <<< penting: hentikan loop di sini
  }

  // ==== kendali sesi 5 menit ====
  if (irValue > FINGER_THRESHOLD && !collecting) {
    collecting = true;
    startTime  = now;
    rrCount    = 0;
    lastBeat   = 0;
    lastNN     = 0;
    for (int i=0;i<RATE_SIZE;i++) rates[i] = 0;
    rateSpot = 0; beatAvg = 0; beatsPerMinute = 0;
    Serial.println("Mulai akuisisi 5 menit...");
  }

  if (collecting && (now - startTime >= MEAS_MS)) {
    collecting = false;

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Analyzing...");

    uint32_t tStart = millis();

    // === hitung HRV ===
    HRVFeatures hrv = calculateHRV();
    // === prediksi ===
    int prediction = predictCancer(hrv);

    uint32_t tEnd = millis();
    uint32_t elapsed = tEnd - tStart; // waktu total klasifikasi

    // tampilkan hasil + waktu klasifikasi di Serial
    Serial.println("\n=== HASIL HRV & KLASIFIKASI ===");
    Serial.print("pNN50 = "); Serial.println(hrv.pnn50);
    Serial.print("HF    = "); Serial.println(hrv.hf);
    Serial.print("Pred  = "); Serial.println(prediction == 0 ? "NORMAL" : "CANCER");
    Serial.print("t     = "); Serial.print(elapsed); Serial.println(" ms");

    // tampil ke LCD: result dulu
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(prediction == 0 ? "Result: NORMAL" : "Result: CANCER");
    lcd.setCursor(0, 1);
    lcd.print("t=");
    lcd.print(elapsed);
    lcd.print("ms");
    delay(10000);

    // siap sesi berikutnya
    rrCount = 0; 
    lastBeat = 0;
    lastNN = 0;
    for (int i=0; i<RATE_SIZE; i++) rates[i] = 0;
    rateSpot = 0; beatAvg = 0; beatsPerMinute = 0;

    lcd.clear();
    lcd.print("Place finger...");
  }

  // debug stream sederhana ke Serial Plotter
  Serial.print(irValue);
  Serial.print(",");
  Serial.print(beatsPerMinute);
  Serial.print(",");
  Serial.print(beatAvg);
  Serial.print(",");
  Serial.println(lastNN);

  delay(10);
}