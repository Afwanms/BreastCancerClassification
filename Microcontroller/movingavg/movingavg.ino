#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "MAX30105.h"
#include <arduinoFFT.h>
#include <math.h>
#include "decisiontree_model.h"

// === perangkat ===
LiquidCrystal_I2C lcd(0x27, 16, 2);
MAX30105 sensor;

// === konfigurasi sistem ===
#define MAX_RR_INTERVALS 512
float rrIntervals[MAX_RR_INTERVALS];
int rrCount = 0;

const long FINGER_THRESHOLD = 50000;
const int MIN_RR_INTERVAL = 300;
const int MAX_RR_INTERVAL = 2000;
unsigned long startTime = 0;
bool collecting = false;

// ====== Peak detection (moving average + threshold) ======
const int SMOOTH_N = 4;
long smoothBuf[SMOOTH_N];
int smoothIdx = 0;
long prevAvg = 0;

long peakThreshold = 10000;          // kalibrasi sesuai kondisi
unsigned long lastPeakTime = 0;

// === buffer FFT ===
float vReal[MAX_RR_INTERVALS];
float vImag[MAX_RR_INTERVALS];
ArduinoFFT<float> FFT(vReal, vImag, MAX_RR_INTERVALS, 1.0);

// (opsional) normalisasi fitur
float meanVals[2]  = {7.7733f, 20.1531f};
float scaleVals[2] = {9.4457f, 15.7324f};

// === model ===
Eloquent::ML::Port::DecisionTree classifier;

// === struktur fitur HRV ===
struct HRVFeatures { float pnn50; float hf; };

// === BPM rolling average ===
const int BPM_BUF = 8;          // banyaknya detak untuk rata-rata
float bpmBuf[BPM_BUF];
int bpmIdx = 0, bpmCount = 0;

void recordBPM(float bpm) {
  bpmBuf[bpmIdx] = bpm;
  bpmIdx = (bpmIdx + 1) % BPM_BUF;
  if (bpmCount < BPM_BUF) bpmCount++;
}
float getAvgBPM() {
  if (bpmCount == 0) return 0;
  float sum = 0;
  for (int i = 0; i < bpmCount; i++) sum += bpmBuf[i];
  return sum / bpmCount;
}

// === deklarasi ===
HRVFeatures calculateHRV();
int predictCancer(HRVFeatures hrv);
void displayResult(HRVFeatures hrv, int prediction);
void printMMSS(uint32_t ms);

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  lcd.init(); lcd.backlight();
  lcd.clear(); lcd.setCursor(0, 0); lcd.print("HRV Cancer");
  lcd.setCursor(0, 1); lcd.print("Detection");
  delay(1500);

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    lcd.clear(); lcd.print("Sensor Error!"); while (1);
  }
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeGreen(0);

  long seed = sensor.getIR();
  for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = seed;
  prevAvg = seed;
  lastPeakTime = 0;

  // init buffer bpm
  for (int i = 0; i < BPM_BUF; i++) bpmBuf[i] = 0;

  lcd.clear(); lcd.print("Place finger...");
}

void loop() {
  unsigned long now = millis();
  long irValue = sensor.getIR();

  if (irValue > FINGER_THRESHOLD) {
    if (!collecting) {
      collecting = true;
      rrCount = 0;
      bpmIdx = 0; bpmCount = 0;
      for (int i = 0; i < BPM_BUF; i++) bpmBuf[i] = 0;

      startTime = now; lastPeakTime = 0;
      for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = irValue;
      prevAvg = irValue;

      lcd.clear(); lcd.print("Start 5min...");
      Serial.println("Mulai akuisisi 5 menit...");
    }

    // Moving average
    smoothBuf[smoothIdx] = irValue;
    smoothIdx = (smoothIdx + 1) % SMOOTH_N;
    long avgValue = 0; for (int i=0;i<SMOOTH_N;i++) avgValue += smoothBuf[i];
    avgValue /= SMOOTH_N;

    // Deteksi puncak sederhana
    if (avgValue > peakThreshold && avgValue > prevAvg) {
      if (lastPeakTime == 0) {
        lastPeakTime = now;  // anchor
      } else {
        unsigned long peakInterval = now - lastPeakTime; // ms
        if (peakInterval > MIN_RR_INTERVAL && peakInterval < MAX_RR_INTERVAL) {
          if (rrCount < MAX_RR_INTERVALS) {
            rrIntervals[rrCount++] = (float)peakInterval;
            float bpmInst = 60000.0f / (float)peakInterval;

            // simpan ke rolling average
            recordBPM(bpmInst);
            float bpmAvg = getAvgBPM();

            // LCD: tampilkan BPM rata-rata
            lcd.clear();
            lcd.setCursor(0, 0); lcd.print("Collecting...  ");
            lcd.setCursor(0, 1); lcd.print("BPM:");
            lcd.print(bpmAvg, 0);
            lcd.print("  ");
            lcd.print(rrCount); lcd.print("/"); lcd.print(MAX_RR_INTERVALS);

            Serial.print("RR="); Serial.print(peakInterval);
            Serial.print(" ms | BPM_inst="); Serial.print(bpmInst,1);
            Serial.print(" | BPM_avg="); Serial.println(bpmAvg,1);
          }
          lastPeakTime = now;
        } else if (peakInterval >= MAX_RR_INTERVAL) {
          Serial.println("Artefak: interval terlalu panjang, re-anchor");
          lastPeakTime = now;
        }
      }
    }

    // Timer mm:ss
    lcd.setCursor(12, 0);
    printMMSS(now - startTime);

    // Selesai 5 menit
    if (collecting && (now - startTime) >= 5UL * 60UL * 1000UL) {
      collecting = false;
      lcd.clear(); lcd.setCursor(0,0); lcd.print("Analyzing...");

      HRVFeatures hrv = calculateHRV();
      Serial.println("\n=== HRV Features ===");
      Serial.print("pNN50: "); Serial.println(hrv.pnn50);
      Serial.print("HF: ");    Serial.println(hrv.hf);

      int prediction = predictCancer(hrv);
      displayResult(hrv, prediction);

      rrCount = 0;
      delay(15000);
      lcd.clear(); lcd.print("Place finger...");
    }

    prevAvg = avgValue;
  } else {
    // no finger
    lcd.setCursor(0,0); lcd.print("No finger       ");
    lcd.setCursor(0,1); lcd.print("detected        ");
    rrCount = 0; collecting = false;
    for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = irValue;
    prevAvg = irValue; lastPeakTime = 0;
    delay(100);
    return;
  }

  delay(5);
}

// helper waktu
void printMMSS(uint32_t ms) {
  uint32_t sec = ms / 1000UL;
  uint16_t m = sec / 60U;
  uint8_t  s = sec % 60U;
  char buf[6];
  snprintf(buf, sizeof(buf), "%u:%02u", m, s);
  lcd.print(buf);
}

// HRV (pNN50 + HF via FFT, zero-pad)
HRVFeatures calculateHRV() {
  HRVFeatures hrv;
  if (rrCount < 10) { hrv.pnn50 = 0; hrv.hf = 0; return hrv; }

  int nn50 = 0;
  for (int i = 1; i < rrCount; i++) {
    float diff = fabsf(rrIntervals[i] - rrIntervals[i - 1]);
    if (diff > 50.0f) nn50++;
  }
  hrv.pnn50 = 100.0f * ((float)nn50 / (float)(rrCount - 1));

  float totalTimeMs = 0; for (int i=0;i<rrCount;i++) totalTimeMs += rrIntervals[i];
  float totalTime = totalTimeMs / 1000.0f;
  float fs = (totalTime > 0) ? (rrCount / totalTime) : 1.0f;

  float meanRR = 0.0f; for (int i=0;i<rrCount;i++) meanRR += rrIntervals[i];
  meanRR /= rrCount;

  for (int i=0;i<rrCount;i++){ vReal[i] = rrIntervals[i] - meanRR; vImag[i] = 0.0f; }
  for (int i=rrCount;i<MAX_RR_INTERVALS;i++){ vReal[i] = 0.0f; vImag[i] = 0.0f; }

  FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();

  const int N = MAX_RR_INTERVALS;
  const float freqRes = fs / N;
  const int halfN = N / 2;

  float hfPower = 0.0f;
  for (int i=0;i<halfN;i++){
    float f = i * freqRes;
    if (f >= 0.15f && f <= 0.40f) {
      float mag = vReal[i];
      hfPower += mag * mag;
    }
  }
  hrv.hf = hfPower;

  Serial.print("fs(auto): "); Serial.print(fs,3); Serial.println(" Hz");
  return hrv;
}

// prediksi
int predictCancer(HRVFeatures hrv) {
  float raw[2] = {hrv.pnn50, hrv.hf};
  float input[2];
  for (int i = 0; i < 2; i++) input[i] = (raw[i] - meanVals[i]) / scaleVals[i];
  return (int)classifier.predict(input);
}

// tampil hasil
void displayResult(HRVFeatures hrv, int prediction) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(prediction == 0 ? "Result: NORMAL" : "Result: CANCER");
  lcd.setCursor(0, 1);
  lcd.print("p:");  lcd.print((int)(hrv.pnn50 + 0.5)); lcd.print("% ");
  lcd.print("HF:"); lcd.print((int)(hrv.hf + 0.5));

  Serial.println("\n=== PREDICTION ===");
  Serial.println(prediction == 0 ? "NORMAL" : "CANCER");
}
