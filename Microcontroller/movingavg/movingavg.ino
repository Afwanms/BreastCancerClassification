#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "MAX30105.h"
#include "heartRate.h"          // tidak dipakai lagi untuk checkForBeat, tapi biarin tetap include kalau mau
#include <arduinoFFT.h>
#include "decisiontree_model.h" // hasil export micromlgen

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

float beatsPerMinute = 0;

// ====== Peak detection (moving average + threshold) ======
const int SMOOTH_N = 4;
long smoothBuf[SMOOTH_N];
int smoothIdx = 0;
long prevAvg = 0;

long peakThreshold = 10000;          // kalibrasi sesuai kondisi
unsigned long lastPeakTime = 0;
const unsigned long minPeakInterval = 600; // ms (~100 bpm max)

// === buffer FFT ===
float vReal[MAX_RR_INTERVALS];
float vImag[MAX_RR_INTERVALS];
ArduinoFFT<float> FFT(vReal, vImag, MAX_RR_INTERVALS, 1.0);

// (opsional) normalisasi fitur kalau model dilatih terstandarisasi
float meanVals[2]  = {7.7733, 20.1531};      // mean pNN50, HF
float scaleVals[2] = {9.4457, 15.7324};      // scale pNN50, HF

// === model ===
Eloquent::ML::Port::DecisionTree classifier;

// === struktur fitur HRV ===
struct HRVFeatures {
  float pnn50;
  float hf;
};

// === deklarasi fungsi ===
HRVFeatures calculateHRV();
int predictCancer(HRVFeatures hrv);
void displayResult(HRVFeatures hrv, int prediction);

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("HRV Cancer");
  lcd.setCursor(0, 1);
  lcd.print("Detection");
  delay(1500);

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    lcd.clear();
    lcd.print("Sensor Error!");
    while (1);
  }
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeGreen(0);

  // init smoother
  for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = 0;
  prevAvg = 0;
  lastPeakTime = 0;

  lcd.clear();
  lcd.print("Place finger...");
}

void loop() {
  unsigned long now = millis();
  long irValue = sensor.getIR();

  // deteksi jari
  if (irValue > FINGER_THRESHOLD) {
    if (!collecting) {
      collecting = true;
      rrCount = 0;
      startTime = now;
      lastPeakTime = 0;
      // reset smoother
      for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = irValue;
      prevAvg = irValue;

      lcd.clear();
      lcd.print("Start 5min...");
      Serial.println("Mulai akuisisi 5 menit...");
    }

    // ====== Moving average untuk meredam noise ======
    smoothBuf[smoothIdx] = irValue;
    smoothIdx = (smoothIdx + 1) % SMOOTH_N;

    long avgValue = 0;
    for (int i = 0; i < SMOOTH_N; i++) avgValue += smoothBuf[i];
    avgValue /= SMOOTH_N;

    // ====== Deteksi puncak sederhana (rising + threshold + refractory) ======
    if (avgValue > peakThreshold && avgValue > prevAvg) {
      if (lastPeakTime == 0) {
        lastPeakTime = now;  // puncak pertama sebagai anchor
      } else {
        unsigned long peakInterval = now - lastPeakTime; // ms
        if (peakInterval > MIN_RR_INTERVAL && peakInterval < MAX_RR_INTERVAL) {
          if (rrCount < MAX_RR_INTERVALS) {
            rrIntervals[rrCount++] = (float)peakInterval;

            beatsPerMinute = 60000.0f / (float)peakInterval;

            // LCD status singkat
            lcd.clear();
            lcd.setCursor(0, 0);
            lcd.print("Collecting...   ");
            lcd.setCursor(0, 1);
            lcd.print("HR:");
            lcd.print(beatsPerMinute, 0);
            lcd.print("  ");
            lcd.print(rrCount);
            lcd.print("/");
            lcd.print(MAX_RR_INTERVALS);
            Serial.print("RR[");
            Serial.print(rrCount);
            Serial.print("] = ");
            Serial.print(peakInterval);
            Serial.print(" ms | HR = ");
            Serial.println(beatsPerMinute);
          }
          lastPeakTime = now; // update anchor puncak
        } else if (peakInterval >= MAX_RR_INTERVAL) {
          // artefak: puncak terlewat / gerak besar
          Serial.println("Artefak terdeteksi, interval diabaikan.");
          lastPeakTime = now; // re-anchor biar tidak panjang terus
        }
      }
    }

    // tampilkan waktu berjalan (mm:ss)
    if (collecting) {
      unsigned long elapsed = now - startTime;
      int minutes = elapsed / 60000UL;
      int seconds = (elapsed / 1000UL) % 60;
      lcd.setCursor(12, 0);
      lcd.printf("%d:%02d", minutes, seconds);
    }

    // selesai 5 menit -> hitung HRV + prediksi
    if (collecting && (now - startTime) >= 5UL * 60UL * 1000UL) {
      collecting = false;
      lcd.clear();
      lcd.print("Analyzing...");

      HRVFeatures hrv = calculateHRV();

      Serial.println("\n=== HRV Features ===");
      Serial.print("pNN50: "); Serial.println(hrv.pnn50);
      Serial.print("HF: ");    Serial.println(hrv.hf);

      int prediction = predictCancer(hrv);
      displayResult(hrv, prediction);

      rrCount = 0; // siap sesi berikutnya
      delay(15000);
      lcd.clear();
      lcd.print("Place finger...");
    }

    prevAvg = avgValue;  // update slope referensi
  }
  else {
    // tidak ada jari
    lcd.setCursor(0, 0);
    lcd.print("No finger       ");
    lcd.setCursor(0, 1);
    lcd.print("detected        ");
    rrCount = 0;
    collecting = false;

    // reset smoother pelan2
    for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = irValue;
    prevAvg = irValue;
    lastPeakTime = 0;

    delay(100);
    return;
  }

  delay(5);
}

// === hitung HRV (pNN50 + HF dengan auto fs) ===
HRVFeatures calculateHRV() {
  HRVFeatures hrv;
  if (rrCount < 10) {
    hrv.pnn50 = 0;
    hrv.hf = 0;
    return hrv;
  }

  // --- 1. pNN50 ---
  int nn50 = 0;
  for (int i = 1; i < rrCount; i++) {
    float diff = fabs(rrIntervals[i] - rrIntervals[i - 1]);
    if (diff > 50) nn50++;
  }
  hrv.pnn50 = (float)nn50 / (rrCount - 1) * 100.0;

  // --- 2. fs aktual (Hz) dari total waktu ---
  float totalTimeMs = 0;
  for (int i = 0; i < rrCount; i++) totalTimeMs += rrIntervals[i];
  float totalTime = totalTimeMs / 1000.0f;
  float fs = rrCount / totalTime; // ~ detak/second → “sample rate” tachogram

  // --- 3. FFT pada tachogram (RR-mean) ---
  float meanRR = 0;
  for (int i = 0; i < rrCount; i++) meanRR += rrIntervals[i];
  meanRR /= rrCount;

  for (int i = 0; i < rrCount; i++) {
    vReal[i] = rrIntervals[i] - meanRR;
    vImag[i] = 0;
  }

  FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();

  float freqRes = fs / rrCount;
  float hfPower = 0;
  for (int i = 0; i < rrCount / 2; i++) {
    float freq = i * freqRes;
    if (freq >= 0.15 && freq <= 0.40) {
      // magnitude^2 / N^2 → perkiraan power (unit ~ ms^2)
      hfPower += (vReal[i] * vReal[i]) / (rrCount * rrCount);
    }
  }
  hrv.hf = hfPower;

  Serial.print("fs (auto): ");
  Serial.print(fs);
  Serial.println(" Hz");

  return hrv;
}

// === prediksi ===
int predictCancer(HRVFeatures hrv) {
  float input[2];
  float raw[2] = {hrv.pnn50, hrv.hf};

  // standarize kalau model dilatih dgn scaler
  for (int i = 0; i < 2; i++)
    input[i] = (raw[i] - meanVals[i]) / scaleVals[i];

  int pred = classifier.predict(input);
  return pred; // 0=Normal, 1=Cancer
}

// === tampilkan hasil ===
void displayResult(HRVFeatures hrv, int prediction) {
  lcd.clear();
  lcd.setCursor(0, 0);
  if (prediction == 0){
    lcd.print("Result: NORMAL");
  } else {
    lcd.print("Result: CANCER");
  }
  lcd.setCursor(0, 1);
  lcd.print("p:");
  lcd.print(hrv.pnn50, 0);
  lcd.print("% HF:");
  lcd.print(hrv.hf, 0);

  Serial.println("\n=== PREDICTION ===");
  if (prediction == 0){
    Serial.println("NORMAL");
  } else {
    Serial.println("CANCER");
  }
}