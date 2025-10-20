#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <arduinoFFT.h>
#include "decisiontree_model.h"   // hasil export micromlgen

// === perangkat ===
LiquidCrystal_I2C lcd(0x27, 16, 2);
MAX30105 sensor;

// === konfigurasi sistem ===
#define MAX_RR_INTERVALS 256
float rrIntervals[MAX_RR_INTERVALS];
int rrCount = 0;

const long FINGER_THRESHOLD = 50000;
const int MIN_RR_INTERVAL = 300;
const int MAX_RR_INTERVAL = 2000;

unsigned long lastBeat = 0;
float beatsPerMinute = 0;

// === buffer FFT ===
float vReal[MAX_RR_INTERVALS];
float vImag[MAX_RR_INTERVALS];
ArduinoFFT<float> FFT(vReal, vImag, MAX_RR_INTERVALS, 1.0);

// === scaling dari Python (contoh) ===
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

  lcd.clear();
  lcd.print("Place finger...");
}

void loop() {
  long irValue = sensor.getIR();
  Serial.println(irValue);
  // deteksi jari
  if (irValue < FINGER_THRESHOLD) {
    lcd.setCursor(0, 0);
    lcd.print("No finger       ");
    lcd.setCursor(0, 1);
    lcd.print("detected        ");
    rrCount = 0;
    delay(100);
    return;
  }

  // deteksi detak jantung
  if (checkForBeat(irValue)) {
    long now = millis();
    long delta = now - lastBeat;
    lastBeat = now;

    if (delta > MIN_RR_INTERVAL && delta < MAX_RR_INTERVAL) {
      if (rrCount < MAX_RR_INTERVALS)
        rrIntervals[rrCount++] = delta;

      beatsPerMinute = 60.0 / (delta / 1000.0);

      lcd.setCursor(0, 0);
      lcd.print("Collecting...   ");
      lcd.setCursor(0, 1);
      lcd.print("HR:");
      lcd.print(beatsPerMinute, 1);
      lcd.print(" ");
      lcd.print(rrCount);
      lcd.print("/");
      lcd.print(MAX_RR_INTERVALS);

      Serial.print("RR[");
      Serial.print(rrCount);
      Serial.print("] = ");
      Serial.print(delta);
      Serial.print(" ms | HR=");
      Serial.println(beatsPerMinute);
    }
  }

  // setelah data cukup
  if (rrCount >= MAX_RR_INTERVALS) {
    lcd.clear();
    lcd.print("Analyzing...");
    HRVFeatures hrv = calculateHRV();

    Serial.println("\n=== HRV Features ===");
    Serial.print("pNN50: "); Serial.println(hrv.pnn50);
    Serial.print("HF: "); Serial.println(hrv.hf);

    int prediction = predictCancer(hrv);
    displayResult(hrv, prediction);

    rrCount = 0;
    delay(5000);
    lcd.clear();
    lcd.print("Place finger...");
  }

  delay(10);
}

// === hitung HRV (pNN50 + HF dengan auto fs) ===
HRVFeatures calculateHRV() {
  HRVFeatures hrv;
  if (rrCount < 10) {
    hrv.pnn50 = 0;
    hrv.hf = 0;
    return hrv;
  }

  // --- 1. Hitung pNN50 ---
  int nn50 = 0;
  for (int i = 1; i < rrCount; i++) {
    float diff = fabs(rrIntervals[i] - rrIntervals[i - 1]);
    if (diff > 50) nn50++;
  }
  hrv.pnn50 = (float)nn50 / (rrCount - 1) * 100.0;

  // --- 2. Hitung sampling rate aktual ---
  float totalTime = 0;
  for (int i = 0; i < rrCount; i++) totalTime += rrIntervals[i];
  totalTime /= 1000.0;
  float fs = rrCount / totalTime; // sample rate aktual (Hz)

  // --- 3. Siapkan data FFT ---
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

  // --- 4. Hitung HF power (0.15–0.4 Hz) ---
  float freqRes = fs / rrCount;
  float hfPower = 0;
  for (int i = 0; i < rrCount / 2; i++) {
    float freq = i * freqRes;
    if (freq >= 0.15 && freq <= 0.4)
      hfPower += vReal[i];
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

  for (int i = 0; i < 2; i++)
    input[i] = (raw[i] - meanVals[i]) / scaleVals[i];

  int pred = classifier.predict(input);
  return pred; // 0=Normal, 1=Cancer
}

// === tampilkan hasil ===
void displayResult(HRVFeatures hrv, int prediction) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(prediction == 0 ? "Result: NORMAL " : "Result: CANCER ");
  lcd.setCursor(0, 1);
  lcd.print("p:");
  lcd.print(hrv.pnn50, 0);
  lcd.print("% HF:");
  lcd.print(hrv.hf, 0);

  Serial.println("\n=== PREDICTION ===");
  Serial.println(prediction == 0 ? "NORMAL" : "CANCER");
}
