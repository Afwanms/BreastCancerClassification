#include <Wire.h>
#include "MAX30105.h"
#include <LiquidCrystal_I2C.h>
#include "DecisionTree.h"
#include <math.h>

Eloquent::ML::Port::DecisionTree clf;
MAX30105 particleSensor;

// === LCD I2C ===
LiquidCrystal_I2C lcd(0x27, 16, 2);   // ganti 0x27 -> 0x3F jika modulmu pakai alamat lain

// === Buffer & Deteksi Puncak ===
const int bufferSize = 4;
long buffer[bufferSize];
int bufferIndex = 0;
long previousValue = 0;

long peakThreshold = 20000;           // kalibrasi sesuai kondisi
unsigned long lastPeakTime = 0;
unsigned long minPeakInterval = 600;  // ms (≈100 bpm)

const int nnBufferSize = 100;
unsigned long nnIntervals[nnBufferSize];
int nnIndex = 0;

// rate-limit tampilan
unsigned long lastLcd = 0;
const unsigned long lcdInterval = 200; // update setiap 200 ms

// === Deklarasi ===
float calculateAVNN();
float calculateSDNN(float avnn);
void displayResult(float avnn, float sdnn, int stressDetected, unsigned long compMs);

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");

  // LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Init MAX30105...");
  lcd.setCursor(0,1); lcd.print("Tunggu sebentar");

  // MAX30105
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30105 not found. Check wiring/power.");
    lcd.clear();
    lcd.setCursor(0,0); lcd.print("MAX30105 ERR");
    lcd.setCursor(0,1); lcd.print("Cek wiring!");
    while (1);
  }

  // --- Konfigurasi hemat arus & stabil ---
  byte ledBrightness = 0x07; // lebih rendah dari 0x1F
  byte sampleAverage = 4;
  byte ledMode = 2;          // IR only (lebih irit dari Red+IR)
  int  sampleRate = 50;      // 50 Hz cukup untuk HR/NN
  int  pulseWidth = 118;     // 118 atau 215
  int  adcRange = 2048;      // lebih kecil = irit
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);

  // Buffer init
  for (int i = 0; i < bufferSize; i++) buffer[i] = 0;
  for (int i = 0; i < nnBufferSize; i++) nnIntervals[i] = 0;

  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Ready");
  lcd.setCursor(0,1); lcd.print("Letakkan jari");
}

void loop() {
  long currentValue = particleSensor.getIR();
  unsigned long currentTime = millis();

  // moving average sederhana
  buffer[bufferIndex] = currentValue;
  bufferIndex = (bufferIndex + 1) % bufferSize;
  long avgValue = 0;
  for (int i = 0; i < bufferSize; i++) avgValue += buffer[i];
  avgValue /= bufferSize;

  // deteksi puncak
  if (currentValue > peakThreshold &&
      currentValue > previousValue &&
      (currentTime - lastPeakTime) > minPeakInterval) {

    if (lastPeakTime > 0) {
      unsigned long peakInterval = currentTime - lastPeakTime;

      // filter artefak (interval biologis wajar)
      if (peakInterval > 300 && peakInterval < 2000) {
        nnIntervals[nnIndex] = peakInterval;
        nnIndex = (nnIndex + 1) % nnBufferSize;

        // hitung fitur + klasifikasi
        unsigned long t0 = millis();
        float avnn = calculateAVNN();     // ms
        float sdnn = calculateSDNN(avnn); // ms
        float features[] = { avnn, sdnn };
        int stressDetected = clf.predict(features);
        unsigned long compMs = millis() - t0;

        displayResult(avnn, sdnn, stressDetected, compMs);
      } else {
        Serial.println("Artefak terdeteksi, interval diabaikan.");
      }
    }
    lastPeakTime = currentTime;
  }

  previousValue = avgValue;
  unsigned long peakInterval = currentTime - lastPeakTime;
  nnIntervals[nnIndex] = peakInterval;
  nnIndex = (nnIndex + 1) % nnBufferSize;
}

float calculateAVNN() {
  unsigned long total = 0; int count = 0;
  for (int i = 0; i < nnBufferSize; i++) {
    if (nnIntervals[i] > 0) { total += nnIntervals[i]; count++; }
  }
  if (count == 0) return 0;
  return (float)total / count;
}

float calculateSDNN(float avnn) {
  if (avnn <= 0) return 0;
  float variance = 0.0; int count = 0;
  for (int i = 0; i < nnBufferSize; i++) {
    if (nnIntervals[i] > 0) {
      float d = (float)nnIntervals[i] - avnn;
      variance += d * d; count++;
    }
  }
  if (count == 0) return 0;
  return sqrt(variance / count);
}

void displayResult(float avnn, float sdnn, int stressDetected, unsigned long compMs) {
  // debug serial
  Serial.print("AVNN: "); Serial.print(avnn); Serial.println(" ms");
  Serial.print("SDNN: "); Serial.print(sdnn); Serial.println(" ms");
  Serial.print("Comp: "); Serial.print(compMs); Serial.println(" ms");
  Serial.print("Status: "); Serial.println(stressDetected == 1 ? "Stres" : "Tidak Stres");

  // rate-limit update LCD & tanpa clear berulang
  if (millis() - lastLcd >= lcdInterval) {
    // Baris 1: AV & SD (dibulatkan agar muat)
    lcd.setCursor(0,0);
    lcd.print("AV:"); lcd.print((int)round(avnn));
    lcd.print(" SD:"); lcd.print((int)round(sdnn));
    // padding untuk nutup sisa karakter lama
    lcd.print("   ");

    // Baris 2: Status
    lcd.setCursor(0,1);
    if (stressDetected == 1) {
      lcd.print("Status: STRES  ");
    } else {
      lcd.print("Status: NORMAL ");
    }
    lastLcd = millis();
  }
}
