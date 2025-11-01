#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "MAX30105.h"
#include <arduinoFFT.h>
#include "decisiontree_model.h" // micromlgen export

// ==== Perangkat ====
LiquidCrystal_I2C lcd(0x27, 16, 2);
MAX30105 sensor;

// ==== Konfigurasi RR ====
#define MAX_RR_INTERVALS 512
float rrIntervals[MAX_RR_INTERVALS];
int   rrCount = 0;

const long FINGER_THRESHOLD = 50000;
const int  MIN_RR_INTERVAL  = 300;   // ms
const int  MAX_RR_INTERVAL  = 2000;  // ms
unsigned long startTime = 0;
bool collecting = false;

// ==== Peak adaptif (smoothing + EMA baseline/noise) ====
const int SMOOTH_N = 4;
long smoothBuf[SMOOTH_N];
int  smoothIdx = 0;
long prevAvg   = 0;

float emaBaseline = 0;
float emaNoise    = 1000;
const float A_BASE = 0.01f;     // 0.005–0.02
const float A_NOISE = 0.05f;    // 0.03–0.10
const float K_THR = 3.8f;       // 3–6
long   minPeakAmp = 1200;

unsigned long lastPeakTime = 0;
const unsigned long minPeakInterval = 600;   // ms (~100 bpm)
const unsigned long maxPeakInterval = 2000;  // ms
int prevSlope = 0;                            // -1/0/1

// ==== Jam tampilan ====
unsigned long lastClockUpdate = 0;
const unsigned long CLOCK_PERIOD = 500;  // ms

// ==== FFT (HF 0.15–0.40 Hz) ====
float vReal[MAX_RR_INTERVALS];
float vImag[MAX_RR_INTERVALS];
ArduinoFFT<float> FFT(vReal, vImag, MAX_RR_INTERVALS, 1.0);

// ==== Scaler fitur (opsional, samakan dgn training) ====
float meanVals[2]  = {7.7733, 20.1531};    // mean pNN50, HF
float scaleVals[2] = {9.4457, 15.7324};    // scale pNN50, HF

// ==== Model ====
Eloquent::ML::Port::DecisionTree classifier;

// ==== Struktur fitur ====
struct HRVFeatures { float pnn50; float hf; };

// ==== Deklarasi ====
HRVFeatures calculateHRV();
int  predictCancer(const HRVFeatures& hrv);
void displayResult(const HRVFeatures& hrv, int prediction);

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);              // ESP32: SDA=21, SCL=22

  lcd.init(); lcd.backlight();
  lcd.clear(); lcd.setCursor(0,0); lcd.print("HRV Cancer");
  lcd.setCursor(0,1); lcd.print("Detection");
  delay(1200);

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    lcd.clear(); lcd.print("Sensor Error!");
    while (1) { delay(10); }
  }
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeGreen(0);

  // init smoother dari bacaan awal
  long seed = sensor.getIR();
  for (int i=0;i<SMOOTH_N;i++) smoothBuf[i] = seed;
  prevAvg = seed;
  emaBaseline = prevAvg;
  emaNoise = 1000;
  lastPeakTime = 0;
  rrCount = 0;
  collecting = false;

  lcd.clear();
  lcd.print("Place finger...");
}

void loop() {
  unsigned long now = millis();
  long irValue = sensor.getIR();

  // ===== deteksi jari =====
  if (irValue <= FINGER_THRESHOLD) {
    // tampilkan status, jangan reset startTime tiap frame
    lcd.setCursor(0,0); 
    lcd.print("No finger      ");
    lcd.setCursor(0,1); 
    lcd.print("detected       ");
    rrCount = 0; collecting = false; lastPeakTime = 0;
    for (int i=0;i<SMOOTH_N;i++) smoothBuf[i] = irValue;
    prevAvg = irValue;
    delay(80);
    return;
  }

  // mulai koleksi saat jari terdeteksi pertama kali
  if (!collecting) {
    collecting = true;
    rrCount = 0; startTime = now; lastPeakTime = 0;
    for (int i=0;i<SMOOTH_N;i++) smoothBuf[i] = irValue;
    prevAvg = irValue;
    lcd.clear(); 
    lcd.print("Start 5min...");
    Serial.println("Mulai akuisisi 5 menit...");
  }

  // ===== smoothing =====
  smoothBuf[smoothIdx] = irValue;
  smoothIdx = (smoothIdx + 1) % SMOOTH_N;
  long avgValue = 0; for (int i=0;i<SMOOTH_N;i++) avgValue += smoothBuf[i];
  avgValue /= SMOOTH_N;

  // ===== EMA baseline & noise + threshold adaptif =====
  float dev = fabs((float)avgValue - emaBaseline);
  emaBaseline = (1 - A_BASE) * emaBaseline + A_BASE * avgValue;
  emaNoise    = (1 - A_NOISE) * emaNoise    + A_NOISE * dev;
  float dynamicThr = emaBaseline + K_THR * emaNoise;

  // ===== slope =====
  int slope = (avgValue > prevAvg) ? 1 : ((avgValue < prevAvg) ? -1 : 0);

  // ===== kunci puncak saat naik->turun, di atas thr & refractory =====
  if (prevSlope > 0 && slope <= 0) {
    if (avgValue > dynamicThr && (now - lastPeakTime) > minPeakInterval) {
      unsigned long dt = (lastPeakTime == 0) ? 0 : (now - lastPeakTime);
      long amp = avgValue - (long)emaBaseline;

      if (dt == 0 || (dt >= minPeakInterval && dt <= maxPeakInterval)) {
        if (amp > minPeakAmp) {
          if (lastPeakTime > 0 && rrCount < MAX_RR_INTERVALS) {
            rrIntervals[rrCount++] = (float)dt;
            float bpm = 60000.0f / (float)dt;

            // update baris HR & counter tanpa clear
            lcd.clear();
            lcd.setCursor(0,0);
            lcd.print("Collecting...  ");
            lcd.setCursor(0,1);
            lcd.print("HR:"); 
            lcd.print(bpm,0);
            lcd.print("  ");  
            lcd.print(rrCount); 
            lcd.print("/"); 
            lcd.print(MAX_RR_INTERVALS);
            lcd.print("  ");

            Serial.print("RR="); 
            Serial.print(dt); 
            Serial.print("ms, BPM=");
            Serial.println(bpm,1);
          }
          lastPeakTime = now;
        }
      } else if (dt > maxPeakInterval) {
        lastPeakTime = now; // re-anchor
      }
    }
  }
  prevSlope = slope;
  prevAvg   = avgValue;

  // (opsional) kirim ke Serial Plotter utk tuning: RAW,AVG,THR
  Serial.print(irValue); Serial.print(',');
  Serial.print(avgValue); Serial.print(',');
  Serial.println((long)dynamicThr);

  // ===== timer (mm:ss) tanpa clear =====
  if (collecting && (now - lastClockUpdate >= CLOCK_PERIOD)) {
    lastClockUpdate = now;
    unsigned long elapsed = now - startTime;
    int minutes = elapsed / 60000UL;
    int seconds = (elapsed / 1000UL) % 60;

    lcd.setCursor(11,0);
    if (minutes < 10) lcd.print('0');
    lcd.print(minutes);
    lcd.print(':');
    if (seconds < 10) lcd.print('0');
    lcd.print(seconds);
  }

  // ===== selesai 5 menit -> analisis =====
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

  delay(5);
}

// ==== HRV (pNN50 + HF via FFT) ====
HRVFeatures calculateHRV() {
  HRVFeatures hrv; hrv.pnn50 = 0; hrv.hf = 0;
  if (rrCount < 10) return hrv;

  int nn50 = 0;
  for (int i = 1; i < rrCount; i++) {
    float d = fabs(rrIntervals[i] - rrIntervals[i - 1]);
    if (d > 50) nn50++;
  }
  hrv.pnn50 = 100.0f * ((float)nn50 / (float)(rrCount - 1));

  float totalMs = 0; for (int i=0;i<rrCount;i++) totalMs += rrIntervals[i];
  float fs = rrCount / (totalMs / 1000.0f);

  float meanRR = 0; for (int i=0;i<rrCount;i++) meanRR += rrIntervals[i];
  meanRR /= rrCount;
  for (int i=0;i<rrCount;i++){ vReal[i] = rrIntervals[i] - meanRR; vImag[i] = 0; }

  FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();

  float hfPower = 0, freqRes = fs / rrCount;
  for (int i=0;i<rrCount/2;i++){
    float f = i * freqRes;
    if (f >= 0.15f && f <= 0.40f)
      hfPower += (vReal[i] * vReal[i]) / (rrCount * rrCount); // ~ms^2
  }
  hrv.hf = hfPower;

  Serial.print("fs(auto): "); Serial.print(fs,3); Serial.println(" Hz");
  return hrv;
}

// ==== Prediksi ====
int predictCancer(const HRVFeatures& hrv) {
  float raw[2] = {hrv.pnn50, hrv.hf}, input[2];
  for (int i=0;i<2;i++) input[i] = (raw[i] - meanVals[i]) / scaleVals[i];
  return classifier.predict(input); // 0=Normal, 1=Cancer
}

// ==== Tampil hasil ====
void displayResult(const HRVFeatures& hrv, int prediction) {
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print(prediction==0 ? "Result: NORMAL" : "Result: CANCER");
  lcd.setCursor(0,1);
  lcd.print("p:"); lcd.print(hrv.pnn50,0);
  lcd.print("% HF:"); lcd.print(hrv.hf,0);

  Serial.println("\n=== PREDICTION ===");
  Serial.println(prediction==0 ? "NORMAL" : "CANCER");
}