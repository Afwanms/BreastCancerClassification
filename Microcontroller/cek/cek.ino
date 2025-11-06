#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "MAX30105.h"
#include <arduinoFFT.h>
#include <math.h>                  // for fabsf
#include "decisiontree_model.h"    // micromlgen export

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

// ==== Peak adaptif ====
const int SMOOTH_N = 4;
long smoothBuf[SMOOTH_N];
int  smoothIdx = 0;
long prevAvg   = 0;

// === threshold tuning ===
float emaBaseline = 0;
float emaNoise    = 500;
const float A_BASE = 0.01f;
const float A_NOISE = 0.1f;
const float K_THR = 1.5f;
long minPeakAmp = 200;

unsigned long lastPeakTime = 0;
const unsigned long minPeakInterval = 300;
const unsigned long maxPeakInterval = 2000;
int prevSlope = 0;

// ==== BPM Detection (Improved) ====
const int BPM_BUFFER_SIZE = 8;
unsigned long beatTimes[BPM_BUFFER_SIZE];
int beatIdx = 0;
int validBeats = 0;
unsigned long lastBeatTime = 0;
int beatsPerMinute = 0;
int beatAvg = 0;

const unsigned long MIN_BEAT_INTERVAL = 375;   // ~160 BPM max
const unsigned long MAX_BEAT_INTERVAL = 1500;  // ~40 BPM min

// ==== FFT (HF 0.15–0.40 Hz) ====
float vReal[MAX_RR_INTERVALS];
float vImag[MAX_RR_INTERVALS];
ArduinoFFT<float> FFT(vReal, vImag, MAX_RR_INTERVALS, 1.0);

// ==== Scaler fitur (samakan dgn training) ====
float meanVals[2]  = {7.7733f, 20.1531f};    // mean pNN50, HF
float scaleVals[2] = {9.4457f, 15.7324f};    // scale pNN50, HF

// ==== Model ====
Eloquent::ML::Port::DecisionTree classifier;

// ==== Struktur fitur ====
struct HRVFeatures { 
  float pnn50; 
  float hf; 
};

// ==== Deklarasi ====
HRVFeatures calculateHRV();
int  predictCancer(const HRVFeatures& hrv);
void displayResult(const HRVFeatures& hrv, int prediction);
void updateBPMDisplay();
void recordBeat(unsigned long beatTime);
void printMMSS(uint32_t ms);
int  computeAvgBPMFromBeats();

void setup() {
  Serial.begin(115200);
  // ESP32: SDA=21, SCL=22 (kalau UNO/MEGA cukup Wire.begin();)
  Wire.begin(21, 22);

  lcd.init(); 
  lcd.backlight();
  lcd.clear(); 
  lcd.setCursor(0,0); lcd.print("HRV Cancer");
  lcd.setCursor(0,1); lcd.print("Detection");
  delay(1200);

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    lcd.clear(); lcd.print("Sensor Error!");
    while (1) { delay(10); }
  }
  
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeGreen(0);

  // Init smoother
  long seed = sensor.getIR();
  for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = seed;
  prevAvg = seed;
  emaBaseline = prevAvg;
  emaNoise = 1000;
  lastPeakTime = 0;
  rrCount = 0;
  collecting = false;
  
  // Init beat buffer
  for (int i = 0; i < BPM_BUFFER_SIZE; i++) beatTimes[i] = 0;
  beatIdx = 0;
  validBeats = 0;
  lastBeatTime = 0;
  beatsPerMinute = 0;
  beatAvg = 0;

  lcd.clear();
  lcd.print("Place finger...");
}

void loop() {
  unsigned long now = millis();
  long irValue = sensor.getIR();

  if (irValue > FINGER_THRESHOLD) {
    if (!collecting) {
      collecting = true;
      rrCount = 0;
      startTime = now;
      lastPeakTime = 0;
      lastBeatTime = 0;
      validBeats = 0;
      beatIdx = 0;
      for (int i = 0; i < BPM_BUFFER_SIZE; i++) beatTimes[i] = 0;
      
      for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = irValue;
      prevAvg = irValue;

      lcd.clear();
      lcd.print("Start 5min...");
      Serial.println("Mulai akuisisi 5 menit...");
    }

    // ==== Smoothing dan Peak Detection ====
    smoothBuf[smoothIdx] = irValue;
    smoothIdx = (smoothIdx + 1) % SMOOTH_N;
    long avgValue = 0; 
    for (int i = 0; i < SMOOTH_N; i++) avgValue += smoothBuf[i];
    avgValue /= SMOOTH_N;

    float dev = fabsf((float)avgValue - emaBaseline);
    emaBaseline = (1 - A_BASE) * emaBaseline + A_BASE * avgValue;
    emaNoise   = (1 - A_NOISE) * emaNoise   + A_NOISE * dev;
    float dynamicThr = emaBaseline + K_THR * emaNoise;

    // ==== Slope-based Peak Detection ====
    int slope = (avgValue > prevAvg) ? 1 : ((avgValue < prevAvg) ? -1 : 0);
    if (prevSlope > 0 && slope <= 0) {
      unsigned long dt = (lastPeakTime == 0) ? 0 : (now - lastPeakTime);
      long amp = avgValue - (long)emaBaseline;

      if (dt == 0 || (dt >= minPeakInterval && dt <= maxPeakInterval)) {
        if (amp > minPeakAmp) {  // tambahan: cek threshold dinamis
          // ==== Record Beat untuk BPM ====
          if (lastBeatTime == 0) {
            lastBeatTime = now;
            recordBeat(now);
          } else {
            unsigned long beatDelta = now - lastBeatTime;
            if (beatDelta >= MIN_BEAT_INTERVAL && beatDelta <= MAX_BEAT_INTERVAL) {
              beatsPerMinute = 60000UL / beatDelta;
              recordBeat(now);
              lastBeatTime = now;

              beatAvg = computeAvgBPMFromBeats();
              updateBPMDisplay();

              Serial.print("Beat detected! Delta=");
              Serial.print(beatDelta);
              Serial.print("ms, BPM=");
              Serial.print(beatsPerMinute);
              Serial.print(", AvgBPM=");
              Serial.println(beatAvg);
            } else {
              Serial.print("Beat rejected: interval=");
              Serial.println(beatDelta);
            }
          }

          // ==== RR Interval Recording ====
          if (lastPeakTime > 0 && rrCount < MAX_RR_INTERVALS) {
            rrIntervals[rrCount++] = (float)dt;
            Serial.print("RR="); Serial.print(dt); 
            Serial.print("ms, Count="); Serial.println(rrCount);
          }
          lastPeakTime = now;
        }
      } else if (dt > maxPeakInterval) {
        lastPeakTime = now;
      }
    }
    prevSlope = slope;
    prevAvg = avgValue;

    // Serial Plotter data
    Serial.print(irValue); Serial.print(',');
    Serial.print(avgValue); Serial.print(',');
    Serial.println((long)dynamicThr);

    // ==== Timer (mm:ss) ====
    if (collecting) {
      lcd.setCursor(12, 0);
      printMMSS(now - startTime);   // ganti lcd.printf
    }

    // ==== Selesai 5 menit -> Analisis ====
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
      validBeats = 0;
      delay(15000);
      lcd.clear(); lcd.print("Place finger...");
    }

    delay(5);
  } else {
    // Tidak ada jari
    lcd.clear(); 
    lcd.setCursor(0, 0); lcd.print("No finger      ");
    lcd.setCursor(0, 1); lcd.print("detected       ");
    rrCount = 0; 
    collecting = false;
    validBeats = 0;
    lastBeatTime = 0;
    for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = irValue;
    prevAvg = irValue; 
    lastPeakTime = 0;
    delay(100);
    return;
  }
}

// ==== Record Beat ke buffer ====
void recordBeat(unsigned long beatTime) {
  beatTimes[beatIdx] = beatTime;
  beatIdx = (beatIdx + 1) % BPM_BUFFER_SIZE;
  if (validBeats < BPM_BUFFER_SIZE) validBeats++;
}

// Hitung rata-rata BPM dari beatTimes (ring buffer), hanya pakai interval valid
int computeAvgBPMFromBeats() {
  if (validBeats < 2) return 0;
  // Traverse ring buffer dalam urutan kronologis
  int count = 0;
  unsigned long sumBPM = 0;

  // tentukan index awal (elemen tertua)
  int start = (beatIdx - validBeats + BPM_BUFFER_SIZE) % BPM_BUFFER_SIZE;
  unsigned long prev = 0;
  for (int k = 0; k < validBeats; k++) {
    int i = (start + k) % BPM_BUFFER_SIZE;
    unsigned long t = beatTimes[i];
    if (t == 0) continue;
    if (prev != 0) {
      unsigned long interval = t - prev;
      if (interval >= MIN_BEAT_INTERVAL && interval <= MAX_BEAT_INTERVAL) {
        sumBPM += 60000UL / interval;
        count++;
      }
    }
    prev = t;
  }
  if (count == 0) return 0;
  return (int)(sumBPM / count);
}

// ==== Update BPM Display ====
void updateBPMDisplay() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("BPM:");
  lcd.print(beatAvg > 0 ? beatAvg : beatsPerMinute);

  lcd.setCursor(0, 1);
  lcd.print("RR:");
  lcd.print(rrCount); 
  lcd.print("/");
  lcd.print(MAX_RR_INTERVALS);
}

// ==== HRV (pNN50 + HF via FFT) ====
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

  // estimasi fs dari RR
  float totalTimeMs = 0;
  for (int i = 0; i < rrCount; i++) totalTimeMs += rrIntervals[i];
  float totalTime = totalTimeMs / 1000.0f;
  float fs = (totalTime > 0) ? (rrCount / totalTime) : 1.0f;

  // detrend & copy ke buffer FFT
  float meanRR = 0.0f; 
  for (int i=0; i<rrCount; i++) meanRR += rrIntervals[i];
  meanRR /= rrCount;

  for (int i=0; i<rrCount; i++){ 
    vReal[i] = rrIntervals[i] - meanRR; 
    vImag[i] = 0.0f; 
  }
  // zero pad sisa buffer
  for (int i=rrCount; i<MAX_RR_INTERVALS; i++){ 
    vReal[i] = 0.0f; 
    vImag[i] = 0.0f; 
  }

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

// ==== Prediksi ====
int predictCancer(const HRVFeatures& hrv) {
  float raw[2] = {hrv.pnn50, hrv.hf};
  float input[2];
  for (int i = 0; i < 2; i++)
    input[i] = (raw[i] - meanVals[i]) / scaleVals[i];
  return (int)classifier.predict(input); // 0=Normal, 1=Cancer
}

// ==== Tampil hasil ====
void displayResult(const HRVFeatures& hrv, int prediction) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(prediction == 0 ? "Result: NORMAL" : "Result: CANCER");
  lcd.setCursor(0, 1);
  lcd.print("p:"); lcd.print((int)(hrv.pnn50 + 0.5)); lcd.print("% ");
  lcd.print("HF:"); lcd.print((int)(hrv.hf + 0.5));

  Serial.println("\n=== PREDICTION ===");
  Serial.println(prediction == 0 ? "NORMAL" : "CANCER");
}

// util: cetak mm:ss
void printMMSS(uint32_t ms) {
  uint32_t sec = ms / 1000UL;
  uint16_t m = sec / 60U;
  uint8_t  s = sec % 60U;
  char buf[6];
  snprintf(buf, sizeof(buf), "%u:%02u", m, s);
  lcd.print(buf);
}