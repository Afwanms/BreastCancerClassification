#include <Wire.h>
#include "MAX30105.h"   // Gunakan library SparkFun MAX3010x
#include "heartRate.h"  // Optional

MAX30105 sensor;

const int bufferSize = 100;
uint32_t irBuffer[bufferSize];
int bufferIndex = 0;

float threshold = 100000;   // Sesuaikan dengan kondisi sensor/jari
bool peakDetected = false;
unsigned long lastPeakTime = 0;
unsigned long NNInterval = 0;  // dalam ms

void setup() {
  Serial.begin(9600);
  Wire.begin();

  Serial.println("Inisialisasi MAX30102...");

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("Sensor tidak terdeteksi. Periksa kabel!");
    while (1);
  }

  // Konfigurasi sensor
  sensor.setup();                  
  sensor.setPulseAmplitudeRed(0x0A);   // Matikan LED RED (IR only)
  sensor.setPulseAmplitudeIR(0x1F);    // Nyalakan LED IR

  Serial.println("Sensor siap. Letakkan jari di atas sensor.");
}

void loop() {
  long irValue = sensor.getIR();

  // 🟡 Deteksi keberadaan jari
  bool fingerDetected = (irValue > 100000); // threshold ini bisa kamu kalibrasi

  if (fingerDetected) {
    // Jalankan deteksi puncak hanya kalau jari terdeteksi
    detectPeak(irValue);

    if (NNInterval > 0) {
      float HR = 60000.0 / NNInterval;
      Serial.print("IR: "); Serial.print(irValue);
      Serial.print("\tNN Interval: "); Serial.print(NNInterval);
      Serial.print("\tHR: "); Serial.println(HR);
    } else {
      Serial.print("IR: "); Serial.print(irValue);
      Serial.println("\tMendeteksi..."); // belum ada puncak
    }

  } else {
    // Reset variabel saat jari dilepas
    NNInterval = 0;
    lastPeakTime = 0;
    peakDetected = false;

    Serial.print("IR: "); Serial.print(irValue);
    Serial.println("\tJari tidak terdeteksi.");
  }

  delay(10);  // sampling sekitar 100 Hz
}

void detectPeak(long irValue) {
  static long prevValue = 0;
  static bool rising = false;

  // Deteksi kenaikan
  if (irValue > threshold && irValue > prevValue) {
    rising = true;
  }
  // Deteksi puncak ketika nilai mulai turun setelah naik
  if (rising && irValue < prevValue) {
    rising = false;
    unsigned long currentTime = millis();

    if (lastPeakTime > 0) {
      NNInterval = currentTime - lastPeakTime; // selisih waktu antar puncak
    }
    lastPeakTime = currentTime;
    peakDetected = true;
  }
  prevValue = irValue;
}
