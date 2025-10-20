#include <Wire.h>
#include "MAX30105.h"         // Library SparkFun MAX3010x
#include <LiquidCrystal_I2C.h>

MAX30105 sensor;
LiquidCrystal_I2C lcd(0x27, 16, 2);  // alamat LCD kamu

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);         // Gunakan bus utama SDA=21, SCL=22

  // Inisialisasi sensor MAX30102
  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 tidak terdeteksi. Cek koneksi!");
    while (1);
  }
  Serial.println("MAX30102 terdeteksi di alamat 0x57");

  // Inisialisasi LCD
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("I2C OK!");
}

void loop() {
  long irValue = sensor.getIR();

  lcd.setCursor(0, 1);
  lcd.print("IR: ");
  lcd.print(irValue);
  delay(500);
}
