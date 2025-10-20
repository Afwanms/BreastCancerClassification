#include <Wire.h>

void setup() {
  Wire.begin(21, 22);  // SDA, SCL default
  Serial.begin(115200);
  delay(1000);
  Serial.println("🔍 Scanning I2C devices...");
  
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("✅ Found device at 0x");
      Serial.println(addr, HEX);
    }
  }
  Serial.println("✅ Scan complete.");
}

void loop() {}
