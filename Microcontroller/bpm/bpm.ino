#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);

  // Initialize sensor
  if (particleSensor.begin() == false) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1)
      ;
  }

  particleSensor.setup();  //Configure sensor. Use 6.4mA for LED drive
}

void loop() {
  Serial.print(particleSensor.getRed());
  Serial.print(", ");
  Serial.println(particleSensor.getIR());
}