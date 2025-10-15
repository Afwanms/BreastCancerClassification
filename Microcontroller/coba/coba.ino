#include "decisiontree_model.h"

Eloquent::ML::Port::DecisionTree model;

void setup() {
  Serial.begin(9600);
}

void loop() {
  // contoh input: [pNN50, HF] dalam satuan ter-scale (pakai scaler Python)
  float features[] = {0.32, -1.15};   // nilai contoh
  int prediction = model.predict(features);

  Serial.print("Prediksi Kelas: ");
  if (prediction == 0) Serial.println("Sehat");
  else Serial.println("Kanker");

  delay(1000);
}