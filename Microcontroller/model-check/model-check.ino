#include "decisiontree_model.h"

Eloquent::ML::Port::DecisionTree model;

void setup() {
  Serial.begin(9600);
}

void loop() {
  // contoh input: [pNN50, HF] dalam satuan ter-scale (pakai scaler Python)
  float features[2] = {1, 58};   // nilai contoh
  float meanVals[2]  = {6.758620689655173f, 203.151724137931f};    // mean pNN50, HF
  float scaleVals[2] = {8.80695325772255f, 320.2095161314936f}; 
  float input[2];
  for (int i = 0; i < 2; i++)
    input[i] = (features[i] - meanVals[i]) / scaleVals[i];
  int prediction = model.predict(input);

  Serial.print("Prediksi Kelas: ");
  if (prediction == 0) Serial.println("Sehat");
  else Serial.println("Kanker");

  delay(1000);
}