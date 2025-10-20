#include "decisiontree_model.h"

Eloquent::ML::Port::DecisionTree model;

float mean[]  = {6.758620689655173, 203.151724137931};
float scale[] = {8.80695325772255, 320.2095161314936};

void setup() {
  Serial.begin(9600);
}

void loop() {
  float raw[] = {};
  float features[3];

  for (int i = 0; i < 3; i++) {
    features[i] = (raw[i] - mean[i]) / scale[i];
  }

  int prediction = model.predict(features);

  Serial.print("Prediksi: ");
  if (prediction == 0) Serial.println("Sehat");
  else Serial.println("Kanker");

  delay(2000);
}