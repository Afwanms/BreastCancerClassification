#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <LiquidCrystal.h>

MAX30105 sensor;
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int bufferSize = 100;
uint32_t irBuffer[bufferSize];
int bufferIndex = 0;
unsigned long lastPeakTime = 0;
unsigned long NNinterval = 0;

#define MAX_NN 50
unsigned long NN[MAX_NN];
int NNcount = 0;

float countSDNN(unsigned long arr[], int n){
    if (n < 2)
        return 0;
    float mean = 0;

    for (int i = 0; i < n; i++)
        mean += arr[i];
    mean /= n;
    float sum = 0;

    for (int i = 0; i < n; i++)
        sum += (arr[i] - mean) * (arr[i] - mean);
    return sqrt(sum / (n - 1));
}

float countRMSSD(unsigned long arr[], int n){
    if (n < 2)
        return 0;
    float sum = 0;

    for (int i = 1; i < n; i++){
        float diff = arr[i] - arr[i - 1];
        sum += diff * diff;
    }
    return sqrt(sum / (n - 1));
}

float countpNN50(unsigned long arr[], int n){
    if (n < 2)
        return 0;
    int count50 = 0;

    for (int i = 1; i < n; i++){
        if (abs((long)(arr[i] - arr[i - 1])) > 50)
            count50++;
    }
    return (float)count50 / (n - 1) * 100;
}

void displayHRV(unsigned long arr[], int n){
    float SDNN = countSDNN(arr, n);
    float RMSSD = countRMSSD(arr, n);
    float pNN50 = countpNN50(arr, n);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Unclassified..");

    lcd.setCursor(0, 1);
    lcd.print("S:");
    lcd.print(SDNN, 1);
    lcd.print("R:");
    lcd.print(RMSSD, 1);
    lcd.print("P:");
    lcd.print(pNN50, 0);
    delay(1000);
}

void setup(){
    Serial.begin(9600);
    lcd.begin(16, 2);
    lcd.print("Initializing...");
    
    if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
        lcd.clear();
        lcd.print("Sensor Error");
        while (1);
    }

    sensor.setup();
    lcd.print("Sensor Ready");
    delay(1000);
    lcd.clear();
}

void loop(){
    long irValue = sensor.getIR();
    Serial.print("IR:");
    Serial.println(irValue);
    
    if (irValue > 300){
        if(checkForBeat(irValue)){
            unsigned long currentTime = millis();
            NNinterval = currentTime - lastPeakTime;
            lastPeakTime = currentTime;

            if (NNinterval > 300 && NNinterval < 2000){
                if (NNcount < MAX_NN){
                    NN[NNcount++] = NNinterval;
                } else {
                    for (int i = 1; i < MAX_NN; i++){
                        NN[i - 1] = NN[i];
                    }
                    NN[MAX_NN - 1] = NNinterval;
                }
            }
            float HR = 60000.0 / NNinterval;
            Serial.print("NN: ");
            Serial.println(NNinterval);
            Serial.print("ms | HR: ");
            Serial.println(HR);

            if (NNcount >= 10){
                displayHRV(NN, NNcount);
                NNcount = 0;
            }
        }
    } else{
        lcd.setCursor(0, 0);
        lcd.print("No Finger Detected");
        delay(500);
    }
    delay(20);
}