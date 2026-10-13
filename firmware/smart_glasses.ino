/*
 * Smart Glasses - 내장 IMU(LSM6DS3TR-C)
 * Board: Seeed XIAO nRF52840 Sense
 * Output: Teleplot 포맷
 */
#include <Wire.h>
#include "LSM6DS3.h"

LSM6DS3 myIMU(I2C_MODE, 0x6A);

const uint32_t SAMPLE_INTERVAL_US = 10000;  // 100 Hz
uint32_t lastSampleTime = 0;

void plot(const char* name, uint32_t ts, float value) {
  Serial.print(">"); Serial.print(name);
  Serial.print(":"); Serial.print(ts);
  Serial.print(":"); Serial.println(value, 4);
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}

  Wire.begin();
  Wire.setClock(400000);

  if (myIMU.begin() != 0) Serial.println("# IMU init FAILED");
  else                    Serial.println("# IMU init OK");

  Serial.println("# Setup complete - streaming...");
  lastSampleTime = micros();
}

void loop() {
  uint32_t now = micros();
  if (now - lastSampleTime < SAMPLE_INTERVAL_US) return;
  lastSampleTime += SAMPLE_INTERVAL_US;

  uint32_t ts = millis();

  float ax = myIMU.readFloatAccelX();
  float ay = myIMU.readFloatAccelY();
  float az = myIMU.readFloatAccelZ();
  float gx = myIMU.readFloatGyroX();
  float gy = myIMU.readFloatGyroY();
  float gz = myIMU.readFloatGyroZ();
  float accMag = sqrt(ax*ax + ay*ay + az*az);

  plot("AccX",   ts, ax);
  plot("AccY",   ts, ay);
  plot("AccZ",   ts, az);
  plot("GyrX",   ts, gx);
  plot("GyrY",   ts, gy);
  plot("GyrZ",   ts, gz);
  plot("AccMag", ts, accMag);
}
