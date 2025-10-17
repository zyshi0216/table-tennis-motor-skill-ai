#include "mpu6050.h"

void MPU6050_Init(void){
    I2C_WriteByte(MPU6050_ADDR, PWR_MGMT_1, 0x00);
    I2C_WriteByte(MPU6050_ADDR, SMPLRT_DIV, 0x09);     // 1 kHz / (1+9) = 100 Hz
    I2C_WriteByte(MPU6050_ADDR, CONFIG, 0x03);
    I2C_WriteByte(MPU6050_ADDR, GYRO_CONFIG, 0x18);    // ±2000 °/s
    I2C_WriteByte(MPU6050_ADDR, ACCEL_CONFIG, 0x10);   // ±8 g
}

void MPU6050_Read_Acc(void){
    uint8_t buf[6];
    I2C_ReadBytes(MPU6050_ADDR, ACCEL_XOUT_H, buf, 6);
    MPU6050_Acc.X = (int16_t)(buf[0]<<8|buf[1]);
    MPU6050_Acc.Y = (int16_t)(buf[2]<<8|buf[3]);
    MPU6050_Acc.Z = (int16_t)(buf[4]<<8|buf[5]);
}

void MPU6050_Read_Gyro(void){
    uint8_t buf[6];
    I2C_ReadBytes(MPU6050_ADDR, GYRO_XOUT_H, buf, 6);
    MPU6050_Gyro.X = (int16_t)(buf[0]<<8|buf[1]);
    MPU6050_Gyro.Y = (int16_t)(buf[2]<<8|buf[3]);
    MPU6050_Gyro.Z = (int16_t)(buf[4]<<8|buf[5]);
}
