#include "stm32f10x.h"
#include "mpu6050.h"
#include <stdio.h>

#define LOOP_PERIOD_MS 10   // 100 Hz

// printf 重定向
int fputc(int ch, FILE *f){ USART_SendData(USART1, (uint8_t)ch); while(USART_GetFlagStatus(USART1,USART_FLAG_TXE)==RESET); return ch; }

void delay_ms(uint32_t ms){
    for(uint32_t i=0;i<ms*7200;i++); // 简易延时
}

int main(void){
    SystemInit();
    USART1_Init(115200);
    I2C1_Init();
    MPU6050_Init();

    printf("MPU6050 start...\r\n");

    while(1){
        static uint32_t tick = 0;
        if(++tick >= LOOP_PERIOD_MS){ tick = 0; }

        MPU6050_Read_Acc();
        MPU6050_Read_Gyro();

        float ax = MPU6050_Acc.X / 16384.0f;
        float ay = MPU6050_Acc.Y / 16384.0f;
        float az = MPU6050_Acc.Z / 16384.0f;
        float gx = MPU6050_Gyro.X / 131.0f;
        float gy = MPU6050_Gyro.Y / 131.0f;
        float gz = MPU6050_Gyro.Z / 131.0f;

        uint32_t ts = HAL_GetTick();   // 毫秒时间戳

        // 输出 CSV 行
        printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\r\n",
                ts, ax, ay, az, gx, gy, gz);

        delay_ms(LOOP_PERIOD_MS);
    }
}
