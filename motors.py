# -*- coding: utf-8 -*-
import RPi.GPIO as GPIO
from time import sleep
SERVO_PIN_B = 19
SERVO_PIN_C = 13

PWM_FREQ = 50

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_B, GPIO.OUT)
GPIO.setup(SERVO_PIN_C, GPIO.OUT)
servo_pwm1 = GPIO.PWM(SERVO_PIN_B, PWM_FREQ)
servo_pwm2 = GPIO.PWM(SERVO_PIN_C, PWM_FREQ)

current_duty = 0
current_angle = 90
is_running = False

def angle_to_duty(angle: float) -> float:
    return 2 + (angle / 18.0)

def get_pwm_status():
    return {"angle": current_angle, "duty": current_duty}

def stop():
    global is_running
    servo_pwm1.stop()
    servo_pwm2.stop()
    print("[SERVO] STOPPED")

def move_to(angle: float):
    global current_duty, current_angle, is_running
    if not is_running:
        servo_pwm1.start(0)
        servo_pwm2.start(0)
        is_running = True
    duty = angle_to_duty(angle)
    servo_pwm1.ChangeDutyCycle(duty)
    servo_pwm2.ChangeDutyCycle(duty)
    current_duty = duty
    current_angle = angle
    print(f"[SERVO] Angle={angle:.1f} deg , Duty={duty:.2f}%")
    sleep(0.5)
    servo_pwm1.ChangeDutyCycle(0)
    servo_pwm2.ChangeDutyCycle(0)


if __name__ == "__main__":
    try:
        move_to(90)
        sleep(1)
        move_to(0)
        sleep(1)
        move_to(180)
        sleep(1)
        move_to(90)
        sleep(1)
        move_to(0)
        sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        GPIO.cleanup()
        print("[GPIO] Cleaned up successfully.")


