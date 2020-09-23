import RPi.GPIO as GPIO
import time

pwmPin = 33
pwmRate = 100
initDuty = 5

GPIO.setmode(GPIO.BOARD)
GPIO.setup(pwmPin,GPIO.OUT)

pwm = GPIO.PWM(pwmPin,pwmRate)
print('Starting, initial duty cycle', initDuty)
pwm.start(initDuty)

pwm.ChangeDutyCycle(50)
print('change duty cycle', 50)
time.sleep(5)
print('Ending')
GPIO.cleanup()