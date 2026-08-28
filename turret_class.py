import serial
import time

class TurretController:
    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.pan = 90
        self.tilt = 90

    def send(self, pan, tilt, laser_on=False):
        pan = int(max(0, min(180, pan)))
        tilt = int(max(0, min(180, tilt)))
        cmd = f"P;{pan};{tilt};{1 if laser_on else 0}\n"
        self.ser.write(cmd.encode())
        self.pan, self.tilt = pan, tilt

    def close(self):
        self.send(90, 90, laser_on=False)
        self.ser.close()