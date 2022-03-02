#!/usr/bin/env python3
import sys
import serial
import requests
from time import sleep

def send(a0, a1, a2, a3):
    url = 'https://actionprojectdatacollector.azurewebsites.net/dtu/wybren/raspberry/measurements/4b0c2338-9a29-11ec-b909-0242ac120002'
    headers = {'Content-Type': 'application/json'}
    data = [
        {
            "id": "4b0c2338-9a29-11ec-b909-0242ac120002",
            "telemetry": {
                "monitoring": {
                    "a0": a0,
                    "a1": a1,
                    "a2": a2,
                    "a3": a3
                }
            }
        }
    ]
    r = requests.post(url, headers=headers, json=data)
    print(r.status_code)

def read():
    port = '/dev/ttyUSB0'
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        port = '/dev/tty.usbmodem12401'

    ser = serial.Serial(port, 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        a0, a1, a2, a3 = None, None, None, None
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            a0, a1, a2, a3 = line.split(',')
        send(a0, a1, a2, a3)
        sleep(2)




if __name__ == '__main__':
    read()