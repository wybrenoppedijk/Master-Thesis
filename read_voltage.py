#!/usr/bin/env python3
import sys

import serial
import requests

from time import time_ns


def send(a0, a1, a2, a3):
    url = 'https://actionprojectdatacollector.azurewebsites.net/dtu/wybren/raspberry/measurements'
    headers = {'Content-Type': 'application/json'}
    timestamp = int(time_ns() / 1000)
    data = [
        {
            "id": "4b0c2338-9a29-11ec-b909-0242ac120002",
            "telemetry": {
                "monitoring": {
                    "a0": a0,
                    "a1": a1,
                    "a2": a2,
                    "a3": a3,
                    "time": timestamp,
                }
            }
        }
    ]
    r = requests.post(url, headers=headers, json=data)
    print("{}\t[{}, {}, {}, {}]\t{}".format(r.status_code, a0, a1, a2, a3, timestamp))


def read():
    port = '/dev/ttyACM0'
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        port = '/dev/tty.usbmodem12401'

    ser = serial.Serial(port, 9600, timeout=1)
    ser.reset_input_buffer()

    send_interval_s = 2
    waiting_time = send_interval_s * 1000 * 1000 * 1000  # seconds to nanoseconds.
    last_sent = time_ns()
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            if time_ns() > last_sent + waiting_time:
                send(*line.split(','))
                last_sent = time_ns()


if __name__ == '__main__':
    read()
