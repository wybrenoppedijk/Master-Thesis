#!/usr/bin/env python3
import sys
import serial

if __name__ == '__main__':
    port = '/dev/ttyUSB0'
    if len(sys.argv)>1 and sys.argv[1] == 'debug':
        port = '/dev/tty.usbmodem12401'

    ser = serial.Serial(port, 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            s_type, s_value = line.split('_')
            if s_type == 'A0':
                print('_________________')
            print("type {}:\t{} volt".format(s_type, s_value))