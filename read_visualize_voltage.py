import numpy as np
import matplotlib.pyplot as plt
import serial
import sys

refresh_rate_hz = 60
buffer_size = 80

def plot():
    fig, ax = plt.subplots(1, 1)
    x = np.array([])
    y = np.array([])
    lines, = ax.plot(x, y)
    ax.set_ylim(-0.1,5.1)

    port = '/dev/ttyUSB0'
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        port = '/dev/tty.usbmodem12401'
    ser = serial.Serial(port, 9600, timeout=1)
    ser.reset_input_buffer()

    #Infinitely plot from here
    count = 0
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            s_type, s_value = line.split('_')
            if s_type == 'A0':
                if count > buffer_size:
                    y = np.delete(y, 0)
                    x = np.delete(x, 0)

                y = np.append(y, float(s_value))
                x = np.append(x, count)
                count+=1
                lines.set_data(x, y)
                ax.set_xlim((x.min(), x.max()))
                plt.pause(1/refresh_rate_hz)

if __name__ == "__main__":
    plot()
