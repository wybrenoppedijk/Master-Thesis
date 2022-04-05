#!/usr/bin/env python

import io
import fcntl
import time
import plotext as plt
I2C_SLAVE = 0x0703


class i2c:

    def __init__(self, device, bus):
        self.fr = io.open("/dev/i2c-" + str(bus), "rb", buffering=0)
        self.fw = io.open("/dev/i2c-" + str(bus), "wb", buffering=0)

        # set device address

        fcntl.ioctl(self.fr, I2C_SLAVE, device)
        fcntl.ioctl(self.fw, I2C_SLAVE, device)

    def write(self, values):
        self.fw.write(bytearray(values))

    def read(self, NumberOfByte):
        valueBytes = self.fr.read(NumberOfByte)
        return list(valueBytes)

    def close(self):
        self.fw.close()
        self.fr.close()


class mcp3428:

    def __init__(self, I2CAddress, bus):
        self.I2CAddress = I2CAddress
        self.dev = i2c(I2CAddress, bus)

        self.STARTCONV = 0b10000000
        self.CHANNEL1 = 0b00000000
        self.CHANNEL2 = 0b00100000
        self.CHANNEL3 = 0b01000000
        self.CHANNEL4 = 0b01100000
        self.BITS12 = 0b00000000
        self.BITS14 = 0b00000100
        self.BITS16 = 0b00001000
        self.ONESHOT = 0b00000000
        self.CONTINOUS = 0b00010000
        self.GAIN1 = 0b00000000
        self.GAIN2 = 0b00000001
        self.GAIN4 = 0b00000010
        self.GAIN8 = 0b00000011

    def __del__(self):
        self.dev.close()

    def readSingle(self, channel=1, gain=1):
        # set one shot new conversion 16 bits
        control = self.BITS16 | self.ONESHOT

        # set gain
        if gain == 8:
            control = control | self.GAIN8
        elif gain == 4:
            control = control | self.GAIN4
        elif gain == 2:
            control = control | self.GAIN2
        else:
            control = control | self.GAIN1
            gain = 1

        # set channels
        if channel == 4:
            control = control | self.CHANNEL4
        elif channel == 3:
            control = control | self.CHANNEL3
        elif channel == 2:
            control = control | self.CHANNEL2
        else:
            channel = 1
            control = control | self.CHANNEL1

        # apply the change but no conversion
        self.dev.write([control])

        # let's wait a little to stabilize the A/D capacitor
        time.sleep(0.001)

        # let's start a conversion
        control = control | self.STARTCONV
        self.dev.write([control])

        # now wait for conversion
        while True:
            time.sleep(0.001)
            info = self.dev.read(3)
            if (info[2] & 0x80) == 0:
                break
        # ok let read the value
        info = self.dev.read(3)
        UnsignedV = info[0] * 256 + info[1]

        # let check the sign
        if (UnsignedV > 32767):
            UnsignedV = UnsignedV - 65536

        # now the value in current
        return UnsignedV * 0.0009


if __name__ == "__main__":
    a2d = mcp3428(0x68, 1)
    results = []
    window = 100
    while True:
        results.append(a2d.readSingle(1, 1))
        if len(results) > window:
            results = results[1:]
            plt.clf()
            plt.plot(results)
            plt.show()
