'''**************************************************************************/
Controleverything.com- MCP3427 I2C 16-Bit 2-Channel Analog to Digital Converter I2C Mini Module

Firmware v1.0 - Python
Author - Amanpal Singh

2-Channel ADC with 16-Bit Resolution
2 Differential Analog Inputs
Programmable x1-x8 Gain Amplifier
Upto 4 Devices per I2C Port
Up to 3.4MHz Communication Speed
0x68 I2C Start Address 

Hardware Version - Rev A.
Platform - Raspberry Pi

/**************************************************************************/'''
# Setting Up smbus libraries
import smbus
import time

bus = smbus.SMBus(1)

# MCP3426, MCP3427 & MCP3428 addresses are controlled by address lines A0 and A1
# each address line can be low (GND), high (VCC) or floating (FLT)
MCP3427_DEFAULT_ADDRESS = 0x68
MCP3427_CONF_A0GND_A1GND = 0x68
MCP3427_CONF_A0GND_A1FLT = 0x69
MCP3427_CONF_A0GND_A1VCC = 0x6A
MCP3427_CONF_A0FLT_A1GND = 0x6B
MCP3427_CONF_A0VCC_A1GND = 0x6C
MCP3427_CONF_A0VCC_A1FLT = 0x6D
MCP3427_CONF_A0VCC_A1VCC = 0x6E
MCP3427_CONF_A0FLT_A1VCC = 0x6F

# /RDY bit definition
MCP3427_CONF_NO_EFFECT = 0x00
MCP3427_CONF_RDY = 0x80

# Conversion mode definitions
MCP3427_CONF_MODE_ONESHOT = 0x00
MCP3427_CONF_MODE_CONTINUOUS = 0x10

# Channel definitions
# MCP3425 have only the one channel
# MCP3426 & MCP3427 have two channels and treat 3 & 4 as repeats of 1 & 2 respectively
# MCP3428 have all four channels
MCP3427_CONF_CHANNEL_1 = 0x00
MCP3427_CHANNEL_2 = 0x20
'''MCP342X_CHANNEL_3			= 0x40
   MCP342X_CHANNEL_4			= 0x60
'''

# Sample size definitions - these also affect the sampling rate
# 12-bit has a max sample rate of 240sps
# 14-bit has a max sample rate of  60sps
# 16-bit has a max sample rate of  15sps
MCP3427_CONF_SIZE_12BIT = 0x00
MCP3427_CONF_SIZE_14BIT = 0x04
MCP3427_CONF_SIZE_16BIT = 0x08
# MCP342X_CONF_SIZE_18BIT		= 0x0C

# Programmable Gain definitions
MCP3427_CONF_GAIN_1X = 0x00
MCP3427_CONF_GAIN_2X = 0x01
MCP3427_CONF_GAIN_4X = 0x02
MCP3427_CONF_GAIN_8X = 0x03

# Default values for the sensor
ready = MCP3427_CONF_RDY
channel = MCP3427_CONF_CHANNEL_1
mode = MCP3427_CONF_MODE_CONTINUOUS
rate = MCP3427_CONF_SIZE_12BIT
gain = MCP3427_CONF_GAIN_1X
VRef = 2.048  # 2.048 Volts


# Power on and prepare for general usage.
def initialise():
    # Default :Channel 1,Sample Rate 15SPS(16- bit),Gain x1 Selected

    setRate(ready)
    setChannel(channel)
    setMode(mode)
    setSample(rate)
    setGain(gain)


# Set Ready Bit 
# In read mode ,it indicates the output register has been updated with a new conversion.
# In one-shot Conversion mode,writing initiates a new conversion.
def setRate(ready):
    bus.write_byte(MCP3427_DEFAULT_ADDRESS, ready)


# Set Channel Selection
# As the channels are not used for MCP3427
def setChannel(channel):
    bus.write_byte(MCP3427_DEFAULT_ADDRESS, channel)


# Set Conversion Mode
# 1= Continous Conversion Mode
# 0 = One-shot Conversion Mode
def setMode(mode):
    bus.write_byte(MCP3427_DEFAULT_ADDRESS, mode)


# Set Sample rate selection bit
# 00 : 240 SPS-12 bits
# 01 : 60 SPS 14 bits
# 10 : 15 SPS 16 bits
def setSample(rate):
    bus.write_byte(MCP3427_DEFAULT_ADDRESS, rate)


# Set the PGA gain
# 00 : 1 V/V
# 01 : 2 V/V
# 10 : 4 V/V
# 11 : 8 V/V
def setGain(gain):
    bus.write_byte(MCP3427_DEFAULT_ADDRESS, gain)


# Get the measurement for the ADC values  from the register
# using the General Calling method

def getadcread():
    data = bus.read_i2c_block_data(MCP3427_DEFAULT_ADDRESS, 0x00, 2)
    value = ((data[0] << 8) | data[1])
    if (value >= 32768):
        value = 65536 - value
    return value


# The output code is proportional to the voltage difference b/w two analog points
# Checking the conversion value
# Conversion of the raw data into
# Shows the output codes of input level using 16-bit conversion mode

def getconvert():
    code = getadcread()
    # setSample(rate)
    N = 16  # resolution,number of bits
    voltage = (2 * VRef * code) / (2 ** N)
    return voltage


# Initialising the Device.
initialise()

voltage = 0.00

while True:
    time.sleep(0.5)
    voltage = getconvert()
    print("Voltage : %.2f" % voltage)
    "		MCP3427 Readings "