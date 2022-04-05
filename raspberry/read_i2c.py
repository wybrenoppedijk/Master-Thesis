import smbus
import time
import plotext as plt
import numpy as np

# Get I2C bus
bus = smbus.SMBus(1)

# I2C address of the device
MCP3428_DEFAULT_ADDRESS = 0x68

# MCP3428 Configuration Command Set
MCP3428_CMD_NEW_CNVRSN = 0x80  # Initiate a new conversion(One-Shot Conversion mode only)
MCP3428_CMD_CHNL_1 = 0x00  # Channel-1 Selected
MCP3428_CMD_CHNL_2 = 0x20  # Channel-2 Selected
MCP3428_CMD_CHNL_3 = 0x40  # Channel-3 Selected
MCP3428_CMD_CHNL_4 = 0x60  # Channel-4 Selected
MCP3428_CMD_MODE_CONT = 0x10  # Continuous Conversion Mode
MCP3428_CMD_MODE_ONESHOT = 0x00  # One-Shot Conversion Mode
MCP3428_CMD_SPS_240 = 0x00  # 240 SPS (12-bit)
MCP3428_CMD_SPS_60 = 0x04  # 60 SPS (14-bit)
MCP3428_CMD_SPS_15 = 0x08  # 15 SPS (16-bit)
MCP3428_CMD_GAIN_1 = 0x00  # PGA Gain = 1V/V
MCP3428_CMD_GAIN_2 = 0x01  # PGA Gain = 2V/V
MCP3428_CMD_GAIN_4 = 0x02  # PGA Gain = 4V/V
MCP3428_CMD_GAIN_8 = 0x03  # PGA Gain = 8V/V
MCP3428_CMD_READ_CNVRSN = 0x00  # Read Conversion Result Data


class MCP3428():
    def __init__(self):
        self.channel = 1


    def config_command(self):
        """Select the Configuration Command from the given provided values"""
        if self.channel == 1:
            CONFIG_CMD = (MCP3428_CMD_MODE_CONT | MCP3428_CMD_SPS_240 | MCP3428_CMD_GAIN_2 | MCP3428_CMD_CHNL_1)
        elif self.channel == 2:
            CONFIG_CMD = (MCP3428_CMD_MODE_CONT | MCP3428_CMD_SPS_240 | MCP3428_CMD_GAIN_1 | MCP3428_CMD_CHNL_2)
        elif self.channel == 3:
            CONFIG_CMD = (MCP3428_CMD_MODE_CONT | MCP3428_CMD_SPS_240 | MCP3428_CMD_GAIN_1 | MCP3428_CMD_CHNL_3)
        elif self.channel == 4:
            CONFIG_CMD = (MCP3428_CMD_MODE_CONT | MCP3428_CMD_SPS_15 | MCP3428_CMD_GAIN_2 | MCP3428_CMD_CHNL_4)
        bus.write_byte(MCP3428_DEFAULT_ADDRESS, CONFIG_CMD)

    def read_adc(self):
        """Read data back from MCP3428_CMD_READ_CNVRSN(0x00), 2 bytes
        raw_adc MSB, raw_adc LSB"""
        data = bus.read_i2c_block_data(MCP3428_DEFAULT_ADDRESS, MCP3428_CMD_READ_CNVRSN, 2)
        # Convert the data to 16 bits
        raw_adc = data[0] * 256 + data[1]

        if raw_adc > 32767:
            raw_adc -= 65535
        return data, raw_adc


from read_i2c import MCP3428

mcp3428 = MCP3428()
i = 1
results = []
window = 100
while True:
    mcp3428.config_command()
    time.sleep(0.1)
    adc, raw_adc = mcp3428.read_adc()
    results.append(raw_adc)
    if len(results) > window:
        results = results[1:]
        print(results)
        print(f"{np.mean(results)}")