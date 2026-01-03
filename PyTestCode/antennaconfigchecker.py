from lora_e32 import LoRaE32, print_configuration, Configuration
from lora_e32_operation_constant import ResponseStatusCode
import serial

from lora_e32_constants import OperatingFrequency

loraSerial = serial.Serial('/dev/serial0') #, baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
lora = LoRaE32('433T20D', loraSerial, aux_pin=24, m0_pin=18, m1_pin=23)
code = lora.begin()

print("Initialization: {}", ResponseStatusCode.get_description(code))

code, configuration = lora.get_configuration()

def main():

    while True:
        command = input("> ").strip()

        if command == "exit":
            print("closing...")
            break
        elif command.startswith("set CHAN "):
            configuration_to_set = Configuration('433T20D')
            configuration_to_set.OPTION.operatingFrequency = OperatingFrequency.FREQUENCY_433
            lora.set_configuration(configuration_to_set)
        elif command == "show config":
            print("Retrieve configuration: {}", ResponseStatusCode.get_description(code))

            print_configuration(configuration)
        else:
            print("Unknown command")

if __name__ == "__main__":
    main()

