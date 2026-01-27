from lora_e32 import LoRaE32, print_configuration, Configuration
from lora_e32_operation_constant import ResponseStatusCode
import serial
from time import sleep
from lora_e32_constants import OperatingFrequency

loraSerial = serial.Serial('/dev/serial0') #, baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
lora = LoRaE32('433T20D', loraSerial, aux_pin=24, m0_pin=18, m1_pin=23)
code = lora.begin()

print("Initialization: {}", ResponseStatusCode.get_description(code))

code, configuration = lora.get_configuration()
print("""
Available Commands:
  set frequency to [num]   Set the LoRa RF channel, where 410<num<441
  show config       Display current LoRa configuration
  reset 			Reset to default parameters
  exit              Exit the program
""")
def main():

    while True:
        command = input("> ").strip()

        if command == "exit":
            print("closing...")
            break
        elif command.startswith("set frequency to "):
            try:
                num = int(command.split()[3])
                if not 410 <= num <= 441:
                    print("frequency must be between 410 MHz and 441 MHz")
                    continue
                configuration_to_set = Configuration('433T20D')
                configuration_to_set.CHAN = num
                lora.set_configuration(configuration_to_set)
                print("frequency set to num")
            except:
                print("failed")
        elif command == "show config":
            code = lora.begin()
            code, configuration = lora.get_configuration()	
            print("Retrieve configuration: {}", ResponseStatusCode.get_description(code))
            print_configuration(configuration)
        elif command == "reset":
            configuration_to_set = Configuration('433T20D')
            code, confSetted = lora.set_configuration(configuration_to_set)
            print(ResponseStatusCode.get_description(code))
            print_configuration(confSetted)
        else:	
            print("Unknown command")

if __name__ == "__main__":
    main()

