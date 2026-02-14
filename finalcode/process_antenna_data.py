"""
Process sensor data received from antenna via PuTTY or live serial connection
Decodes compact messages and outputs human-readable format

Usage:
    From PuTTY log file:
        python process_antenna_data.py --file putty_log.txt
    
    From live serial port:
        python process_antenna_data.py --port COM3 --baud 9600
"""

import argparse
import serial
import sys
from pathlib import Path
from sensor_decoder import SensorDecoder, format_data


def process_log_file(filepath):
    """Process a PuTTY log file and decode all sensor messages"""
    print(f"Processing log file: {filepath}\n")
    
    decoder = SensorDecoder()
    message_count = {'BME': 0, 'GPS': 0}
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines only
                if not line or line.startswith('['):
                    continue
                
                # Extract potential message (look for B: or G: pattern)
                messages = extract_messages(line)
                if messages:
                    for msg in messages:
                        decoded = decoder.decode_message(msg)
                        if decoded:
                            msg_type = decoded.get('type')
                            message_count[msg_type] = message_count.get(msg_type, 0) + 1
                            print(f"Line {line_num}: {format_data(decoded)}")
                else:
                    # Print raw line if no sensor messages found
                    print(f"Line {line_num}: {line}")
        
        print(f"\n--- Summary ---")
        print(f"BME messages: {message_count['BME']}")
        print(f"GPS messages: {message_count['GPS']}")
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)


def process_serial(port, baudrate=9600):
    """Process live data from serial connection (antenna via USB/UART)"""
    print(f"Connecting to {port} at {baudrate} baud...")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        decoder = SensorDecoder()
        
        print(f"Connected! Listening for sensor data...\n")
        
        while True:
            try:
                data = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if data:
                    # Extract messages from the line
                    messages = extract_messages(data)
                    if messages:
                        for msg in messages:
                            decoded = decoder.decode_message(msg)
                            if decoded:
                                print(format_data(decoded))
                    else:
                        # Print raw line if no sensor messages found
                        print(data)
            
            except KeyboardInterrupt:
                print("\n\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)
    finally:
        if ser.is_open:
            ser.close()

def extract_messages(line):
    import re
    return re.findall(r'[BGC]:[^BGC]+', line)

#def extract_messages(line):
 #   """Extract compact messages (B:... or G:...) from a line"""
  #  messages = []
    
    # Find all occurrences of B: or G: followed by data
   # import re
  #  pattern = r'[BG]:\d+\.?\d*,[^\s,]+,[^\s,]+,\d+'
    #pattern = r'[BGC]:[^\s]+'
    #matches = re.findall(pattern, line)
    
    #return matches


def main():
    parser = argparse.ArgumentParser(
        description='Decode antenna sensor data from PuTTY logs or live serial'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', help='PuTTY log file to process')
    group.add_argument('--port', '-p', help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    
    parser.add_argument('--baud', '-b', type=int, default=9600, 
                       help='Baud rate for serial connection (default: 9600)')
    
    args = parser.parse_args()
    
    if args.file:
        process_log_file(args.file)
    elif args.port:
        process_serial(args.port, args.baud)


if __name__ == "__main__":
    main()
