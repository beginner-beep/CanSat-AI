
from datetime import datetime

class SensorDecoder:

    
    @staticmethod
    def decode_message(message):

        if not message or len(message) < 2:
            return None
        
        message_type = message[0]
        data_str = message[2:] 
        
        try:
            if message_type == 'B':
                return SensorDecoder._decode_bme(data_str)
            elif message_type == 'G':
                return SensorDecoder._decode_gps(data_str)
            elif message_type == 'C':
                return SensorDecoder._decode_contour(data_str)
            else:
                print(f"Unknown message type: {message_type}")
                return None
            
        except Exception as e:
            print(f"Error decoding message '{message}': {e}")
            return None
    
    @staticmethod
    def _decode_bme(data_str):
        """Decode BME (temperature, humidity, pressure) data"""
        parts = data_str.split(',')
        if len(parts) != 4:
            raise ValueError(f"BME data should have 4 values, got {len(parts)}")
        
        temperature = float(parts[0])
        humidity = float(parts[1])
        pressure = float(parts[2])
        unix_timestamp = int(parts[3])
        
        return {
            'type': 'BME',
            'temperature_c': temperature,
            'humidity_percent': humidity,
            'pressure_hpa': pressure,
            'timestamp': unix_timestamp,
            'datetime': datetime.fromtimestamp(unix_timestamp).isoformat(),
            'raw': f"Temp: {temperature}°C, Humidity: {humidity}%, Pressure: {pressure} hPa"
        }
    @staticmethod
    def _decode_contour(data_str):
        parts = [p for p in data_str.split(',') if p] 
        coords = list(map(int, parts))

        if len(coords) % 2 != 0:
            coords = coords[:-1]

        points = []
        for i in range(0, len(coords), 2):
           points.append((coords[i], coords[i+1]))

        return {
            'type': 'CONTOUR',
            'points': points
        }

    @staticmethod
    def _decode_gps(data_str):
        """Decode GPS (latitude, longitude, altitude) data"""
        parts = data_str.split(',')
        if len(parts) != 4:
            raise ValueError(f"GPS data should have 4 values, got {len(parts)}")
        
        latitude = float(parts[0])
        longitude = float(parts[1])
        altitude = float(parts[2])
        unix_timestamp = int(parts[3])
        
        return {
            'type': 'GPS',
            'latitude': latitude,
            'longitude': longitude,
            'altitude_m': altitude,
            'timestamp': unix_timestamp,
            'datetime': datetime.fromtimestamp(unix_timestamp).isoformat(),
            'raw': f"Lat: {latitude}°, Lon: {longitude}°, Alt: {altitude}m"
        }


def format_data(decoded_data):
    """Format decoded data for display or logging"""
    if not decoded_data:
        return "Invalid data"
    
    msg_type = decoded_data.get('type')
    timestamp = decoded_data.get('datetime')
    
    if msg_type == 'BME':
        return (f"[{timestamp}] BME: "
                f"Temp={decoded_data['temperature_c']}°C, "
                f"Humidity={decoded_data['humidity_percent']}%, "
                f"Pressure={decoded_data['pressure_hpa']} hPa")
    
    elif msg_type == 'GPS':
        return (f"[{timestamp}] GPS: "
                f"Lat={decoded_data['latitude']}°, "
                f"Lon={decoded_data['longitude']}°, "
                f"Alt={decoded_data['altitude_m']}m")
    elif msg_type == 'CONTOUR':
     return f"[CONTOUR] Points: {decoded_data['points']}"

    
    return "Unknown data type"

if __name__ == "__main__":
    test_messages = [
        "B:23.5,45.1,1013.2,1739567445",
        "G:40.7,-74.0,324.5,1739567445",
        "B:22.1,48.3,1012.8,1739567446",
    ]
    
    print("Sensor Data Decoder - Test Output\n")
    decoder = SensorDecoder()
    
    for msg in test_messages:
        decoded = decoder.decode_message(msg)
        if decoded:
            print(f"Raw: {msg}")
            print(f"Decoded: {format_data(decoded)}")
            print()
