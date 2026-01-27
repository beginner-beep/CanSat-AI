import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import StringIO

def parse_putty_log(log_text):
    """Parse PuTTY log format and extract sensor data."""
    # Regex to extract temperature, humidity, pressure, and time
    temp_pattern = re.compile(
        r"Temperature(?P<temp>[\d.]+),\s*humidity(?P<hum>[\d.]+),pressure(?P<pres>[\d.]+),time:\s*(?P<time>.+)"
    )
    # Regex to extract GPS coordinates
    gps_pattern = re.compile(
        r"Latitude=(?P<lat>[-\d.]+)and\s*Longitude=(?P<lon>[-\d.]+)"
    )
    
    records = []
    current_record = {}
    
    for line in log_text.split('\n'):
        # Try to match temperature/humidity/pressure data
        temp_match = temp_pattern.search(line)
        if temp_match:
            if current_record and 'temp' in current_record:
                records.append(current_record)
            current_record = temp_match.groupdict()
        
        # Try to match GPS data
        gps_match = gps_pattern.search(line)
        if gps_match:
            current_record.update(gps_match.groupdict())
    
    # Don't forget the last record
    if current_record and 'temp' in current_record:
        records.append(current_record)
    
    return records

def create_plots(df):
    """Create matplotlib plots for the sensor data."""
    # Create subplots - 5 plots (temp, humidity, pressure, GPS 2D, GPS 3D)
    fig = plt.figure(figsize=(14, 16))
    
    # Temperature plot
    ax0 = plt.subplot(5, 1, 1)
    ax0.plot(df.index, df["temp"], color='red', marker='o', linestyle='-', linewidth=2)
    ax0.set_ylabel("Temperature (°C)", fontsize=12)
    ax0.set_title("Temperature Over Time", fontsize=14, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='x', rotation=45)
    
    # Humidity plot
    ax1 = plt.subplot(5, 1, 2)
    ax1.plot(df.index, df["hum"], color='blue', marker='s', linestyle='-', linewidth=2)
    ax1.set_ylabel("Humidity (%)", fontsize=12)
    ax1.set_title("Humidity Over Time", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Pressure plot
    ax2 = plt.subplot(5, 1, 3)
    ax2.plot(df.index, df["pres"], color='green', marker='^', linestyle='-', linewidth=2)
    ax2.set_ylabel("Pressure (hPa)", fontsize=12)
    ax2.set_title("Pressure Over Time", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # GPS 2D plot - trajectory
    ax3 = plt.subplot(5, 1, 4)
    if 'lat' in df.columns and 'lon' in df.columns:
        # Convert to float, handling non-numeric values
        lat = pd.to_numeric(df["lat"], errors='coerce')
        lon = pd.to_numeric(df["lon"], errors='coerce')
        
        # Only plot points with valid coordinates
        valid = ~(lat.isna() | lon.isna())
        if valid.any():
            scatter = ax3.scatter(lon[valid], lat[valid], c=range(valid.sum()), cmap='viridis', s=100, alpha=0.7)
            ax3.plot(lon[valid], lat[valid], 'k-', alpha=0.3, linewidth=1)
            ax3.set_xlabel("Longitude", fontsize=12)
            ax3.set_ylabel("Latitude", fontsize=12)
            ax3.set_title("GPS Trajectory (2D)", fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid GPS data', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title("GPS Trajectory 2D (No valid data)", fontsize=14, fontweight='bold')
    
    # GPS 3D plot - trajectory with time as z-axis
    ax4 = plt.subplot(5, 1, 5, projection='3d')
    if 'lat' in df.columns and 'lon' in df.columns:
        lat = pd.to_numeric(df["lat"], errors='coerce')
        lon = pd.to_numeric(df["lon"], errors='coerce')
        
        # Convert time to numeric values (seconds since first timestamp)
        time_numeric = (df.index - df.index[0]).total_seconds()
        
        # Only plot points with valid coordinates
        valid = ~(lat.isna() | lon.isna())
        if valid.any():
            # Create 3D scatter plot
            scatter = ax4.scatter(lon[valid], lat[valid], time_numeric[valid], 
                                 c=time_numeric[valid], cmap='plasma', s=100, alpha=0.7)
            ax4.plot(lon[valid], lat[valid], time_numeric[valid], 'k-', alpha=0.3, linewidth=1)
            ax4.set_xlabel("Longitude", fontsize=10)
            ax4.set_ylabel("Latitude", fontsize=10)
            ax4.set_zlabel("Time (seconds)", fontsize=10)
            ax4.set_title("GPS Trajectory (3D - Time as Height)", fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4, pad=0.1, shrink=0.8)
            cbar.set_label("Time (seconds)", fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No valid GPS data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title("GPS Trajectory 3D (No valid data)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Read from putty.log file
    log_file = "putty.log"
    
    try:
        with open(log_file, 'r') as f:
            log_data = f.read()
    except FileNotFoundError:
        print(f"Error: {log_file} not found in current directory.")
        print("Make sure putty.log is in the same directory as this script.")
        exit(1)
    
    # Parse the log data
    records = parse_putty_log(log_data)
    
    if not records:
        print("Error: No sensor data found in log file.")
        exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    df["temp"] = df["temp"].astype(float)
    df["hum"] = df["hum"].astype(float)
    df["pres"] = df["pres"].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    
    # Set time as index
    df.set_index("time", inplace=True)
    
    # Display data summary
    print("Data Summary:")
    print(df)
    print("\nStatistics:")
    print(df.describe())
    
    # Create and show plots
    fig = create_plots(df)
    plt.show()
