import minimalmodbus
import serial
import time


def read_soil_data():
    try:
        instrument = minimalmodbus.Instrument(
            'COM3', 1)  # Update COM port if needed
        instrument.serial.baudrate = 9600
        instrument.serial.bytesize = 8
        instrument.serial.parity = serial.PARITY_NONE
        instrument.serial.stopbits = 1
        instrument.serial.timeout = 2
        instrument.mode = minimalmodbus.MODE_RTU

        print("Waiting for sensor to stabilize...")
        time.sleep(3)  # Give sensor time to power up

        print("Reading sensor data...\n")
        data = instrument.read_registers(0, 14)
        print(f"Raw register data: {data}")

        # Interpret registers with corrected NPK mapping
        soil_temp = data[0] / 10.0     # Temperature in °C
        soil_moisture = data[2] / 10.0  # Moisture in %
        soil_ec = data[3]              # Conductivity in µS/cm
        soil_ph = data[4] / 10.0       # pH value

        # Updated NPK mappings based on testing
        nitrogen = data[1] / 10.0      # N in mg/kg
        phosphorus = data[5] / 10.0    # P in mg/kg
        potassium = data[6] / 10.0     # K in mg/kg

        print("🌱 Soil Sensor Readings:")
        print(f"  Temperature     : {soil_temp:.1f} °C")
        print(f"  Moisture        : {soil_moisture:.1f} %")
        print(f"  Conductivity    : {soil_ec} µS/cm")
        print(f"  pH              : {soil_ph:.1f}")
        print(f"  Nitrogen (N)    : {nitrogen:.1f} mg/kg")
        print(f"  Phosphorus (P)  : {phosphorus:.1f} mg/kg")
        print(f"  Potassium (K)   : {potassium:.1f} mg/kg")

    except IOError:
        print("❌ I/O Error: Check wiring and sensor power.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    read_soil_data()
