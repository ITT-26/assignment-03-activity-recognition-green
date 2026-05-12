import csv
import time
from DIPPID import SensorUDP

PORT = 5700
ACTIVITY_DURATION = 10
ACTIVITY_DICT = {
    '0': 'running',
    '1': 'rowing',
    '2': 'jumping_jacks',
    '3': 'lifting'
}
PLACEMENT_DICT = {
    '0': 'hand',
    '1': 'pocket'
}
SAMPLE_RATE_DICT = {
    '0': 20,
    '1': 100
}
OUTPUT_DIR = 'data'

request_start = False
is_recording = False

sensor = SensorUDP(PORT)

# Helpers
def get_input():
    name = input("Enter your name: ")
    name = name.strip().lower()
    name = name.replace(" ", "_")
    return name


def countdown(seconds=3):
    for n in range(seconds, 0, -1):
        print(f"Recording starts in {n}...")
        time.sleep(1)


def handle_button_press(value):
    global request_start
    if value == 1:
        request_start = True


def wait_for_button_press():
    global request_start
    request_start = False
    while not request_start:
        time.sleep(0.1)
    while sensor.get_value("button_1") == 1:
        time.sleep(0.1)
    request_start = False

# write csv file
def save_csv(filepath, data):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'timestamp', 'acc_x', 'acc_y',
                        'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        writer.writerows(data)


# get data
def record(sample_rate):
    data = []
    interval = 1 / sample_rate
    total_samples = int(ACTIVITY_DURATION * sample_rate)
    data_id = 1
    start_time = time.perf_counter()

    # wait for valid sensor streams
    while not sensor.get_value('accelerometer') or not sensor.get_value('gyroscope'):
        time.sleep(0.1)

    for sample_index in range(total_samples):
        target_time = start_time + (sample_index * interval)
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        # get sensor data
        acc_data = sensor.get_value('accelerometer')
        gyro_data = sensor.get_value('gyroscope')

        # only record if we have valid data from both sensors
        if acc_data and gyro_data:
            timestamp = time.time()

            # get values from sensor
            acc_x = acc_data.get('x', 0)
            acc_y = acc_data.get('y', 0)
            acc_z = acc_data.get('z', 0)
            gyro_x = gyro_data.get('x', 0)
            gyro_y = gyro_data.get('y', 0)
            gyro_z = gyro_data.get('z', 0)

            # append data row
            data.append([data_id, timestamp, acc_x, acc_y,
                        acc_z, gyro_x, gyro_y, gyro_z])
            data_id += 1
    return data


def main():
    global is_recording
    is_recording = False
    name = get_input()

    # Button 1 callback
    sensor.register_callback('button_1', handle_button_press)

    for activity in ACTIVITY_DICT.values():
        for placement in PLACEMENT_DICT.values():
            for sample_rate in SAMPLE_RATE_DICT.values():
                for iteration in range(5):
                    print(
                        f"Next up: {activity}, {placement} at {sample_rate} Hz ({iteration+1}/5)")
                    # if button 1 is pressed, start recording data for 10 seconds
                    wait_for_button_press()

                    is_recording = True
                    countdown(3)
                    print(
                        f"Recording: {activity}, {placement}, {sample_rate} Hz ({iteration+1}/5)")

                    data = record(sample_rate)
                    is_recording = False

                    filepath = f"{OUTPUT_DIR}/{name}-{activity}-{sample_rate}Hz-{placement}-{iteration+1}.csv"
                    save_csv(filepath, data)
                    print(f"Saved: {filepath}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sensor.disconnect()
        print("Interrupted")
