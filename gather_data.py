# this program gathers sensor data

# Recommended Packages: DIPPID, pandas
# Create a program called gather_data.py which captures sensor data (accelerometer and
# gyroscope) from the DIPPID device and saves it to a CSV file. This sensor data will be
# used to train a machine learning classifier later.
# Record data sets for each of the following activities: running, rowing, lifting,
# jumpingjacks. A depitcion of those activities can be seen in Figure 1. To add some
# variety in data and capture a big enough data set,- vary sampling rate: 20Hz and 100Hz- vary placement of DIPPID device: in hand and in pocket of pants- recordatleast 5 data sets for each activity, sample rate, and placement (4×2×2×5=80)- for each recording, perform the activity for at least 10 seconds
# It should be possible to use the DIPPID device to start capturing data. Data logging
# stops automatically when a fixed time has passed. Each recorded data set is saved sep
# arately to a CSV file. The CSV files should be sturctured as follows:
# File Name: your_name-activity-sample_rate-placement_number.csv
# (e.g., susi-running-20Hz-pocket-1.csv)
# Columns: id,timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z

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

is_recording = False

sensor = SensorUDP(PORT)

# Helpers


def get_input():
    name = input("Enter your name: ")
    name = name.strip().lower()
    activity = input(
        "Enter activity \n0: Running \n1: Rowing \n2: Jumping Jacks \n3: Lifting:\n")
    if activity in ACTIVITY_DICT:
        activity = ACTIVITY_DICT[activity]
    else:
        raise ValueError("Invalid activity")
    placement = input("Enter placement \n0: Hand \n1: Pocket:\n")
    if placement in PLACEMENT_DICT:
        placement = PLACEMENT_DICT[placement]
    else:
        raise ValueError("Invalid placement")
    sample_rate = input("Enter sample rate \n 0: 20Hz \n 1: 100Hz\n")
    if sample_rate in SAMPLE_RATE_DICT:
        sample_rate = SAMPLE_RATE_DICT[sample_rate]
    else:
        raise ValueError("Invalid sample rate")
    return name, activity, placement, sample_rate


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
    data_id = 0
    start_time = time.perf_counter()

    # wait for valid sensor streams
    while not sensor.get_value('accelerometer') or not sensor.get_value('gyroscope'):
        time.sleep(1)

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
            # TODO: timestamp
            timestamp = sample_index * interval

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


# DIPPID callbacks
def handle_button_press(data):
    global is_recording
    # on release
    if int(data) == 0:
        is_recording = True
    else:
        is_recording = False


def main():
    global is_recording
    iteration = 0
    is_recording = False
    name, activity, placement, sample_rate = get_input()

    sensor.register_callback('button_1', handle_button_press)

    # if button 1 is pressed, start recording data for 10 seconds
    while not is_recording:
        time.sleep(0.1)

    print(
        f"Recording: {activity}, {placement} at {sample_rate} Hz ({iteration}/5)")
    data = record(sample_rate)

    filepath = f"{OUTPUT_DIR}/{name}-{activity}-{sample_rate}-{placement}-{iteration}.csv"
    save_csv(filepath, data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sensor.disconnect()
        print("Interrupted")
