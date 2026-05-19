# this program recognizes activities

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ACTIVITIES = ['jumping_jacks', 'lifting', 'rowing', 'running']
SENSOR_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
CSV_COLUMNS = ['id', 'timestamp'] + SENSOR_COLUMNS

WINDOW_SECONDS = 2
DEFAULT_SAMPLE_RATE = 100
DEFAULT_SENSORS = 'all'
DEFAULT_PREPROCESSING = 'stats_fft'
DEFAULT_KERNEL = 'linear'

FILE_PATTERN = re.compile(
    r'(.+)-(running|rowing|lifting|jumping[_]?jacks)-(20|100)Hz-(hand|pocket)-(\d+)\.csv$',
    re.IGNORECASE,
)


# Helpers
def clean_activity_name(name):
    return name.lower().replace('jumpingjacks', 'jumping_jacks')


def parse_recording(path):
    # get labels from file name
    match = FILE_PATTERN.match(Path(path).name)
    if not match:
        return None

    person, activity, sample_rate, placement, recording_id = match.groups()
    return {
        'path': Path(path),
        'person': person.lower(),
        'activity': clean_activity_name(activity),
        'sample_rate': int(sample_rate),
        'placement': placement.lower(),
        'recording_id': int(recording_id),
    }


# read csv files
def discover_recordings(data_dir):
    recordings = []
    for path in sorted(Path(data_dir).rglob('*.csv')):
        recording = parse_recording(path)
        if recording:
            recordings.append(recording)
    return recordings


def load_recording(recording):
    frame = pd.read_csv(recording['path'])
    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{recording['path']} is missing columns: {missing}")

    frame[SENSOR_COLUMNS] = frame[SENSOR_COLUMNS].apply(pd.to_numeric, errors='coerce')
    frame = frame.dropna(subset=SENSOR_COLUMNS)
    return frame[CSV_COLUMNS].reset_index(drop=True)


# get features
def choose_sensor_columns(samples, sensors):
    samples = np.asarray(samples, dtype=float)

    if sensors == 'acc':
        return samples[:, :3]
    if sensors == 'gyro':
        if samples.shape[1] >= 6:
            return samples[:, 3:6]
        return samples[:, :3]
    if sensors == 'all':
        return samples[:, :6]

    raise ValueError('sensors must be acc, gyro, or all')


def extract_features(
    samples,
    sample_rate,
    sensors=DEFAULT_SENSORS,
    preprocessing=DEFAULT_PREPROCESSING,
):
    values = choose_sensor_columns(samples, sensors)
    feature_values = []

    stat_functions = [
        np.mean,
        np.std,
        np.min,
        np.max,
        np.ptp,
        lambda column: np.sqrt(np.mean(column * column)),
    ]

    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        for stat_function in stat_functions:
            feature_values.append(float(stat_function(column)))

    if sensors == 'all':
        magnitude_blocks = [values[:, :3], values[:, 3:6]]
    else:
        magnitude_blocks = [values]

    for block in magnitude_blocks:
        magnitude = np.linalg.norm(block, axis=1)
        for stat_function in stat_functions:
            feature_values.append(float(stat_function(magnitude)))

        if preprocessing == 'stats_fft':
            spectrum = np.abs(np.fft.rfft(magnitude - magnitude.mean()))
            frequencies = np.fft.rfftfreq(len(magnitude), d=1 / sample_rate)

            if len(spectrum) > 1:
                spectrum = spectrum[1:]
                frequencies = frequencies[1:]
                dominant_frequency = frequencies[np.argmax(spectrum)]
                spectral_energy = np.mean(spectrum * spectrum)
            else:
                dominant_frequency = 0.0
                spectral_energy = 0.0

            feature_values.extend([float(dominant_frequency), float(spectral_energy)])

    return np.asarray(feature_values, dtype=float)


def recording_windows(recording, sensors, preprocessing):
    frame = load_recording(recording)
    data = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    window_size = int(WINDOW_SECONDS * recording['sample_rate'])

    features = []
    labels = []
    for start in range(0, len(data) - window_size + 1, window_size):
        window = data[start : start + window_size]
        features.append(
            extract_features(
                window,
                recording['sample_rate'],
                sensors=sensors,
                preprocessing=preprocessing,
            )
        )
        labels.append(recording['activity'])

    return features, labels


def build_dataset(
    recordings,
    sample_rate=DEFAULT_SAMPLE_RATE,
    sensors=DEFAULT_SENSORS,
    preprocessing=DEFAULT_PREPROCESSING,
):
    all_features = []
    all_labels = []

    for recording in recordings:
        if sample_rate is not None and recording['sample_rate'] != sample_rate:
            continue

        features, labels = recording_windows(recording, sensors, preprocessing)
        all_features.extend(features)
        all_labels.extend(labels)

    if not all_features:
        raise ValueError('No usable training windows found')

    return np.vstack(all_features), np.asarray(all_labels)


def split_recordings(recordings, sample_rate=DEFAULT_SAMPLE_RATE):
    filtered = [
        recording
        for recording in recordings
        if sample_rate is None or recording['sample_rate'] == sample_rate
    ]
    labels = [recording['activity'] for recording in filtered]

    return train_test_split(filtered, test_size=0.2, random_state=42, stratify=labels)


# train classifier
def train_and_test(
    data_dir,
    sample_rate=DEFAULT_SAMPLE_RATE,
    sensors=DEFAULT_SENSORS,
    preprocessing=DEFAULT_PREPROCESSING,
    kernel=DEFAULT_KERNEL,
):
    recordings = discover_recordings(data_dir)
    train_files, test_files = split_recordings(recordings, sample_rate)

    x_train, y_train = build_dataset(train_files, sample_rate, sensors, preprocessing)
    x_test, y_test = build_dataset(test_files, sample_rate, sensors, preprocessing)

    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma='scale'))
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        'model': model,
        'accuracy': float(accuracy_score(y_test, predictions)),
        'confusion_matrix': confusion_matrix(y_test, predictions, labels=ACTIVITIES),
        'sample_rate': sample_rate,
        'sensors': sensors,
        'preprocessing': preprocessing,
        'kernel': kernel,
    }


def train_activity_model(
    data_dir,
    sample_rate=DEFAULT_SAMPLE_RATE,
    sensors=DEFAULT_SENSORS,
    preprocessing=DEFAULT_PREPROCESSING,
    kernel=DEFAULT_KERNEL,
):
    result = train_and_test(data_dir, sample_rate, sensors, preprocessing, kernel)
    recordings = discover_recordings(data_dir)
    x_all, y_all = build_dataset(recordings, sample_rate, sensors, preprocessing)

    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma='scale'))
    model.fit(x_all, y_all)
    result['model'] = model
    return result


def predict_window(trained_model, samples):
    features = extract_features(
        np.asarray(samples, dtype=float),
        trained_model['sample_rate'],
        sensors=trained_model['sensors'],
        preprocessing=trained_model['preprocessing'],
    )
    return trained_model['model'].predict([features])[0]


# compare classifier settings
def compare_configurations(
    data_dirs,
    kernels,
    sample_rates,
    sensor_sets,
    preprocessings,
):
    rows = []

    for dataset_name, data_dir in data_dirs.items():
        for sample_rate in sample_rates:
            for sensors in sensor_sets:
                for preprocessing in preprocessings:
                    for kernel in kernels:
                        result = train_and_test(data_dir, sample_rate, sensors, preprocessing, kernel)
                        rows.append(
                            {
                                'dataset': dataset_name,
                                'accuracy': result['accuracy'],
                                'kernel': kernel,
                                'sample_rate': 'both' if sample_rate is None else sample_rate,
                                'sensors': sensors,
                                'preprocessing': preprocessing,
                            }
                        )

    return pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
