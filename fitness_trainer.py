# this program visualizes activities with pyglet

from collections import deque
from pathlib import Path

import pyglet

pyglet.options['dpi_scaling'] = 'stretch'

from pyglet import gl, shapes

import activity_recognizer as activity
from DIPPID import SensorUDP


PORT = 5700
DATA_DIR = 'data'
TARGET_SECONDS = 10
ACTIVITY_ORDER = ['running', 'rowing', 'lifting', 'jumping_jacks']
ACTIVITY_LABELS = {
    'running': 'Running',
    'rowing': 'Rowing',
    'lifting': 'Lifting',
    'jumping_jacks': 'Jumping jacks',
}
ACTIVITY_IMAGES = {
    'running': 'running_1.png',
    'rowing': 'rowing_1.png',
    'lifting': 'lifting_1.png',
    'jumping_jacks': 'jumpingjack_1.png',
}


# train model
def train_startup_model():
    print(f"Training model from {DATA_DIR} ...")
    model = activity.train_activity_model(
        DATA_DIR,
        sample_rate=activity.DEFAULT_SAMPLE_RATE,
        sensors=activity.DEFAULT_SENSORS,
        preprocessing=activity.DEFAULT_PREPROCESSING,
        kernel=activity.DEFAULT_KERNEL,
    )

    print(f"Test accuracy: {model['accuracy']:.3f}")
    return model


# fitness trainer window
class FitnessTrainer:
    def __init__(self, model, port, target_seconds, sample_rate):
        self.model = model
        self.target_seconds = target_seconds
        self.sample_rate = sample_rate
        self.window_size = activity.WINDOW_SECONDS * sample_rate
        self.samples = deque(maxlen=self.window_size)
        self.target_index = 0
        self.correct_seconds = 0
        self.last_prediction = 'waiting'
        self.status = 'Waiting for DIPPID data ...'
        self.prediction_timer = 0

        self.sensor = SensorUDP(port)
        self.window = pyglet.window.Window(900, 620, caption='Fitness Trainer', resizable=False)
        self.batch = pyglet.graphics.Batch()

        shapes.Rectangle(0, 0, 900, 620, color=(235, 238, 242), batch=self.batch)
        shapes.Rectangle(210, 185, 480, 350, color=(255, 255, 255), batch=self.batch)
        self.progress_back = shapes.Rectangle(170, 72, 560, 28, color=(190, 198, 210), batch=self.batch)
        self.progress_front = shapes.Rectangle(170, 72, 0, 28, color=(70, 160, 100), batch=self.batch)

        self.title = pyglet.text.Label(
            'Fitness Trainer',
            x=450,
            y=570,
            anchor_x='center',
            font_size=28,
            color=(30, 35, 42, 255),
            batch=self.batch,
        )
        self.target_label = pyglet.text.Label(
            '',
            x=450,
            y=140,
            anchor_x='center',
            font_size=24,
            color=(30, 35, 42, 255),
            batch=self.batch,
        )
        self.prediction_label = pyglet.text.Label(
            '',
            x=450,
            y=112,
            anchor_x='center',
            font_size=16,
            color=(65, 74, 86, 255),
            batch=self.batch,
        )
        self.status_label = pyglet.text.Label(
            '',
            x=450,
            y=42,
            anchor_x='center',
            font_size=14,
            color=(85, 96, 110, 255),
            batch=self.batch,
        )

        self.images = self.load_images()
        self.sprite = None
        self.update_target_image()
        self.update_labels()

        self.window.push_handlers(on_draw=self.on_draw, on_close=self.on_close)
        pyglet.clock.schedule_interval(self.update, 1 / sample_rate)

    def current_target(self):
        return ACTIVITY_ORDER[self.target_index % len(ACTIVITY_ORDER)]

    def load_images(self):
        images = {}
        for activity_name, filename in ACTIVITY_IMAGES.items():
            path = Path('img') / filename
            if path.exists():
                images[activity_name] = pyglet.image.load(str(path))
        return images

    def update_target_image(self):
        image = self.images.get(self.current_target())
        if image is None:
            self.sprite = None
            return

        self.sprite = pyglet.sprite.Sprite(image)
        self.sprite.scale = min(430 / image.width, 330 / image.height)
        self.sprite.x = 450 - (image.width * self.sprite.scale) / 2
        self.sprite.y = 195

    def update_labels(self):
        target_name = ACTIVITY_LABELS.get(self.current_target(), self.current_target())
        prediction_name = ACTIVITY_LABELS.get(self.last_prediction, self.last_prediction)
        progress = min(self.correct_seconds / self.target_seconds, 1)

        self.progress_front.width = int(560 * progress)
        self.target_label.text = f"Target: {target_name}"
        self.prediction_label.text = f"Recognized: {prediction_name}"
        self.status_label.text = self.status

    # get sensor data
    def read_sensor_values(self):
        acc_data = self.sensor.get_value('accelerometer')
        gyro_data = self.sensor.get_value('gyroscope')

        try:
            if self.model['sensors'] == 'acc':
                if not acc_data:
                    return None
                return float(acc_data['x']), float(acc_data['y']), float(acc_data['z'])

            if self.model['sensors'] == 'gyro':
                if not gyro_data:
                    return None
                return float(gyro_data['x']), float(gyro_data['y']), float(gyro_data['z'])

            if not acc_data or not gyro_data:
                return None
            return (
                float(acc_data['x']),
                float(acc_data['y']),
                float(acc_data['z']),
                float(gyro_data['x']),
                float(gyro_data['y']),
                float(gyro_data['z']),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def update(self, dt):
        sensor_values = self.read_sensor_values()
        if sensor_values is None:
            self.status = 'Waiting for DIPPID data ...'
            self.update_labels()
            return

        self.samples.append(sensor_values)
        self.status = f"Live window: {len(self.samples)}/{self.window_size} samples"
        self.prediction_timer += dt

        if len(self.samples) == self.window_size and self.prediction_timer >= 0.25:
            self.prediction_timer = 0
            self.last_prediction = activity.predict_window(self.model, self.samples)

        if self.last_prediction == self.current_target():
            self.correct_seconds += dt
        else:
            self.correct_seconds = 0

        if self.correct_seconds >= self.target_seconds:
            self.target_index += 1
            self.correct_seconds = 0
            self.samples.clear()
            self.last_prediction = 'waiting'
            self.status = 'Good job. Next exercise.'
            self.update_target_image()

        self.update_labels()

    def on_draw(self):
        gl.glClearColor(0.92, 0.93, 0.95, 1.0)
        self.window.clear()
        self.batch.draw()
        if self.sprite:
            self.sprite.draw()

    def on_close(self):
        self.sensor.disconnect()
        pyglet.app.exit()


def main():
    model = train_startup_model()
    FitnessTrainer(model, PORT, TARGET_SECONDS, activity.DEFAULT_SAMPLE_RATE)
    pyglet.app.run()


if __name__ == "__main__":
    main()
