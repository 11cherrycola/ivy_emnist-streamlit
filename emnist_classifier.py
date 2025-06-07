# emnist_classifier.py

import tensorflow as tf
import tensorflow_datasets as tfds

# --- LOAD DATASET ---
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# --- PREPROCESSING: normalisasi & sesuaikan label ---
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # normalize ke 0–1
    image = tf.expand_dims(image, -1)           # (28, 28, 1)
    label = label - 1                           # dari 1–26 jadi 0–25
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 128

train_ds = (ds_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(10000)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_ds = (ds_test
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# --- DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# --- CNN MODEL ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(26, activation='softmax')  # 26 huruf A-Z
])

# --- KOMPILE MODEL ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- TRAINING ---
model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds
)

# --- EVALUASI ---
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
model.save('huruf_model_terbaik.h5') 