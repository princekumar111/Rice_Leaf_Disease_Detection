import tensorflow as tf

# Load old h5 model
model = tf.keras.models.load_model(
    "rice_model.h5",
    compile=False
)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization (reduce size)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save new model
with open("rice_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully")