import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPU(s) found: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True) # Optional: only allocate GPU memory as needed
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        print(f"Runtime error during GPU initialization: {e}")
else:
    print("No GPU found. TensorFlow will run on CPU.")