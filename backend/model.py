# imports
from xml.parsers.expat import model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt  # visualize digits during development


print(tf.__version__)


# tensorflow + keras CNN model --> exported to ONNX --> load at app startup

# 1. loading MNIST  dataset to work off of; will augment with my own data later
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training samples: {x_train.shape}")   # (60000, 28, 28)
print(f"Test samples:     {x_test.shape}")    # (10000, 28, 28)
print(f"Labels look like: {y_train[:10]}")    # [5 0 4 1 9 2 1 3 1 4]

# 2. normalize
# pixel values are 0-255, we squish them to 0.0-1.0
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# 3. reshape for CNN
# CNN expects (samples, height, width, channels)
# MNIST is grayscale so channels = 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

print(f"Shape after reshape: {x_train.shape}")  # (60000, 28, 28, 1)


# 4. peeking at the data
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.show()


# 5. building the CNN
model = keras.Sequential([

    # --- block 1: first conv layer ---
    # 32 filters, each 3x3, looks for basic features (edges, curves)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),   # shrinks the image by half

    # --- block 2: second conv layer ---
    # 64 filters, looks for more complex features (loops, corners)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # --- flatten + classify ---
    layers.Flatten(),              # unrolls 2D into 1D
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),           # randomly turns off 50% of neurons during training
                                   # prevents memorizing, forces generalization
    layers.Dense(10, activation='softmax')  # 10 outputs, one per digit 0-9
])

model.summary()  # prints a table showing each layer's shape + param count


# 6. compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # for integer labels (0-9)
    metrics=['accuracy']
)




# 7. train on mnist
history = model.fit(
    x_train, y_train,
    epochs=5,              # 5 passes through the full dataset
    batch_size=64,         # process 64 images at a time
    validation_split=0.1   # hold back 10% of training data to check progress
)



# 8. evaluate on test
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")  # should be ~99%




# 9. save model
model.save('backend/data/mnist_model.keras')
print("Model saved.")