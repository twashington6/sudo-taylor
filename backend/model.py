# FIRST STEP: GET THE BASE MODEL
# tensorflow + keras CNN model on MNIST data --> exported

# imports
from xml.parsers.expat import model

import tensorflow as tf
# TensorFlow is the ML FRAMEWORK -- think of it as the engine.
# It handles all the math (matrix multiplications, gradients)

from tensorflow import keras
# Keras is the HIGH-LEVEL API built into TensorFlow.
# TensorFlow alone is very low-level (like assembly language).
# Keras is like a friendly wrapper -- it lets one say
# "add a convolutional layer" instead of writing the math themselves.

from tensorflow.keras import layers
# The actual building blocks of the neural network.
# Conv2D, Dense, Dropout etc. all live here.

import numpy as np
# NumPy = numerical Python. Neural networks are just arrays of numbers.
# np handles array manipulation, reshaping, math operations.

import matplotlib.pyplot as plt  # visualize digits during development
# Visualization library. Useful during model development to peek at the data and see how the model is doing.

print(tf.__version__)



# 1. loading MNIST  dataset to work off of; will augment with my own data later
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# MNIST is a famous benchmark dataset: 70,000 handwritten digits.
# Already split into train (60k) and test (10k).
# x = the images (pixel data)
# y = the labels (the actual digit each image represents)


print(f"Training samples: {x_train.shape}")   # (60000, 28, 28)
print(f"Test samples:     {x_test.shape}")    # (10000, 28, 28)
print(f"Labels look like: {y_train[:10]}")    # [5 0 4 1 9 2 1 3 1 4]

# 2. normalize
# pixel values are 0-255, we squish them to 0.0-1.0
x_train = x_train / 255.0
x_test  = x_test  / 255.0
# WHY: Neural networks learn via gradient descent --
# they adjust weights by tiny amounts each step.
# Large input values (like 255) make the gradients unstable
# and training slow. Small values (0-1) keep everything well-behaved.
# Normalizing inputs is almost ALWAYS step 1 in any ML pipeline.


# 3. reshape for CNN
# CNN expects (samples, height, width, channels)
# MNIST is grayscale so channels = 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)
# CNNs expect input shape: (samples, height, width, channels)
# MNIST images are 28x28 pixels, grayscale (1 channel).
# Color images would be (samples, height, width, 3) -- RGB.
# The -1 means "figure out this dimension automatically"
# (so -1 becomes 60000 for train, 10000 for test).

print(f"Shape after reshape: {x_train.shape}")  # (60000, 28, 28, 1)


# 4. peeking at the data
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.show()
# Want to confirm: are the images readable? Are labels correct?
# Garbage in = garbage out. No amount of model tuning fixes bad data.


# 5. building the CNN
model = keras.Sequential([
# Sequential = layers stacked in a line, one after another.
# Data flows: input -> layer1 -> layer2 -> ... -> output.
# The alternative is functional API (for branching/complex architectures)

    # --- block 1: first conv layer ---
    # 32 filters, each 3x3, looks for basic features (edges, curves)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # THE KEY LAYER. Conv2D slides a small window (3x3) across the image,
    # looking for patterns. 32 means 32 different filters --
    # each filter learns to detect something different
    # (one might find horizontal edges, another finds curves, etc).
    # "relu" = Rectified Linear Unit. The activation function.
    # It just does max(0, x) -- kills negative values.
    # WHY relu: without activation functions, stacking layers does nothing
    # (linear + linear = linear). relu adds non-linearity,
    # letting the network learn complex patterns.
    
    layers.MaxPooling2D((2, 2)),   # shrinks the image by half
    # Shrinks the image by half by taking the MAX value in each 2x2 block.
    # WHY: makes the network less sensitive to exact position of features.
    # A "7" is still a "7" whether it's slightly left or right.
    # Also reduces computation for the next layers.

    # --- block 2: second conv layer ---
    # 64 filters, looks for more complex features (loops, corners)
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Second conv layer with 64 filters (more than first).
    # WHY more filters: early layers detect simple things (edges).
    # Later layers combine those into complex things (loops, corners).
    # More filters = more capacity to learn complex combinations.

    layers.MaxPooling2D((2, 2)),
    

    # --- flatten + classify ---
    layers.Flatten(), # unrolls 2D into 1D
    # Unrolls the 2D feature maps into a 1D array.
    # Think of it as: we've extracted all the visual features,
    # now we need to feed them into a regular classifier.

    layers.Dense(128, activation='relu'),
    # A fully-connected layer -- every neuron connects to every input.
    # This is where the network combines all the features it found
    # and starts making sense of them as a whole digit.
    # 128 = number of neurons (a hyperparameter that can be tuned).

    layers.Dropout(0.5), # randomly turns off 50% of neurons during training
                         # prevents memorizing, forces generalization
    # During training, randomly turns off 50% of neurons each pass.
    # WHY: forces the network not to rely on any single neuron.
    # Prevents OVERFITTING -- memorizing training data instead of
    # learning generalizable patterns.
    # Think of it as: studying with random flashcards removed
    # forces one to actually understand, not just memorize.

    layers.Dense(10, activation='softmax')  # 10 outputs, one per digit 0-9
    # Output layer. 10 neurons = one per digit (0-9).
    # softmax converts raw scores to PROBABILITIES that sum to 1.
    # e.g. [0.01, 0.02, 0.90, 0.03, ...] → model thinks it's a 2.
    # The highest probability is the prediction.
])

model.summary()  # prints a table showing each layer's shape + param count


# 6. compile
model.compile(
    optimizer='adam',
    # Adam = Adaptive Moment Estimation.
    # The algorithm that UPDATES the weights during training.
    # It figures out HOW MUCH to adjust each weight after each batch.
    # Adam is the safe default -- works well on almost everything.

    loss='sparse_categorical_crossentropy',  # for integer labels (0-9)
    # The LOSS FUNCTION measures how wrong the model is.
    # Training = minimizing this number.
    # sparse_categorical = labels are integers (0,1,2...)
    # vs categorical_crossentropy = labels are one-hot ([0,0,1,0,...])
    # Cross-entropy specifically penalizes confident wrong answers heavily.

    metrics=['accuracy']
    # What to DISPLAY during training. Doesn't affect the model --
    # just for visibility. Accuracy = % of correct predictions.
)



# 7. train on mnist
history = model.fit(
    x_train, y_train,
    epochs=5, # 5 passes through the full dataset
    # One epoch = the model sees every training sample once.
    # 5 epochs = 5 full passes through 60,000 images.
    # More epochs isn't always better -- can overfit.

    batch_size=64, # process 64 images at a time
    # Instead of updating weights after every single image (slow)
    # or after all 60k images (memory intensive),
    # we update after every 64 images. Good balance.

    validation_split=0.1   # hold back 10% of training data to check progress
    # Hold back 10% of training data to check progress
    # on data the model hasn't trained on.
    # If train accuracy >> val accuracy, overfitting.
)



# 8. evaluate on test
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")  # should be ~99%




# 9. save model
model.save('backend/data/mnist_model.keras')
print("Model saved.")