# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage as ndi

# model metrics imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# load the saved MNIST model
model = keras.models.load_model('backend/data/mnist_model.keras')

model.summary()  # confirm it loaded correctly

# load handwriting data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,       # slightly rotate images for variety
    width_shift_range=0.1,   # slight horizontal shift
    height_shift_range=0.1,  # slight vertical shift
    zoom_range=0.1,          # slight zoom
    validation_split=0.2
)

# create training set
train_data = datagen.flow_from_directory(
    'backend/data/handwriting',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='training'
)

# create validation set
val_data = datagen.flow_from_directory(
    'backend/data/handwriting',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='validation'
)

# freeze early layers to preserve learned features
# the early conv layers already know how to see edges/curves from MNIST
# we only want to adjust the final classification layers for your style
model.layers[0].trainable = False  # freeze conv block 1
model.layers[2].trainable = False  # freeze conv block 2

# recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # smaller lr for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# fine-tune on handwriting data
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)


# confusion matrix on validation set
# get true labels and predictions from validation data
val_data.reset()  # reset to start of data
y_true = []
y_pred = []

for i in range(len(val_data)):
    x_batch, y_batch = val_data[i]
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(y_batch.astype(int))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# plot it
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion matrix — my handwriting validation set')
plt.tight_layout()
plt.show()

# print the worst offenders
print("\nMost confused pairs:")
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)  # ignore correct predictions
for _ in range(5):
    idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    print(f"  True: {idx[0]}  →  Predicted as: {idx[1]}  ({cm_copy[idx]} times)")
    cm_copy[idx] = 0

# save the final model
model.save('backend/data/final_model.keras')
print("Fine-tuned model saved.")