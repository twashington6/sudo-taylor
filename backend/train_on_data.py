# THIRD STEP
# train model from model.py on the handwriting data collected with collect_data.py --> obtain fine-tuned model final_model.keras

# Fine-tuning = take a model that already knows something,
# teach it something more specific without forgetting what it knew.
# Classic example: a model trained on ImageNet (1M photos)
# fine-tuned to recognize specific dog breeds.
# MY case: MNIST -> my specific handwriting style.

# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage as ndi

# model metrics imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight

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

# calculate weights to compensate for imbalance
labels = train_data.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(weights))


# freeze early layers to preserve learned features
# the early conv layers already know how to see edges/curves from MNIST
# we only want to adjust the final classification layers for style
#model.layers[0].trainable = False  # freeze conv block 1
#model.layers[2].trainable = False  # freeze conv block 2

# only unfreeze the dense layers at the end
for layer in model.layers:
    layer.trainable = False


# WHY freeze early layers:
# Conv layers learn VISUAL PRIMITIVES: edges, curves, corners.
# These are universal -- a curve is a curve regardless of who drew it.
# No need to relearn from scratch. Freezing preserves this knowledge.
# Only the final Dense layers need to adjust to personal style.

model.layers[-1].trainable = True  # output layer
model.layers[-2].trainable = True  # dropout
model.layers[-3].trainable = True  # dense 128

# recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # smaller lr for fine-tuning
    # WHY a smaller learning rate for fine-tuning (0.0001 vs 0.001):
    # The model already has good weights from MNIST.
    # A large learning rate would take big steps and overwrite that knowledge.
    # Small steps = subtle adjustments = keeps MNIST knowledge intact.
    
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,           # stop if no improvement for 3 epochs
    restore_best_weights=True  # revert to best epoch automatically
)

# fine-tune on handwriting data
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    class_weight=class_weights, # handling class imbalance
    # class_weight -- BALANCING:
    # If have 100 zeros but only 40 fives,
    # the model sees zeros 2.5x more often and biases toward them.
    # class_weight tells the model "mistakes on rare classes cost more"
    # effectively leveling the playing field.

    callbacks=[early_stop]
    # EarlyStopping:
    # Monitors val_accuracy. If it doesn't improve for "patience" epochs,
    # training stops and reverts to the best weights.
    # Prevents the model from training past its peak and overfitting.
    # restore_best_weights=True is crucial -- without it you keep
    # the final weights, not the best ones.
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


# overall accuracy on validation set
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
total = len(y_true)
print(f"Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%")

# per digit accuracy
print("\nPer digit accuracy:")
for digit in range(10):
    digit_mask = y_true == digit
    digit_correct = sum((y_true[digit_mask] == y_pred[digit_mask]))
    digit_total = sum(digit_mask)
    print(f"  Digit {digit}: {digit_correct}/{digit_total} = {digit_correct/digit_total*100:.1f}%")

# save the final model
model.save('backend/data/final_model.keras')
print("Fine-tuned model saved.")