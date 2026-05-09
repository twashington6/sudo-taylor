import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

train_data = datagen.flow_from_directory(
    'backend/data/my_handwriting',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='training'
)

val_data = datagen.flow_from_directory(
    'backend/data/my_handwriting',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='validation'
)

# 11. FREEZE EARLY LAYERS, RETRAIN LAST FEW
# the early conv layers already know how to see edges/curves from MNIST
# we only want to adjust the final classification layers for your style
model.layers[0].trainable = False  # freeze conv block 1
model.layers[2].trainable = False  # freeze conv block 2

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # smaller lr for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 12. FINE-TUNE
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# 13. SAVE FINAL MODEL
model.save('backend/data/final_model.keras')
print("Fine-tuned model saved.")