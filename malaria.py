import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import tensorflow_datasets as tfds
dataset = tfds.load("malaria", split = "train")
batch_size = 32
def preprocess_data(example):
    image = example['image']
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0
    label = example['label']
    return image, label

dataset = dataset.map(preprocess_data)

dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
vgg19 = tf.keras.applications.vgg19.VGG19(include_top = False, input_shape = (32, 32, 3), weights = 'imagenet')



model_vgg19 = Sequential()
for layer in vgg19.layers:
    layer.trainable = False      ### To turn off VGG19's trainable parameters


model_vgg19.add(vgg19)
model_vgg19.add(Flatten())      
model_vgg19.add(Dense(4096, activation = 'relu'))
model_vgg19.add(Dropout(0.2))
model_vgg19.add(Dense(1024, activation = 'relu'))
model_vgg19.add(Dense(2, activation = 'softmax'))


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)    
optimizer = tf.keras.optimizers.Adam()
model_vgg19.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])


checkpoint = ModelCheckpoint(
    'malaria.h5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='auto'
)
csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)
reduceLR = ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)
callbacks = [checkpoint, earlystop, csvlogger,reduceLR]

history = model_vgg19.fit(
    dataset, 
    epochs = 10,
    callbacks = callbacks,
    shuffle = True
)