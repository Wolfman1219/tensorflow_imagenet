import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
import tensorflow_datasets as tfds
import numpy as np
dataset = tfds.load("siscore/rotation", split = "test")
dataset = tfds.load("siscore/rotation", split = "test")
batch_size = 32
dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# class MyModel(Model):
#     def __init__(self, input_shape):
#         super(MyModel, self).__init__()
#         self.conv1 = Conv2D(32, 3, activation='relu', input_shape=input_shape)
#         # self.conv1 = Conv2D(32, 3, activation='relu', )
#         self.max1 = MaxPool2D((2, 2))
#         self.conv2 = Conv2D(16, (3, 3), activation='relu')
#         self.flatten = Flatten()
#         self.d1 = Dense(128, activation='relu')
#         self.d2 = Dense(1000)

#     def call(self, x):
#         x = self.conv1(x)
#         x = self.max1(x)
#         x = self.conv2(x)
#         x = self.flatten(x)
#         x = self.d1(x)
#         return self.d2(x)





class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.conv1 = Conv2D(64, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv1_1 = Conv2D(64, kernel_size = (3, 3),  activation='relu', padding = 'same')
        
        self.conv2 = Conv2D(128, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv2_1 = Conv2D(128, kernel_size = (3, 3),  activation='relu', padding = 'same')
        
        self.conv3 = Conv2D(256, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv3_1 = Conv2D(256, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv3_2 = Conv2D(256, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv3_3 = Conv2D(256, kernel_size = (3, 3),  activation='relu', padding = 'same')
        
        self.conv4 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv4_1 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv4_2 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv4_3 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        
        self.conv5 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv5_1 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv5_2 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        self.conv5_3 = Conv2D(512, kernel_size = (3, 3),  activation='relu', padding = 'same')
        
        self.pooling = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')
        
        self.drop = Dropout(rate = 0.2)
        self.flatten = Flatten()
#         self.dense1 = Dense(4096, activation='relu')
#         self.dense2 = Dense(2048, activation='relu')
        self.dense3 = Dense(1024, activation='relu')
#         self.dense4 = Dense(512, activation='relu')
        self.dense_output = Dense(1000, activation = 'softmax')

    def call(self, x):
        
        # Block 1
        cnn1_1 = self.conv1(x)
        cnn1_2 = self.conv1_1(cnn1_1)
        pool = self.pooling(cnn1_2)

        
        #  Block 2
        cnn2_1 = self.conv2(pool)
        cnn2_2 = self.conv2_1(cnn2_1)
        pool = self.pooling(cnn2_2)
        
        # Block 3
        cnn3_1 = self.conv3(pool)
        cnn3_2 = self.conv3_1(cnn3_1)
        cnn3_3 = self.conv3_2(cnn3_2)
        cnn3_4 = self.conv3_3(cnn3_3)
        pool = self.pooling(cnn3_4)
        
        # Block 4
        cnn4_1 = self.conv4(pool)
        cnn4_2 = self.conv4_1(cnn4_1)
        cnn4_3 = self.conv4_2(cnn4_2)
        cnn4_4 = self.conv4_3(cnn4_3)
        pool = self.pooling(cnn4_4)
        
        # Block 5
        cnn5_1 = self.conv5(pool)
        cnn5_2 = self.conv5_1(cnn5_1)
        cnn5_3 = self.conv5_2(cnn5_2)
        cnn5_4 = self.conv5_3(cnn5_3)
        pool = self.pooling(cnn5_4)
        
        # Block 6   
        flatten_1  = self.flatten(pool)
#         hidden_1 = self.dense1(flatten_1)
#         drop_1 = self.drop(hidden_1)
#         hidden_2 = self.dense2(drop_1)
#         drop_2 = self.drop(hidden_2)
        hidden_3 = self.dense3(flatten_1)
#         hidden_4 = self.dense4(hidden_3)
        return self.dense_output(hidden_3)

# Create an instance of the model
model = MyModel()




# Create an instance of the model
# model = MyModel((512, 512, 3))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 5
for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  for data in dataset:
      images = data["image"]
      labels = data["label"]
      # print(images.shape)
      # print(labels/255.0)
      train_step(tf.cast(images, tf.float32) / 255.0, tf.cast(labels, tf.float32) / 255.0)


  # for exampe in tfds.as_numpy(img['test']):
  #     if sanoq !=32:
  #       image.append(exampe['image']/255.0)
  #       label.append(exampe['label']/255.0)
  #       continue
  #     sanoq = 0
  #     image_np = np.array(image)
  #     label_np = np.array(label)
  #     train_step(image_np, label_np)
  #     image.clear()
  #     label.clear()


#   for test_images, test_labels in test_ds:
#     test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )