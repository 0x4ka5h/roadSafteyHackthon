import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import numpy as np

os.chdir("DenseofDepth")

#os.chdir('DenseofDepth')
list_ = os.listdir()

x=[]
y=[]

def imgArray(img):
	image = tf.io.read_file(img)
	img = tf.image.decode_png(image)
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, [128, 128])
	return np.asarray(img)
def label(l):
	return int(l.split("-")[0])

for i in list_:
	x.append(imgArray(i))
	y.append(label(i))
x = np.asarray(x)
y = np.asarray(y)

#y.shape

y.dtype = np.float32

s = []
for i in range(len(y)):
    if i%2==0:
      s.append(y[i])

y = np.asarray(s)

xTrain, xTest, yTrain, yTest = train_test_split(x, y)

model= Sequential([
    tf.keras.layers.Input(shape = xTrain.shape[1:]),
    Dense(300, activation='relu'),  #activation layer is linear with 200 epoch is used for final trained model
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='softmax'),
    Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')

model.compile(
    loss=loss_fn,
    optimizer=Adam()
)

history = model.fit(xTrain, yTrain, epochs=20, validation_data=(xTest, yTest))

print(history.predict(xTest[0]))



