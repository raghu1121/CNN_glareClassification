from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import numpy as np




datagen=image_utils.ImageDataGenerator(rescale=1. / 255,validation_split=0.2)

generator=datagen.flow_from_directory('fisheye_bmp',save_to_dir='transformed_pngs',target_size=(224,224),batch_size=10)

#def cnnModel():
model=Sequential()
model.add(Conv2D(62,(5,5),strides=(2, 2),input_shape=(224,224,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (5, 5),strides=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(generator,steps_per_epoch=20,epochs=10)