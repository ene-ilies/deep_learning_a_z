#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 12:39:09 2018

@author: bogdan-ilies
"""

# Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend

# Building the CNN
classifier = Sequential()

# Adding convolution layer
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
# Adding max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully Connected
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the images to CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=50,
        validation_data=test_set,
        validation_steps=800/32)

real_img_gen = ImageDataGenerator(rescale=1./255)

image_generator = real_img_gen.flow_from_directory(
	'dataset/single_prediction',
	target_size=(128, 128),
	class_mode=None,
	shuffle=False)

real_image_pred = classifier.predict_generator(image_generator,
	steps=1)

backend.clear_session()

# flip class indices
label_map = dict((v, k) for k,v in training_set.class_indices.items())

class_idx_mapper = lambda val: (0, 1)[bool(val > 0.5)]
category_mapper = lambda val: label_map[class_idx_mapper(val)]
real_image_pred = [category_mapper(pred) for pred in real_image_pred]

print("Predictions: ")
print(real_image_pred)
