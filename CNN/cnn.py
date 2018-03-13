#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 12:39:09 2018

@author: bogdan-ilies
"""

# Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend

# Building the CNN
classifier = Sequential()

# Adding convolution layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
# Adding max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully Connected
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

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
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=800)

real_img_gen = ImageDataGenerator(rescale=1./255)

image_generator = real_img_gen.flow_from_directory(
	'dataset/single_prediction',
	target_size=(64,64),
	class_mode=None,
	shuffle=False)

real_image_pred = classifier.predict_generator(image_generator,
	val_samples=2)

backend.clear_session()

# flip class indices
label_map = dict((v, k) for k,v in training_set.class_indices.items())

class_idx_mapper = lambda val: (0, 1)[bool(val > 0.5)]
category_mapper = lambda val: label_map[class_idx_mapper(val)]
real_image_pred = [category_mapper(pred) for pred in real_image_pred]

print("Predictions: ")
print(real_image_pred)
