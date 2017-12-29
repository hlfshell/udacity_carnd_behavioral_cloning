import csv
import cv2
import numpy as np
from random import random

import sklearn.utils
import sklearn.model_selection

from keras.layers import Dense, Flatten, Lambda, Conv2D, Cropping2D, Dropout
from keras.models import Sequential

batch_size = 24
epochs = 3
dropout_rate = 0.3

straight_limit = 0.90
straight_drop_percentage = 0.8

side_camera_correction = 0.25

base_path = "./data/"

samples_to_use = [ \
    "course_1",\
    "course_1_reverse",\
    # "course_2",\
    # "course_2_reverse",\
    "course_1_dirt_turn_straight",\
    "course_1_extreme_right_water_straight",\
    "course_1_extreme_corrections"\
]

dropped_data = 0

def load_sample_data(path_to_samples):
    global dropped_data
    samples = []
    with open(path_to_samples + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            steering_angle = abs(float(line[3]))
            if(steering_angle <= straight_limit):
                if(random() >= straight_drop_percentage):
                    samples.append(line)
                else:
                    dropped_data += 1
            else:
                samples.append(line)

    print("Total of {0} samples loaded for {1}".format(len(samples), path_to_samples))
    return samples

samples = []

for sample_path in samples_to_use:
    samples.extend(load_sample_data(base_path + sample_path))

print("Total of {0} samples, with {1} data points dropped".format(len(samples), dropped_data))

def generator(samples):
    
    while 1:
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []

            for sample in batch_samples:
                source_path = sample[0]
                # filename = source_path.split('/')[-1]
                # current_path = './data/IMG/' + filename

                center_image = cv2.imread(source_path)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                images.append(center_image)

                angle = float(sample[3])
                steering_angles.append(angle)

                #Flip the center image
                flipped_image = cv2.flip(center_image, 1)
                images.append(flipped_image)
                steering_angles.append(-1.0 * angle)

                #Handle the left camera image
                source_path = sample[1]
                # filename = source_path.split('/')[-1]
                # current_path = './data/IMG/' + filename 
                left_image = cv2.imread(source_path)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                images.append(left_image)

                left_steering_angle = angle + side_camera_correction
                steering_angles.append(left_steering_angle)

                #Flip the left image
                flipped_left_image = cv2.flip(left_image, 1)
                images.append(flipped_left_image)
                steering_angles.append(angle - side_camera_correction)

                #Handle the right camera image
                source_path = sample[2]
                # filename = source_path.split('/')[-1]
                # current_path = './data/IMG/' + filename 
                right_image = cv2.imread(source_path)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                images.append(right_image)

                right_steering_angle = angle - side_camera_correction
                steering_angles.append(right_steering_angle)

                #Flip the right image
                flipped_right_image = cv2.flip(right_image, 1)
                images.append(flipped_right_image)
                steering_angles.append(angle + side_camera_correction)

            
            X_train = np.array(images)
            y_train = np.array(steering_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

training_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size = 0.2)
training_generator = generator(training_samples)
validation_generator = generator(validation_samples)


model = Sequential()

#Crop our images - we only care about a certain window of pixels, so why add in unneeded data and muddy our NN?
model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape = (160, 320, 3)))
# #Resize the image back to expected size so drive.py can use it
# def resize_image(input):
#     from keras.backend import tf as ktf
#     return ktf.image.resize_images(input, (new_height, new_width))
# model.add(Lambda(lambda x: resize_image))
#Normalize the image
model.add(Lambda(lambda x: (x / 255) - 0.5))

#The following is a copy of the nvidia architecture found in nvidia-architecture.png
#Magic numbers (number of convolutions, neurons, etc) come from there

#Now we do the convolutional networks
#The first three convolutional layers have a 5x5 kernel w/ a 2x2 stride 
model.add(Conv2D(24, 5, 5, subsample = (2, 2), activation='relu'))
if dropout_rate is not None:
    model.add(Dropout(dropout_rate))
model.add(Conv2D(36, 5, 5, subsample = (2, 2), activation='relu'))
if dropout_rate is not None:
    model.add(Dropout(dropout_rate))
model.add(Conv2D(48, 5, 5, subsample = (2, 2), activation='relu'))
if dropout_rate is not None:
    model.add(Dropout(dropout_rate))

#The last three are non strided 3x3 kernel size conv nets
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))

#Flatten
model.add(Flatten())

# Now we go through our fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(model.summary())

# model.load_weights('chauffeur_weights.h5')

history_object = model.fit_generator(training_generator, \
    samples_per_epoch = len(training_samples) * 6, \
    nb_epoch = epochs, verbose = 1, \
    validation_data = validation_generator, \
    nb_val_samples = len(validation_samples) * 6)

model.save('chauffeur.h5')
model.save_weights('chauffeur_weights.h5')

print("Training complete - model saved")

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])