# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goal of this project is to create a deep neural net with convolutional net layers to clone my driving behavior within a simulator. This document is the writeup for lessons learned during the project.

## Getting started

Before getting too deep into the project, I created `simple_test.py`. The goal of this file was to test the entire pipeline of

1) Getting the training data loaded
2) Creating a keras model
3) Training the keras model
4) Using the model to control the autonomous car

To do this, simple_test.py creates a simple dense neural network - no convolutional nets or anything else special. The goal here is to not create a good autonomous driver - in fact, it didn't create a good driver at all. It was to ensure that each step worked and I could create a model that would control the simulator car autonomously. This limits debugging time later to trying to figure out the simulator/data input instead of focussing my time on choosing an architecutre and training it.

## Model Architecture and Training Strategy

### NVidia Architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

### Changes to the NVidia Architecture

### Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

## 4. Model Training and Data Gathering strategies

### How I organized the data

### Straight data biases

#### 3. Creation of the Training Set & Training Process

### Generator

## Results

## Additional thoughts during training
