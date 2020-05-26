# Installing Keras,Tensorflow

# Part 1 -

# Building CNN

# Importing keras packages
# Initialising NN
from keras.models import Sequential
# Adding convulational layer for 2d 
from keras.layers import Conv2D
# Pooling layers
from keras.layers import MaxPooling2D
# Flattening 
from keras.layers import Flatten
# Adding layers to NN
from keras.layers import Dense

# Step -1
# Initaialising the CNN
classifier = Sequential()

# Step -2 Convoluation
# Adding layers to CNN
# Specify the image format in input_shape, so that we have same inputs
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step -3 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step -4 Flattening
# If we flatten directly without above steps, each node will be one pixel of image
# we dont get info about how this image is spatialliy conected with other pixels
# Also feature detector detects special features in the image which is important to predict 
classifier.add(Flatten())

# Step 4 - Full connection
# Hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 256, activation = 'relu'))
# Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# adam = Stochastic gradient descent 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part -2 Fit the CNN to the images

# Image augumentation to prevent overfitting Train set good test set bad accuracy
# Using keras documentation
# overfitting happens when we have few models to train on, so the model finds
# correlations in the few obs of the trai set but fails to generalize the correlations
# hence augumentation does transformation like shearing roatating etc to find correlations

# Image augumentation part where we apply several transformation
from keras.preprocessing.image import ImageDataGenerator

# Image Augumentation on training set 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Image augumentation for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Applying augumentation and resizing our images and batch size to 32
training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

# Applying augumentation and resizing our images and batch size to 32
test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
                            training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)
