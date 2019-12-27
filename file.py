# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout,Reshape


# import zipfile
# with zipfile.ZipFile('friends.zip', 'r') as zip_ref:
#     zip_ref.extractall('My Drive')

# cd "./drive/My Drive/My Drive/friends"

# import os
# os.getcwd()

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128,  input_dim=7 ,activation = 'relu'))
classifier.add(Dense(units = 7, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('./training_set',
                                                 target_size = (64,64),
                                                 batch_size = 1,
                                                 class_mode='categorical')


test_set = test_datagen.flow_from_directory('./test_set',
                                            target_size = (64,64),
                                            batch_size = 1,
                                            class_mode = 'categorical')

# del(classifier)

classifier.fit_generator(training_set,
                         samples_per_epoch = 33,
                         epochs = 15,
                         validation_data = test_set,validation_steps = 10)

# cd "../"

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('./test_set/denis/denis1.jpeg', target_size = (64,64))
test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

np.argmax(result[0][0])
