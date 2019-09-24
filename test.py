
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot as plt
from keras_preprocessing.image import ImageDataGenerator


classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 1)))
#adding extra layer
classifier.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
classifier.add(Conv2D(64, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#classifier.add(Flatten())
#classifier.add(Dense(1000, activation='relu'))
#classifier.add(Dense(2, activation='softmax'))

classifier.add(Flatten())
classifier.add(Dropout(.5))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('Test_set', target_size = (64, 64), batch_size = 30, class_mode = 'binary')
history = classifier.fit_generator(training_set, steps_per_epoch = 50, epochs = 20, validation_data = test_set, validation_steps = 25)

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/Tassi/Dropbox/Tabassum/Conference Submissions/Data Analysis/Test Images/Attentive/20.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'inattentive'
else:
    prediction = 'attentive'

print(prediction)

# plot the training loss and accuracy
##list all data in history
print(history.history.keys())
# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()