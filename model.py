from keras import layers
from keras.applications import VGG16
from keras import models


class Model:
    def create_model(self):
        model = models.Sequential()
        #
        # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        # model.add(layers.MaxPooling2D((2, 2)))
        #
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPool2D((2, 2)))
        #
        # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(layers.MaxPool2D((2, 2)))
        #
        # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(layers.MaxPool2D((2, 2)))
        #
        # model.add(layers.Flatten())
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(512, activation='relu'))
        #
        # model.add(layers.Dense(1, activation='sigmoid'))
        model_vgg_base = VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(150, 150, 3),)

        model.add(model_vgg_base)

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model_vgg_base.trainable = False
        return model
