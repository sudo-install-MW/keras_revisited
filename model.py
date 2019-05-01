from keras import layers
from keras.applications import VGG16
from keras import models
from keras.applications import MobileNetV2


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
        # model_vgg_base = VGG16(weights='imagenet',
        #                        include_top=False,
        #                        input_shape=(150, 150, 3),)
        #
        # model.add(model_vgg_base)
        # model_vgg_base.trainable = True
        # set_trainable = False
        #
        # for layer in model_vgg_base.layers:
        #     if layer.name == 'block5_conv1':
        #         set_trainable = True
        #     if set_trainable:
        #         layer.trainable = True
        #     else:
        #         layer.trainable = False
        #
        # model.add(layers.Flatten())
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dense(84, activation='sigmoid'))

        model_mobilenet_v2 = MobileNetV2(input_shape=(224, 224, 3),
                                         alpha=1.0,
                                         include_top=False,
                                         weights='imagenet',
                                         input_tensor=None,
                                         pooling=None,)

        model.add(model_mobilenet_v2)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(83, activation='softmax'))
        print(model.summary())

        return model
