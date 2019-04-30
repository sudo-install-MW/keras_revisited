from keras.preprocessing.image import ImageDataGenerator
from keras_revisited.model import Model
from keras import optimizers
import matplotlib.pyplot as plt

train_dir = "/media/mash-compute/eatsmart/DeepLearning/keras_revisited/data/data_processed/train"
val_dir = "/media/mash-compute/eatsmart/DeepLearning/keras_revisited/data/data_processed/validation"

# preprocess image
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=128,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(val_dir,
                                                        target_size=(150, 150),
                                                        batch_size=128,
                                                        class_mode='binary')

classifier = Model()
model = classifier.create_model()

print(model.trainable_weights)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

# plotting accuracy and training graph
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()