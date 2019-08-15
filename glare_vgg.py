import functools
from keras.metrics import top_k_categorical_accuracy
from keras.applications import *
from keras_preprocessing import image
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from train_val_split import createTrainValSplit
#image_size=299
image_size=224
seed = 7
np.random.seed(seed)
inputpath = 'fisheye_bmp_90'
# createTrainValSplit(inputpath,0.1)
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#vgg_conv = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
# Freeze all the layers
for layer in vgg_conv.layers[:-14]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

train_dir = inputpath + '/train'
validation_dir = inputpath + '/val'

# nTrain = 549
# nVal = 137
#
# datagen = image.ImageDataGenerator(rescale=1. / 255)
# batch_size = 10
#
# train_features = np.zeros(shape=(nTrain, 7, 7, 512))
# train_labels = np.zeros(shape=(nTrain, 4))
#
# train_generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode='categorical', shuffle=True
# )
# i = 0
# for inputs_batch, labels_batch in train_generator:
#     features_batch = vgg_conv.predict(inputs_batch)
#     train_features[i * batch_size: (i + 1) * batch_size] = features_batch
#     train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
#     i += 1
#     if i * batch_size >= nTrain:
#         break
#
# train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))
#
# validation_features = np.zeros(shape=(nVal, 7, 7, 512))
# validation_labels = np.zeros(shape=(nVal, 4))
#
# validation_generator = datagen.flow_from_directory(
#     validation_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode='categorical', shuffle=False
# )
#
# i = 0
# for inputs_batch, labels_batch in validation_generator:
#     features_batch = vgg_conv.predict(inputs_batch)
#
#     validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
#     validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
#     i += 1
#     if i * batch_size >= nVal:
#         break
#
# validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))

model = models.Sequential()
# Add the vgg convolutional base model
model.add(vgg_conv)

#model.add(layers.Dense(1024, activation='relu', input_dim=7 * 7 * 512))
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512,activation='sigmoid'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128,activation='tanh'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32,activation='tanh'))
model.add(layers.Dropout(0.3))
# model.add(layers.Dense(16,activation='sigmoid'))
# model.add(layers.Dropout(0.3))
model.add(layers.Dense(4, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# train_datagen = image.ImageDataGenerator(
#       rescale=1./255,
#       rotation_range=20,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest')

# No Data augmentation
train_datagen = image.ImageDataGenerator(rescale=1./255)
validation_datagen = image.ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 20
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

top2_acc = functools.partial(top_k_categorical_accuracy, k=2)

top2_acc.__name__ = 'top2_acc'

model.compile(optimizer=optimizers.RMSprop(lr=0.00001),
              loss='categorical_crossentropy',
              metrics=['acc',top2_acc])

# history = model.fit(train_features,
#                     train_labels,
#                     epochs=50,
#                     batch_size=batch_size,
#                     validation_data=(validation_features, validation_labels))
# Train the Model
# NOTE that we have multiplied the steps_per_epoch by 2. This is because we are using data augmentation.
outputFolder = '/media/raghu/6A3A-B7CD/glare_resnet_models'
filepath = outputFolder + "/models-{val_acc:.4f}-{val_top2_acc:.4f}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
                             save_best_only=True, save_weights_only=False, \
                             mode='auto', period=1)
callbacks=[checkpoint]
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=60,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,callbacks=callbacks)

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
top2_acc = history.history['top2_acc']
val_top2_acc = history.history['val_top2_acc']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.plot(epochs, top2_acc, 'b', label='Training top2 acc')
plt.plot(epochs, val_top2_acc, 'r', label='Validation top2 acc')
plt.title('Training and validation top2 accuracy')
plt.legend()

plt.show()

# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator,
                                      steps=validation_generator.samples / validation_generator.batch_size, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = image.load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
    plt.figure(figsize=[7, 7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.savefig('/media/raghu/6A3A-B7CD/wrong_predictions/'+str(i)+'.png')
    plt.clf()

