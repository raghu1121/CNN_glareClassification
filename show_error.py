import functools
from keras.metrics import top_k_categorical_accuracy
from keras.applications import *
from keras_preprocessing import image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import optimizers
#image_size=299
image_size=224
seed = 7
np.random.seed(seed)
inputpath = 'reclassified'

# top2_acc = functools.partial(top_k_categorical_accuracy, k=2)
#
# top2_acc.__name__ = 'top2_acc'

def top2_acc(y_true, y_pred):
    return functools.partial(top_k_categorical_accuracy, k=2)

model=load_model('/media/raghu/6A3A-B7CD/glare_resnet_models/models-0.7209-0.7751.hdf5')

validation_dir = inputpath + '/val'


validation_datagen = image.ImageDataGenerator(rescale=1./255)
val_batchsize = 10


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
