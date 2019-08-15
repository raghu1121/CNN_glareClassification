from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image")
args=vars(ap.parse_args())

orig=cv2.imread(args["image"])
print("[INFO] loading and preprocessing image...")
image=image_utils.load_img(args["image"],target_size=(224,224))
image=image_utils.img_to_array(image)

image =np.expand_dims(image,axis=0)
image = preprocess_input(image)

print("[INFO] loading network...")
model=VGG16(weights='imagenet')

print("[INFO] classifying image...")
preds =model.predict(image)
P=decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)