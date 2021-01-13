
# Creator ===== Mayank Goyal

#import all the dependencies
from tensorflow.keras.preprocessing.image import ImageDataGenerator ,img_to_array ,load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D ,Dropout ,Flatten ,Dense ,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import os


# First we define the learning rate
# Higher the epoches higher the training model

init_lr = 1e-4  ### Initial learning
EPOCHS = 20   ####### more dataset more epochs ####### less dataset less epochs
BS = 20       #####  Batch size



# Path of the dataset
directory = r"C:/Users/MGOYA\Desktop/Face_Mask_Detector_Mayank/dataset"
categories = ["Mask","No Mask"]


# load the images
print(" [INFO] Loading images......")

data =[]     # data list for dataset 
labels =[]   #labels list for categories

for category in categories:
    paths = os.path.join(directory,category)
    for img in os.listdir(paths):
        img_path = os.path.join(paths,img)

         # load the input image (224x224) and preprocess it
        image = load_img(img_path , target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)



         # update the data and labels lists, respectively
        data.append(image)
        labels.append(category)


# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)


 #partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

(trainX , testX , trainY , testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


# Image data generator to data augumentation 
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# BASE MODEL
# load the mobileNetv2     ## it will ignore the head part , it covers upto our nose
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))  #(224 as size of image and 3 is rgb value)


# HEAD MODEL
# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)




# we define base model /// we define head model 
# Now we define actual model 

# Input as the BASE MODEL ///// And Output as HEAD MODEL 

# ACTUAL MODEL
model = Model(inputs=baseModel.input, outputs=headModel)


# loop over all layers in the base model and freeze them so they will
# *not* be updated images during the first training process
for layer in baseModel.layers:
    layer.trainable = False # Not update the images


# compile our model
print("[INFO] Compiling model.....")
opt = Adam(learning_rate=init_lr, decay=init_lr / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# training of the head model##
print("[INFO] Training HEAD MODEL.....")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


# make predictions on the testing set
print("[INFO] Evaluating Network......")
predIdxs = model.predict(testX, batch_size=BS)


# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)


# show/Print a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_)) # lb is define in Label Binarizer function


# serialize the model to disk //// Save the model to disk
print("[INFO] Saving Mask Detector Model...")
model.save("mask_detector.h5")        #, save_format="h5")


# Plot training vs loss graph 
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label=["train_loss"])
plt.plot(np.arange(0,N), H.history["val_loss"], label=["val_loss"])    # Valuable loss
plt.plot(np.arange(0,N), H.history["accuracy"], label=["train_acc"])
plt.plot(np.arange(0,N), H.history["val_accuracy"], label=["val_accuracy"])   
plt.title("Training Loss and Accuracy")      # Epochs higher /// loss lower /// accuracy higher
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

