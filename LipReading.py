import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
# Non-Binary Image Classification using Convolution Neural Networks
from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils

path = 'LRWDataset'
'''
labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            label = getID(name)
            X_train.append(im2arr)
            Y_train.append(label)
            print(name+" "+root+"/"+directory[j])
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)

figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
axis[0].set_title("Word Begin")
axis[1].set_title("Word Choose")
axis[0].imshow(cv2.imread("LRWDataset/01/color_004.jpg"))
axis[1].imshow(cv2.imread("LRWDataset/02/color_004.jpg"))
figure.tight_layout()
plt.show()

X_train = X_train.astype('float32')
X_train = X_train/255
    

indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)
np.save('model/X.txt',X_train)
np.save('model/Y.txt',Y_train)
'''
X_train = np.load('model/X.txt.npy')
Y_train = np.load('model/Y.txt.npy')

print(Y_train)
if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[249] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    classifier = Sequential()
    classifier.add(Convolution2D(64, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = 10, activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=250, shuffle=True, verbose=2)
    classifier.save_weights('model/model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[249] * 100
    print("Training Model Accuracy = "+str(accuracy))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('lip_detector.dat')

words = ['Begin','Choose','Connection','Navigation','Next','Previous','Start','Stop','Hello','Web']


facial_features_cordinates = {}

FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68))
    
])


def shape_to_numpy_array(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    plate = None
    x = 0
    y = 0
    w = 0
    h = 0
    overlay = image.copy()
    output = image.copy()
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]    
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts
        if name == "Jaw":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.rectangle(overlay, ptA, ptB, colors[i], 2)
        else:
            hull = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull], -1, colors[i], -1)
            x, y, w, h = cv2.boundingRect(hull)
            plate = image[y:y + h, x:x + w]
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.imshow("aa",image)
            #cv2.waitKey(0)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return plate,x, y, w, h

cap = cv2.VideoCapture(0)
while True:
    _, orig_img = cap.read()
    height, width, channels = orig_img.shape
    faceROI = None
    mouth = None
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        for (fX, fY, fW, fH) in faces:
            faceROI = gray[fY:fY+ fH, fX:fX + fW]
        cv2.imwrite("mouth.png",orig_img) 
        image = cv2.imread('mouth.png')
        image = imutils.resize(image, width=500)
        orig_img = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_numpy_array(shape)
            mouth,x, y, w, h = visualize_facial_landmarks(image, shape)
            if mouth is not None:
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imwrite("mouth.png",mouth)
                image = cv2.imread('mouth.png')
                img = cv2.resize(image, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,64,64,3)
                img = np.asarray(im2arr)
                img = img.astype('float32')
                img = img/255
                preds = classifier.predict(img)
                predict = np.argmax(preds)
                if np.amax(preds) > 0.98:
                    print(str(words[predict])+" "+str(np.amax(preds)))
                    cv2.putText(orig_img, "Word Identified as : "+words[predict], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
                else:
                    print("unable to identify")
    cv2.imshow("Image", orig_img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()    


