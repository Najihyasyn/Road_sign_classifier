###########################################      project reconnaissance des panneaux routiers     ##############################
##########################################       NAJIH yasyn                         #############################
##########################################       E-mail:yasynnajih@gmail.com             #############################
##########################################       yasyn.najih@edu.uca.ma                 ###############################
#################################################################################################################################

import numpy as np# pour faire des processusses mathematique
import pandas as pd # pour  la manipulation et l'analyse des données
import matplotlib.pyplot as plt #pour tous qu'est figure (les diagramme, les images ...)
import cv2 #
import tensorflow as tf
from PIL import Image# pour travailler sur les images :comme l'ouverture, le redimentionnement d'une image etc
import os #pour utiliser notre systeme
from sklearn.model_selection import train_test_split #pour diviser les donnees en train et test
from tensorflow.keras.utils import to_categorical # encodage chaud
from tensorflow.keras.models import Sequential, load_model # pour l'utilisation et choiser le mode de notre model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout #pour appliquer les etapes de CNN
#la declaration des quelque va
data = []
labels = []
classes = 43
cur_path = os.getcwd()# pour obtenir le chemin actuel de ce fichier
#ce block permet d'importe les classes ainsi que les images de chaque clesse de notre Dataset,
# et affiche un message en cas d'erreur
#et permet de ranger les images dans "data" et les labels dans "labels"
## Récupération des images et de leurs étiquettes
#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
##Convertir des listes en tableaux numpy
#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting ensemble de données de formation et de test
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
##Convertir les étiquettes en un seul encodage à chaud
# c-a-d en un vecteur qui a tous les valeurs sont nuls sauf la composant qui corespond a la classe actuel
#exemple : si la classe est 1 donc le vecteur qu'on va obtenir sera (0 1 0 .......0)
#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


##Construire le modèle
#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))#
model.add(Dropout(rate=0.25))
model.add(Flatten())
#???? 256
model.add(Dense(256, activation='relu'))#pour costruire la couche d'entree qui a le nombre de neurone est 256 (16*16)
model.add(Dropout(rate=0.5))#on va desactiver 50% des neurones pour
model.add(Dense(43, activation='softmax'))#pour construire la couche de sortie qui a le nombre de neurone est le nombre des classes celui 43 classes
#Compilation du modèle
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("model.summary::::::::::::::\n")
model.summary()

epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
#model.save("my_model.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test=np.array(data)

pred = model.predict_classes(X_test)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
#En fin de compte, nous allons enregistrer le modèle que nous avons formé
# à l'aide de la fonction Keras model.save ().
model.save( 'traffic_classifier.h5' )
