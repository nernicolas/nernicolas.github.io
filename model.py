from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils as np
import tensorflowjs as tfjs

#utilisation du set de données mnist inclus avec keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#traitement des données
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")
x_train /= 255
x_test /= 255
y_train = np.to_categorical(y_train)
y_test = np.to_categorical(y_test)

#le modèle CNN 2 couches de convolution
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=10, activation="softmax"))
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#entrainement du modèle
classifier.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=200, epochs=10)

#evalutation
scores = classifier.evaluate(x_test, y_test, verbose=0)
print("Error: {:.2f}%".format((1-scores[1])*100))

#utilisation de tensorflow.js pour sauvegarder le modele et le réutiliser dans un script
# javascript
tfjs.converters.save_keras_model(classifier, "model")
