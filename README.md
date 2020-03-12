# nernicolas.github.io

Built for a web application class project, this is a digit recognizer using deep learning, especially convolutional neural nerwork.

To test it, you can either download the zip file and exec index.html in browser, or go to https://nernicolas.github.io .

model.py is a python script that is used to create, train, evaluate and save the model ( using tensorflow.js which allows us to use it in a following .js file). 
model.json is the file describing our model, created with the call to tf.save().
index.html is our main page and consist of two canvas and one clear button.
The script.js allows the user to draw in the canvas, and use an https request to load the CNN-model, it gets the info from the drawing in the canvas and reshape the pixel data to be of the same shape as the training data. It will then use the model (tensorflowjs)) to make and choose a prediction that will display in the other canvas. Calls to predictions methods are asynchronous so you may have to wait a bit for it to appear.
