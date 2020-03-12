const canvas = document.getElementById('main-canvas');
const smallCanvas = document.getElementById('small-canvas');
const displayBox = document.getElementById('prediction');

const inputBox = canvas.getContext('2d');
const smBox = smallCanvas.getContext('2d');

let isDrawing = false;
let model;

/* Load le model entrainé */
async function init() {
  model = await tf.loadModel('https://nernicolas.github.io/model.json');
}

canvas.addEventListener('mousedown', event => {
  isDrawing = true;

  inputBox.strokeStyle = 'white';
  inputBox.lineWidth = '15';
  inputBox.lineJoin = inputBox.lineCap = 'round';
  inputBox.beginPath();
});

canvas.addEventListener('mousemove', event => {
  if (isDrawing) drawStroke(event.clientX, event.clientY);
});

canvas.addEventListener('mouseup', event => {
  isDrawing = false;
  updateDisplay(predict());
});

/* Pour dessiner dans les canvas */
function drawStroke(clientX, clientY) {
  // get mouse coordinates on canvas
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  inputBox.lineTo(x, y);
  inputBox.stroke();
  inputBox.moveTo(x, y);
}

/* Predictions */
function predict() {
  let values = getPixelData();
  let predictions = model.predict(values).dataSync();

  return predictions;
}

/* Avoir les informations (pixels) du canvas*/
function getPixelData() {
  smBox.drawImage(inputBox.canvas, 0, 0, smallCanvas.width, smallCanvas.height);
  const imgData = smBox.getImageData(0, 0, smallCanvas.width, smallCanvas.height);

  // les mettre sous le meme format que les données d'entrainement du modele
  let values = [];
  for (let i = 0; i < imgData.data.length; i += 4) {
    values.push(imgData.data[i] / 255);
  }
  values = tf.reshape(values, [1, 28, 28, 1]);
  return values;
}

/* Afficher les prédictions */
function updateDisplay(predictions) {
  // Find index of best prediction, which corresponds to the predicted value
  const bestPred = predictions.indexOf(Math.max(...predictions));
  displayBox.innerText = bestPred;
}

document.getElementById('erase').addEventListener('click', erase);

/* Clears canvas */
function erase() {
  inputBox.fillStyle = 'black';
  inputBox.fillRect(0, 0, canvas.width, canvas.height);
  displayBox.innerText = '';
}

erase();
init();
