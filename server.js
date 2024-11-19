// server.js

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Parameters
const numSamples = 50000;    // Number of data points
const epochs = 150;          // Increased epochs to 150
const batchSize = 1024;      // Batch size for training

// Generate a random complex parameter 'c' for the Julia set
const cRe = Math.random() * 2 - 1;  // Real part in [-1, 1]
const cIm = Math.random() * 2 - 1;  // Imaginary part in [-1, 1]
const c = { re: cRe, im: cIm };

console.log(`Using Julia set parameter c = ${c.re.toFixed(4)} + ${c.im.toFixed(4)}i`);

// Generate dataset based on the Julia set
function generateJuliaData(numSamples, c) {
  const xsArray = [];
  const ysArray = [];

  for (let i = 0; i < numSamples; i++) {
    // Randomly sample points in the complex plane
    const re = Math.random() * 3.5 - 2.5;  // Real part: [-2.5, 1]
    const im = Math.random() * 2 - 1;      // Imaginary part: [-1, 1]
    xsArray.push([re, im]);

    // Determine if the point escapes to infinity
    const isInSet = julia(re, im, c, 50);
    ysArray.push(isInSet ? 1 : 0);
  }

  return { xsArray, ysArray };
}

// Julia set function
function julia(re, im, c, maxIter) {
  let zRe = re;
  let zIm = im;
  let n = 0;

  while (zRe * zRe + zIm * zIm <= 4 && n < maxIter) {
    const tempRe = zRe * zRe - zIm * zIm + c.re;
    const tempIm = 2 * zRe * zIm + c.im;
    zRe = tempRe;
    zIm = tempIm;
    n++;
  }

  return n === maxIter;
}

// Generate training and validation data
const { xsArray, ysArray } = generateJuliaData(numSamples, c);
const xs = tf.tensor2d(xsArray);
const ys = tf.tensor1d(ysArray);

// Split data into training (80%) and validation (20%) sets
const splitIndex = Math.floor(xsArray.length * 0.8);
const trainXs = xs.slice([0, 0], [splitIndex, 2]);
const trainYs = ys.slice([0], [splitIndex]);
const valXs = xs.slice([splitIndex, 0], [xsArray.length - splitIndex, 2]);
const valYs = ys.slice([splitIndex], [ysArray.length - splitIndex]);

// Define a complex neural network model with increased depth
const model = tf.sequential();
model.add(tf.layers.dense({ units: 256, inputShape: [2], activation: 'relu', kernelInitializer: 'heNormal' }));
model.add(tf.layers.batchNormalization());
model.add(tf.layers.dense({ units: 512, activation: 'relu', kernelInitializer: 'heNormal' }));
model.add(tf.layers.batchNormalization());
model.add(tf.layers.dropout({ rate: 0.3 }));  // Dropout to prevent overfitting
model.add(tf.layers.dense({ units: 512, activation: 'relu', kernelInitializer: 'heNormal' }));
model.add(tf.layers.batchNormalization());
model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model with a slightly increased learning rate
model.compile({
  optimizer: tf.train.adam(0.0001),  // Increased learning rate for better convergence
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

// Real-time training and visualization
(async () => {
  console.log('Starting training...');
  await model.fit(trainXs, trainYs, {
    epochs: epochs,
    batchSize: batchSize,
    shuffle: true,
    validationData: [valXs, valYs],   // Include validation data to monitor performance
    verbose: 0, // Turn off default logging
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${(logs.acc || logs.accuracy).toFixed(4)} - Val Loss: ${logs.val_loss.toFixed(4)} - Val Accuracy: ${(logs.val_acc || logs.val_accuracy).toFixed(4)}`
        );

        // Generate grid data for visualization
        const gridResolution = 600; // Increased resolution for better detail
        const gridData = [];
        const reStart = -2.5;
        const reEnd = 1;
        const imStart = -1;
        const imEnd = 1;

        for (let y = 0; y < gridResolution; y++) {
          for (let x = 0; x < gridResolution; x++) {
            const re = reStart + (x / gridResolution) * (reEnd - reStart);
            const im = imStart + (y / gridResolution) * (imEnd - imStart);
            gridData.push([re, im]);
          }
        }

        const gridTensor = tf.tensor2d(gridData);
        const predictions = model.predict(gridTensor).dataSync();

        // Prepare data to send to client
        const visualizationData = {
          epoch: epoch + 1,
          totalEpochs: epochs,
          logs,
          gridResolution,
          predictions: Array.from(predictions),
          c, // Send the complex parameter to display
        };

        // Emit data to client
        io.emit('update', visualizationData);

        // Slow down training for visualization purposes
        await new Promise((resolve) => setTimeout(resolve, 50)); // Adjust delay as needed
      },
    },
  });
  console.log('Training completed.');
})();

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
