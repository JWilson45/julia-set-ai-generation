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
const epochs = 50;
const batchSize = 1024;

// Generate dataset based on the Mandelbrot set
function generateMandelbrotData(numSamples) {
  const xsArray = [];
  const ysArray = [];

  for (let i = 0; i < numSamples; i++) {
    // Randomly sample points in the complex plane
    const re = Math.random() * 3.5 - 2.5;  // Real part: [-2.5, 1]
    const im = Math.random() * 2 - 1;      // Imaginary part: [-1, 1]
    xsArray.push([re, im]);

    // Determine if the point is in the Mandelbrot set
    const isInSet = mandelbrot(re, im, 50);
    ysArray.push(isInSet ? 1 : 0);
  }

  return { xsArray, ysArray };
}

// Mandelbrot function
function mandelbrot(re, im, maxIter) {
  let zRe = 0;
  let zIm = 0;
  let n = 0;

  while (zRe * zRe + zIm * zIm <= 4 && n < maxIter) {
    const tempRe = zRe * zRe - zIm * zIm + re;
    const tempIm = 2 * zRe * zIm + im;
    zRe = tempRe;
    zIm = tempIm;
    n++;
  }

  return n === maxIter;
}

const { xsArray, ysArray } = generateMandelbrotData(numSamples);
let xs = tf.tensor2d(xsArray);
let ys = tf.tensor1d(ysArray);

// Define a complex neural network model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 128, inputShape: [2], activation: 'relu' }));
model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model
model.compile({
  optimizer: tf.train.adam(0.0001),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

// Real-time training and visualization
(async () => {
  console.log('Starting training...');
  await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: batchSize,
    shuffle: true,
    verbose: 0, // Turn off default logging
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${(
            logs.acc || logs.accuracy
          ).toFixed(4)}`
        );

        // Generate grid data for visualization
        const gridResolution = 400; // Increase for higher resolution
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
