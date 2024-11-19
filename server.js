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
const numSamples = 1000;    // Adjust for desired data size
const inputSize = 2;
const epochs = 20;
const batchSize = 32;

// Generate synthetic data with patterns
const xsArray = [];
const ysArray = [];

for (let i = 0; i < numSamples; i++) {
  const x = Math.random() * 2 - 1;
  const y = Math.random() * 2 - 1;
  xsArray.push([x, y]);

  // Label: 1 if inside circle of radius 0.5, else 0
  const label = x * x + y * y < 0.5 * 0.5 ? 1 : 0;
  ysArray.push(label);
}

const xs = tf.tensor2d(xsArray);
const ys = tf.tensor1d(ysArray);

// Define a neural network model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, inputShape: [inputSize], activation: 'relu' }));
model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

// Real-time training and visualization
(async () => {
  console.log('Starting training...');
  await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: batchSize,
    verbose: 0, // Turn off default logging
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${(
            logs.acc || logs.accuracy
          ).toFixed(4)}`
        );

        // Generate data for visualization
        const gridSize = 50;
        const gridData = [];
        for (let i = 0; i < gridSize; i++) {
          for (let j = 0; j < gridSize; j++) {
            const x = (i / gridSize) * 2 - 1;
            const y = (j / gridSize) * 2 - 1;
            gridData.push([x, y]);
          }
        }

        const gridTensor = tf.tensor2d(gridData);
        const predictions = model.predict(gridTensor).dataSync();

        // Prepare data to send to client
        const visualizationData = {
          epoch: epoch + 1,
          totalEpochs: epochs,
          logs,
          gridData,
          predictions: Array.from(predictions),
          xsArray,
          ysArray,
        };

        // Emit data to client
        io.emit('update', visualizationData);

        // Slow down training for visualization purposes
        await new Promise((resolve) => setTimeout(resolve, 500)); // Adjust delay as needed
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
