// ai_test.js

const tf = require('@tensorflow/tfjs-node');
const { createCanvas } = require('canvas');
const fs = require('fs');

// Parameters
const numSamples = 10000;    // Reduced for demonstration purposes
const inputSize = 2;         // 2D data for visualization
const epochs = 10;           // Number of epochs to train
const batchSize = 256;       // Batch size for training

// Generate synthetic data with patterns
const xsArray = [];
const ysArray = [];

for (let i = 0; i < numSamples; i++) {
  const x = Math.random() * 2 - 1;  // Random number between -1 and 1
  const y = Math.random() * 2 - 1;  // Random number between -1 and 1
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

// Function to generate and save visualization
async function generateVisualization(epoch) {
  const canvasSize = 400;
  const canvas = createCanvas(canvasSize, canvasSize);
  const ctx = canvas.getContext('2d');

  // Generate a grid of points
  const gridSize = 100;
  const gridData = [];
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const x = (i / gridSize) * 2 - 1;
      const y = (j / gridSize) * 2 - 1;
      gridData.push([x, y]);
    }
  }

  const gridTensor = tf.tensor2d(gridData);
  const predictions = model.predict(gridTensor).reshape([gridSize, gridSize]);

  const imageData = ctx.createImageData(canvasSize, canvasSize);
  const data = imageData.data;

  // Map predictions to colors
  const predArray = await predictions.array();
  for (let i = 0; i < canvasSize; i++) {
    for (let j = 0; j < canvasSize; j++) {
      const predValue = predArray[Math.floor((i / canvasSize) * gridSize)][Math.floor((j / canvasSize) * gridSize)];
      const color = predValue * 255;

      const index = (i * canvasSize + j) * 4;
      data[index] = color;        // R
      data[index + 1] = 0;        // G
      data[index + 2] = 255 - color;  // B
      data[index + 3] = 255;      // A
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Draw training data points
  ctx.fillStyle = 'white';
  xsArray.forEach((point, idx) => {
    const x = ((point[0] + 1) / 2) * canvasSize;
    const y = ((point[1] + 1) / 2) * canvasSize;
    const label = ysArray[idx];
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fillStyle = label === 1 ? 'yellow' : 'black';
    ctx.fill();
  });

  // Save the image
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(`training_visualization_epoch_${epoch}.png`, buffer);
  console.log(`Visualization for epoch ${epoch} saved.`);
}

// Train the model
(async () => {
  console.log('Starting training...');
  await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: batchSize,
    verbose: 1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${(
            logs.acc || logs.accuracy
          ).toFixed(4)}`
        );
        await generateVisualization(epoch + 1);
      },
    },
  });
  console.log('Training completed.');
})();
