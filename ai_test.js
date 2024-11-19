// ai_test.js

const tf = require('@tensorflow/tfjs-node');
const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const fs = require('fs');

// Parameters
const numSamples = 100000;    // Increase this number for more intensity
const inputSize = 100;        // Size of each input vector
const epochs = 10;            // Number of epochs to train
const batchSize = 256;        // Batch size for training

// Generate synthetic data
const xs = tf.randomNormal([numSamples, inputSize]);
const ys = tf.randomUniform([numSamples, 1]);

// Define a deep neural network model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 512, inputShape: [inputSize], activation: 'relu' }));
model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1 }));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'meanSquaredError',
  metrics: ['mse'],
});

// Arrays to store loss and mse values
const lossValues = [];
const mseValues = [];

// Train the model
(async () => {
  console.log('Starting training...');
  await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: batchSize,
    verbose: 1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - MSE: ${logs.mse.toFixed(4)}`);
        lossValues.push(logs.loss);
        mseValues.push(logs.mse);
      },
    },
  });
  console.log('Training completed.');

  // Generate a chart of loss and mse over epochs
  const width = 800; // width of the image
  const height = 600; // height of the image
  const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height });

  const configuration = {
    type: 'line',
    data: {
      labels: Array.from({ length: epochs }, (_, i) => i + 1),
      datasets: [
        {
          label: 'Loss',
          data: lossValues,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          fill: false,
        },
        {
          label: 'MSE',
          data: mseValues,
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          fill: false,
        },
      ],
    },
    options: {
      plugins: {
        title: {
          display: true,
          text: 'Training Progress',
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch',
          },
        },
        y: {
          title: {
            display: true,
            text: 'Value',
          },
        },
      },
    },
  };

  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration);
  fs.writeFileSync('training_progress.png', imageBuffer);
  console.log('Training progress chart saved as training_progress.png');
})();
