// ai_test.js

const tf = require('@tensorflow/tfjs-node');

// Parameters
const numSamples = 1000000;    // Increase this number for more intensity
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

// Train the model
(async () => {
  console.log('Starting training...');
  const history = await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: batchSize,
    verbose: 1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - MSE: ${logs.mse.toFixed(4)}`);
      },
    },
  });
  console.log('Training completed.');
})();

