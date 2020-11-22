
// When we have multiple features for our input, we should perform normalization (aka feature scaling) for each separately (keeping hold of separate min and max values for denormalization).
// We can visualise 2 input features for a on a scatter plot.

// global variables
let points;
let normalizedFeature, normalizedLabel;
let trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
let model;

run();

async function plotPoints(pointsArray, priceTag) {
const allSeries = {};
// Add each class as a series
pointsArray.forEach(p => {
    // Add each point to the series for the class it is in
    if(p.z > 400){
    const seriesName = `${priceTag}: ${p.z}`;
    let series = allSeries[seriesName];
    if (!series) {
        series = [];
        allSeries[seriesName] = series;
    }
    series.push(p);
    }

});

tfvis.render.scatterplot(
    {
    name: `Square feet vs Bedrooms`,
    styles: { width: "100%", height: "100%" },
    },
    {
    values: Object.values(allSeries),
    series: Object.keys(allSeries),
    },
    {
    xLabel: "Square feet",
    yLabel: "Bedrooms"
    }
);
}

function normalize(tensor, previousMin = null, previousMax = null) {
const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];

if (featureDimensions && featureDimensions > 1) {
    // More than one feature

    // Split into separate tensors
    const features = tf.split(tensor, featureDimensions, 1);

    // normalize and find min/max values for each feature
    const normalizedFeatures = features.map((featureTensor, i) =>
    normalize(featureTensor,
        previousMin ? previousMin[i] : null,
        previousMax ? previousMax[i] : null,
    )
    );

    // Prepare return values
    const returnTensor = tf.concat(normalizedFeatures.map(f => f.tensor), 1);
    const min = normalizedFeatures.map(f => f.min);
    const max = normalizedFeatures.map(f => f.max);

    return { tensor: returnTensor, min, max };
}
else {
    // Just one feature
    const min = previousMin || tensor.min();
    const max = previousMax || tensor.max();
    const normalizedTensor = tensor.sub(min).div(max.sub(min));
    return {
    tensor: normalizedTensor,
    min,
    max
    };
}
}

function denormalize(tensor, min, max) {
const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];

if (featureDimensions && featureDimensions > 1) {
    // More than one feature

    // Split into separate tensors
    const features = tf.split(tensor, featureDimensions, 1);

    const denormalized = features.map((featureTensor, i) => denormalize(featureTensor, min[i], max[i]));

    const returnTensor = tf.concat(denormalized, 1);
    return returnTensor;
}
else {
    // Just one feature
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalizedTensor;
}
}

function createModel () {
model = tf.sequential();

// using sigmoid
model.add(tf.layers.dense({
    units: 10,
    useBias: true,
    activation: 'sigmoid',
    inputDim: 2,
}));
model.add(tf.layers.dense({
    units: 10,
    useBias: true,
    activation: 'sigmoid',
}));
model.add(tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'sigmoid',
}));

const optimizer = tf.train.adam();
model.compile({
    // for binary classification
    loss: 'meanSquaredError',
    optimizer,
});

return model;
}

async function trainModel (model, trainingFeatureTensor, trainingLabelTensor) {

const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
    { name: "Training Performance" },
    ['loss']
);

return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    batchSize: 32,
    epochs: 30,
    validationSplit: 0.2,
    callbacks: {
    onEpochEnd,
    onEpochBegin: async function () {
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({ name: "Layer 1" }, layer);
    }
    }
});
}

async function predict(){
const predictionInputOne = parseInt(document.getElementById("prediction-input-1").value);
const predictionInputTwo = parseInt(document.getElementById("prediction-input-2").value);
if (isNaN(predictionInputOne) || isNaN(predictionInputTwo)) {
    alert("Please enter a valid number");
}
else if (predictionInputOne < 200) {
    alert("Please enter a value above 200 sqft");
}
else if (predictionInputTwo < 1) {
    alert("Please enter a value for bedrooms");
}
else {
    tf.tidy(() => {
    const inputTensor = tf.tensor2d([[predictionInputOne, predictionInputTwo]]);
    const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
    const normalizedOutputTensor = model.predict(normalizedInput.tensor);
    const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
    const outputValue = outputTensor.dataSync()[0];
    const outputValuePercent = (outputValue/50).toFixed(0)*50;
    document.getElementById("prediction-output").innerHTML = `The predicted price is <br>`
        + `<span style="font-size: 2em">${outputValuePercent}</span>`;
    });
}
}

const storageID = "nyc_housing_multiple";
async function save () {
const saveResults = await model.save(`localstorage://${storageID}`);
document.getElementById("model-status").innerHTML = `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;
}

async function load () {
const storageKey = `localstorage://${storageID}`;
const models = await tf.io.listModels();
const modelInfo = models[storageKey];
if (modelInfo) {
    model = await tf.loadLayersModel(storageKey)
    tfvis.show.modelSummary({ name: "Model summary" }, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1" }, layer);

    document.getElementById("model-status").innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
    document.getElementById("predict-button").removeAttribute("disabled");
}
else {
    alert("Could not load: no saved model found");
}
}

async function test () {
const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
const loss = (await lossTensor.dataSync())[0];
console.log(`Testing set loss: ${loss}`);

document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

async function train () {
// Disable all buttons and update status
["train", "test", "load", "predict", "save"].forEach(id => {
    document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
});
document.getElementById("model-status").innerHTML = "Training...";

const model = createModel();
tfvis.show.modelSummary({ name: "Model summary" }, model);
const layer = model.getLayer(undefined, 0);
tfvis.show.layer({ name: "Layer 1" }, layer);

const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor)
console.log(result);
const trainingLoss = result.history.loss.pop();
console.log(`Training set loss: ${trainingLoss}`);
const validationLoss = result.history.val_loss.pop();
console.log(`Validation set loss: ${validationLoss}`);

document.getElementById("model-status").innerHTML = "Trained (unsaved)\n<br>"
    + `Loss: ${trainingLoss.toPrecision(5)}\n<br>`
    + `Validation loss: ${validationLoss.toPrecision(5)}`;
document.getElementById("test-button").removeAttribute("disabled");
document.getElementById("save-button").removeAttribute("disabled");
document.getElementById("predict-button").removeAttribute("disabled");
}

async function toggleVisor () {
tfvis.visor().toggle();
}

async function run () {
// Ensure backend has initialized
await tf.ready();

// Import from CSV
const houseSalesDataset = tf.data.csv("./nyc_housing.csv");

// BEGIN
// Extract x and y values to plot
const pointsDataset = houseSalesDataset.map(record => ({
    x: record.sqfeet,
    y: record.beds,
    z: record.price,
}));
points = await pointsDataset.toArray();
if(points.length % 2 !== 0) { // If odd number of elements
    points.pop(); // remove one element
}
console.log(points);
tf.util.shuffle(points);
plotPoints(points, "price");

// Extract Features (inputs)
const featureValues = points.map(p => [p.x, p.y]);
const featureTensor = tf.tensor2d(featureValues);

// Extract Labels (outputs)
const labelValues = points.map(p => p.z);
const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

// normalize features and labels
normalizedFeature = normalize(featureTensor);
normalizedLabel = normalize(labelTensor);
featureTensor.dispose();
labelTensor.dispose();

[trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
[trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

// Update status and enable train button
document.getElementById("model-status").innerHTML = "No model trained";
document.getElementById("train-button").removeAttribute("disabled");
document.getElementById("load-button").removeAttribute("disabled");
}
