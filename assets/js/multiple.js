
// When we have multiple features for our input, we should perform normalization (aka feature scaling) for each separately (keeping hold of separate min and max values for denormalization).
// We can visualise 2 input features for a on a scatter plot.

// global variables
let points, normalizedFeature, normalizedLabel, trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor, model;
const storageID = "nyc_housing_multiple";

run();

async function toggle() {
    tfvis.visor().toggle();
    }


async function save() {
    const savedModel = await model.save(`localstorage://${storageID}`);
    document.getElementById("model-status").innerHTML = `Trained (saved ${savedModel.modelArtifactsInfo.dateSaved})`;
    }

async function plotPoints(pointsArray, priceTag) {
    const allpricePoints = {};
    // Process the points received
    pointsArray.forEach(p => {
        // Add points per rent
        if(p.z > 400){
            const pricePointsName = `${priceTag}: ${p.z}`;
            let pricePoints = allpricePoints[pricePointsName];
            if (!pricePoints) {
                pricePoints = [];
                allpricePoints[pricePointsName] = pricePoints;
            }
            pricePoints.push(p);
        }

    });

    // draw them
    tfvis.render.scatterplot(
        {
        name: `Square feet vs Bedrooms`,
        styles: { width: "100%", height: "100%" },
        },
        {
        values: Object.values(allpricePoints),
        series: Object.keys(allpricePoints),
        },
        {
        xLabel: "Square feet",
        yLabel: "Bedrooms"
        }
    );
}

function normalize(tensor, previousMin = null, previousMax = null) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];

    // If We got more than one feature
    if (featureDimensions && featureDimensions > 1) {

        // Put them into split tensors
        const features = tf.split(tensor, featureDimensions, 1);

        // Normalize and find min max
        const normalizedFeatures = features.map((featureTensor, i) =>
            normalize(featureTensor,
                previousMin ? previousMin[i] : null,
                previousMax ? previousMax[i] : null,
            )
        );

        // concat to tf
        const returnTensor = tf.concat(normalizedFeatures.map(f => f.tensor), 1);
        const min = normalizedFeatures.map(f => f.min);
        const max = normalizedFeatures.map(f => f.max);

        return { tensor: returnTensor, min, max };
    }
    else {
        // else we got just one feature
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

// function to denormalize with multiple features
function denormalize(tensor, min, max) {
    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];

    if (featureDimensions && featureDimensions > 1) {
        // We got more than one feature

        // put them into separate tensors to process
        const features = tf.split(tensor, featureDimensions, 1);

        // denormalized
        const denormalized = features.map((featureTensor, i) => denormalize(featureTensor, min[i], max[i]));

        // concat to tf
        const returnTensor = tf.concat(denormalized, 1);
        return returnTensor;
    }
    else {
        // regular one feature
        const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
        return denormalizedTensor;
    }
}

// function to create our model
function createModel () {
    model = tf.sequential();

    // using linear
    model.add(tf.layers.dense({
        units: 10,
        useBias: true,
        activation: 'linear',
        inputDim: 5,
    }));
    model.add(tf.layers.dense({
        units: 10,
        useBias: true,
        activation: 'linear',
    }));
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
    }));

    const optimizer = tf.train.adam();
    model.compile({
        // using MSE
        loss: 'meanSquaredError',
        optimizer,
    });

    return model;
}

// function to train our model
async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training" },
        ['loss']
    );
    // return the fit model
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

// async func to predict
async function predict(){

    const predictionInputOne = parseInt(document.getElementById("prediction-input-1").value);
    const predictionInputTwo = parseInt(document.getElementById("prediction-input-2").value);

    const predictionInputThree = parseInt(document.getElementById("prediction-input-3").value);
    const predictionInputFour = parseInt(document.getElementById("prediction-input-4").value);
    const predictionInputFive = parseInt(document.getElementById("prediction-input-5").value);

    if (isNaN(predictionInputOne) || isNaN(predictionInputTwo) || isNaN(predictionInputThree) || isNaN(predictionInputFour) || isNaN(predictionInputFive)) {
        alert("Please enter a valid input");
    }
    else if (predictionInputOne < 200) {
        alert("Please enter a value above 200 sqft");
    }
    else if (predictionInputTwo < 0) {
        alert("Please enter a valid number for bedrooms");
    }
    else if (predictionInputThree < 0) {
        alert("Please enter a valid number for Bathrooms");
    }
    else if (predictionInputFour < 0) {
        alert("Please enter a valid number for Dogs Allowed");
    }
    else if (predictionInputFive < 0) {
        alert("Please enter a valid number for Cats Allowed");
    }
    else {
        // use tidy for memory cleaning
        tf.tidy(() => {
            const inputTensor = tf.tensor2d([[predictionInputOne, predictionInputTwo, predictionInputThree, predictionInputFour, predictionInputFive]]);
            const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
            const normalizedOutputTensor = model.predict(normalizedInput.tensor);
            const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
            const outputValue = outputTensor.dataSync()[0];
            const outputValuePercent = (outputValue/50).toFixed(0)*50;
            document.getElementById("prediction-output").innerHTML = `The predicted rent is <br>`
                + `<span style="font-size: 2em">${outputValuePercent}</span>`;
        });
    }
}

async function load() {
    // func to load previously saved model
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

async function test(){
    // func to test
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = (await lossTensor.dataSync())[0];
    console.log(`Testing set loss: ${loss}`);

    document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

async function train(){
    // func to train our model
    // Disable all buttons and update status
    ["train", "test", "load", "predict", "save"].forEach(id => {
        document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
    });
    document.getElementById("model-status").innerHTML = "Training...";

    // create model
    const model = createModel();
    tfvis.show.modelSummary({ name: "Model summary" }, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({ name: "Layer 1" }, layer);

    // await the resuld from train model
    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor)
    console.log(result);
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);
    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    document.getElementById("model-status").innerHTML = "Trained (unsaved)\n<br>"
        + `Training Loss: ${trainingLoss.toPrecision(5)}\n<br>`
        + `Validation loss: ${validationLoss.toPrecision(5)}`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("save-button").removeAttribute("disabled");
    document.getElementById("predict-button").removeAttribute("disabled");
}

async function run () {

    // make sure tf is ready to operate on web
    await tf.ready();

    // Import from our CSV file
    const houseSalesDataset = tf.data.csv("./nyc_housing.csv");

    // Extract x and y values to plot
    const pointsDataset = houseSalesDataset.map(record => ({
        x: record.sqfeet,
        y: record.beds,
        w: record.baths,
        v: record.dogs_allowed,
        s: record.cats_allowed,
        z: record.price,
    }));
    points = await pointsDataset.toArray();

    // If odd number of elements remove one element
    if(points.length % 2 !== 0) { 
        points.pop(); 
    }
    console.log(points);
    tf.util.shuffle(points);
    plotPoints(points, "price");

    // get Features
    const featureValues = points.map(p => [p.x, p.y, p.w, p.v, p.s]);
    const featureTensor = tf.tensor2d(featureValues);

    // get Labels
    const labelValues = points.map(p => p.z);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // normalize features and labels
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);
    featureTensor.dispose();
    labelTensor.dispose();

    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    // Update results on UI
    document.getElementById("model-status").innerHTML = "No model trained";
    document.getElementById("train-button").removeAttribute("disabled");
    document.getElementById("load-button").removeAttribute("disabled");
}
