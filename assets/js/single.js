

// train model, evaluate loss, update and optimize model and repeat for next epoch.
// test is not iterative, just evaluate against test dataset and evaluate loss and compare with train
// if close, well trained. if too diff, may overfitted or data was not prepared properly, not shuffled
// validation dataset for frequent evaluation of model independent from train set, testing reserved for final stage


// http-server

// TensorFlow.js can import data from CSV file via HTTP (or the local file system when using Node.js).
// The tfjs-vis library can to help you visualise data when using TensorFlow.js
// Features and labels should be stored in 2D tensors to feed into the model.
// Features and labels should be normalized to a range between 0 and 1.
// You can normalize using min-max normalization, implemented with tensor operations.
// You need to keep track of the minimum and maximum values to later denormalize.
// Tensors can be split into sub tensors in any ratio using tf.split(...)

// global variables
let model, points, normalizedFeature, normalizedLabel,trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor, testingLabelTensor;
const storageID = "nyc_housing_price";

run();

function normalize(tensor, previousMin = null, previousMax = null) {
    // method: https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    //  every value be in range 0 and 1
    // prioratize if already given and specified
    const min = previousMin || tensor.min();
    const max = previousMax || tensor.max();
    const normalizedTensor = tensor.sub(min).div(max.sub(min));
    return {
        tensor: normalizedTensor,
        min,
        max
    };
}

function denormalize(tensor, min, max) {
    // denormalize tensors if needed later
    const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalizedTensor;
}

// runs on save button click
// saves our model into the localstorage
async function save(){
    const saveResults = await model.save(`localstorage://${storageID}`);
    // update UI with date time of it
    document.getElementById("model-status").innerHTML = `Saved Trained Model (on ${saveResults.modelArtifactsInfo.dateSaved})`;
}

// async func to plot and visualize our data 
async function plot(pointsArray, featureName, predictedPointsArray = null) {
    const values = [pointsArray.slice(0, 1000)];
    const series = ["original"];

    // if predicted points array is provided then plot those as well
    if (Array.isArray(predictedPointsArray)) {
        values.push(predictedPointsArray);
        series.push("predicted");
    }

    tfvis.render.scatterplot(
        { name: `${featureName} vs Rent` },
        { values, series },
        {
        xLabel: featureName,
        yLabel: "Rent",
        height: 300,
        }
    )
}

// open or close tf visor
async function toggle() {
    tfvis.visor().toggle();
}

// func to plot the prediction line
async function plotPredictionLine() {

    // prepare points in equal distance from each other using linspace
    const [xs, ys] = tf.tidy(() => {
        const normalizedXs = tf.linspace(0, 1, 100);
        // predict y points
        const normalizedYs = model.predict(normalizedXs.reshape([100, 1]));
        
        // denormalize to show them on plot
        const xs = denormalize(normalizedXs, normalizedFeature.min, normalizedFeature.max);
        const ys = denormalize(normalizedYs, normalizedLabel.min, normalizedLabel.max);

        return [ xs.dataSync(), ys.dataSync() ];
    });

    // build the array of predicted points
    const predictedPoints = Array.from(xs).map((val, index) => {
        return { x: val, y: ys[index] };
    });

    // plot them
    await plot(points, "Square feet", predictedPoints);
}

function createModel() {
    // sequential model, output of a layer is the input of next layer
    model = tf.sequential();

    // single layer, single node, linear
    // dense, all inputs and outputs are connected
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1,
    }));
    // model.add(tf.layers.dense({
    //   units: 1,
    //   useBias: true,
    //   activation: 'linear',
    // }));
    // model.add(tf.layers.dense({
    //   units: 1,
    //   useBias: true,
    //   activation: 'linear',
    // }));

    // do it non-linearly with signmoid activ func
    // model.add(tf.layers.dense({
    //   units: 1,
    //   useBias: true,
    //   activation: 'sigmoid',
    //   inputDim: 1,
    // }));

    // add multiple layers to model
    // speeds up training process, but more vulnerable to overfit
    // model.add(tf.layers.dense({
    //   units: 10,
    //   useBias: true,
    //   activation: 'sigmoid',
    //   inputDim: 1,
    // }));
    // model.add(tf.layers.dense({
    //   units: 10,
    //   useBias: true,
    //   activation: 'sigmoid',
    // }));
    // model.add(tf.layers.dense({
    //   // only one label
    //   units: 1,
    //   useBias: true,
    //   activation: 'sigmoid',
    // }));

    // optimizer
    // stochastic gradient descent builtin optimizer, with a learning rate of 0.01
    // 0.1: good starting point
    // 0.01: reduced loss, but slightly more number of epochs to train
    // 0.001: taking longer to min
    // 0.5: wobbly moving away from min, bit too large
    // 1.0: very much too large, diverging away from min
    // adam: different optimizer algorithm, works effectively without learning rate, the algo adapts leraning rate over epochs
    const optimizer = tf.train.sgd(0.01);

    // prepares model for training and testing, chose loss func from builtin tf.loss.meanSquaredError
    // RMSE with already normalized data is overkill, so we just use MSE
    model.compile({
        loss: 'meanSquaredError',
        optimizer,
    });

    return model;
}

// async func to train model
async function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {

    // visualize the training progress - loss
    const { onBatchEnd, onEpochEnd } = tfvis.show.fitCallbacks(
        { name: "Training" },
        ['loss']
    );

    // fit method to train for a certain number of epochs, used 20 first, then 30
    //  depends on training dataset and learning rate
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        // in each epoch, there'll be number batches of data to be optimized
        batchSize: 32,
        epochs: 30,
        // 20% of dataset for validation for safeguard to check overfitting
        // calculating loss independently from training data
        validationSplit: 0.2,
        callbacks: {
        onEpochEnd,
        // to log loss values on each epochs
        // onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        onEpochBegin: async function () {
            await plotPredictionLine();
            // const layer = model.getLayer(undefined, 0);
            // tfvis.show.layer({ name: "Layer 1" }, layer);
        }
        }
    });
}

// runs on predict button click
// predicts price
async function predict(){

    // get input for sqft
    const predictionInput = parseInt(document.getElementById("prediction-input").value);
    if (isNaN(predictionInput)) {
        alert("Please enter a valid number");
    }
    // does not make sense for user or a house to be smaller than 200sqft
    // not much data to predict it
    else if (predictionInput < 200) {
        alert("Please enter a value above 200 sqft");
    }
    else {
        // tidy to free up memory usage afterwards
        tf.tidy(() => {
        const inputTensor = tf.tensor1d([predictionInput]);
        // normalize
        const normalizedInput = normalize(inputTensor, normalizedFeature.min, normalizedFeature.max);
        // predict
        const normalizedOutputTensor = model.predict(normalizedInput.tensor);
        // denormalize the output
        const outputTensor = denormalize(normalizedOutputTensor, normalizedLabel.min, normalizedLabel.max);
        // extract the denormalized value
        const outputValue = outputTensor.dataSync()[0];
        // round to nearest 50 number for userfriendliness
        const outputValueRounded = (outputValue/50).toFixed(0)*50;
        // update the user interface with the found result
        document.getElementById("prediction-output").innerHTML = `The predicted house rent is <br>`
            + `<span style="font-size: 2em">\$${outputValueRounded}</span>`;
        });
    }
}

// runs on load button click
// runs the previously saved model
async function load(){
    // try to find it by the key
    const storageKey = `localstorage://${storageID}`;
    //  list of models saved
    const models = await tf.io.listModels();
    // get the model assign the object to modelInfo
    const modelInfo = models[storageKey];

    // if model exists (saved)
    if (modelInfo) {
        // model found
        model = await tf.loadLayersModel(storageKey);

        // plot the line
        await plotPredictionLine();

        // update user interface with saved model loaded
        document.getElementById("model-status").innerHTML = `Trained (saved ${modelInfo.dateSaved})`;
        document.getElementById("predict-button").removeAttribute("disabled");
    }
    else {
        // there is no saved model
        alert("Could not load: no saved model found");
    }
}

// runs on test button click
async function test(){
    // get the testing set loss and print it to console
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
    const loss = (await lossTensor.dataSync())[0];
    console.log(`Testing set loss: ${loss}`);

    // update user interface with loss result
    document.getElementById("testing-status").innerHTML = `Testing set loss: ${loss.toPrecision(5)}`;
}

// runs on train button click, train our model
async function train(){

    // Disable all buttons and update status
    ["train", "test", "load", "predict", "save"].forEach(id => {
        document.getElementById(`${id}-button`).setAttribute("disabled", "disabled");
    });
    document.getElementById("model-status").innerHTML = "Training...";

    // create model
    const model = createModel();

    // plot the prediction line
    await plotPredictionLine();

    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
    console.log(result);

    // print loss for training set result to console
    const trainingLoss = result.history.loss.pop();
    console.log(`Training set loss: ${trainingLoss}`);

    // print loss for validation set result to console
    const validationLoss = result.history.val_loss.pop();
    console.log(`Validation set loss: ${validationLoss}`);

    document.getElementById("model-status").innerHTML = "Trained (unsaved)\n\n<br>"
        + `Training Loss: ${trainingLoss.toPrecision(5)}\n<br>`
        + `Validation loss: ${validationLoss.toPrecision(5)}`;
    document.getElementById("test-button").removeAttribute("disabled");
    document.getElementById("save-button").removeAttribute("disabled");
    document.getElementById("predict-button").removeAttribute("disabled");
}

async function run() {
    // Ensure tensorflow is initialized and ready
    await tf.ready();

    // Import from csv file
    const housingData = tf.data.csv("http://127.0.0.1:8080/nyc_housing.csv");

    // Extract x and y values to plot
    const pointsDataset = housingData.map(record => ({
        x: record.sqfeet,
        y: record.price,
    }));
    points = await pointsDataset.toArray();

    // to fix error of evenly split data into training and testing, we remove one element if odd
    if(points.length % 2 !== 0) { // If odd number of elements
        points.pop(); // remove one element
    }

    // shuffle data to avoid patterns in order of data or prices by area
    // so no split of data gets more than others on a category or group of values
    tf.util.shuffle(points);
    plot(points, "Square Feet");

    // Extract Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // Extract Labels (outputs)
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

    // featureTensor.print()
    // labelTensor.print()

    // Normalize features and labels
    normalizedFeature = normalize(featureTensor);
    normalizedLabel = normalize(labelTensor);

    // keep memory usage under control
    featureTensor.dispose();
    labelTensor.dispose();

    // normalizedFeature.tensor.print()
    // normalizedLabel.tensor.print()

    // split train and test into 50 - 50 , equally sized datasets
    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel.tensor, 2);

    // trainingFeatureTensor.print()
    // trainingLabelTensor.print()


    // Update status and enable train button
    document.getElementById("model-status").innerHTML = "No trained model yet";
    document.getElementById("train-button").removeAttribute("disabled");
    document.getElementById("load-button").removeAttribute("disabled");

}