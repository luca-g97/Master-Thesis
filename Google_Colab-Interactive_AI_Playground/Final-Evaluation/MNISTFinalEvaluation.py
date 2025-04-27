import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
import numpy as np
import pickle
import logging
import os
import RENNFinalEvaluation as RENN
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import spearmanr, kendalltau, pearsonr
from IPython.display import clear_output, display
from collections import defaultdict, Counter, OrderedDict
import concurrent.futures
import json
import time
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import plotly.graph_objects as go
import plotly.subplots as sp
import itertools
import functools # For caching
import hashlib
import math
import traceback
import collections.abc # For checking nested structures more robustly

CHECKPOINT_DIR = "checkpoints_mnist_viz" # Directory to store checkpoint files

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

mnist, to_categorical, nn, DataLoader, pd, optuna, device, metricsEvaluation = "", "", "", "", "", "", "", True
train_dataloader, test_dataloader, eval_dataloader, trainDataSet, testDataSet, trainSubset, testSubset, x_train, y_train, x_test, y_test, x_eval, y_eval = "", "", "", "", "", "", "", "", "", "", "", "", ""
model, criterion_class, chosen_optimizer, layers = "", "", "", ""
train_samples, eval_samples, test_samples = 1, 1, 1
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource = [], [], [], [], [], []

def initializePackages(mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, pdPackage, optunaPackage, devicePackage):
    global mnist, to_categorical, nn, DataLoader, pd, optuna, device

    mnist, to_categorical, nn, DataLoader, pd, optuna, device = mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, pdPackage, optunaPackage, devicePackage

def createTrainAndTestSet():
    global trainDataSet, testDataSet, x_train, y_train
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the training data
    x_train = x_train.astype('float32') / 255.0
    y_train = to_categorical(y_train)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    # Preprocess the testing data
    x_test = x_test.astype('float32') / 255.0
    y_test = to_categorical(y_test)

    x_eval = x_test
    y_eval = y_test
    x_eval = torch.from_numpy(x_eval)
    y_eval = torch.from_numpy(y_eval)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    trainDataSet = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_train, y_train)]
    testDataSet = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_test, y_test)]

    print(f"Created {len(trainDataSet)} Trainsamples & {len(testDataSet)} Testsamples")
    return trainDataSet, testDataSet

def initializeDatasets(train_samplesParameter, test_samplesParameter, eval_samplesParameter, batch_size_training, batch_size_test, seed=""):
    global train_samples, test_samples, eval_samples, np, torch
    global train_dataloader, test_dataloader, eval_dataloader, trainSubset, testSubset
    train_samples, test_samples, eval_samples = train_samplesParameter, test_samplesParameter, eval_samplesParameter

    if(seed != ""):
        print("Setting seed number to", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else: print("Setting random seed")

    trainSubset = Subset(trainDataSet, range(train_samples))
    testSubset = Subset(testDataSet, range(test_samples))
    evalSubset = Subset(testDataSet, range(eval_samples))
    train_dataloader = DataLoader(trainSubset, batch_size=batch_size_training, shuffle=False)
    test_dataloader = DataLoader(testSubset, batch_size=batch_size_test, shuffle=False)
    eval_dataloader = DataLoader(evalSubset, batch_size=1, shuffle=False)
    print("Created all dataloaders")

def initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate):
    global model, criterion_class, chosen_optimizer, layers
    input_size = torch.flatten(train_dataloader.dataset[0][0]).numel()
    output_size = torch.flatten(train_dataloader.dataset[0][1]).numel()

    model = RENN.CustomizableRENN(input_size, hidden_sizes, output_size)
    model.to(device)
    layers = np.array(RENN.layers)

    if(loss_function == "MSE"):
        criterion_class = nn.MSELoss()  # For regression
    elif(loss_function == "Cross-Entropy"):
        criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

    if(optimizer == "Adam"):
        chosen_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer == "SGD"):
        chosen_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(epochs=10):
    global model, chosen_optimizer, criterion_class
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, classification in train_dataloader:
            images = images.float()
            images = images.to(device)
            classification = classification.float()
            classification = classification.to(device)
            chosen_optimizer.zero_grad()
            output = model(images)
            loss = criterion_class(output, classification)
            loss.backward()
            chosen_optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, classification in test_dataloader:
                images = images.float()
                images = images.to(device)
                classification = classification.float()
                classification = classification.to(device)
                output = model(images)
                loss = criterion_class(output, classification)
                val_loss += loss.item()

        # Print statistics
        if epoch == epochs-1:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataloader)}, Validation Loss: {val_loss/len(test_dataloader)}')

def trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs):
    initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate)
    print("Model initialized, Starting training")
    train(epochs=epochs)
    print("Training finished")

def initializeHook(hidden_sizes, train_samples):
    global trainSubset

    hookDataLoader = DataLoader(trainSubset, batch_size=1, shuffle=False)
    RENN.initializeHook(hookDataLoader, model, hidden_sizes, train_samples, metricsEvaluation)

def showIndividualImagesPlotly(images, layer, closestSources, showClosestMostUsedSources, mode):
    num_images = len(images)
    num_cols = 5  # Number of columns
    if num_images == 10:
        num_cols = 5
    num_rows = num_images // num_cols if (num_images // num_cols >= 1) else 1  # Number of rows
    if num_images % num_cols != 0:
        num_rows = num_images // num_cols + 1  # Number of rows

    fig = sp.make_subplots(rows=num_rows, cols=num_cols)

    for i, (image, sources, title) in enumerate(images):
        row_index = i // num_cols
        col_index = i % num_cols
        fig.add_trace(go.Image(name=f'<b>{title}</b><br>Closest {showClosestMostUsedSources} Sources compared by {mode}:<br>{sources}', z=image), row=row_index + 1, col=col_index + 1)

    fig.update_layout(
        title=f'Blended closest {closestSources} sources for each neuron in layer {layer} (compared to their {mode} output)',
        grid={'rows': num_rows, 'columns': num_cols},
        height=225 * num_rows,  # Adjust the height of the plot
        width=225 * num_cols,
        hoverlabel=dict(namelength=-1)
    )

    #fig.show()
    display(fig)

def showImagesUnweighted(name, originalImage, blendedSourceImageActivation, blendedSourceImageSum, closestMostUsedSourceImagesActivation, closestMostUsedSourceImagesSum):
    fig, axes = plt.subplots(1, 5, figsize=(35, 35))
    plt.subplots_adjust(hspace=0.5)

    # Display original image
    axes[0].set_title(f"{name}: BLENDED -> Original: {originalImage[1]}")
    axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))

    # Display blendedSourceImageActivation
    axes[1].set_title(f"A=Activation - Closest Sources/Neuron (Most Used)")
    axes[1].imshow(blendedSourceImageActivation[0])

    # Display blendedSourceImageSum
    axes[2].set_title(f"S=Sum - Closest Sources/Neuron (Most Used)")
    axes[2].imshow(blendedSourceImageSum[0])

    # Display weightedSourceImageActivation
    axes[3].set_title(f"WA=WeightedActivation - Closest Sources/Neuron (Weighted)")
    axes[3].imshow(Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA"))

    # Display weightedSourceImageSum
    axes[4].set_title(f"WS=WeigthedSum - Closest Sources/Neuron (Weighted)")
    axes[4].imshow(Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA"))

    plt.show()

    # Display closestMostUsedSourceImagesActivation
    fig, axes = plt.subplots(1, len(closestMostUsedSourceImagesActivation)+2, figsize=(35, 35))
    axes[0].set_title(f"NON-LINEAR - Original: {originalImage[1]}")
    axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))
    axes[1].set_title(f"A - Closest Sources/Neuron (Most Used)")
    axes[1].imshow(blendedSourceImageActivation[0])
    for i, source in enumerate(closestMostUsedSourceImagesActivation):
        image = createImageWithPrediction(x_train[source[0]], y_train[source[0]], predict(x_train[source[0]]))
        axes[i+2].set_title(f"A - Source {source[0]} ({(blendedSourceImageActivation[1][i]*100.0):.2f}%): {image[1]}")
        axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
    plt.show()

    # Display closestMostUsedSourceImagesSum
    fig, axes = plt.subplots(1, len(closestMostUsedSourceImagesSum)+2, figsize=(35, 35))
    axes[0].set_title(f"LINEAR - Original: {originalImage[1]}")
    axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))
    axes[1].set_title(f"S - Closest Sources/Neuron (Most Used)")
    axes[1].imshow(blendedSourceImageSum[0])
    for i, source in enumerate(closestMostUsedSourceImagesSum):
        image = createImageWithPrediction(x_train[source[0]], y_train[source[0]], predict(x_train[source[0]]))
        axes[i+2].set_title(f"S - Source {source[0]} ({(blendedSourceImageSum[1][i]*100.0):.2f}%): {image[1]}")
        axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
    plt.show()

#Import dataclass package to create a class based approach instead of normal arrays
from dataclasses import dataclass
@dataclass(order=True)
class WeightedSource:
    source: int
    difference: float

def getMostUsedPerLayer(sources):
    mostUsed = []
    sourceCounter = 0
    for src in sources:
        mostUsed.append(src.source)
        sourceCounter += 1
    return sourceCounter, mostUsed

def getClosestSourcesPerNeuronAndLayer(sources, layersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, mode=""):
    for cLayer, layer in enumerate(sources):
        weightedSourcesPerLayer = []
        totalDifferencePerLayer = 0
        imagesPerLayer = []

        for cNeuron, neuron in enumerate(layer):
            if(cNeuron < layers[cLayer][1].out_features):
                weightedSourcesPerNeuron = []
                totalDifferencePerNeuron = 0
                for sourceNumber, value, difference in neuron:
                    baseWeightedSource = {'source': sourceNumber, 'difference': difference}
                    totalDifferencePerNeuron += difference
                    totalDifferencePerLayer += difference
                    weightedSourcesPerNeuron.append(WeightedSource(**baseWeightedSource))
                    weightedSourcesPerLayer.append(WeightedSource(**baseWeightedSource))
                if not(visualizationChoice == "Custom" and ((cNeuron < int(visualizeCustom[cLayer][0][0])) or (cNeuron > int(visualizeCustom[cLayer][0][1])))):
                    imagesPerLayer.append([blendIndividualImagesTogether(weightedSourcesPerNeuron, closestSources), [f"Source: {source.source}, Difference: {source.difference:.10f}<br>" for source in weightedSourcesPerNeuron][:showClosestMostUsedSources], f"{mode} - Layer: {int(layersToCheck[cLayer]/2)}, Neuron: {cNeuron}"])

        if not(visualizationChoice == "Per Layer Only"):
            if not(mode == "Activation" and visualizationChoice == "Custom" and visualizeCustom[cLayer][1] == False):
                showIndividualImagesPlotly(imagesPerLayer, int(layersToCheck[cLayer]/2), closestSources, showClosestMostUsedSources, mode)

        if not(visualizationChoice == "Per Neuron Only"):
            if not(mode == "Activation" and visualizationChoice == "Custom" and visualizeCustom[cLayer][1] == False):
                weightedSourcesPerLayer = sorted(weightedSourcesPerLayer, key=lambda x: x.difference)
                sourceCounter, mostUsed = getMostUsedPerLayer(weightedSourcesPerLayer)
                counter = Counter(mostUsed)
                image = blendIndividualImagesTogether(counter.most_common()[:closestSources], closestSources, True)

                plt.figure(figsize=(28,28))
                plt.imshow(image)
                plt.title(f"{mode} - Layer:  {int(layersToCheck[cLayer]/2)}, {closestSources} most used Sources")
                plt.show()

"""# Evaluation: Visual Blending"""

def blendImagesTogether(mostUsedSources, mode):
    image_numpy = np.zeros(shape=[28,28], dtype=np.float64)
    weights = []
    total = 0

    if mode == "Not Weighted":
        for sourceNumber, counter in mostUsedSources:
            total += counter

        for sourceNumber, counter in mostUsedSources:
            image_numpy += (counter/total) * x_train[sourceNumber].numpy()*255
            weights.append(counter/total)

    image = Image.fromarray(image_numpy).convert("RGBA")
    return (image, weights)

def blendIndividualImagesTogether(mostUsedSources, closestSources, layer=False):
    image_numpy = np.zeros(shape=[28,28], dtype=np.float64)
    #image = Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA")

    total = 0
    for source in mostUsedSources:
        if(layer):
            total += source[1]
        else:
            total += source.difference

    for wSource in mostUsedSources:
        #TODO: NORMALIZATION!!!
        if(total > 0):
            if(closestSources < 2):
                if(layer):
                    image_numpy += x_train[int(wSource[0])].numpy()*255
                else:
                    image_numpy += x_train[int(wSource.source)].numpy()*255
            else:
                if(layer):
                    image_numpy += (x_train[int(wSource[0])].numpy()*255 * wSource[1] / total)
                else:
                    #print(f"Diff: {wSource.difference}, Total: {total}, Calculation: {(1 - (wSource.difference / total)) / closestSources}")
                    image_numpy += (x_train[int(wSource.source)].numpy()*255 * (1 - (wSource.difference / total)) / closestSources)

    image = Image.fromarray(image_numpy).convert("RGBA")
    return image
"""# Evaluation: Prediction"""

def predict(sample):
    with torch.no_grad():
        sample = sample.to(device)
        model.eval()
        output = model(torch.flatten(sample))
    normalizedPredictions = normalizePredictions(output.cpu().numpy())
    return np.argmax(normalizedPredictions), normalizedPredictions[np.argmax(normalizedPredictions)]

def createImageWithPrediction(sample, true, prediction):
    sample = sample.to(device)
    true = true.to(device)
    prediction, probability = predict(sample)
    true_class = int(torch.argmax(true.cpu()))  # Move `true` tensor to CPU and then get the index of the maximum value
    return [sample, f"pred: {prediction}, prob: {probability:.2f}, true: {true_class}"]

def normalizePredictions(array):
    min = np.min(array)
    max = np.max(array)
    return ((array - min) / (max - min)) if (max-min) > 0.0 else np.zeros_like(array)

"""# Evaluation: Code"""

original_image_similarity, metrics_image_similarity, mt_image_similarity = [], [], []
def evaluateImageSimilarity(name, sample, mostUsed):
    global original_image_similarity, metrics_image_similarity, mt_image_similarity

    # Flatten and reshape the sample
    sample = np.asarray(sample.flatten().reshape(1, -1))

    blended_image = blendIndividualImagesTogether(mostUsed, len(mostUsed), layer=True)
    # Compute similarity for the blended image with sample
    blended_image_flat = np.asarray(blended_image.convert('L')).flatten() / 255.0
    blended_image_flat = blended_image_flat.reshape(1, -1)

    cosine_similarity, euclidean_distance, manhattan_distance, jaccard_similarity, hamming_distance, pearson_correlation = computeSimilarity(sample, blended_image_flat)

    if not (np.isnan(sample).any() or np.isnan(blended_image_flat).any() or np.std(sample) < EPSILON or np.std(blended_image_flat) < EPSILON):
        kendall_tau, _ = kendalltau(sample, blended_image_flat)
        spearman_rho, _ = spearmanr(sample, blended_image_flat)
    else:
        kendall_tau = np.nan
        spearman_rho = np.nan

    results = {
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "manhattan_distance": manhattan_distance,
        "jaccard_similarity": jaccard_similarity if jaccard_similarity is not None else np.nan,
        "hamming_distance": hamming_distance,
        "pearson_correlation": pearson_correlation if pearson_correlation is not None else np.nan,
    }

    # --- Print Results ---
    # print(f"\n--- Blended Image Similarity Scores ({name})---")
    # print(f"Kendall's Tau: {kendall_tau:.2f}")
    # print(f"Spearman's Rho: {spearman_rho:.2f}")
    # print(f"Cosine Similarity: {cosine_similarity:.4f}")
    # print(f"Euclidean Distance: {euclidean_distance:.4f}")
    # print(f"Manhattan Distance: {manhattan_distance:.4f}")
    # print(f"Jaccard Similarity: {jaccard_similarity:.4f}" if jaccard_similarity is not None else "Jaccard Similarity: N/A")
    # print(f"Hamming Distance: {hamming_distance:.4f}")
    # print(f"Pearson Correlation: {pearson_correlation:.4f}" if pearson_correlation is not None else "Pearson Correlation: N/A")

    if name == "":
        original_image_similarity.append(results)
    elif name == "Metrics":
        metrics_image_similarity.append(results)
    elif name == "MT":
        mt_image_similarity.append(results)
        
    return results

def compute_cosine_similarity(image1, image2):
    """Compute cosine similarity between two images."""
    vec1 = image1.flatten().reshape(1, -1)
    vec2 = image2.flatten().reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

EPSILON = 1e-9 # Small tolerance for variance check
def computeSimilarity(sample, train_sample):
    # Compute similarities
    cosine_similarity = compute_cosine_similarity(sample, train_sample)
    euclidean_distance = np.linalg.norm(sample - train_sample)  # Euclidean
    manhattan_distance = np.sum(np.abs(sample - train_sample))  # Manhattan
    jaccard_similarity = (
        np.sum(np.minimum(sample, train_sample)) / np.sum(np.maximum(sample, train_sample))
        if np.sum(np.maximum(sample, train_sample)) > 0 else None
    )
    hamming_distance = np.mean(sample != train_sample)  # Hamming
    if not (np.isnan(sample).any() or np.isnan(train_sample).any() or np.std(sample) < EPSILON or np.std(train_sample) < EPSILON):
        pearson_correlation, _ = pearsonr(sample.flatten(), train_sample.flatten())  # Pearson
    else:
        pearson_correlation = np.nan

    return cosine_similarity, euclidean_distance, manhattan_distance, jaccard_similarity, hamming_distance, pearson_correlation

# Global storage (optional)
original_activation_similarity, metrics_activation_similarity, mt_activation_similarity = [], [], []
# --- blendActivations function with added NaN checks ---
def blendActivations(name, mostUsed, evaluationActivations, layerNumbersToCheck, store_globally=False, overallEvaluation=True):
    """
    Blends activations based on most used sources and computes similarity metrics.
    Includes checks for NaNs generated during the blending process.
    """
    global original_activation_similarity, metrics_activation_similarity, mt_activation_similarity # Ensure RENN is accessible

    if not layerNumbersToCheck:
        print(f"WARNING blendActivations ({name}): layerNumbersToCheck is empty. Returning NaNs.")
        return {key: np.nan for key in ACTIVATION_METRIC_KEYS}

    # Initialize blendedActivations based on the shape of the first layer's activations
    # This assumes all layers in evaluationActivations[layerNumbersToCheck] have the same shape.
    try:
        # Get the shape of the target activations for the layers being checked
        target_shape = evaluationActivations[layerNumbersToCheck].shape
        blendedActivations = np.zeros(target_shape, dtype=np.float64) # Use float64 for accumulation
    except IndexError:
        print(f"ERROR blendActivations ({name}): Invalid layer indices in layerNumbersToCheck: {layerNumbersToCheck} for evaluationActivations shape {evaluationActivations.shape}")
        return {key: np.nan for key in ACTIVATION_METRIC_KEYS}
    except Exception as e:
        print(f"ERROR blendActivations ({name}): Could not initialize blendedActivations. Error: {e}")
        return {key: np.nan for key in ACTIVATION_METRIC_KEYS}

    nan_introduced_during_blending = False # Flag

    try:
        if overallEvaluation:
            # Handle BTO case where mostUsed might be nested one level deeper
            if "BTO" in name and isinstance(mostUsed, list) and len(mostUsed) == 1 and isinstance(mostUsed[0], list):
                currentMostUsed = mostUsed[0] # Use the inner list for BTO
            else:
                currentMostUsed = mostUsed # Use directly otherwise
            
            #print("currentMostUsed", currentMostUsed)
            if not isinstance(currentMostUsed, list) or not all(isinstance(item, (tuple, list)) and len(item) == 2 for item in currentMostUsed):
                raise TypeError(f"Invalid structure for mostUsed in overallEvaluation=True. Expected list of (source, count), got {type(currentMostUsed)}")

            # Calculate totalSources, checking for NaN counts
            valid_counts = [count for _, count in currentMostUsed if isinstance(count, (float, int, np.number)) and math.isfinite(count)]
            if len(valid_counts) != len(currentMostUsed):
                print(f"WARNING blendActivations ({name}, Overall): NaN/Inf found in source counts. Using only finite counts.")
                nan_introduced_during_blending = True # Counts had issues

            if not valid_counts:
                print(f"ERROR blendActivations ({name}, Overall): No valid source counts found. Cannot blend.")
                nan_introduced_during_blending = True
                totalSources = 0 # Avoid division by zero later, but result will be wrong
            else:
                totalSources = sum(valid_counts)

            # Check if totalSources is zero or NaN
            if totalSources < EPSILON or math.isnan(totalSources):
                print(f"WARNING blendActivations ({name}, Overall): totalSources is near zero ({totalSources}) or NaN. Blending may produce NaN/Inf.")
                nan_introduced_during_blending = True # Mark potential issue
                # Avoid division by zero later if totalSources is zero
                totalSources = EPSILON if totalSources < EPSILON else totalSources


            for (source, count) in currentMostUsed:
                source = int(source)
                # Skip if count is invalid
                if not (isinstance(count, (float, int, np.number)) and math.isfinite(count)):
                    continue

                # Ensure source is a valid key/index for RENN.activationsBySources
                if RENN.activationsBySources[source] is None:
                    print(f"WARNING blendActivations ({name}, Overall): Source '{source}' not found in RENN.activationsBySources. Skipping.")
                    print(RENN.activationsBySources)
                    continue

                activationsBySources = RENN.activationsBySources[source]
                for layerIdx, layerNumber in enumerate(layerNumbersToCheck):
                    # Ensure layerNumber is valid for the fetched activations
                    if layerNumber >= len(activationsBySources):
                        print(f"WARNING blendActivations ({name}, Overall): layerNumber {layerNumber} invalid for source '{source}' activations (len: {len(activationsBySources)}). Skipping layer.")
                        continue

                    neurons = np.asarray(activationsBySources[layerNumber], dtype=np.float64) # Ensure numpy array and float64

                    # Check neurons for NaN/Inf before blending
                    if check_activations_for_nan(neurons):
                        print(f"WARNING blendActivations ({name}, Overall): NaN/Inf found in 'neurons' for source '{source}', layer {layerNumber}. Skipping contribution.")
                        nan_introduced_during_blending = True
                        continue # Skip adding NaN/Inf values

                    # Perform blending if totalSources is valid
                    if not math.isnan(totalSources) and totalSources > EPSILON:
                        blendedActivations[layerIdx] += neurons * (count / totalSources)
                    # else: # Already warned about totalSources issue

        else: # Not overallEvaluation (per-layer blending)
            if len(mostUsed) != len(layerNumbersToCheck):
                raise ValueError(f"Length mismatch: mostUsed ({len(mostUsed)}) vs layerNumbersToCheck ({len(layerNumbersToCheck)}) for per-layer blending.")

            for layerIdx, mostUsedSourcesPerLayer in enumerate(mostUsed):
                if not isinstance(mostUsedSourcesPerLayer, list) or not all(isinstance(item, (tuple, list)) and len(item) == 2 for item in mostUsedSourcesPerLayer):
                    raise TypeError(f"Invalid structure for mostUsedSourcesPerLayer at layerIdx {layerIdx}. Expected list of (source, count), got {type(mostUsedSourcesPerLayer)}")

                layerNumber = layerNumbersToCheck[layerIdx]

                # Calculate totalSources for this layer, checking for NaN counts
                valid_counts = [count for _, count in mostUsedSourcesPerLayer if isinstance(count, (float, int, np.number)) and math.isfinite(count)]
                if len(valid_counts) != len(mostUsedSourcesPerLayer):
                    print(f"WARNING blendActivations ({name}, Layer {layerNumber}): NaN/Inf found in source counts. Using only finite counts.")
                    nan_introduced_during_blending = True # Counts had issues

                if not valid_counts:
                    print(f"ERROR blendActivations ({name}, Layer {layerNumber}): No valid source counts found. Cannot blend for this layer.")
                    nan_introduced_during_blending = True
                    totalSources = 0 # Avoid division by zero later
                else:
                    totalSources = sum(valid_counts)

                # Check if totalSources is zero or NaN
                if totalSources < EPSILON or math.isnan(totalSources):
                    print(f"WARNING blendActivations ({name}, Layer {layerNumber}): totalSources is near zero ({totalSources}) or NaN. Blending may produce NaN/Inf.")
                    nan_introduced_during_blending = True # Mark potential issue
                    totalSources = EPSILON if totalSources < EPSILON else totalSources


                for source, count in mostUsedSourcesPerLayer:
                    # Skip if count is invalid
                    if not (isinstance(count, (float, int, np.number)) and math.isfinite(count)):
                        continue

                    if RENN.activationsBySources[int(source)] is None:
                        print(f"WARNING blendActivations ({name}, Layer {layerNumber}): Source '{source}' not found in RENN.activationsBySources. Skipping.")
                        continue

                    activationsBySources = RENN.activationsBySources[source]
                    if layerNumber >= len(activationsBySources):
                        print(f"WARNING blendActivations ({name}, Layer {layerNumber}): layerNumber {layerNumber} invalid for source '{source}' activations (len: {len(activationsBySources)}). Skipping layer contribution.")
                        continue

                    neurons = np.asarray(activationsBySources[layerNumber], dtype=np.float64)

                    # Check neurons for NaN/Inf before blending
                    if check_activations_for_nan(neurons):
                        print(f"WARNING blendActivations ({name}, Layer {layerNumber}): NaN/Inf found in 'neurons' for source '{source}'. Skipping contribution.")
                        nan_introduced_during_blending = True
                        continue

                    # Perform blending if totalSources is valid
                    if not math.isnan(totalSources) and totalSources > EPSILON:
                        blendedActivations[layerIdx] += neurons * (count / totalSources)
                    # else: # Already warned about totalSources issue

    except Exception as e:
        print(f"ERROR blendActivations ({name}): Unexpected error during blending loop: {e}")
        traceback.print_exc()
        nan_introduced_during_blending = True # Mark as potential issue

    # Check final blendedActivations for NaN/Inf
    if check_activations_for_nan(blendedActivations):
        print(f"WARNING blendActivations ({name}): Final 'blendedActivations' contains NaN/Inf.")
        nan_introduced_during_blending = True # Confirm issue

    # Flatten and reshape for similarity computation
    try:
        eval_activations_subset = evaluationActivations[layerNumbersToCheck]
        # Check before flattening
        if check_activations_for_nan(eval_activations_subset):
            print(f"WARNING blendActivations ({name}): 'evaluationActivations' subset for layers {layerNumbersToCheck} contains NaN/Inf before flattening.")
            nan_introduced_during_blending = True

        eval_flat = eval_activations_subset.flatten().reshape(1, -1).astype(np.float64)
        blend_flat = blendedActivations.flatten().reshape(1, -1).astype(np.float64) # Already checked blendedActivations

        # Final check before computeSimilarity
        eval_flat_has_nan = check_activations_for_nan(eval_flat)
        blend_flat_has_nan = check_activations_for_nan(blend_flat)

        if eval_flat_has_nan:
            print(f"WARNING blendActivations ({name}): 'eval_flat' contains NaN/Inf right before computeSimilarity.")
        if blend_flat_has_nan:
            print(f"WARNING blendActivations ({name}): 'blend_flat' contains NaN/Inf right before computeSimilarity.")

    except Exception as e:
        print(f"ERROR blendActivations ({name}): Error during flattening/reshaping: {e}")
        traceback.print_exc()
        # Cannot compute similarity if flattening failed
        return {key: np.nan for key in ACTIVATION_METRIC_KEYS}


    # --- Compute Metrics ---
    # Proceed even if NaNs were detected, computeSimilarity should handle/warn
    try:
        cosine_sim, euclidean_dist, manhattan_dist, jaccard_sim, hamming_dist, pearson_corr = computeSimilarity(eval_flat, blend_flat)
    except Exception as cs_e:
        print(f"ERROR blendActivations ({name}): computeSimilarity failed: {cs_e}")
        traceback.print_exc()
        # Assign NaNs if computeSimilarity fails catastrophically
        cosine_sim, euclidean_dist, manhattan_dist, jaccard_sim, hamming_dist, pearson_corr = [np.nan] * 6


    # Compute rank correlations only if inputs are finite and have variance
    # Use the flags checked just before computeSimilarity
    if not (eval_flat_has_nan or blend_flat_has_nan or np.std(eval_flat) < EPSILON or np.std(blend_flat) < EPSILON):
        try:
            kendall_tau, _ = kendalltau(eval_flat.squeeze(), blend_flat.squeeze())
        except Exception as e:
            print(f"ERROR blendActivations ({name}): kendalltau failed: {e}")
            kendall_tau = np.nan
        try:
            spearman_rho, _ = spearmanr(eval_flat.squeeze(), blend_flat.squeeze())
        except Exception as e:
            print(f"ERROR blendActivations ({name}): spearmanr failed: {e}")
            spearman_rho = np.nan
    else:
        # Print info if skipping due to NaN/Inf or no variance
        if eval_flat_has_nan or blend_flat_has_nan:
            print(f"INFO blendActivations ({name}): Skipping rank correlations due to NaN/Inf in flattened vectors.")
        elif np.std(eval_flat) < EPSILON or np.std(blend_flat) < EPSILON:
            print(f"INFO blendActivations ({name}): Skipping rank correlations due to zero variance in flattened vectors.")
        kendall_tau = np.nan
        spearman_rho = np.nan

    # --- Store Results ---
    results = {
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist,
        "manhattan_distance": manhattan_dist,
        "jaccard_similarity": jaccard_sim if jaccard_sim is not None else np.nan, # Handle None from computeSimilarity
        "hamming_distance": hamming_dist,
        "pearson_correlation": pearson_corr if pearson_corr is not None else np.nan, # Handle None from computeSimilarity
    }

    # --- Store Globally (Optional) ---
    if store_globally:
        # This part needs access to these lists, ensure they are global or passed
        try:
            if name == "":
                original_activation_similarity.append(results)
            elif name == "Metrics":
                metrics_activation_similarity.append(results)
            elif name == "MT":
                mt_activation_similarity.append(results)
        except NameError as ne:
            print(f"ERROR blendActivations ({name}): Global list for storing results not found: {ne}")


    # --- Print Results (Optional) ---
    #print("\n--- Blended Activation Similarity Scores ---")
    #for metric, value in results.items():
    #    print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return results  # Return for immediate use

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, evaluation_name, analyze=False):
    """
    Performs analysis with checkpointing and prepares data for visualization.
    """
    # Use global variables as defined in the original function
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, original_image_similarity, metrics_image_similarity, mt_image_similarity, original_activation_similarity, metrics_activation_similarity, mt_activation_similarity, resultDataframe

    # --- Reset/Initialize global lists/dictionaries ---
    original_image_similarity, metrics_image_similarity, mt_image_similarity = [], [], []
    original_activation_similarity, metrics_activation_similarity, mt_activation_similarity = [], [], []
    # Make sure RENN, eval_dataloader, eval_samples, model are accessible here
    try:
        dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource = RENN.initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model)
    except NameError as e:
        print(f"Error: Required variable '{e.name}' not defined. Cannot initialize hooks.")
        return # Or handle appropriately

    # --- Configuration & Checkpointing Setup ---
    checkpoint_dir = "checkpoints" # Directory to store checkpoint files
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists

    # Generate a unique ID for this parameter set to avoid checkpoint conflicts
    # Include parameters that define the scope of the analysis loop
    param_string = f"{evaluation_name}-{hidden_sizes}-{closestSources}"
    # Add more parameters to the string if they affect the loop's iterations or results
    run_id = hashlib.md5(param_string.encode()).hexdigest()[:8] # Short hash

    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{run_id}_{evaluation_name}.csv")
    print(f"Using checkpoint file: {checkpoint_file}")

    # --- Load existing checkpoint data ---
    processed_combinations = set()
    results_list_from_checkpoint = [] # Store dataframes loaded from checkpoint

    if os.path.exists(checkpoint_file):
        print(f"Found existing checkpoint file: {checkpoint_file}. Resuming...")
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            # Assuming 'name' column uniquely identifies a completed iteration within this run
            if 'name' in checkpoint_df.columns:
                # Ensure we only load combinations for the *current* evaluation_name if the file might be shared (though run_id helps)
                # This check might be redundant if run_id is sufficiently unique
                if 'evaluation_name' in checkpoint_df.columns:
                    processed_combinations = set(checkpoint_df[checkpoint_df['evaluation_name'] == evaluation_name]['name'].unique())
                else:
                    # If evaluation_name column doesn't exist assume all names are for this run
                    processed_combinations = set(checkpoint_df['name'].unique())

                # Store the loaded dataframe
                # Filter again to be absolutely sure, if evaluation_name exists
                if 'evaluation_name' in checkpoint_df.columns:
                    results_list_from_checkpoint.append(checkpoint_df[checkpoint_df['evaluation_name'] == evaluation_name])
                else:
                    results_list_from_checkpoint.append(checkpoint_df)

                print(f"Loaded {len(processed_combinations)} previously completed combinations for '{evaluation_name}'.")
            else:
                print("Warning: Checkpoint file found but lacks 'name' column. Starting fresh for this run.")
                # Optionally delete the corrupt checkpoint file here
                # os.remove(checkpoint_file)
        except pd.errors.EmptyDataError:
            print("Checkpoint file is empty. Starting fresh for this run.")
        except Exception as e:
            print(f"Error reading checkpoint file: {e}. Starting fresh for this run.")
            # Optionally backup/delete the corrupt checkpoint file here
            # os.rename(checkpoint_file, checkpoint_file + ".error")

    # --- Analysis Section ---
    results_this_run = [] # Holds results generated only in the current execution

    if analyze:
        print(f"\n--- Starting Analysis for: {evaluation_name} ---")
        try:
            METRICS_COMBINATIONS = RENN.create_global_metric_combinations(2, 2, True)
        except NameError:
            print("Error: RENN object not defined. Cannot create metric combinations.")
            return # Or handle appropriately

        # Layer selection logic - ensure 'layers' is defined
        try:
            linearLayers = [idx * 2 for idx, (name, layerNumber, activation) in enumerate(layers)]
        except NameError:
            print("Error: 'layers' variable not defined. Cannot determine linear layers.")
            return # Or handle appropriately

        all_combinations_to_process = []
        # --- Pre-calculate all combination names ---
        for blendType in ["BTO", "BTL"]:
            for countType in ["-CTW", "-CTA"]:
                for distanceType in ["-DTE", "-DTA"]:
                    for normalizationType in ["-NTS", "-NTA","-NTZ", "-NTM"]:
                        mode = "Activation" if normalizationType == "-NTA" else "Sum"
                        name = blendType+countType+distanceType+normalizationType
                        all_combinations_to_process.append({'name': name, 'mode': mode})

        total_combinations = len(all_combinations_to_process)
        print(f"Total combinations to process: {total_combinations}")
        print(f"Already processed: {len(processed_combinations)}")

        processed_count_this_run = 0
        # --- Iterate through combinations ---
        for i, combo in enumerate(all_combinations_to_process):
            name = combo['name']
            mode = combo['mode']

            if name in processed_combinations:
                # print(f"Skipping already processed: {name}")
                continue # Skip this iteration

            print(f"Processing combination {i+1-len(processed_combinations)}/{total_combinations-len(processed_combinations)} (Overall {i+1}): {name} (Mode: {mode})")

            try:
                # --- Call the evaluation function ---
                single_result_df = evaluate_metric_combinations_overall(
                    evaluation_name=evaluation_name,
                    name=name,
                    linearLayers=linearLayers,
                    closestSources=closestSources,
                    all_metric_combinations=METRICS_COMBINATIONS,
                    mode=mode # Change to Activation for Overall Activation-Evaluation
                )

                # --- Check if result is valid (basic check) ---
                if single_result_df is None or single_result_df.empty:
                    print(f"Warning: Evaluation for {name} returned empty result. Skipping checkpoint write.")
                    continue

                # --- Append result to the checkpoint file ---
                # Write header only if file doesn't exist or is empty
                header = not os.path.exists(checkpoint_file) or os.path.getsize(checkpoint_file) == 0
                # Append the new result
                single_result_df.to_csv(checkpoint_file, mode='a', header=header, index=False)

                # --- Update state for this run ---
                processed_combinations.add(name) # Add to set to avoid re-processing if loop continues
                results_this_run.append(single_result_df) # Keep track of results from *this* run
                processed_count_this_run += 1
                # print(f"-> Successfully processed and checkpointed: {name}")


            except Exception as e:
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"ERROR during evaluation or checkpointing for: {name}")
                print(f"Evaluation Name: {evaluation_name}")
                print(f"Error details: {e}")
                print(f"Run interrupted. Rerun the script to resume from the last checkpoint.")
                print(f"Checkpoint file: {checkpoint_file}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                # Re-raise the exception to stop the script
                raise e

        print(f"\nFinished processing all combinations for {evaluation_name}.")
        print(f"Processed {processed_count_this_run} new combinations in this run.")

    # --- Combine results into the global resultDataframe ---
    # Combine results loaded from checkpoint (if any) and results from this run
    all_results_list = results_list_from_checkpoint + results_this_run

    if all_results_list:
        # Reload the definitive checkpoint file after the loop finishes for consistency
        try:
            print(f"Loading final results from checkpoint file: {checkpoint_file}")
            # Read the whole file again to ensure consistency
            final_df_from_checkpoint = pd.read_csv(checkpoint_file)
            # Filter specifically for the current evaluation run if the column exists
            if 'evaluation_name' in final_df_from_checkpoint.columns:
                resultDataframe = final_df_from_checkpoint[final_df_from_checkpoint['evaluation_name'] == evaluation_name].copy()
            else:
                resultDataframe = final_df_from_checkpoint.copy()

            print(f"Successfully loaded {len(resultDataframe)} total results for '{evaluation_name}'.")
        except Exception as e:
            print(f"Error loading final results from checkpoint file: {e}. Using in-memory results.")
            # Fallback to in-memory combined list if loading fails
            if all_results_list:
                # Need to filter the list again in case checkpoint contained other runs
                filtered_list = [df for df in all_results_list if evaluation_name in df['evaluation_name'].unique()]
                if filtered_list:
                    resultDataframe = pd.concat(filtered_list, ignore_index=True)
                    print(f"Concatenated {len(resultDataframe)} in-memory results for '{evaluation_name}'.")
                else:
                    print(f"No in-memory results found for '{evaluation_name}'. Creating empty DataFrame.")
                    resultDataframe = pd.DataFrame()
            else:
                resultDataframe = pd.DataFrame() # Create empty DF
    elif not analyze:
        print("Analysis was skipped (analyze=False). No results generated or loaded.")
        resultDataframe = pd.DataFrame() # Ensure it's empty
    else:
        print(f"Warning: No results were generated or loaded for '{evaluation_name}'. Creating empty DataFrame.")
        resultDataframe = pd.DataFrame() # Create an empty DF if no results

    # --- Generate Timestamped Filename for Final Output ---
    local_time_struct = time.localtime()
    formatted_time = time.strftime("%Y%m%d_%H%M%S", local_time_struct) # Format for filename
    # Construct filename with parameters
    try:
        # Make sure these variables are defined in the accessible scope
        filename_params = f"Results_{evaluation_name}_E{eval_samples}"
        output_csv_filename = f"{filename_params}-{formatted_time}.csv"
    except NameError as e:
        print(f"Warning: Could not create detailed filename due to missing variable ({e}), using default.")
        output_csv_filename = f"{evaluation_name}_metric_evaluation_results_{run_id}_{formatted_time}.csv" # Fallback includes run_id
    except IndexError:
        print(f"Warning: Could not access hidden_sizes[0][1] for filename, using default.")
        output_csv_filename = f"{evaluation_name}_metric_evaluation_results_{run_id}_{formatted_time}.csv" # Fallback includes run_id
    except Exception as e:
        print(f"Warning: An unexpected error occurred creating filename: {e}, using default.")
        output_csv_filename = f"{evaluation_name}_metric_evaluation_results_{run_id}_{formatted_time}.csv" # Fallback includes run_id


    # --- Export Full Results to CSV ---
    if not resultDataframe.empty:
        try:
            resultDataframe.to_csv(output_csv_filename, index=False)
            print(f"\nFull results for '{evaluation_name}' successfully exported to: {output_csv_filename}")

            # --- Clean up checkpoint file upon successful completion of THIS evaluation run ---
            # Only remove if the analysis actually ran and finished
            if analyze and os.path.exists(checkpoint_file):
                # Check if all combinations for *this* run are in the final dataframe
                # This is a safety check before removing the checkpoint
                if len(resultDataframe['name'].unique()) == total_combinations:
                    print(f"Analysis complete. Removing checkpoint file: {checkpoint_file}")
                    try:
                        os.remove(checkpoint_file)
                    except OSError as e:
                        print(f"Warning: Could not remove checkpoint file '{checkpoint_file}': {e}")
                else:
                    print(f"Warning: Final result count ({len(resultDataframe['name'].unique())}) doesn't match expected total ({total_combinations}). Checkpoint file '{checkpoint_file}' kept.")


        except Exception as e:
            print(f"\nWarning: Failed to export results to CSV file '{output_csv_filename}': {e}")
            print(f"Checkpoint file '{checkpoint_file}' has been kept for recovery.")
    else:
        print(f"\nResult DataFrame for '{evaluation_name}' is empty. No final CSV file generated.")
        # Decide whether to remove an empty/partial checkpoint file
        # if analyze and os.path.exists(checkpoint_file):
        #     print(f"Removing potentially empty/partial checkpoint file: {checkpoint_file}")
        #     os.remove(checkpoint_file)

# ==============================================================
# Helper Functions for Metric Calculations
# ==============================================================

def aggregate_source_layers(multi_layer_list):
    if not multi_layer_list:
        return []

    aggregated_scores = defaultdict(float)
    valid_input_structure = True # Flag to track if input looks like list of lists

    for i, layer_list in enumerate(multi_layer_list):
        # Check if the outer list actually contains lists/tuples
        if i == 0 and not isinstance(layer_list, (list, tuple)):
            valid_input_structure = False
            # print(f"DEBUG: Input to aggregate_source_layers doesn't seem nested. First item type: {type(layer_list)}")
            break # Stop processing if structure is wrong

        if not isinstance(layer_list, (list, tuple)):
            # print(f"Warning: Expected list/tuple for layer {i}, got {type(layer_list)}. Skipping layer.")
            continue

        try:
            for item in layer_list:
                if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)):
                    source_id = str(item[0]) # Standardize ID
                    score = item[1]
                    if np.isfinite(score): # Only add finite scores
                        aggregated_scores[source_id] += score
                # else: # Optional: print warning about invalid items *within* a layer
                # print(f"Warning: Invalid item format {repr(item)} in layer list {i}. Skipping item.")
        except Exception as e:
            print(f"Error processing layer list {i}: {e}. Skipping layer.")
            continue

    # If input wasn't nested list of lists, maybe it was already flat?
    # Return empty list in that case, as aggregation wasn't possible/meaningful.
    # Or handle differently if flat input should be processed directly? For now, return empty.
    if not valid_input_structure:
        print("Warning: aggregate_source_layers received non-nested input. Returning empty list.")
        return [] # Or raise error?

    # Convert back to list of tuples, sorted by score descending
    final_list = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
    return final_list

def calculate_source_cosine_similarity(list1, list2):
    """ Calculates standard cosine similarity between source count/score vectors. """
    vec1, vec2 = None, None
    norm1, norm2 = 0.0, 0.0
    try:
        # Ensure inputs are lists of tuples before dict conversion
        if not isinstance(list1, list) or not isinstance(list2, list): return 0.0
        counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
        all_source_ids_set = counts1.keys() | counts2.keys()
        if not all_source_ids_set: return 1.0 # Both empty
        all_source_ids = sorted(list(all_source_ids_set))
        vec1 = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float).reshape(1, -1)
        vec2 = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float).reshape(1, -1)
        norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    except (TypeError, ValueError) as e:
        # print(f"DEBUG: Cosine Sim TypeError/ValueError during vector creation: {e}")
        return 0.0 # Error during data prep
    except Exception as e:
        # print(f"DEBUG: Cosine Sim unexpected error during vector creation: {e}")
        return 0.0

    # Calculation part
    try:
        if not np.isfinite(norm1) or not np.isfinite(norm2): return 0.0 # Handle NaN/inf norms
        if norm1 == 0 and norm2 == 0: return 1.0
        if norm1 == 0 or norm2 == 0: return 0.0
        if cosine_similarity is None: raise ImportError("cosine_similarity not available")
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return np.clip(similarity if np.isfinite(similarity) else 0.0, 0.0, 1.0)
    except Exception as e:
        # print(f"DEBUG: Cosine Sim calculation error: {e}")
        return 0.0

def calculate_log_cosine_similarity(list1, list2):
    """ Calculates cosine similarity between log2(score+1) transformed source vectors. """
    vec1, vec2 = None, None
    norm1, norm2 = 0.0, 0.0
    try:
        if not isinstance(list1, list) or not isinstance(list2, list): return 0.0
        counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
        all_source_ids_set = counts1.keys() | counts2.keys()
        if not all_source_ids_set: return 1.0
        all_source_ids = sorted(list(all_source_ids_set))
        # Use np.log2 for potential vectorization and better NaN handling
        vec1_vals = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float)
        vec2_vals = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float)
        # Add 1 before log, handle potential negative counts if they exist
        vec1 = np.log2(np.maximum(vec1_vals, 0) + 1).reshape(1, -1)
        vec2 = np.log2(np.maximum(vec2_vals, 0) + 1).reshape(1, -1)
        norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    except (TypeError, ValueError) as e:
        # print(f"DEBUG: Log Cosine Sim TypeError/ValueError during vector creation: {e}")
        return 0.0
    except Exception as e:
        # print(f"DEBUG: Log Cosine Sim unexpected error during vector creation: {e}")
        return 0.0

    # Calculation part
    try:
        if not np.isfinite(norm1) or not np.isfinite(norm2): return 0.0
        if norm1 == 0 and norm2 == 0: return 1.0
        if norm1 == 0 or norm2 == 0: return 0.0
        if cosine_similarity is None: raise ImportError("cosine_similarity not available")
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return np.clip(similarity if np.isfinite(similarity) else 0.0, 0.0, 1.0)
    except Exception as e:
        # print(f"DEBUG: Log Cosine Sim calculation error: {e}")
        return 0.0

def calculate_jsd(list1, list2):
    """ Calculates Jensen-Shannon Distance (sqrt(JSD), base 2) between distributions. """
    p, q = None, None
    try:
        if not isinstance(list1, list) or not isinstance(list2, list): return 1.0 # Max distance
        counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
        all_source_ids_set = counts1.keys() | counts2.keys()
        if not all_source_ids_set: return 0.0 # Zero distance if both empty
        all_source_ids = sorted(list(all_source_ids_set))
        vec1 = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float)
        vec2 = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float)
        # Ensure non-negative values before summing
        vec1 = np.maximum(vec1, 0); vec2 = np.maximum(vec2, 0)
        sum1 = np.sum(vec1); sum2 = np.sum(vec2)
        if not np.isfinite(sum1) or not np.isfinite(sum2): return 1.0 # Max distance if sums are non-finite
        if sum1 == 0 and sum2 == 0: return 0.0
        if sum1 == 0 or sum2 == 0: return 1.0 # Max distance if one is empty
        p = vec1 / sum1; q = vec2 / sum2
        # Normalize again to ensure sum is exactly 1 due to potential float issues
        p /= np.sum(p); q /= np.sum(q)
    except (TypeError, ValueError) as e:
        # print(f"DEBUG: JSD TypeError/ValueError during vector creation: {e}")
        return 1.0
    except Exception as e:
        # print(f"DEBUG: JSD unexpected error during vector creation: {e}")
        return 1.0

    # Calculation part
    try:
        if distance is None: raise ImportError("scipy.spatial.distance not available")
        js_distance = distance.jensenshannon(p, q, base=2.0)
        # Ensure result is finite, otherwise return max distance
        return js_distance if np.isfinite(js_distance) else 1.0
    except Exception as e:
        # print(f"DEBUG: JSD calculation error: {e}")
        return 1.0 # Max distance on error

# --- MODIFIED calculate_rank_correlation ---
def calculate_rank_correlation(list1, list2):
    """ Calculates Spearman and Kendall rank correlation based on scores. Returns (0.0, 0.0) on error/insufficient data or if correlation is NaN. """
    spearman_corr, kendall_tau = np.nan, np.nan # Start with NaN
    try:
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Inputs must be lists")
        # Ensure items are tuples and scores are extractable and numeric
        dict1 = {item[0]: item[1] for item in list1 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and math.isfinite(item[1])}
        dict2 = {item[0]: item[1] for item in list2 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and math.isfinite(item[1])}
        if not dict1 or not dict2:
            # print("DEBUG RankCorr: One or both input dicts empty after filtering.") # Reduce noise
            return 0.0, 0.0 # Return 0 if no data

        common_ids = sorted(list(set(dict1.keys()) & set(dict2.keys()))) # Sort for consistency
        # --- MODIFICATION: Require >= 2 common points ---
        if len(common_ids) < 2:
            # print(f"DEBUG RankCorr: Insufficient common IDs ({len(common_ids)}). Returning 0.") # Reduce noise
            return 0.0, 0.0 # Return 0 if insufficient common points
        # --- END MODIFICATION ---

        # Let's use ranks as per original code intent:
        sorted_list1 = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
        sorted_list2 = sorted(dict2.items(), key=lambda x: x[1], reverse=True)
        rank_map1 = {id: rank + 1 for rank, (id, _) in enumerate(sorted_list1)}
        rank_map2 = {id: rank + 1 for rank, (id, _) in enumerate(sorted_list2)}
        ranks1 = [rank_map1[id] for id in common_ids]
        ranks2 = [rank_map2[id] for id in common_ids]

        if spearmanr is None or kendalltau is None: raise ImportError("spearmanr or kendalltau not available")

        # Calculate correlations on ranks
        try:
            temp_spearman, _ = spearmanr(ranks1, ranks2)
            # Assign only if calculation succeeds and is finite
            if np.isfinite(temp_spearman):
                spearman_corr = temp_spearman
            # else: # Optional: Warn if scipy returns non-finite
            # print(f"DEBUG RankCorr: Spearman returned non-finite: {temp_spearman}")
        except Exception as spearman_e:
            print(f"ERROR RankCorr: spearmanr failed: {spearman_e}")
            # Keep spearman_corr as np.nan

        try:
            temp_kendall, _ = kendalltau(ranks1, ranks2)
            # Assign only if calculation succeeds and is finite
            if np.isfinite(temp_kendall):
                kendall_tau = temp_kendall
            # else: # Optional: Warn if scipy returns non-finite
            # print(f"DEBUG RankCorr: Kendall returned non-finite: {temp_kendall}")
        except Exception as kendall_e:
            print(f"ERROR RankCorr: kendalltau failed: {kendall_e}")
            # Keep kendall_tau as np.nan


    except (IndexError, TypeError, ValueError) as e:
        print(f"ERROR Rank correlation (Input/Processing): {e}") # Keep error prints
        pass # Return default NaNs
    except ImportError as e:
        print(f"ERROR Rank correlation: {e}") # Keep error prints
        pass # Return default NaNs
    except Exception as e:
        print(f"ERROR Unexpected Rank correlation: {e}") # Keep error prints
        traceback.print_exc()
        pass # Return default NaNs

    # --- MODIFICATION: Return 0.0 if result is still NaN ---
    # Ensure tuple is always returned, substituting NaN with 0.0
    final_spearman = spearman_corr if np.isfinite(spearman_corr) else 0.0
    final_kendall = kendall_tau if np.isfinite(kendall_tau) else 0.0
    return final_spearman, final_kendall

def calculate_top_k_overlap(list1, list2, k):
    """ Calculates Intersection@k, Precision@k, Recall@k based on scores. Includes debugging. """
    intersection_count, precision_at_k, recall_at_k = 0, 0.0, 0.0 # Defaults
    try:
        if k <= 0:
            # print(f"DEBUG TopK: k={k} is non-positive, returning defaults.") # Reduce noise
            return 0.0, 0.0, 0.0 # Return floats
        if not isinstance(list1, list) or not isinstance(list2, list):
            # --- Added Debug for TopK Input Validation ---
            print(f"DEBUG TopK (k={k}): Invalid input types! list1: {type(list1)}, list2: {type(list2)}")
            raise ValueError("Inputs must be lists")

        # Ensure items are tuples and scores are sortable and finite
        valid_list1 = [item for item in list1 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and math.isfinite(item[1])]
        valid_list2 = [item for item in list2 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and math.isfinite(item[1])]

        # If either list becomes empty after validation, overlap is 0
        if not valid_list1 or not valid_list2:
            # print(f"DEBUG TopK: One or both lists empty after validation (len1={len(valid_list1)}, len2={len(valid_list2)}).") # Reduce noise
            return 0.0, 0.0, 0.0 # Return floats

        # Sort by score (second element) descending
        sorted_list1 = sorted(valid_list1, key=lambda x: x[1], reverse=True)
        sorted_list2 = sorted(valid_list2, key=lambda x: x[1], reverse=True)

        # Get top K IDs as strings
        top_k_ids1 = set(str(item[0]) for item in sorted_list1[:k])
        top_k_ids2 = set(str(item[0]) for item in sorted_list2[:k])

        intersection_set = top_k_ids1 & top_k_ids2
        intersection_count = len(intersection_set) # This should be an integer

        # Precision: Intersection / k
        precision_at_k = float(intersection_count) / k if k > 0 else 0.0

        # Recall: Intersection / number of items in the ground truth's top k
        # Assuming list1 is the ground truth
        num_relevant_in_top_k1 = len(top_k_ids1) # Number of unique items in list1's top k
        recall_at_k = float(intersection_count) / num_relevant_in_top_k1 if num_relevant_in_top_k1 > 0 else 0.0

        # --- >>> ADDED NaN Check before returning <<< ---
        if not np.isfinite(intersection_count) or not np.isfinite(precision_at_k) or not np.isfinite(recall_at_k):
            print(f"DEBUG TopK (k={k}): Returning non-finite value!")
            print(f"  Input list1 len (valid): {len(valid_list1)}")
            print(f"  Input list2 len (valid): {len(valid_list2)}")
            print(f"  top_k_ids1 (len={len(top_k_ids1)}): {list(top_k_ids1)[:5]}...")
            print(f"  top_k_ids2 (len={len(top_k_ids2)}): {list(top_k_ids2)[:5]}...")
            print(f"  intersection_count: {intersection_count}")
            print(f"  precision_at_k: {precision_at_k}")
            print(f"  recall_at_k: {recall_at_k}")
            # Force return NaN if any component is non-finite
            return np.nan, np.nan, np.nan


    except (IndexError, TypeError, ValueError) as e:
        print(f"ERROR TopK overlap (k={k}): {e}") # Print error if it occurs
        return np.nan, np.nan, np.nan # Return NaNs on error
    except Exception as e:
        print(f"ERROR Unexpected TopK overlap (k={k}): {e}") # Print error if it occurs
        traceback.print_exc()
        return np.nan, np.nan, np.nan # Return NaNs on error

    # Ensure float return types for all
    return float(intersection_count), float(precision_at_k), float(recall_at_k)


def calculate_vector_distances(list1, list2):
    """ Calculates Euclidean (L2) and Manhattan (L1) distances. Returns (NaN, NaN) on error. """
    euclidean_dist, manhattan_dist = np.nan, np.nan # Default return
    try:
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Inputs must be lists")
        counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
        all_source_ids_set = counts1.keys() | counts2.keys()
        if not all_source_ids_set: return 0.0, 0.0 # Zero distance if both empty
        # If only one is empty, result depends on interpretation. Let's return NaN.
        if not list1 or not list2: return np.nan, np.nan

        all_source_ids = sorted(list(all_source_ids_set))
        vec1 = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float)
        vec2 = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float)

        diff = vec1 - vec2
        temp_euclidean = np.linalg.norm(diff, ord=2)
        temp_manhattan = np.linalg.norm(diff, ord=1)

        # Assign only if finite
        if np.isfinite(temp_euclidean): euclidean_dist = temp_euclidean
        if np.isfinite(temp_manhattan): manhattan_dist = temp_manhattan
    except (TypeError, ValueError) as e:
        # print(f"DEBUG: Vector distance ValueError/TypeError: {e}")
        pass # Return default NaNs
    except Exception as e:
        # print(f"DEBUG: Unexpected Vector distance error: {e}")
        pass # Return default NaNs
    # Ensure tuple is always returned
    return euclidean_dist, manhattan_dist

def calculate_ruzicka_similarity(list1, list2):
    """ Calculates Ruzicka similarity (Generalized Jaccard for counts/scores). """
    similarity = np.nan # Default
    try:
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Inputs must be lists")
        counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
        all_source_ids = counts1.keys() | counts2.keys()
        if not all_source_ids: return 1.0 # Identical empty lists

        sum_min = 0.0; sum_max = 0.0
        for id_ in all_source_ids:
            # Ensure scores are treated as numbers, default to 0 if missing/invalid
            c1 = counts1.get(id_, 0); c2 = counts2.get(id_, 0)
            c1 = c1 if isinstance(c1, (int, float)) and np.isfinite(c1) else 0
            c2 = c2 if isinstance(c2, (int, float)) and np.isfinite(c2) else 0
            # Ensure non-negative for min/max interpretation
            c1 = max(c1, 0); c2 = max(c2, 0)
            sum_min += min(c1, c2)
            sum_max += max(c1, c2)

        # Avoid division by zero; if sum_max is 0, similarity is 1
        similarity = sum_min / sum_max if sum_max > 0 else 1.0
    except (TypeError, ValueError) as e:
        # print(f"DEBUG: Ruzicka ValueError/TypeError: {e}")
        pass # Return default NaN
    except Exception as e:
        # print(f"DEBUG: Unexpected Ruzicka error: {e}")
        pass # Return default NaN
    return similarity

def calculate_symmetric_difference_size(list1, list2):
    """ Calculates the number of items present in one list ID set but not the other. """
    diff_size = np.nan # Default
    try:
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Inputs must be lists")
        # Extract IDs robustly, handle non-tuple items, convert to string
        ids1 = set(str(item[0]) for item in list1 if isinstance(item, (tuple, list)) and len(item) > 0)
        ids2 = set(str(item[0]) for item in list2 if isinstance(item, (tuple, list)) and len(item) > 0)
        diff_size = float(len(ids1.symmetric_difference(ids2)))
    except (IndexError, TypeError, ValueError) as e:
        # print(f"DEBUG: SymmDiff ValueError/TypeError: {e}")
        pass # Return default NaN
    except Exception as e:
        # print(f"DEBUG: Unexpected SymmDiff error: {e}")
        pass # Return default NaN
    return diff_size

# ==============================================================
# Constants for Metric Keys
# ==============================================================
# Using names consistent with the worker function code provided by user
ACTIVATION_METRIC_KEYS = ["kendall_tau", "spearman_rho", "cosine_similarity", "euclidean_distance", "manhattan_distance", "jaccard_similarity", "hamming_distance", "pearson_correlation"]
IMAGE_SIM_KEY_MAP = {'Cosine Sim': 'cosine_sim', 'Euclidean Dst': 'euclidean_dst', 'Manhattan Dst': 'manhattan_dst', 'Jaccard Sim': 'jaccard_sim', 'Hamming Dst': 'hamming_dst', 'Pearson Corr': 'pearson_corr', 'Kendall Tau': 'kendall_tau', 'Spearman Rho': 'spearman_rho'}
IMG_SIM_PREFIX = "img_"
IMAGE_METRIC_KEYS_PREFIXED = [f"{IMG_SIM_PREFIX}{v}" for v in IMAGE_SIM_KEY_MAP.values()]
SOURCE_COSINE_METRIC = "source_cosine_similarity"
SOURCE_LOG_COSINE_METRIC = "source_log_cosine_similarity"
SOURCE_JSD_METRIC = "source_jsd"
SOURCE_SPEARMAN_METRIC = "source_spearman_rank_corr"
SOURCE_KENDALL_METRIC = "source_kendall_rank_tau"
SOURCE_INTERSECT_K_METRIC = "source_intersect_at_k"
SOURCE_PRECISION_K_METRIC = "source_precision_at_k"
SOURCE_RECALL_K_METRIC = "source_recall_at_k"
SOURCE_EUCLIDEAN_METRIC = "source_euclidean_dist"
SOURCE_MANHATTAN_METRIC = "source_manhattan_dist"
SOURCE_RUZICKA_METRIC = "source_ruzicka_similarity"
SOURCE_SYMM_DIFF_METRIC = "source_symmetric_diff_size"
SOURCE_METRIC_KEYS = [
    SOURCE_COSINE_METRIC, SOURCE_LOG_COSINE_METRIC, SOURCE_JSD_METRIC,
    SOURCE_SPEARMAN_METRIC, SOURCE_KENDALL_METRIC, SOURCE_INTERSECT_K_METRIC,
    SOURCE_PRECISION_K_METRIC, SOURCE_RECALL_K_METRIC, SOURCE_EUCLIDEAN_METRIC,
    SOURCE_MANHATTAN_METRIC, SOURCE_RUZICKA_METRIC, SOURCE_SYMM_DIFF_METRIC
]
OPTIMIZATION_METRIC = "source_intersect_at_k" # Default for user's logic if needed
BEST_CLOSEST_SOURCES = 'best_closest_sources' # Key for storing k value in user's logic
BEST_CLOSEST_SOURCES_FOR_ORIGINAL = 'best_closest_sources_for_original'
ALL_METRIC_KEYS_FOR_AGGREGATION = ACTIVATION_METRIC_KEYS + IMAGE_METRIC_KEYS_PREFIXED + SOURCE_METRIC_KEYS + [BEST_CLOSEST_SOURCES, BEST_CLOSEST_SOURCES_FOR_ORIGINAL]

# ==============================================================
# Worker Function (incorporating user's structure and debugging)
# ==============================================================
# --- Helper functions for NaN checks (Copied/adapted from previous context) ---
def check_list_for_nan_scores(source_list, list_name="Unnamed List"):
    """Checks a list of (id, score) tuples for NaN scores using math.isnan."""
    if not isinstance(source_list, list):
        return False
    nan_found = False
    for i, item in enumerate(source_list):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            score = item[1]
            if isinstance(score, (float, np.floating)) and math.isnan(score): # Check numpy floats too
                nan_found = True
                # print(f"DEBUG NaN Check: NaN found in {list_name} at index {i}, item: {item}") # Uncomment for deep debug
                break # Found one, no need to check further for this call
        # else: # Optional: Warn about malformed items if needed
        # print(f"DEBUG NaN Check: Item at index {i} in {list_name} is not a valid (id, score) tuple: {item}")
    return nan_found

def check_nested_list_for_nan_scores(nested_list):
    """Checks a potentially nested list for NaN scores using math.isnan."""
    if not isinstance(nested_list, collections.abc.Iterable) or isinstance(nested_list, (str, bytes)):
        return False
    for item in nested_list:
        if isinstance(item, tuple) and len(item) == 2:
            score = item[1]
            if isinstance(score, (float, np.floating)) and math.isnan(score): # Check numpy floats too
                return True
        elif isinstance(item, collections.abc.Iterable) and not isinstance(item, (str, bytes)):
            if check_nested_list_for_nan_scores(item):
                return True
    return False

def check_activations_for_nan(activations):
    """Checks activation data (assuming NumPy array or similar) for NaNs."""
    if activations is None: # Handle None case
        return False
    if isinstance(activations, np.ndarray):
        return np.isnan(activations).any()
    elif isinstance(activations, list):
        try:
            # Use a generator expression for potentially better memory efficiency
            def flatten(items):
                for x in items:
                    if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
                        yield from flatten(x)
                    else:
                        yield x
            # Check if any numeric item is NaN
            for item in flatten(activations):
                if isinstance(item, (float, np.floating)) and math.isnan(item): # Check numpy floats too
                    return True
            return False # No NaNs found
        except (TypeError, ValueError):
            return False # Cannot easily check non-numeric lists
    # Add checks for other types like tensors if necessary
    # elif torch.is_tensor(activations):
    #    return torch.isnan(activations).any()
    return False # Default if type is unknown or not checkable

# --- Main Worker Function ---
def process_sample_evaluation(args):
    """
    Processes evaluation for a single sample across metric combinations and k values.
    Includes targeted warnings printed ONLY if NaNs are detected in calculated results.
    """
    # --- Function arguments ---
    (pos, sample, originalMostUsedSources_input, evaluationActivations, # Renamed input arg
     metricsSampleActivations, linearLayers, all_metric_combinations, closestSources,
     mode, name, bestClosestSourcesForOriginal
     ) = args

    # --- Function results dict ---
    sample_results = {}

    # --- Initial Checks ---
    if metricsSampleActivations is None: return None
    
    # --- Prepare/Validate/Aggregate original sources ---
    originalSources_processed = None
    original_sources_contain_nan = False # Flag
    try:
        # print(f"DEBUG WORKER (S:{pos}): Raw originalMostUsedSources_input type={type(originalMostUsedSources_input)}, len={len(originalMostUsedSources_input) if hasattr(originalMostUsedSources_input, '__len__') else 'N/A'}")
        # if isinstance(originalMostUsedSources_input, list): print(f"DEBUG WORKER (S:{pos}): Raw originalMostUsedSources_input (first 5): {originalMostUsedSources_input[:5]}")
        is_original_nested = isinstance(originalMostUsedSources_input, list) and \
                             len(originalMostUsedSources_input) > 0 and \
                             all(isinstance(inner_item, (list, tuple)) for inner_item in originalMostUsedSources_input)
        if is_original_nested:
            # print(f"DEBUG WORKER (S:{pos}): Processing original sources via aggregate_source_layers (Nested Input)...")
            originalSources_processed = aggregate_source_layers(originalMostUsedSources_input)
            if "BTO" in name and "-CTW" in name:
                originalSources_processed = [(str(item[0]), item[1]) for item in originalMostUsedSources_input[0]]
                
        elif isinstance(originalMostUsedSources_input, list): # Input is likely already flat
            # print(f"DEBUG WORKER (S:{pos}): Processing original sources via flat list standardization...")
            originalSources_processed = [(str(item[0]), item[1]) for item in originalMostUsedSources_input if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and math.isfinite(item[1])]
        else:
            print(f"ERROR: Sample {pos} received unexpected type for originalMostUsedSources_input: {type(originalMostUsedSources_input)}. Setting processed to empty list.")
            originalSources_processed = []
        original_sources_contain_nan = check_list_for_nan_scores(originalSources_processed, f"S:{pos} Original Processed")
        if originalSources_processed is None: raise ValueError("Original sources became None")
        # print(f"DEBUG WORKER (S:{pos}): Processed originalSources_processed type={type(originalSources_processed)}, len={len(originalSources_processed) if hasattr(originalSources_processed, '__len__') else 'N/A'}")
        # if isinstance(originalSources_processed, list): print(f"DEBUG WORKER (S:{pos}): Processed originalSources_processed (first 5): {originalSources_processed[:5]}")
        # print(f"DEBUG WORKER (S:{pos}): original_sources_contain_nan flag: {original_sources_contain_nan}")
        if original_sources_contain_nan: print(f"WARNING: Sample {pos} - Processed original sources contain NaN scores (before loops).")
    except Exception as e:
        print(f"ERROR: Sample {pos} failed preparing original sources: {e}. Returning None.")
        traceback.print_exc()
        return None


    # --- Loop through metric combinations ---
    for metric_combination in all_metric_combinations:
        combination_str = str(metric_combination)
        currentBestValue = -np.inf
        current_combination_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION}

        # --- Loop through k values ---
        for currentClosestSources in range(max(1, closestSources-25), closestSources+25):
            current_k_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION}
            mostUsedMetricSources_from_renn = None
            mostUsedMetricSources_processed_flat = None
            mostUsedMetricSources_for_blend = None
            activation_sim_dict = None
            printed_nan_skip_warning_k = False
            printed_blend_skip_warning_k = False

            try:
                # --- RENN Calls ---
                metricSources, layerNumbersToCheck = RENN.identifyClosestSourcesByMetricCombination(
                    name, currentClosestSources, metricsSampleActivations, metric_combination, mode=mode
                )
                if not layerNumbersToCheck: continue
                mostUsedMetricSources_from_renn = RENN.getMostUsedSourcesByMetrics(
                    name, metricSources, currentClosestSources, weightedMode=mode
                )

                # --- Prepare/Validate/Aggregate metric sources ---
                if mostUsedMetricSources_from_renn is not None:
                    is_metric_nested = isinstance(mostUsedMetricSources_from_renn, list) and \
                                       len(mostUsedMetricSources_from_renn) > 0 and \
                                       all(isinstance(inner_item, (list, tuple)) for inner_item in mostUsedMetricSources_from_renn)
                    metric_processing_path = "Unknown"
                    try:
                        if "BTL" in name:
                            if is_metric_nested:
                                metric_processing_path = "BTL - Aggregate (Nested Input)"
                                mostUsedMetricSources_for_blend = mostUsedMetricSources_from_renn
                                mostUsedMetricSources_processed_flat = aggregate_source_layers(mostUsedMetricSources_from_renn)
                            else:
                                metric_processing_path = "BTL - Keep Raw (Flat Input)"
                                # print(f"WARNING (S:{pos}, K:{currentClosestSources}): BTL mode expected nested input from RENN, but received flat. Using flat list for all calculations.") # Reduce noise
                                mostUsedMetricSources_processed_flat = [(str(item[0]), item[1]) for item in mostUsedMetricSources_from_renn if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and np.isfinite(item[1])]
                                mostUsedMetricSources_for_blend = mostUsedMetricSources_processed_flat
                        else: # BTO or other modes
                            if is_metric_nested:
                                metric_processing_path = "BTO/Other - Aggregate (Nested Input)"
                                mostUsedMetricSources_processed_flat = aggregate_source_layers(mostUsedMetricSources_from_renn)
                                mostUsedMetricSources_for_blend = mostUsedMetricSources_processed_flat
                            else:
                                metric_processing_path = "BTO/Other - Validate/Standardize (Flat Input)"
                                mostUsedMetricSources_processed_flat = [(str(item[0]), item[1]) for item in mostUsedMetricSources_from_renn if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float, np.number)) and np.isfinite(item[1])]
                                mostUsedMetricSources_for_blend = mostUsedMetricSources_processed_flat
                        if mostUsedMetricSources_processed_flat is None: raise ValueError("Metric sources (flat) became None")
                        if mostUsedMetricSources_for_blend is None: raise ValueError("Metric sources (for blend) became None")
                        nan_in_flat_list = check_list_for_nan_scores(mostUsedMetricSources_processed_flat, f"S:{pos} K:{currentClosestSources} Metric Flat")
                        nan_in_blend_list = check_nested_list_for_nan_scores(mostUsedMetricSources_for_blend) if is_metric_nested and "BTL" in name else check_list_for_nan_scores(mostUsedMetricSources_for_blend, f"S:{pos} K:{currentClosestSources} Metric Blend")
                        if nan_in_flat_list: print(f"WARNING: Sample {pos}, k={currentClosestSources} - Processed metric sources (flat) contain NaN scores.")
                        if nan_in_blend_list: print(f"WARNING: Sample {pos}, k={currentClosestSources} - Processed metric sources (for blend) contain NaN scores.")
                    except Exception as e:
                        print(f"ERROR: Sample {pos}, k={currentClosestSources} failed preparing metric sources (Path: {metric_processing_path}): {e}. Skipping k.")
                        traceback.print_exc()
                        continue
                else: continue
                
                #print(originalSources_processed)
                #print(mostUsedMetricSources_for_blend)

                # --- Calculations use the appropriate PROCESSED lists ---
                valid_original_structure = isinstance(originalSources_processed, list) and (not originalSources_processed or isinstance(originalSources_processed[0], (tuple, list)))
                valid_metric_flat_structure = isinstance(mostUsedMetricSources_processed_flat, list) and (not mostUsedMetricSources_processed_flat or isinstance(mostUsedMetricSources_processed_flat[0], (tuple, list)))
                # if not valid_original_structure: print(f"DEBUG CALC CHECK (S:{pos}, K:{currentClosestSources}): originalSources_processed has invalid structure for calculations: type={type(originalSources_processed)}") # Reduce noise
                # if not valid_metric_flat_structure: print(f"DEBUG CALC CHECK (S:{pos}, K:{currentClosestSources}): mostUsedMetricSources_processed_flat has invalid structure for calculations: type={type(mostUsedMetricSources_processed_flat)}") # Reduce noise

                if mostUsedMetricSources_processed_flat is not None and originalSources_processed is not None and \
                        len(mostUsedMetricSources_processed_flat) > 0 and len(originalSources_processed) > 0 and \
                        valid_original_structure and valid_metric_flat_structure:

                    nan_in_metric_sources_processed = check_list_for_nan_scores(mostUsedMetricSources_processed_flat, f"S:{pos} K:{currentClosestSources} Metric Flat (Pre-Calc Check)")
                    nan_in_inputs = original_sources_contain_nan or nan_in_metric_sources_processed

                    # --- Calculate Source Similarities/Distances (using FLAT list) ---
                    source_cos_sim = np.nan if nan_in_inputs else calculate_source_cosine_similarity(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(source_cos_sim): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in {SOURCE_COSINE_METRIC}: {source_cos_sim}")
                    current_k_results[SOURCE_COSINE_METRIC] = source_cos_sim

                    source_log_cos_sim = np.nan if nan_in_inputs else calculate_log_cosine_similarity(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(source_log_cos_sim): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in {SOURCE_LOG_COSINE_METRIC}: {source_log_cos_sim}")
                    current_k_results[SOURCE_LOG_COSINE_METRIC] = source_log_cos_sim

                    source_jsd = np.nan if nan_in_inputs else calculate_jsd(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(source_jsd): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in {SOURCE_JSD_METRIC}: {source_jsd}")
                    current_k_results[SOURCE_JSD_METRIC] = source_jsd

                    # --- Call Rank Correlation ---
                    spearman, kendall = calculate_rank_correlation(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(spearman): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in spearman: {spearman}")
                    if not np.isfinite(kendall): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in kendall: {kendall}")
                    current_k_results[SOURCE_SPEARMAN_METRIC] = spearman
                    current_k_results[SOURCE_KENDALL_METRIC] = kendall

                    # --- Call Top K ---
                    intersect_k, precision_k, recall_k = calculate_top_k_overlap(originalSources_processed, mostUsedMetricSources_processed_flat, currentClosestSources)
                    current_k_results[SOURCE_INTERSECT_K_METRIC] = intersect_k
                    current_k_results[SOURCE_PRECISION_K_METRIC] = precision_k
                    current_k_results[SOURCE_RECALL_K_METRIC] = recall_k
                    if not np.isfinite(current_k_results.get(SOURCE_INTERSECT_K_METRIC, np.nan)): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): Assigned NaN/Inf for {SOURCE_INTERSECT_K_METRIC}")

                    # --- Call Vector Distances ---
                    euclidean, manhattan = calculate_vector_distances(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(euclidean): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in source_euclidean: {euclidean}")
                    if not np.isfinite(manhattan): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in source_manhattan: {manhattan}")
                    current_k_results[SOURCE_EUCLIDEAN_METRIC] = euclidean
                    current_k_results[SOURCE_MANHATTAN_METRIC] = manhattan

                    # --- Call Ruzicka ---
                    source_ruzicka = calculate_ruzicka_similarity(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(source_ruzicka): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in source_ruzicka: {source_ruzicka}")
                    current_k_results[SOURCE_RUZICKA_METRIC] = source_ruzicka

                    # --- Call Symmetric Difference ---
                    source_symm_diff = calculate_symmetric_difference_size(originalSources_processed, mostUsedMetricSources_processed_flat)
                    if not np.isfinite(source_symm_diff): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in source_symm_diff: {source_symm_diff}")
                    current_k_results[SOURCE_SYMM_DIFF_METRIC] = source_symm_diff

                    if nan_in_inputs and not printed_nan_skip_warning_k: printed_nan_skip_warning_k = True

                    current_k_results[BEST_CLOSEST_SOURCES] = currentClosestSources
                    current_k_results[BEST_CLOSEST_SOURCES_FOR_ORIGINAL] = bestClosestSourcesForOriginal

                    # --- Calculate Image Similarity (using FLAT list) ---
                    if evaluationActivations is not None:
                        try:
                            image_sim_dict = evaluateImageSimilarityByMetrics(name="Metrics", combination=combination_str, sample=sample, mostUsed=mostUsedMetricSources_processed_flat, storeGlobally=False)
                            if image_sim_dict:
                                for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items():
                                    if eval_key in image_sim_dict:
                                        val = image_sim_dict[eval_key]
                                        if not np.isfinite(val): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in image_sim {eval_key}: {val}")
                                        current_k_results[f"{IMG_SIM_PREFIX}{store_suffix}"] = val
                            else:
                                for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items(): current_k_results[f"{IMG_SIM_PREFIX}{store_suffix}"] = np.nan
                        except Exception as img_sim_e:
                            print(f"ERROR: S:{pos} K:{currentClosestSources} - evaluateImageSimilarityByMetrics call failed: {img_sim_e}")
                            for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items(): current_k_results[f"{IMG_SIM_PREFIX}{store_suffix}"] = np.nan

                    # --- Calculate Activation Similarity (using NESTED or FLAT list based on BTL/BTO) ---
                    if layerNumbersToCheck:
                        required_most_used_for_blend = mostUsedMetricSources_for_blend # Use the correct variable
                        eval_activations_contain_nan_this_k = check_activations_for_nan(evaluationActivations)
                        blend_sources_have_nan = check_nested_list_for_nan_scores(required_most_used_for_blend) if is_metric_nested and "BTL" in name else check_list_for_nan_scores(required_most_used_for_blend)

                        if eval_activations_contain_nan_this_k or blend_sources_have_nan:
                            if not printed_blend_skip_warning_k:
                                reason = "evalActivations" if eval_activations_contain_nan_this_k else "processed metric sources for blend"
                                # print(f"INFO: S:{pos} K:{currentClosestSources} - Skipping blendActivations due to NaN found in {reason}.") # Reduce noise
                                printed_blend_skip_warning_k = True
                            for act_key in ACTIVATION_METRIC_KEYS: current_k_results[act_key] = np.nan
                            activation_sim_dict = None
                        else:
                            try:
                                overall_eval_flag = "BTO" in name
                                activation_sim_dict = blendActivations(
                                    name=f"metric_combo_{combination_str}_k{currentClosestSources}",
                                    mostUsed=required_most_used_for_blend,
                                    evaluationActivations=evaluationActivations,
                                    layerNumbersToCheck=linearLayers,
                                    store_globally=False,
                                    overallEvaluation=overall_eval_flag
                                )
                                if activation_sim_dict:
                                    for act_key in ACTIVATION_METRIC_KEYS:
                                        if act_key in activation_sim_dict:
                                            val = activation_sim_dict[act_key]
                                            if not np.isfinite(val): print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): NaN/Inf detected in activation_sim {act_key}: {val}")
                                            current_k_results[act_key] = val
                                        else: current_k_results[act_key] = np.nan
                                else:
                                    # print(f"WARNING: S:{pos} K:{currentClosestSources} - blendActivations returned None.") # Reduce noise
                                    for act_key in ACTIVATION_METRIC_KEYS: current_k_results[act_key] = np.nan
                            except Exception as blend_e:
                                print(f"\n--- ERROR inside blendActivations call (S:{pos}, K:{currentClosestSources}) --- Error: {blend_e}")
                                # traceback.print_exc()
                                for act_key in ACTIVATION_METRIC_KEYS: current_k_results[act_key] = np.nan
                                activation_sim_dict = None
                else:
                    current_k_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION}

            # --- Catch block for errors OUTSIDE the blendActivations call ---
            except Exception as e:
                if not ('blend_e' in locals() and isinstance(e, blend_e.__class__)):
                    print(f"\n--- WORKER ERROR (PID {os.getpid()}) processing k={currentClosestSources}, combination {combination_str} for sample {pos} ---")
                    print(f"Error in main calculation try block (outside blendActivations): {e}")
                    # traceback.print_exc()
                    current_k_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION}
                    activation_sim_dict = None
                    print(f"--- WORKER (PID {os.getpid()}) continuing ---")
                    continue

            # --- User's Logic Block for Storing Results ---
            update_combination_results = False
            try:
                if current_k_results is not None and OPTIMIZATION_METRIC in current_k_results:
                    current_optimization_score = current_k_results.get(OPTIMIZATION_METRIC, np.nan)
                    if not np.isfinite(current_optimization_score):
                        print(f"DEBUG WORKER ({name} -> S:{pos}, K:{currentClosestSources}, C:{combination_str}): Optimization metric '{OPTIMIZATION_METRIC}' is non-finite: {current_optimization_score}")
                    if np.isfinite(current_optimization_score) and current_optimization_score > currentBestValue:
                        update_combination_results = True
                        currentBestValue = current_optimization_score
                # else: # Reduce noise
                # print(f"DEBUG WORKER (S:{pos}, K:{currentClosestSources}, C:{combination_str}): Optimization metric '{OPTIMIZATION_METRIC}' key missing in current_k_results.")
            except Exception as if_block_e:
                print(f"DEBUG: Error in final 'if' check k={currentClosestSources}, sample {pos}: {if_block_e}")
                pass

            if update_combination_results:
                current_combination_results = current_k_results.copy()
            # --- End of User's Logic Block ---

        # --- Storing results for the combination ---
        for metric_key, value in current_combination_results.items():
            if not np.isfinite(value) and not math.isnan(value): # Check for Inf only
                print(f"DEBUG WORKER (S:{pos}, C:{combination_str}): Storing non-finite value for {metric_key}: {value}")
            sample_results[(combination_str, metric_key)] = value

    return sample_results
# ==============================================================
# Main Evaluation Function (Multiprocessing Version - User Provided Structure)
# ==============================================================
def evaluate_metric_combinations_overall(evaluation_name, name, linearLayers, closestSources, all_metric_combinations, mode="Sum", max_workers=None):
    # --- Setup ---
    start_time = time.time()
    print(f"\nStarting evaluation with multiprocessing (max_workers={max_workers or os.cpu_count()})... for {evaluation_name} ({name})")
    print("Metrics: Activation Sim, Image Sim, Source Sims (Cos, LogCos, JSD, Rank, TopK, Dist, Ruzicka, SymmDiff)")
    print(f"Finding best original k via external call around center value: {closestSources}")

    # --- Input Validation ---
    if max_workers is None: max_workers = os.cpu_count(); print(f"Using default max_workers = {max_workers}")
    if not all_metric_combinations: print("Error: No metric combinations provided."); return None
    # Check for existence of necessary globals/functions
    required_globals = ['eval_dataloader', 'dictionaryForSourceLayerNeuron', 'metricsDictionaryForSourceLayerNeuron', 'findBestOriginalValues', 'process_sample_evaluation', 'ALL_METRIC_KEYS_FOR_AGGREGATION', 'BEST_CLOSEST_SOURCES_FOR_ORIGINAL', 'BEST_CLOSEST_SOURCES', 'ACTIVATION_METRIC_KEYS', 'IMAGE_SIM_KEY_MAP', 'IMG_SIM_PREFIX'] # Add others as needed
    for req_glob in required_globals:
        if req_glob not in globals() or globals()[req_glob] is None:
            print(f"Error: Required global variable/function '{req_glob}' is not defined or is None.")
            return None

    futures = []
    original_k_results_list = [] # List to store similarity results for the best original k for each sample

    # --- Process Pool ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_samples = 0
        # Submit tasks for each sample
        for pos, (sample, true) in enumerate(eval_dataloader):
            if processed_samples >= eval_samples: break

            bestClosestSourcesForOriginal = np.nan # Initialize
            originalMostUsedSources = None # Initialize

            # --- Get Sample-Specific Data & Find Best Original k ---
            try:
                # Get original activations
                evaluationActivations = dictionaryForSourceLayerNeuron[pos] # Original activations
                if evaluationActivations is None:
                    # print(f"Warning (S:{pos}): Original activations not found. Skipping sample.") # Reduce noise
                    continue

                # --- Call external function to find best original k ---
                original_k_search_result = findBestOriginalValues(name, sample, evaluationActivations, closestSources, mode)

                if original_k_search_result is None:
                    # print(f"Warning (S:{pos}): findBestOriginalValues returned None. Skipping sample.") # Reduce noise
                    continue

                # Extract results needed
                originalMostUsedSources = original_k_search_result.get('mostUsedSources')
                bestClosestSourcesForOriginal = original_k_search_result.get('closestSources')
                # Extract the NESTED dictionaries for storing similarities
                best_k_activation_results_dict = original_k_search_result.get('activation_similarity')
                best_k_image_results_dict = original_k_search_result.get('image_similarity')

                # Validate extracted results
                if originalMostUsedSources is None:
                    # print(f"Warning (S:{pos}): 'mostUsedSources' missing from findBestOriginalValues result. Skipping sample.") # Reduce noise
                    continue
                # Check bestClosestSourcesForOriginal validity
                if bestClosestSourcesForOriginal is None or (isinstance(bestClosestSourcesForOriginal, float) and math.isnan(bestClosestSourcesForOriginal)):
                    # print(f"Warning (S:{pos}): No valid 'closestSources' (best_k_original) found/returned by findBestOriginalValues. Skipping sample.") # Reduce noise
                    continue
                else:
                    # Ensure it's an integer if valid
                    try:
                        bestClosestSourcesForOriginal = int(bestClosestSourcesForOriginal)
                    except (ValueError, TypeError):
                        print(f"Warning (S:{pos}): Could not convert best_k_original '{bestClosestSourcesForOriginal}' to int. Skipping sample.")
                        continue


                # --- Store the original k similarity results for overall averaging later ---
                original_k_data = {'sample_pos': pos} # Include sample identifier if needed later
                # Extract from the nested activation_similarity dict
                if isinstance(best_k_activation_results_dict, dict):
                    for key in ACTIVATION_METRIC_KEYS: # Iterate through expected activation keys
                        original_k_data[f"orig_{key}"] = best_k_activation_results_dict.get(key, np.nan)
                else:
                    # print(f"Warning (S:{pos}): 'activation_similarity' missing or not a dict in findBestOriginalValues result.") # Reduce noise
                    for key in ACTIVATION_METRIC_KEYS: original_k_data[f"orig_{key}"] = np.nan

                # Extract from the nested image_similarity dict using the map
                if isinstance(best_k_image_results_dict, dict):
                    for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items():
                        target_key = f"orig_{IMG_SIM_PREFIX}{store_suffix}"
                        if eval_key in best_k_image_results_dict:
                            original_k_data[target_key] = best_k_image_results_dict[eval_key]
                        else: original_k_data[target_key] = np.nan
                else:
                    # print(f"Warning (S:{pos}): 'image_similarity' missing or not a dict in findBestOriginalValues result.") # Reduce noise
                    for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items():
                        original_k_data[f"orig_{IMG_SIM_PREFIX}{store_suffix}"] = np.nan

                original_k_results_list.append(original_k_data)


                # --- Prepare data for the worker (evaluating metric combinations) ---
                metricsSampleActivations = metricsDictionaryForSourceLayerNeuron[pos]
                if metricsSampleActivations is None:
                    # print(f"Warning (S:{pos}): metricsSampleActivations is None. Skipping task submission.") # Reduce noise
                    continue
                if not hasattr(metricsSampleActivations, 'shape'):
                    # print(f"Warning (S:{pos}): metricsSampleActivations is not array-like. Skipping task submission.") # Reduce noise
                    continue

                # Ensure sample data is tensor (placeholder logic)
                if hasattr(sample, 'float'): sample_data = sample.float()
                else: sample_data = sample

            except Exception as data_prep_e:
                print(f"Warning: Data preparation or findBestOriginalValues call failed for sample {pos}: {data_prep_e}. Skipping.")
                traceback.print_exc()
                continue

            # --- Arguments for worker ---
            args = (pos, sample_data, originalMostUsedSources, evaluationActivations,
                    metricsSampleActivations, linearLayers, all_metric_combinations,
                    closestSources, # Pass the original center k
                    mode, name, bestClosestSourcesForOriginal # Pass the found best k
                    )
            futures.append(executor.submit(process_sample_evaluation, args))
            processed_samples += 1

        print(f"Submitted {len(futures)} tasks to workers.")
        # --- Retrieve Results ---
        results_list = [] # Re-initialize here
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result is not None: results_list.append(result)
            except Exception as e:
                print(f"\n--- ERROR retrieving result for task {i+1}: {e} ---")
                # traceback.print_exc()

    # --- Aggregation of Worker Results (Metric Combinations) ---
    print(f"\nCollected results from {len(results_list)} completed worker tasks.")
    if not results_list: print("Error: No results collected from workers."); return None

    print("Aggregating worker results...")
    results_aggregator = defaultdict(lambda: defaultdict(list))

    # --- >>> DEBUGGING AGGREGATION <<< ---
    try:
        valid_keys_for_agg = set(ALL_METRIC_KEYS_FOR_AGGREGATION)
        print(f"DEBUG AGG: Expected keys (set, first 10): {list(valid_keys_for_agg)[:10]}")
    except NameError:
        print("ERROR: ALL_METRIC_KEYS_FOR_AGGREGATION is not defined globally!")
        return None
    except Exception as e:
        print(f"ERROR: Problem with ALL_METRIC_KEYS_FOR_AGGREGATION: {e}")
        return None

    printed_sample_result_keys = False # Flag to print only once
    keys_found_count = 0
    scores_appended_count = 0 # <<< Initialize new counter
    keys_not_found = set() # Track keys from worker not in expected set

    for sample_result_dict in results_list:
        if not isinstance(sample_result_dict, dict):
            print(f"Warning: Worker returned non-dict result: {type(sample_result_dict)}")
            continue

        if not printed_sample_result_keys and sample_result_dict:
            print(f"DEBUG AGG: First non-empty sample_result_dict keys (first 10): {list(sample_result_dict.keys())[:10]}")
            # Extract and print the metric_key part from the first key tuple
            first_key_tuple = next(iter(sample_result_dict), None)
            if isinstance(first_key_tuple, tuple) and len(first_key_tuple) == 2:
                print(f"DEBUG AGG: Example metric_key from worker: '{first_key_tuple[1]}' (Type: {type(first_key_tuple[1])})")
            else:
                print(f"DEBUG AGG: First key is not a (combo_str, metric_key) tuple: {first_key_tuple}")
            printed_sample_result_keys = True

        for key_tuple, score in sample_result_dict.items():
            # Check if the key is the expected tuple format
            if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                combination_str, metric_key = key_tuple
                # Check if the extracted metric_key is in the valid set
                if metric_key in valid_keys_for_agg:
                    keys_found_count += 1 # Increment if a match is found
                    # --- Original Aggregation Logic ---
                    appended = False # Flag to check if append happened
                    if metric_key == BEST_CLOSEST_SOURCES_FOR_ORIGINAL:
                        if score is not None and not (isinstance(score, float) and math.isnan(score)):
                            try:
                                results_aggregator[combination_str][metric_key].append(int(score))
                                appended = True
                            except (ValueError, TypeError): print(f"Warning: Could not convert score '{score}' for key '{metric_key}' to int.")
                    elif metric_key == BEST_CLOSEST_SOURCES:
                        if score is not None and not (isinstance(score, float) and math.isnan(score)):
                            try:
                                results_aggregator[combination_str][metric_key].append(int(score))
                                appended = True
                            except (ValueError, TypeError): print(f"Warning: Could not convert score '{score}' for key '{metric_key}' to int.")
                    elif score is not None and np.isfinite(score):
                        results_aggregator[combination_str][metric_key].append(score)
                        appended = True
                    # --- End Original Aggregation Logic ---
                    if appended:
                        scores_appended_count += 1 # <<< Increment counter if append occurred
                else:
                    # Track keys from worker that are not expected
                    if metric_key not in keys_not_found:
                        keys_not_found.add(metric_key)
            else:
                print(f"Warning: Worker returned result with unexpected key format: {key_tuple}")

    print(f"DEBUG AGG: Total valid key occurrences aggregated: {keys_found_count}")
    print(f"DEBUG AGG: Total scores actually appended: {scores_appended_count}") # <<< Print the new counter
    if keys_not_found:
        print(f"DEBUG AGG: Keys received from worker but NOT in ALL_METRIC_KEYS_FOR_AGGREGATION: {keys_not_found}")
    # --- >>> END DEBUGGING AGGREGATION <<< ---

    # --- >>> ADDED DEBUG CHECK for results_aggregator <<< ---
    print(f"DEBUG AGG: Size of results_aggregator after aggregation loop: {len(results_aggregator)}")
    if results_aggregator:
        print(f"DEBUG AGG: Example keys in results_aggregator (first 5): {list(results_aggregator.keys())[:5]}")
    # --- >>> END ADDED DEBUG CHECK <<< ---

    # --- Calculate Averages for Worker Results ---
    final_results = {}
    for combination_str, metric_scores_dict in results_aggregator.items(): # This loop should now execute if keys_found_count > 0
        avg_scores = {}
        for metric_key in ALL_METRIC_KEYS_FOR_AGGREGATION: # Iterate through expected keys
            scores_list = metric_scores_dict.get(metric_key, []) # Get scores if key was found
            if scores_list:
                # Use mean for integer k values, nanmean for others
                if metric_key in [BEST_CLOSEST_SOURCES, BEST_CLOSEST_SOURCES_FOR_ORIGINAL]:
                    avg_scores[f"avg_{metric_key}"] = np.mean(scores_list) # Average k value
                else:
                    avg_scores[f"avg_{metric_key}"] = np.nanmean(scores_list) # nanmean ignores NaNs
            else: avg_scores[f"avg_{metric_key}"] = np.nan # No valid scores recorded for this key
        final_results[combination_str] = avg_scores

    if not final_results:
        print("Error: Aggregation resulted in empty dictionary (final_results). Check aggregation logic and worker return values.")
        # Add extra debug info here if this still happens
        print(f"DEBUG AGG: results_aggregator was size {len(results_aggregator)} before averaging loop.")
        return None

    # --- Create DataFrame from Aggregated Worker Results ---
    try:
        results_df = pd.DataFrame.from_dict(final_results, orient='index')
        print(f"Aggregated Worker Results DataFrame shape: {results_df.shape}")
        if results_df.empty: print("Error: Worker results DataFrame is empty after aggregation."); return None
    except Exception as e: print(f"Error creating worker results DataFrame: {e}"); return None

    # --- Calculate Overall Averages for Original Best K Similarities ---
    print("\nCalculating overall averages for best original k similarities...")
    overall_original_k_avg_results = {}
    if original_k_results_list:
        original_k_df = pd.DataFrame(original_k_results_list)
        # Identify columns containing original similarity results (start with 'orig_')
        original_sim_cols = [col for col in original_k_df.columns if col.startswith('orig_')]
        if original_sim_cols:
            # Calculate mean for each original similarity column, ignoring NaNs
            overall_original_k_avg_results = original_k_df[original_sim_cols].mean(axis=0, skipna=True).to_dict()
            print(f"Calculated averages for {len(overall_original_k_avg_results)} original similarity metrics.")
            # print(overall_original_k_avg_results) # Optional: print averages
        else:
            print("Warning: No original similarity columns found in original_k_results_list (Columns checked: start with 'orig_').")
    else:
        print("Warning: No results collected for original k similarities.")

    # --- Add Overall Original Averages to the Main DataFrame ---
    if overall_original_k_avg_results:
        print("Adding overall original similarity averages to the main DataFrame...")
        for col_name, avg_value in overall_original_k_avg_results.items():
            # Add the average value as a new column, repeating for all rows
            # The col_name already includes 'orig_', add 'overall_' prefix
            results_df[f"overall_{col_name}"] = avg_value
        print(f"Added {len(overall_original_k_avg_results)} new columns.")


    # --- Column Ordering (User Provided Logic - Needs Update for new columns) ---
    prefix = f"avg_" # Prefix for worker aggregated results
    overall_prefix = f"overall_orig_" # Prefix for the new overall original averages

    # Define base ordered lists (without prefixes)
    # Activation keys expected in the 'activation_similarity' dict
    act_sim_ordered_base = ACTIVATION_METRIC_KEYS
    # Image sim keys derived from the IMAGE_SIM_KEY_MAP values (suffixes)
    img_sim_ordered_base = list(IMAGE_SIM_KEY_MAP.values())

    source_sim_ordered_base = [
        SOURCE_COSINE_METRIC, SOURCE_LOG_COSINE_METRIC, SOURCE_RUZICKA_METRIC,
        SOURCE_SPEARMAN_METRIC, SOURCE_KENDALL_METRIC,
        SOURCE_PRECISION_K_METRIC, SOURCE_RECALL_K_METRIC, SOURCE_INTERSECT_K_METRIC
    ]
    source_dist_ordered_base = [
        SOURCE_JSD_METRIC, SOURCE_EUCLIDEAN_METRIC, SOURCE_MANHATTAN_METRIC,
        SOURCE_SYMM_DIFF_METRIC, BEST_CLOSEST_SOURCES, BEST_CLOSEST_SOURCES_FOR_ORIGINAL
    ]

    ordered_columns = []
    # Add overall original averages first (if they exist)
    # Sort activation sim and image sim separately if desired
    overall_act_cols = sorted([col for col in results_df.columns if col.startswith(f"{overall_prefix}") and not col.startswith(f"{overall_prefix}{IMG_SIM_PREFIX}")])
    overall_img_cols = sorted([col for col in results_df.columns if col.startswith(f"{overall_prefix}{IMG_SIM_PREFIX}")])
    ordered_columns.extend(overall_act_cols)
    ordered_columns.extend(overall_img_cols)


    # Add worker aggregated averages
    # Use the base lists defined above
    ordered_columns.extend([f'{prefix}{m}' for m in act_sim_ordered_base if f'{prefix}{m}' in results_df.columns])
    # Note: act_dist_ordered_base was missing, added based on common metrics
    act_dist_ordered_base = ['euclidean_distance', 'manhattan_distance', 'hamming_distance']
    ordered_columns.extend([f'{prefix}{m}' for m in act_dist_ordered_base if f'{prefix}{m}' in results_df.columns])
    ordered_columns.extend([f'{prefix}{IMG_SIM_PREFIX}{m}' for m in img_sim_ordered_base if f'{prefix}{IMG_SIM_PREFIX}{m}' in results_df.columns])
    # Note: img_dist_ordered_base was missing, added based on common metrics
    img_dist_ordered_base = ['euclidean_dst', 'manhattan_dst', 'hamming_dst']
    ordered_columns.extend([f'{prefix}{IMG_SIM_PREFIX}{m}' for m in img_dist_ordered_base if f'{prefix}{IMG_SIM_PREFIX}{m}' in results_df.columns])
    ordered_columns.extend([f'{prefix}{m}' for m in source_sim_ordered_base if f'{prefix}{m}' in results_df.columns])
    ordered_columns.extend([f'{prefix}{m}' for m in source_dist_ordered_base if f'{prefix}{m}' in results_df.columns])

    # Find remaining columns and add them sorted
    results_df_cols = set(results_df.columns)
    ordered_cols_set = set(ordered_columns)
    remaining_cols = sorted(list(results_df_cols - ordered_cols_set))
    ordered_columns.extend(remaining_cols)

    # Filter list to only include columns actually present in DataFrame (redundant check but safe)
    final_ordered_columns = [col for col in ordered_columns if col in results_df_cols]
    # Ensure no duplicates if column names overlap somehow
    final_ordered_columns = list(dict.fromkeys(final_ordered_columns))


    try:
        results_df = results_df[final_ordered_columns]
        print("Reordered DataFrame columns.")
    except KeyError as e:
        print(f"Warning: Column reordering failed due to missing key: {e}. Using default order.")
        # print(f"Available columns: {results_df.columns.tolist()}") # Debugging line
        # print(f"Attempted order: {final_ordered_columns}") # Debugging line
    except Exception as e:
        print(f"Warning: Column reordering failed: {e}. Using default order.")


    # --- Row Sorting Priority (User Provided Logic - Needs Update for new columns) ---
    # Define priority list including potential new overall columns
    metrics_priority_list = [
        # Add overall original metrics if desired for sorting
        f'{overall_prefix}cosine_similarity',
        f'{overall_prefix}{IMG_SIM_PREFIX}cosine_sim', # Adjust key based on IMAGE_SIM_KEY_MAP
        # Original priority list with 'avg_' prefix
        f'{prefix}{SOURCE_INTERSECT_K_METRIC}',
        f'{prefix}cosine_similarity', # Activation Cosine
        f'{prefix}{IMG_SIM_PREFIX}cosine_sim',
        f'{prefix}{SOURCE_COSINE_METRIC}',
        # ... (rest of the original priority list) ...
        f'{prefix}{SOURCE_KENDALL_METRIC}',
        f'{prefix}kendall_tau', # Activation kendall
    ]
    # (Keep similarity_keywords and distance_keywords definitions)
    similarity_keywords = ['similarity', 'correlation', 'tau', 'rho', 'spearman', 'kendall', 'precision', 'recall', 'intersect', 'ruzicka', 'cosine']
    distance_keywords = ['distance', 'dst', 'jsd', 'euclidean', 'manhattan', 'diff', 'hamming']


    sort_by_columns = []; sort_ascending_flags = []; sort_descriptions = []
    for metric_col in metrics_priority_list:
        if metric_col not in results_df.columns: continue # Skip if column doesn't exist

        ascending_order = False # Default: Higher is better
        sort_type = " (Desc)"
        # Determine metric name without prefix for keyword checking
        metric_name_lower = metric_col.replace(prefix, '').replace(overall_prefix, '').lower()

        # Check if it's explicitly a distance metric where lower is better
        is_distance = any(keyword in metric_name_lower for keyword in distance_keywords)
        # Check source distance keys specifically (without prefix)
        is_explicit_source_dist = any(dist_key in metric_col for dist_key in [
            SOURCE_JSD_METRIC, SOURCE_EUCLIDEAN_METRIC, SOURCE_MANHATTAN_METRIC, SOURCE_SYMM_DIFF_METRIC
        ])

        if is_explicit_source_dist or (is_distance and not any(sim_key in metric_name_lower for sim_key in similarity_keywords)):
            ascending_order = True; sort_type = " (Asc)"
        # No need for elif, default is Descending/False

        sort_by_columns.append(metric_col); sort_ascending_flags.append(ascending_order)
        sort_descriptions.append(f"{metric_col.replace(prefix, '').replace(overall_prefix, '')}{sort_type}")

    # --- Perform Final Row Sort ---
    final_df_to_return = results_df # Default if sorting fails
    if not sort_by_columns:
        print("\nWarning: No valid columns found for sorting rows. Returning unsorted DataFrame.")
    else:
        #print(f"\n--- Final Ranking Sorted By Rows: {' -> '.join(sort_descriptions)} ---")
        try:
            final_df_to_return = results_df.sort_values(by=sort_by_columns, ascending=sort_ascending_flags, na_position='last')
        except Exception as e:
            print(f"\nError during final row sorting: {e}. Returning unsorted DataFrame.")
            final_df_to_return = results_df # Return unsorted on error

    end_time = time.time()
    print(f"\nEvaluation for {evaluation_name} ({name}) complete in {end_time - start_time:.2f} seconds.")
    print(f"Returning final DataFrame ({final_df_to_return.shape}).")
    return final_df_to_return

best_image_similarity = []
def evaluateImageSimilarityByMetrics(name, combination, sample, mostUsed, storeGlobally=True):
    global best_image_similarity # Declare intent to modify the global list

    sample_flat = np.asarray(sample.flatten().reshape(1, -1))

    blended_image = blendIndividualImagesTogether(mostUsed, len(mostUsed), layer=True)
    # Compute similarity for the blended image with sample
    blended_image_flat = np.asarray(blended_image.convert('L')).flatten() / 255.0
    blended_image_flat = blended_image_flat.reshape(1, -1)

    # Compute standard similarity/distance metrics using the placeholder/actual function
    cosine_similarity, euclidean_distance, manhattan_distance, jaccard_similarity, hamming_distance, pearson_correlation = computeSimilarity(sample_flat, blended_image_flat)

    # Compute rank correlation coefficients (require 1D arrays)
    sample_1d = sample_flat.squeeze()
    blended_1d = blended_image_flat.squeeze()

    # Handle potential NaN inputs or zero variance for correlation coefficients
    if not (np.isnan(sample_1d).any() or np.isnan(blended_1d).any() or np.std(sample_1d) < EPSILON or np.std(blended_1d) < EPSILON):
        if np.isnan(sample_1d).any() or np.isnan(blended_1d).any():
            print(f"Warning ({name} - {combination}): NaN values found in vectors, skipping rank correlations.")
        elif np.std(sample_1d) == 0 or np.std(blended_1d) == 0:
            print(f"Warning ({name} - {combination}): Zero variance in vectors, skipping rank correlations.")
        elif len(sample_1d) > 1: # Need more than 1 element for correlation
            try:
                kendall_tau, _ = kendalltau(sample_1d, blended_1d)
            except Exception as e:
                print(f"Error calculating Kendall's Tau for {name} - {combination}: {e}")
            try:
                spearman_rho, _ = spearmanr(sample_1d, blended_1d)
            except Exception as e:
                print(f"Error calculating Spearman's Rho for {name} - {combination}: {e}")
        else:
            print(f"Warning ({name} - {combination}): Input vector length <= 1, skipping rank correlations.")
    else:
        kendall_tau, spearman_rho = np.nan, np.nan

    # --- Store Results ---
    # Append the combination identifier along with all metrics to the list
    result_tuple = (
        combination,           # Identifier for the tested parameters/method
        cosine_similarity,
        euclidean_distance,
        manhattan_distance,
        jaccard_similarity,    # May be None or require specific data types
        hamming_distance,      # May be None or require specific data types
        pearson_correlation,   # May be None if calculation fails
        kendall_tau,           # May be None if calculation fails
        spearman_rho           # May be None if calculation fails
    )
    if storeGlobally:
        best_image_similarity.append(result_tuple)

    results_dict = {
        # Use the exact keys expected by the objective function
        'Cosine Sim': cosine_similarity,
        'Euclidean Dst': euclidean_distance,
        'Manhattan Dst': manhattan_distance,
        'Pearson Corr': pearson_correlation,
        'Kendall Tau': kendall_tau,
        'Spearman Rho': spearman_rho,
        # Include others if needed, but ensure the main 6 are present
        'Jaccard Sim': jaccard_similarity, # Example key name
        'Hamming Dst': hamming_distance    # Example key name
    }

    return results_dict

def findBestOriginalValues(name, sample, original_values, closestSources, mode):
    bestMostUsedSources = {'name': name, 'activation_similarity': {'cosine_similarity': -1}, 'image_similarity': [], 'mostUsedSources': [], 'closestSources': None} # Initialize with -1

    # Define the range based on the input 'closestSources' parameter
    search_range = range(max(1, closestSources - 25), closestSources + 25) # Ensure range starts at 1 minimum

    for currentClosestSourcesValue in search_range:
        try:
            # Ensure RENN methods handle potential errors if currentClosestSourcesValue is invalid
            identifiedClosestSources, outputsToCheck, layerNumbersToCheck = RENN.identifyClosestSourcesForOriginal(original_values, currentClosestSourcesValue, mode)
            mostUsedSources = RENN.getMostUsedSourcesByName(name, identifiedClosestSources, currentClosestSourcesValue, mode)

            overallEvaluation = "BTO" in name # Simpler check
            activationResults = blendActivations(name, mostUsedSources, original_values, layerNumbersToCheck, overallEvaluation=overallEvaluation)
            flattenedMostUsedSources = aggregate_source_layers(mostUsedSources)
            if "BTO" in name:
                imageResults = evaluateImageSimilarity(name, sample, mostUsedSources[0])
            if "BTL" in name:
                imageResults = evaluateImageSimilarity(name, sample, flattenedMostUsedSources)

            # Check if results is valid and contains 'cosine_similarity'
            if activationResults and 'cosine_similarity' in activationResults and activationResults['cosine_similarity'] > bestMostUsedSources['activation_similarity']['cosine_similarity']:
                bestMostUsedSources = {'name': name, 'activation_similarity': activationResults, 'image_similarity': imageResults, 'mostUsedSources': mostUsedSources, 'closestSources': currentClosestSourcesValue}
        except Exception as e:
            # Optional: Log errors if RENN or blendActivations fail for certain values
            print(f"Error during processing for closestSources={currentClosestSourcesValue}: {e}")
            continue # Skip to the next value in the range

    # --- Modification Here ---
    # Store the best result *from this specific run*
    if bestMostUsedSources['closestSources'] is not None: # Only append if a valid result was found
        print(f"Cosine-Similarity: {bestMostUsedSources['activation_similarity']['cosine_similarity']} with {bestMostUsedSources['closestSources']} closestSources")
    else:
        print("Error: No closestSources found!!!")
    # --- End Modification ---

    return bestMostUsedSources # Return the best result for this run