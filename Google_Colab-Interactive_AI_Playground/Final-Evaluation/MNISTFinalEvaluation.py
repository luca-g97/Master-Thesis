import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
import numpy as np
import os
import RENNFinalEvaluation as RENN
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import spearmanr, kendalltau, pearsonr
from IPython.display import clear_output, display
from collections import defaultdict, OrderedDict
import concurrent.futures
import time
import math
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import plotly.graph_objects as go
import plotly.subplots as sp

paretoEvaluation, weightTuning = False, False
if paretoEvaluation or weightTuning:
    RENN.useOnlyBestMetrics = False

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
        #print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataloader)}, Validation Loss: {val_loss/len(test_dataloader)}')

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
                    image_numpy += x_train[wSource[0]].numpy()*255
                else:
                    image_numpy += x_train[wSource.source].numpy()*255
            else:
                if(layer):
                    image_numpy += (x_train[wSource[0]].numpy()*255 * wSource[1] / total)
                else:
                    #print(f"Diff: {wSource.difference}, Total: {total}, Calculation: {(1 - (wSource.difference / total)) / closestSources}")
                    image_numpy += (x_train[wSource.source].numpy()*255 * (1 - (wSource.difference / total)) / closestSources)

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

# Global storage (optional)
original_activation_similarity, metrics_activation_similarity, mt_activation_similarity = [], [], []
def blendActivations(name, mostUsed, evaluationActivations, layerNumbersToCheck, store_globally=False, overallEvaluation=True):
    blendedActivations = np.zeros_like(evaluationActivations[layerNumbersToCheck])
    
    if overallEvaluation:
        if "BTO" in name:
            mostUsed = mostUsed[0]
        totalSources = sum(count for (_, count) in mostUsed)
        for (source, count) in mostUsed:
            activationsBySources = RENN.activationsBySources[source]
            for layerIdx, layerNumber in enumerate(layerNumbersToCheck):
                neurons = activationsBySources[layerNumber]
                blendedActivations[layerIdx] += neurons * (count / totalSources)
    else:
        for layerIdx, mostUsedSourcesPerLayer in enumerate(mostUsed):
            totalSources = sum(count for _, count in mostUsedSourcesPerLayer)
            layerNumber = layerNumbersToCheck[layerIdx]
            for source, count in mostUsedSourcesPerLayer:
                activationsBySources = RENN.activationsBySources[source]
                neurons = activationsBySources[layerNumber]
                blendedActivations[layerIdx] += neurons * (count / totalSources)
    
    # Flatten and reshape for similarity computation
    eval_flat = evaluationActivations[layerNumbersToCheck].flatten().reshape(1, -1).astype(np.float64)
    blend_flat = blendedActivations.flatten().reshape(1, -1).astype(np.float64)

    # --- Compute Metrics ---
    cosine_sim, euclidean_dist, manhattan_dist, jaccard_sim, hamming_dist, pearson_corr = computeSimilarity(eval_flat, blend_flat)

    if not (np.isnan(eval_flat).any() or np.isnan(blend_flat).any() or np.std(eval_flat) < EPSILON or np.std(blend_flat) < EPSILON):
        kendall_tau, _ = kendalltau(eval_flat.squeeze(), blend_flat.squeeze())
        spearman_rho, _ = spearmanr(eval_flat.squeeze(), blend_flat.squeeze())
    else:
        kendall_tau = np.nan
        spearman_rho = np.nan

    # --- Store Results ---
    results = {
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist,
        "manhattan_distance": manhattan_dist,
        "jaccard_similarity": jaccard_sim if jaccard_sim is not None else np.nan,
        "hamming_distance": hamming_dist,
        "pearson_correlation": pearson_corr if pearson_corr is not None else np.nan,
    }

    if store_globally:
        if name == "":
            original_activation_similarity.append(results)
        elif name == "Metrics":
            metrics_activation_similarity.append(results)
        elif name == "MT":
            mt_activation_similarity.append(results)

    # --- Print Results ---
    #print("\n--- Blended Activation Similarity Scores ---")
    #for metric, value in results.items():
    #    print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return results  # Return for immediate use

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, analyze=False):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, original_image_similarity, metrics_image_similarity, mt_image_similarity, original_activation_similarity, metrics_activation_similarity, mt_activation_similarity, resultDataframe

    original_image_similarity, metrics_image_similarity, mt_image_similarity, original_activation_similarity, metrics_activation_similarity, mt_activation_similarity = [], [], [], [], [], []

    #Make sure to set new dictionaries for the hooks to fill - they are global!
    dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource = RENN.initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model)

    if analyze:
        METRICS_COMBINATIONS = RENN.create_global_metric_combinations(3, 3, True)
        #metricsDictionaryForSourceLayerNeuron = createRandomDictionary(metricsDictionaryForSourceLayerNeuron, RENN.metricsActivationsBySources) #Random values per metrics min and max
        #metricsDictionaryForSourceLayerNeuron = np.full(metricsDictionaryForSourceLayerNeuron.shape, 0.5) #Fixed Vector

    mostUsedListSum, mostUsedListActivations = [], []
    mostUsedMetricsList = []

    for pos, (sample, true) in enumerate(eval_dataloader):
        sample = sample.float()
        prediction = predict(sample)
        mostUsedSourcesWithSum = ""
        layersToCheck = []

        if(visualizationChoice == "Weighted"):
            sourcesSum, metricSourcesSum, mtSourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum, mostUsedMetricSourcesWithSum, mostUsedMTSourcesWithSum, mostUsedSourcesPerLayerWithSum = RENN.getMostUsedSources(sourcesSum, metricSourcesSum, mtSourcesSum, closestSources, "Sum")

            #20 because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageSum = blendImagesTogether(mostUsedSourcesWithSum, "Not Weighted")
            blendedMetricSourceImageSum = blendImagesTogether(mostUsedMetricSourcesWithSum, "Not Weighted")
            blendedMTSourceImageSum = blendImagesTogether(mostUsedMTSourcesWithSum, "Not Weighted")
            layersToCheck = layerNumbersToCheck # Switch to another variable to use correct layers for analyzation


            #sourcesActivation, metricSourcesActivation, mtSourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation") #For Overall Sum-Evaluation
            sourcesActivation, metricSourcesActivation, mtSourcesActivation, outputsActivation, layersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation") #Uncomment for Overall Activation-Evaluation
            mostUsedSourcesWithActivation, mostUsedMetricSourcesWithActivation, mostUsedMTSourcesWithActivation, mostUsedSourcesPerLayerWithActivation = RENN.getMostUsedSources(sourcesActivation, metricSourcesActivation, mtSourcesActivation, closestSources, "Activation")
            #20 sources only because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation, "Not Weighted")
            blendedMetricSourceImageActivation = blendImagesTogether(mostUsedMetricSourcesWithActivation, "Not Weighted")
            blendedMTSourceImageActivation = blendImagesTogether(mostUsedMTSourcesWithActivation, "Not Weighted")

            #showImagesUnweighted("Per Neuron", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
            #showImagesUnweighted("Metrics", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedMetricSourceImageActivation, blendedMetricSourceImageSum, mostUsedMetricSourcesWithActivation[:showClosestMostUsedSources], mostUsedMetricSourcesWithSum[:showClosestMostUsedSources])
            #showImagesUnweighted("Magnitude Truncation", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedMTSourceImageActivation, blendedMTSourceImageSum, mostUsedMTSourcesWithActivation[:showClosestMostUsedSources], mostUsedMTSourcesWithSum[:showClosestMostUsedSources])
        else:
            sourcesSum, metricSourcesSum, mtSourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum = getClosestSourcesPerNeuronAndLayer(sourcesSum, metricSourcesSum, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Sum")

            sourcesActivation, metricSourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = getClosestSourcesPerNeuronAndLayer(sourcesActivation, metricSourcesActivation, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Activation")
            #RENN.analyzeData(closestSources, dictionaryForSourceLayerNeuron[pos])

        if(analyze):
            #mostUsed, mostUsedMetrics, mostUsedMT = #RENN.getMostUsedSources(sourcesSum, metricSourcesSum, mtSourcesSum, closestSources)
            #Always use the linear layer values for reference within the evaluation
            mostUsedListSum.append(mostUsedSourcesPerLayerWithSum)
            mostUsedListActivations.append(mostUsedSourcesPerLayerWithActivation)
            blendActivations("", mostUsedSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layerNumbersToCheck, True)
            evaluateImageSimilarity("", sample, mostUsedSourcesWithSum)
            if metricsEvaluation:
                blendActivations("Metrics", mostUsedMetricSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layerNumbersToCheck, True)
                evaluateImageSimilarity("Metrics", sample, mostUsedMetricSourcesWithSum)
            if RENN.mtEvaluation:
                blendActivations("MT", mostUsedMTSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layerNumbersToCheck, True)
                evaluateImageSimilarity("MT", sample, mostUsedMTSourcesWithSum)

    results = []
    if analyze:
        for blendType in ["BTO", "BTL"]:
            for countType in ["-CTW", "-CTA"]:
                for distanceType in ["-DTA", "-DTE"]:
                    for normalizationType in ["-NT", "-NTZ", "-NTM"]:
                        for modeType in ["S", "A"]:
                            mode = "Activation" if modeType == "A" else "Sum"
                            name = blendType+countType+distanceType+normalizationType+modeType
                            mostUsedList = mostUsedListActivations if modeType == "A" else mostUsedListSum

                            results.append(evaluate_metric_combinations_overall(
                                name,
                                mostUsedList,
                                linearLayers=layerNumbersToCheck,
                                hidden_sizes=hidden_sizes,
                                closestSources=closestSources,
                                all_metric_combinations=METRICS_COMBINATIONS,
                                mode=mode #Change to Activation for Overall Activation-Evaluation
                            ))

    # --- Combine results into a single DataFrame ---
    if results:
        resultDataframe = pd.concat(results, ignore_index=True)
        print(f"Successfully concatenated {len(results)} results into DataFrame.")
    else:
        print("Warning: No results were generated during the analysis. Creating empty DataFrame.")
        resultDataframe = pd.DataFrame() # Create an empty DF if no results

    # --- Generate Timestamped Filename ---
    local_time_struct = time.localtime()
    formatted_time = time.strftime("%Y%m%d_%H%M%S", local_time_struct) # Format for filename
    # Example: Construct filename with parameters (ensure variables like train_samples exist)
    try:
        filename_params = f"Results_S{RENN.seed}_E{eval_samples}_T{train_samples}_L{hidden_sizes[0][1]}"
        output_csv_filename = f"{filename_params}-{formatted_time}.csv"
    except NameError:
        print("Warning: Could not create detailed filename, using default.")
        output_csv_filename = f"{formatted_time}_metric_evaluation_results.csv" # Fallback


    # --- Export Full Results to CSV ---
    try:
        resultDataframe.to_csv(output_csv_filename)
        print(f"\nFull results successfully exported to: {output_csv_filename}")
    except Exception as e:
        print(f"\nWarning: Failed to export results to CSV file '{output_csv_filename}': {e}")
        
        

# ----------------------------------------------------------------------------
# Helper Functions for Source List Similarities / Distances
# ----------------------------------------------------------------------------

def createRandomDictionary(original, train_array):
    # 1. Get properties from the original array
    original_array = original
    original_shape = original_array.shape
    original_dtype = original_array.dtype

    # 2. Handle empty array case
    if original_array.size == 0:
        # Create an empty array with the same shape and dtype
        new_random_array_local_range = np.empty(original_shape, dtype=original_dtype)
        print("Warning: Original array was empty. Created an empty array with the same shape.")
    # Handle 0D or 1D array case (reverts to global min/max behavior)
    elif original_array.ndim <= 1:
        print("Warning: Array is 0D or 1D. Using global min/max for range.")
        min_val = np.min(train_array)
        max_val = np.max(train_array)
        rng = np.random.default_rng()
        if np.issubdtype(original_dtype, np.integer):
            # Integers [min_val, max_val]
            if min_val >= max_val: # Handle single value or invalid range after potential floor/ceil
                new_random_array_local_range = np.full(original_shape, np.floor(min_val).astype(original_dtype), dtype=original_dtype)
            else:
                int_min = np.floor(min_val).astype(original_dtype)
                int_max = np.ceil(max_val).astype(original_dtype)
                new_random_array_local_range = rng.integers(int_min, int_max + 1, size=original_shape, dtype=original_dtype)
        else:
            # Floats [min_val, max_val)
            if np.isclose(min_val, max_val):
                new_random_array_local_range = np.full(original_shape, min_val, dtype=original_dtype)
            else:
                random_base = rng.random(size=original_shape)
                scaled_values = min_val + random_base * (max_val - min_val)
                new_random_array_local_range = scaled_values.astype(original_dtype)

    # --- Main logic for 2D or higher arrays ---
    else:
        # 3. Calculate min and max ALONG THE LAST AXIS
        # keepdims=True makes broadcasting work easily later
        min_per_slice = np.min(original_array, axis=-1, keepdims=True)
        max_per_slice = np.max(original_array, axis=-1, keepdims=True)
        # Shape of min/max_per_slice will be (d1, d2, ..., dn-1, 1)

        # 4. Initialize the random number generator
        rng = np.random.default_rng() # Initialize once

        # 5. Generate random values based on dtype and LOCAL range

        if np.issubdtype(original_dtype, np.integer):
            # --- Integer Type ---
            # Ensure min/max slices are integer type for calculations
            int_min_slice = np.floor(min_per_slice).astype(original_dtype)
            int_max_slice = np.ceil(max_per_slice).astype(original_dtype)

            # Calculate integer range per slice [min, max] -> range = max - min
            range_per_slice_int = int_max_slice - int_min_slice

            # Mask where min == max (or min > max after floor/ceil)
            mask_min_eq_max = (range_per_slice_int <= 0)

            # Initialize result array
            new_random_array_local_range = np.empty(original_shape, dtype=original_dtype)

            # Fill constant parts where min >= max for the slice
            # Use np.broadcast_to to fill correctly based on slice min value
            new_random_array_local_range[mask_min_eq_max] = np.broadcast_to(int_min_slice, original_shape)[mask_min_eq_max]

            # Generate random values for slices where min < max
            idx_varied = ~mask_min_eq_max
            if np.any(idx_varied):
                # Generate base floats [0, 1) for the entire shape (simpler than indexing)
                random_base = rng.random(size=original_shape)

                # Scale to [min, max + 1) to cover the integer range after flooring
                # Broadcasting automatically applies the correct slice's min/range
                scaled_floats = int_min_slice + random_base * (range_per_slice_int + 1)

                # Floor and cast to get integers in the range [min, max]
                generated_values = np.floor(scaled_floats).astype(original_dtype)

                # Apply only where the range was > 0
                new_random_array_local_range[idx_varied] = generated_values[idx_varied]

        else:
            # --- Float Type (or other non-integer) ---
            # Calculate float range per slice
            range_per_slice = max_per_slice - min_per_slice

            # Mask where min is close to max for the slice (use tolerance for floats)
            # Use np.broadcast_to to compare shape correctly if needed, but direct comparison should work
            mask_min_eq_max = np.isclose(range_per_slice, 0)

            # Generate base random numbers [0, 1)
            random_base = rng.random(size=original_shape)

            # Scale using the corresponding slice's min and range
            # Broadcasting applies the correct min/range from (d1,...,dn-1,1) to (d1,...,dn)
            # Where range is ~0, multiplication results in ~0, effectively adding min_per_slice
            scaled_random_values = min_per_slice + random_base * range_per_slice

            # Cast to original dtype
            new_random_array_local_range = scaled_random_values.astype(original_dtype)

            # Ensure min==max case is exactly min_val (using np.where for precision)
            # Broadcast min_per_slice to match the shape for np.where
            broadcasted_min = np.broadcast_to(min_per_slice, original_shape)
            broadcasted_mask = np.broadcast_to(mask_min_eq_max, original_shape)
            new_random_array_local_range = np.where(broadcasted_mask, broadcasted_min, new_random_array_local_range)
            # Recast just in case 'where' changed dtype (unlikely but safe)
            new_random_array_local_range = new_random_array_local_range.astype(original_dtype)

    return new_random_array_local_range

# ==============================================================
# Helper Functions for Metric Calculations
# ==============================================================

def aggregate_source_layers(multi_layer_list):
    """
    Aggregates a list of layer results (List[List[Tuple[id, score]]])
    into a single list of (id, aggregated_score), summing scores.
    Handles invalid items within layers.
    """
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

def calculate_rank_correlation(list1, list2):
    """ Calculates Spearman and Kendall rank correlation based on scores. Returns (NaN, NaN) on error/insufficient data. """
    spearman_corr, kendall_tau = np.nan, np.nan # Default return
    try:
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Inputs must be lists")
        # Ensure items are tuples and scores are extractable and numeric
        dict1 = {item[0]: item[1] for item in list1 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)) and np.isfinite(item[1])}
        dict2 = {item[0]: item[1] for item in list2 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)) and np.isfinite(item[1])}
        if not dict1 or not dict2: return np.nan, np.nan # Need data in both

        common_ids = sorted(list(set(dict1.keys()) & set(dict2.keys()))) # Sort for consistency
        if len(common_ids) < 2: return np.nan, np.nan # Need at least 2 common points

        # Get scores for common IDs
        scores1 = [dict1[id] for id in common_ids]
        scores2 = [dict2[id] for id in common_ids]

        # Check for zero variance in scores (can cause issues)
        if np.std(scores1) < EPSILON or np.std(scores2) < EPSILON:
            # If one list is constant and the other isn't, corr is NaN (or 0).
            # If both are constant, corr is undefined (NaN) but often treated as 1 or 0. Let's return 0.
            return 0.0, 0.0 # Or return np.nan, np.nan ? Let's be conservative: 0.0

        if spearmanr is None or kendalltau is None: raise ImportError("spearmanr or kendalltau not available")

        # Calculate directly on scores or ranks? Original did ranks. Let's stick to ranks.
        # Need ranks based on full lists to handle ties correctly
        sorted_list1 = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
        sorted_list2 = sorted(dict2.items(), key=lambda x: x[1], reverse=True)
        # Assign ranks, average for ties (scipy does this internally, simpler to use scores if rank method isn't specified)
        # Reverting to correlation on scores for simplicity unless ranks are strictly needed
        # temp_spearman, _ = spearmanr(scores1, scores2)
        # temp_kendall, _ = kendalltau(scores1, scores2)

        # Let's use ranks as per original code intent:
        rank_map1 = {id: rank + 1 for rank, (id, _) in enumerate(sorted_list1)}
        rank_map2 = {id: rank + 1 for rank, (id, _) in enumerate(sorted_list2)}
        ranks1 = [rank_map1[id] for id in common_ids]
        ranks2 = [rank_map2[id] for id in common_ids]
        if np.std(ranks1) < EPSILON or np.std(ranks2) < EPSILON: return 0.0, 0.0 # Corr=0 if ranks are constant

        temp_spearman, _ = spearmanr(ranks1, ranks2)
        temp_kendall, _ = kendalltau(ranks1, ranks2)

        # Assign only if calculation succeeds and is finite
        if np.isfinite(temp_spearman): spearman_corr = temp_spearman
        if np.isfinite(temp_kendall): kendall_tau = temp_kendall

    except (IndexError, TypeError, ValueError) as e:
        # print(f"DEBUG: Rank correlation ValueError/TypeError: {e}")
        pass # Return default NaNs
    except Exception as e:
        # print(f"DEBUG: Unexpected Rank correlation error: {e}")
        pass # Return default NaNs
    # Ensure tuple is always returned
    return spearman_corr, kendall_tau

def calculate_top_k_overlap(list1, list2, k):
    """ Calculates Intersection@k, Precision@k, Recall@k based on scores. """
    intersection_count, precision_at_k, recall_at_k = 0, 0.0, 0.0 # Defaults
    try:
        if k <= 0: return 0, 0.0, 0.0
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("Inputs must be lists")

        # Ensure items are tuples and scores are sortable
        valid_list1 = [item for item in list1 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)) and np.isfinite(item[1])]
        valid_list2 = [item for item in list2 if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)) and np.isfinite(item[1])]
        # If either list becomes empty after validation, overlap is 0
        if not valid_list1 or not valid_list2: return 0, 0.0, 0.0

        # Sort by score (second element)
        sorted_list1 = sorted(valid_list1, key=lambda x: x[1], reverse=True)
        sorted_list2 = sorted(valid_list2, key=lambda x: x[1], reverse=True)

        top_k_ids1 = set(str(item[0]) for item in sorted_list1[:k]) # Use str IDs
        top_k_ids2 = set(str(item[0]) for item in sorted_list2[:k]) # Use str IDs

        intersection_set = top_k_ids1 & top_k_ids2
        intersection_count = len(intersection_set)
        precision_at_k = intersection_count / k if k > 0 else 0.0
        num_relevant_in_top_k1 = len(top_k_ids1)
        recall_at_k = intersection_count / num_relevant_in_top_k1 if num_relevant_in_top_k1 > 0 else 0.0

    except (IndexError, TypeError, ValueError) as e:
        # print(f"DEBUG: TopK overlap error: {e}")
        pass # Return defaults 0, 0.0, 0.0
    except Exception as e:
        # print(f"DEBUG: Unexpected TopK overlap error: {e}")
        pass # Return defaults 0, 0.0, 0.0
    return intersection_count, precision_at_k, recall_at_k


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
OPTIMIZATION_METRIC = SOURCE_COSINE_METRIC # Default for user's logic if needed
BEST_CLOSEST_SOURCES = 'best_closest_sources' # Key for storing k value in user's logic
BEST_CLOSEST_SOURCES_FOR_ORIGINAL = 'best_closest_sources_for_original'
ALL_METRIC_KEYS_FOR_AGGREGATION = ACTIVATION_METRIC_KEYS + IMAGE_METRIC_KEYS_PREFIXED + SOURCE_METRIC_KEYS + [BEST_CLOSEST_SOURCES, BEST_CLOSEST_SOURCES_FOR_ORIGINAL]

# ==============================================================
# Worker Function (incorporating user's structure and debugging)
# ==============================================================
def process_sample_evaluation(args):
    # --- Function arguments ---
    (pos, sample, originalMostUsedSources_input, evaluationActivations, # Renamed input arg
     metricsSampleActivations, linearLayers, all_metric_combinations, closestSources,
     mode, name, bestClosestSourcesForOriginal
     ) = args

    # --- Function results dict ---
    sample_results = {}

    # --- Initial Checks ---
    if metricsSampleActivations is None:
        print(f"Warning: Sample {pos} has no metricsSampleActivations. Skipping.")
        return None # Need metric scores

    # --- Prepare/Validate/Aggregate original sources ---
    originalSources_processed = None
    try:
        # Check if the input looks nested (list containing lists/tuples)
        is_original_nested = isinstance(originalMostUsedSources_input, (list, tuple)) and \
                             len(originalMostUsedSources_input) > 0 and \
                             isinstance(originalMostUsedSources_input[0], (list, tuple))

        if is_original_nested and "BTL" in name: # Aggregate only if nested AND BTL
            # print(f"DEBUG: S:{pos} - Aggregating original sources (Nested BTL).") # Optional debug
            originalSources_processed = aggregate_source_layers(originalMostUsedSources_input)
        elif not is_original_nested: # Assume it's already flat [(id, score)], just validate/standardize
            # print(f"DEBUG: S:{pos} - Original sources seem flat, validating.") # Optional debug
            originalSources_processed = [(str(item[0]), item[1]) for item in originalMostUsedSources_input if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)) and np.isfinite(item[1])]
            if len(originalSources_processed) != len(originalMostUsedSources_input):
                print(f"Warning: Sample {pos} original sources had invalid flat items.")
        else:
            #print(f"Warning: Sample {pos} has nested original sources but not BTL. Using raw input structure.") # Or aggregate?
            originalSources_processed = originalMostUsedSources_input # Keep original structure if not BTL?

        # Ensure it's not None after processing
        if originalSources_processed is None:
            raise ValueError("Original sources became None after processing.")

    except Exception as e:
        print(f"ERROR: Sample {pos} failed preparing original sources: {e}. Returning None.")
        return None

    # original_source_ids = set(src_id for src_id, count in originalSources_processed if isinstance(originalSources_processed[0], (tuple,list))) # Adjusted check
    # ^ This check might fail if originalSources_processed is nested and not BTL

    # --- Loop through metric combinations ---
    for metric_combination in all_metric_combinations:
        combination_str = str(metric_combination)
        # --- Variables specific to the user's requested logic ---
        currentBestCosineSimilarity = -np.inf # Initialize tracker for the 'best' score found *so far*
        current_combination_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION} # Results for this combination

        # --- Loop through k values ---
        for currentClosestSources in range(max(1, closestSources-20), closestSources+21): # Ensure k>=1
            # --- Initialize for this k ---
            current_k_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION} # Results for THIS k
            mostUsedMetricSources_from_renn = None # Output from RENN
            mostUsedMetricSources_processed = None # Processed/Aggregated version
            activation_sim_dict = None # Output from blendActivations

            try:
                # --- RENN Calls ---
                metricSources, layerNumbersToCheck = RENN.identifyClosestSourcesByMetricCombination(
                    name, currentClosestSources, metricsSampleActivations, metric_combination, mode=mode
                )

                if not layerNumbersToCheck:
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - No layers to check. Skipping k.") # Optional Debug
                    continue # Skip this k

                mostUsedMetricSources_from_renn = RENN.getMostUsedSourcesByMetrics(
                    name, metricSources, currentClosestSources, weightedMode=mode
                )

                # --- Prepare/Validate/Aggregate metric sources ---
                if mostUsedMetricSources_from_renn is not None:
                    # Check if the RENN output looks nested
                    is_metric_nested = isinstance(mostUsedMetricSources_from_renn, (list, tuple)) and \
                                       len(mostUsedMetricSources_from_renn) > 0 and \
                                       isinstance(mostUsedMetricSources_from_renn[0], (list, tuple))
                    try:
                        if is_metric_nested and "BTL" in name: # Aggregate only if nested AND BTL
                            # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Aggregating metric sources (Nested BTL).") # Optional debug
                            mostUsedMetricSources_processed = aggregate_source_layers(mostUsedMetricSources_from_renn)
                        elif not is_metric_nested: # Assume flat, just validate/standardize
                            # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Metric sources seem flat, validating.") # Optional debug
                            mostUsedMetricSources_processed = [(str(item[0]), item[1]) for item in mostUsedMetricSources_from_renn if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)) and np.isfinite(item[1])]
                            if len(mostUsedMetricSources_processed) != len(mostUsedMetricSources_from_renn):
                                print(f"Warning: Sample {pos}, k={currentClosestSources} metric sources had invalid flat items.")
                        else: # Nested but not BTL
                            #print(f"Warning: Sample {pos}, k={currentClosestSources} has nested metric sources but not BTL. Using raw input structure.") # Or aggregate?
                            mostUsedMetricSources_processed = mostUsedMetricSources_from_renn # Keep original structure if not BTL?

                        # Ensure it's not None after processing
                        if mostUsedMetricSources_processed is None:
                            raise ValueError("Metric sources became None after processing.")

                    except Exception as e:
                        print(f"ERROR: Sample {pos}, k={currentClosestSources} failed preparing metric sources: {e}. Skipping k.")
                        continue # Skip this k value
                else:
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - RENN returned None/empty metric sources. Skipping calculations.") # Optional Debug
                    continue # Skip calculations if no sources from RENN

                # --- Calculations use the PROCESSED lists ---
                # Check if processed lists exist and are non-empty
                # Also check if the processed original list structure matches the processed metric list structure if needed by calcs
                if mostUsedMetricSources_processed is not None and originalSources_processed is not None and \
                        len(mostUsedMetricSources_processed) > 0 and len(originalSources_processed) > 0 and \
                        isinstance(originalSources_processed, list) and isinstance(mostUsedMetricSources_processed, list): # Basic list check

                    # --- UNCOMMENT DEBUG PRINTS BELOW TO TRACE ---
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Data exists. Len Orig: {len(originalSources_processed)}, Len Metric: {len(mostUsedMetricSources_processed)}")

                    # --- Calculate ALL Source Similarities/Distances ---
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Cosine Sim")
                    current_k_results[SOURCE_COSINE_METRIC] = calculate_source_cosine_similarity(originalSources_processed, mostUsedMetricSources_processed)

                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Log Cosine Sim")
                    current_k_results[SOURCE_LOG_COSINE_METRIC] = calculate_log_cosine_similarity(originalSources_processed, mostUsedMetricSources_processed)

                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before JSD")
                    current_k_results[SOURCE_JSD_METRIC] = calculate_jsd(originalSources_processed, mostUsedMetricSources_processed) # Use processed!

                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Spearman")
                    # Call directly, handle errors internally or with try/except
                    try:
                        spearman, kendall = calculate_rank_correlation(originalSources_processed, mostUsedMetricSources_processed)
                    except Exception as rank_e:
                        print(f"ERROR: S:{pos} K:{currentClosestSources} - calculate_rank_correlation call failed: {rank_e}")
                        spearman, kendall = np.nan, np.nan
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - After Spearman. Result: {(spearman, kendall)}")
                    current_k_results[SOURCE_SPEARMAN_METRIC] = spearman
                    current_k_results[SOURCE_KENDALL_METRIC] = kendall

                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before TopK")
                    intersect_k, precision_k, recall_k = calculate_top_k_overlap(originalSources_processed, mostUsedMetricSources_processed, currentClosestSources)
                    current_k_results[SOURCE_INTERSECT_K_METRIC] = intersect_k
                    current_k_results[SOURCE_PRECISION_K_METRIC] = precision_k
                    current_k_results[SOURCE_RECALL_K_METRIC] = recall_k

                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Vector Dist")
                    try:
                        euclidean, manhattan = calculate_vector_distances(originalSources_processed, mostUsedMetricSources_processed)
                    except Exception as dist_e:
                        print(f"ERROR: S:{pos} K:{currentClosestSources} - calculate_vector_distances call failed: {dist_e}")
                        euclidean, manhattan = np.nan, np.nan
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - After Vector Dist. Result: {(euclidean, manhattan)}")
                    current_k_results[SOURCE_EUCLIDEAN_METRIC] = euclidean
                    current_k_results[SOURCE_MANHATTAN_METRIC] = manhattan

                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Ruzicka")
                    current_k_results[SOURCE_RUZICKA_METRIC] = calculate_ruzicka_similarity(originalSources_processed, mostUsedMetricSources_processed)
                    # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before SymmDiff")
                    current_k_results[SOURCE_SYMM_DIFF_METRIC] = calculate_symmetric_difference_size(originalSources_processed, mostUsedMetricSources_processed)
                    current_k_results[BEST_CLOSEST_SOURCES] = currentClosestSources # Store k for this successful iteration
                    current_k_results[BEST_CLOSEST_SOURCES_FOR_ORIGINAL] = bestClosestSourcesForOriginal

                    # --- Calculate Image Similarity ---
                    if evaluationActivations is not None:
                        # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Image Sim")
                        image_sim_dict = evaluateImageSimilarityByMetrics(name="Metrics", combination=combination_str, sample=sample, mostUsed=mostUsedMetricSources_processed, storeGlobally=False)
                        if image_sim_dict:
                            for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items():
                                if eval_key in image_sim_dict: current_k_results[f"{IMG_SIM_PREFIX}{store_suffix}"] = image_sim_dict[eval_key]

                    # --- Calculate Activation Similarity ---
                    if layerNumbersToCheck:
                        # *** VERIFY: What structure does blendActivations expect for mostUsed? ***
                        required_most_used_for_blend = mostUsedMetricSources_processed # ASSUMPTION: Needs processed/aggregated. CHANGE if needed!
                        # If it needs the original potentially nested structure from RENN:
                        # required_most_used_for_blend = mostUsedMetricSources_from_renn

                        overall_eval_flag = False
                        if "BTO" in name: overall_eval_flag = True
                        elif "BTL" in name: overall_eval_flag = False

                        # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Before Blend Act")
                        activation_sim_dict = blendActivations(
                            name=f"metric_combo_{combination_str}_k{currentClosestSources}",
                            mostUsed=required_most_used_for_blend, # Use the verified variable
                            evaluationActivations=evaluationActivations,
                            layerNumbersToCheck=linearLayers,
                            store_globally=False,
                            overallEvaluation=overall_eval_flag
                        )
                        # print(f"DEBUG: S:{pos} K:{currentClosestSources} - After Blend Act. Dict created: {activation_sim_dict is not None}")
                        if activation_sim_dict:
                            for act_key in ACTIVATION_METRIC_KEYS:
                                if act_key in activation_sim_dict: current_k_results[act_key] = activation_sim_dict[act_key]

                # else: # Handle case where processed lists are empty or None
                # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Skipping calculations due to empty/None processed lists.")

            except Exception as e:
                print(f"\n--- WORKER ERROR (PID {os.getpid()}) processing k={currentClosestSources}, combination {combination_str} for sample {pos} ---")
                print(f"Error in main calculation try block: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for full traceback
                current_k_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION} # Reset k results
                activation_sim_dict = None # Reset activation dict
                print(f"--- WORKER (PID {os.getpid()}) continuing ---")
                continue # Explicitly continue to next k

            # --- User's Logic Block for Storing Results (Safer Checks) ---
            # This block overwrites current_combination_results if the condition is met.
            # It does NOT find the best k over the whole range.
            update_combination_results = False
            try:
                # Check if activation_sim_dict exists from this iteration and has the score
                if activation_sim_dict is not None and OPTIMIZATION_METRIC in activation_sim_dict:
                    # Use .get for safety, provide NaN default
                    current_activation_score = activation_sim_dict.get(OPTIMIZATION_METRIC, np.nan)

                    # Check if score is valid and better than the tracking variable
                    if not np.isnan(current_activation_score) and current_activation_score > currentBestCosineSimilarity:
                        update_combination_results = True
                        currentBestCosineSimilarity = current_activation_score # Update tracker
            except Exception as if_block_e:
                print(f"DEBUG: Error in final 'if' check k={currentClosestSources}, sample {pos}: {if_block_e}")
                pass # Avoid crashing here, flag remains False

            if update_combination_results:
                # print(f"DEBUG: S:{pos} K:{currentClosestSources} - Updating results for combination {combination_str} based on score {currentBestCosineSimilarity}") # Optional Debug
                current_combination_results = current_k_results.copy() # Store this k's results
            # --- End of User's Logic Block ---

        # --- Storing results for the combination ---
        # Store whatever is in current_combination_results (NaNs or last k that met condition)
        for metric_key, value in current_combination_results.items():
            sample_results[(combination_str, metric_key)] = value
            # print(f"DEBUG: S:{pos} - Storing final result for Combo: {combination_str}, Key: {metric_key}, Value: {value}") # Very verbose

    return sample_results


# ==============================================================
# Main Evaluation Function (Multiprocessing Version - User Provided Structure)
# ==============================================================
def evaluate_metric_combinations_overall(name, mostUsedList, linearLayers, hidden_sizes, closestSources, all_metric_combinations, mode="Sum", max_workers=None):
    # --- Setup ---
    start_time = time.time()
    print(f"\nStarting evaluation with multiprocessing (max_workers={max_workers or os.cpu_count()})... for {name}")
    print("Metrics: Activation Sim, Image Sim, Source Sims (Cos, LogCos, JSD, Rank, TopK, Dist, Ruzicka, SymmDiff)")

    if max_workers is None: max_workers = os.cpu_count(); print(f"Using default max_workers = {max_workers}")
    if not all_metric_combinations: print("Error: No metric combinations provided."); return None

    futures = []; results_list = []
    # --- Process Pool ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_samples = 0
        # Submit tasks for each sample (using placeholder dataloader)
        for pos, (sample, true) in enumerate(eval_dataloader):
            if processed_samples >= eval_samples: break
            
            # --- Get Sample-Specific Data (Using Placeholders) ---
            try:
                # originalMostUsedSources_input = mostUsedList[pos] if pos < len(mostUsedList) else [] # Original line commented out
                # Get original sources using placeholder function - MUST match expected structure (flat/nested)
                originalMostUsedSources_input = findBestOriginalValues(name, dictionaryForSourceLayerNeuron[pos], closestSources, mode)
                originalMostUsedSources, bestClosestSourcesForOriginal = originalMostUsedSources_input['mostUsedSources'], originalMostUsedSources_input['closestSources']
                evaluationActivations = dictionaryForSourceLayerNeuron[pos] # Can be None if pos not in dict
                metricsSampleActivations = metricsDictionaryForSourceLayerNeuron[pos] # Can be None

                if metricsSampleActivations is None:
                    print(f"Warning: Skipping sample {pos}, metricsSampleActivations is None.")
                    continue
                # Basic check if it's array-like, worker handles None check again
                if not hasattr(metricsSampleActivations, 'shape'):
                    print(f"Warning: Skipping sample {pos}, metricsSampleActivations is not array-like.")
                    continue

                # Ensure sample data is tensor (placeholder logic)
                if hasattr(sample, 'float'): sample_data = sample.float()
                else: sample_data = sample # Assume already correct type

            except Exception as data_prep_e:
                print(f"Warning: Data preparation failed for sample {pos}: {data_prep_e}. Skipping.")
                continue

            # --- Arguments for worker ---
            args = (pos, sample_data, originalMostUsedSources, evaluationActivations,
                    metricsSampleActivations, linearLayers, all_metric_combinations, closestSources, mode, name, bestClosestSourcesForOriginal
                    )
            futures.append(executor.submit(process_sample_evaluation, args))
            processed_samples += 1

        print(f"Submitted {len(futures)} tasks to workers.")
        # --- Retrieve Results ---
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            # print(f"DEBUG: Waiting for result {i+1}/{len(futures)}") # Optional progress
            try:
                result = future.result() # This is where the "unpacking" error was reported
                if result is not None: results_list.append(result)
                # print(f"DEBUG: Got result {i+1}/{len(futures)}") # Optional progress
            except Exception as e:
                # This is where the error from the worker is re-raised
                print(f"\n--- ERROR retrieving result for task {i+1}: {e} ---")
                # import traceback # Optional traceback for debugging here
                # traceback.print_exc()

    # --- Aggregation ---
    print(f"\nCollected results from {len(results_list)} completed tasks.")
    if not results_list: print("Error: No results collected."); return None

    print("Aggregating results...")
    results_aggregator = defaultdict(lambda: defaultdict(list))
    valid_keys_for_agg = set(ALL_METRIC_KEYS_FOR_AGGREGATION) # Use set for faster lookup
    for sample_result_dict in results_list:
        if not isinstance(sample_result_dict, dict): continue # Skip if result format is wrong
        for (combination_str, metric_key), score in sample_result_dict.items():
            # Only aggregate keys defined in the list and ignore NaNs during aggregation mean calculation
            if metric_key in valid_keys_for_agg:
                if score is not None and np.isfinite(score): # Check for finite numbers
                    results_aggregator[combination_str][metric_key].append(score)
                elif metric_key == BEST_CLOSEST_SOURCES and not np.isnan(score): # Keep best K value even if NaN metric
                    results_aggregator[combination_str][metric_key].append(int(score)) # Store k as int
                # else: store NaN? No, nanmean handles empty lists later

    # --- Calculate Averages ---
    final_results = {}
    for combination_str, metric_scores_dict in results_aggregator.items():
        avg_scores = {}
        for metric_key in ALL_METRIC_KEYS_FOR_AGGREGATION:
            scores_list = metric_scores_dict.get(metric_key, [])
            if scores_list:
                # Use nanmean for most metrics, mean for integer k value
                if metric_key == BEST_CLOSEST_SOURCES:
                    avg_scores[f"avg_{metric_key}"] = np.mean(scores_list) # Average k value
                else:
                    avg_scores[f"avg_{metric_key}"] = np.nanmean(scores_list) # nanmean ignores NaNs
            else: avg_scores[f"avg_{metric_key}"] = np.nan # No valid scores recorded
        final_results[combination_str] = avg_scores

    if not final_results: print("Error: Aggregation resulted in empty dictionary."); return None

    # --- Create DataFrame ---
    try:
        results_df = pd.DataFrame.from_dict(final_results, orient='index')
        print(f"Aggregated DataFrame shape: {results_df.shape}")
        if results_df.empty: print("Error: DataFrame is empty after aggregation."); return None
    except Exception as e: print(f"Error creating DataFrame: {e}"); return None

    # --- Column Ordering (User Provided Logic) ---
    prefix = f"avg_" # Correct prefix used in aggregation
    ordered_columns = []
    act_sim_ordered = ['cosine_similarity', 'pearson_correlation', 'kendall_tau', 'spearman_rho', 'jaccard_similarity']
    act_dist_ordered = ['euclidean_distance', 'manhattan_distance', 'hamming_distance']
    img_sim_ordered = ['cosine_sim', 'pearson_corr', 'kendall_tau', 'spearman_rho', 'jaccard_sim']
    img_dist_ordered = ['euclidean_dst', 'manhattan_dst', 'hamming_dst']
    source_sim_ordered = [
        SOURCE_COSINE_METRIC, SOURCE_LOG_COSINE_METRIC, SOURCE_RUZICKA_METRIC,
        SOURCE_SPEARMAN_METRIC, SOURCE_KENDALL_METRIC,
        SOURCE_PRECISION_K_METRIC, SOURCE_RECALL_K_METRIC, SOURCE_INTERSECT_K_METRIC
    ]
    source_dist_ordered = [
        SOURCE_JSD_METRIC, SOURCE_EUCLIDEAN_METRIC, SOURCE_MANHATTAN_METRIC,
        SOURCE_SYMM_DIFF_METRIC, BEST_CLOSEST_SOURCES, BEST_CLOSEST_SOURCES_FOR_ORIGINAL # BEST_CLOSEST_SOURCES is the K value
    ]
    # Add avg_ prefix correctly
    ordered_columns.extend([f'{prefix}{m}' for m in act_sim_ordered])
    ordered_columns.extend([f'{prefix}{m}' for m in act_dist_ordered])
    ordered_columns.extend([f'{prefix}{IMG_SIM_PREFIX}{m}' for m in img_sim_ordered])
    ordered_columns.extend([f'{prefix}{IMG_SIM_PREFIX}{m}' for m in img_dist_ordered])
    ordered_columns.extend([f'{prefix}{m}' for m in source_sim_ordered])
    ordered_columns.extend([f'{prefix}{m}' for m in source_dist_ordered])
    # Find remaining columns and add them sorted
    results_df_cols = set(results_df.columns)
    ordered_cols_set = set(ordered_columns)
    remaining_cols = sorted(list(results_df_cols - ordered_cols_set))
    ordered_columns.extend(remaining_cols)
    # Filter list to only include columns actually present in DataFrame
    final_ordered_columns = [col for col in ordered_columns if col in results_df_cols]

    try:
        results_df = results_df[final_ordered_columns]
        print("Reordered DataFrame columns.")
    except Exception as e:
        print(f"Warning: Column reordering failed: {e}. Using default order.")

    # --- Row Sorting Priority (User Provided Logic - Adjusted Prefix) ---
    metrics_priority_list = [
        f'{prefix}{SOURCE_INTERSECT_K_METRIC}',
        f'{prefix}cosine_similarity', # Activation Cosine
        f'{prefix}{IMG_SIM_PREFIX}cosine_sim',
        f'{prefix}{SOURCE_COSINE_METRIC}',
        f'{prefix}{SOURCE_LOG_COSINE_METRIC}',
        f'{prefix}{SOURCE_RUZICKA_METRIC}',
        f'{prefix}{SOURCE_SPEARMAN_METRIC}',
        f'{prefix}{SOURCE_PRECISION_K_METRIC}',
        f'{prefix}{SOURCE_RECALL_K_METRIC}',
        f'{prefix}{SOURCE_JSD_METRIC}', # Lower is better
        f'{prefix}{SOURCE_EUCLIDEAN_METRIC}', # Lower is better
        f'{prefix}{SOURCE_MANHATTAN_METRIC}', # Lower is better
        f'{prefix}{SOURCE_SYMM_DIFF_METRIC}', # Lower is better
        f'{prefix}euclidean_distance', # Activation Euclidean
        f'{prefix}{IMG_SIM_PREFIX}euclidean_dst',
        f'{prefix}{SOURCE_KENDALL_METRIC}',
        f'{prefix}kendall_tau', # Activation kendall
    ]
    similarity_keywords = ['similarity', 'correlation', 'tau', 'rho', 'spearman', 'kendall', 'precision', 'recall', 'intersect', 'ruzicka', 'cosine']
    distance_keywords = ['distance', 'dst', 'jsd', 'euclidean', 'manhattan', 'diff', 'hamming']

    sort_by_columns = []; sort_ascending_flags = []; sort_descriptions = []
    for metric_col in metrics_priority_list:
        if metric_col not in results_df.columns: continue
        ascending_order = False # Default: Higher is better
        sort_type = " (Desc)"; metric_col_lower = metric_col.lower()

        # Check if it's explicitly a distance metric where lower is better
        is_distance = any(keyword in metric_col_lower for keyword in distance_keywords)
        is_explicit_source_dist = any(dist_key in metric_col for dist_key in [
            SOURCE_JSD_METRIC, SOURCE_EUCLIDEAN_METRIC, SOURCE_MANHATTAN_METRIC, SOURCE_SYMM_DIFF_METRIC
        ])

        if is_explicit_source_dist or (is_distance and not any(sim_key in metric_col_lower for sim_key in similarity_keywords)):
            ascending_order = True; sort_type = " (Asc)"
        # No need for elif, default is Descending/False

        sort_by_columns.append(metric_col); sort_ascending_flags.append(ascending_order)
        sort_descriptions.append(f"{metric_col.replace(prefix, '')}{sort_type}")

    # --- Perform Final Row Sort ---
    final_df_to_return = results_df # Default if sorting fails
    if not sort_by_columns:
        print("\nWarning: No valid columns found for sorting rows. Returning unsorted DataFrame.")
    else:
        print(f"\n--- Final Ranking Sorted By Rows: {' -> '.join(sort_descriptions)} ---")
        try:
            final_df_to_return = results_df.sort_values(by=sort_by_columns, ascending=sort_ascending_flags, na_position='last')
        except Exception as e:
            print(f"\nError during final row sorting: {e}. Returning unsorted DataFrame.")
            final_df_to_return = results_df # Return unsorted on error

    end_time = time.time()
    print(f"\nEvaluation for {name} complete in {end_time - start_time:.2f} seconds.")
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

all_run_results = []
def findBestOriginalValues(name, original_values, closestSources, mode):
    # Assume RENN and blendActivations are defined elsewhere
    global all_run_results # Use the renamed global list

    bestMostUsedSources = {'result': {'cosine_similarity': -1}, 'mostUsedSources': [], 'closestSources': None} # Initialize with -1

    # Define the range based on the input 'closestSources' parameter
    search_range = range(max(1, closestSources - 20), closestSources + 20) # Ensure range starts at 1 minimum

    for currentClosestSourcesValue in search_range:
        try:
            # Ensure RENN methods handle potential errors if currentClosestSourcesValue is invalid
            identifiedClosestSources, outputsToCheck, layerNumbersToCheck = RENN.identifyClosestSourcesForOriginal(original_values, currentClosestSourcesValue, mode)
            mostUsedSources = RENN.getMostUsedSourcesByName(name, identifiedClosestSources, currentClosestSourcesValue, mode)

            overallEvaluation = "BTO" in name # Simpler check
            results = blendActivations(name, mostUsedSources, original_values, layerNumbersToCheck, overallEvaluation=overallEvaluation)

            # Check if results is valid and contains 'cosine_similarity'
            if results and 'cosine_similarity' in results and results['cosine_similarity'] > bestMostUsedSources['result']['cosine_similarity']:
                bestMostUsedSources = {'result': results, 'mostUsedSources': mostUsedSources, 'closestSources': currentClosestSourcesValue}
        except Exception as e:
            # Optional: Log errors if RENN or blendActivations fail for certain values
            print(f"Error during processing for closestSources={currentClosestSourcesValue}: {e}")
            continue # Skip to the next value in the range

    # --- Modification Here ---
    # Store the best result *from this specific run*
    if bestMostUsedSources['closestSources'] is not None: # Only append if a valid result was found
        all_run_results.append(bestMostUsedSources)
        print("Cosine-Similarity:", bestMostUsedSources['result']['cosine_similarity'])
    # --- End Modification ---

    return bestMostUsedSources # Return the best result for this run

def get_overall_best_sources_and_results():
    global all_run_results

    if not all_run_results:
        print("No results found in 'all_run_results'. Run 'findBestOriginalValues' first.")
        return None

    # Find the dictionary with the maximum cosine_similarity
    try:
        overall_best_run = max(all_run_results, key=lambda run: run['result']['cosine_similarity'])
    except (KeyError, TypeError) as e:
        print(f"Error accessing result data. Ensure 'all_run_results' has the correct structure: {e}")
        # Optional: Inspect the contents of all_run_results here for debugging
        # print("Contents of all_run_results:", all_run_results)
        return None


    print("\n--- Overall Best Result ---")
    print(f"Achieved Cosine Similarity: {overall_best_run['result']['cosine_similarity']:.4f}")
    print(f"Achieved during run using closestSources = {overall_best_run['closestSources']}")
    # You might want to print other details from overall_best_run['result'] if they exist
    # print(f"Full result dictionary: {overall_best_run['result']}")
    print(f"Corresponding 'mostUsedSources':")
    # Print sources nicely (handle list or other types)
    if isinstance(overall_best_run['mostUsedSources'], list):
        for i, source in enumerate(overall_best_run['mostUsedSources']):
            print(f"  - Source {i+1}: {source}") # Adjust formatting as needed
    else:
        print(f"  - {overall_best_run['mostUsedSources']}")

    return overall_best_run
