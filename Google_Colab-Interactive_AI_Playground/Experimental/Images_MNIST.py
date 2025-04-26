import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
import numpy as np
import os
import Customizable_RENN as RENN
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
        print("Setting seed number to ", seed)
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
    try:
        pearson_correlation, _ = pearsonr(sample.flatten(), train_sample.flatten())  # Pearson
    except ValueError:
        pearson_correlation = None

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

    kendall_tau, _ = kendalltau(sample, blended_image_flat)
    spearman_rho, _ = spearmanr(sample, blended_image_flat)

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
    print(f"\n--- Blended Image Similarity Scores ({name})---")
    print(f"Kendall's Tau: {kendall_tau:.2f}")
    print(f"Spearman's Rho: {spearman_rho:.2f}")
    print(f"Cosine Similarity: {cosine_similarity:.4f}")
    print(f"Euclidean Distance: {euclidean_distance:.4f}")
    print(f"Manhattan Distance: {manhattan_distance:.4f}")
    print(f"Jaccard Similarity: {jaccard_similarity:.4f}" if jaccard_similarity is not None else "Jaccard Similarity: N/A")
    print(f"Hamming Distance: {hamming_distance:.4f}")
    print(f"Pearson Correlation: {pearson_correlation:.4f}" if pearson_correlation is not None else "Pearson Correlation: N/A")

    if name == "":
        original_image_similarity.append(results)
    elif name == "Metrics":
        metrics_image_similarity.append(results)
    elif name == "MT":
        mt_image_similarity.append(results)

    # # Initialize aggregates for overall metrics
    # aggregate_scores = {
    #     "cosine": 0,
    #     "euclidean": 0,
    #     "manhattan": 0,
    #     "jaccard": 0,
    #     "hamming": 0,
    #     "pearson": 0,
    # }
    # count_valid = {
    #     "jaccard": 0,  # Count only valid Jaccard entries (non-zero denominator)
    #     "pearson": 0,  # Count only valid Pearson correlations
    # }
    # 
    # # Compute similarity for each training sample
    # for pos, (train_sample, true) in enumerate(trainDataSet):
    #     train_sample = np.asarray(train_sample.flatten().reshape(1, -1))
    # 
    #     cosine_similarity, euclidean_distance, manhattan_distance, jaccard_similarity, hamming_distance, pearson_correlation = computeSimilarity(sample, train_sample)
    # 
    #     # Accumulate aggregate scores
    #     aggregate_scores["cosine"] += cosine_similarity
    #     aggregate_scores["euclidean"] += euclidean_distance
    #     aggregate_scores["manhattan"] += manhattan_distance
    #     if jaccard_similarity is not None:
    #         aggregate_scores["jaccard"] += jaccard_similarity
    #         count_valid["jaccard"] += 1
    #     aggregate_scores["hamming"] += hamming_distance
    #     if pearson_correlation is not None:
    #         aggregate_scores["pearson"] += pearson_correlation
    #         count_valid["pearson"] += 1
    # 
    #     similarityList.append({
    #         "pos": pos,
    #         "cosine": cosine_similarity,
    #         "euclidean": euclidean_distance,
    #         "manhattan": manhattan_distance,
    #         "jaccard": jaccard_similarity,
    #         "hamming": hamming_distance,
    #         "pearson": pearson_correlation,
    #     })
    # 
    # # Normalize aggregate scores
    # num_samples = len(trainDataSet)
    # for key in ["cosine", "euclidean", "manhattan", "hamming"]:
    #     aggregate_scores[key] /= num_samples
    # if count_valid["jaccard"] > 0:
    #     aggregate_scores["jaccard"] /= count_valid["jaccard"]
    # else:
    #     aggregate_scores["jaccard"] = None
    # if count_valid["pearson"] > 0:
    #     aggregate_scores["pearson"] /= count_valid["pearson"]
    # else:
    #     aggregate_scores["pearson"] = None
    # 
    # # Sort similarityList by cosine similarity in descending order
    # similarityList.sort(key=lambda x: x["cosine"], reverse=True)
    # 
    # # Extract top sources from similarity list
    # topSources = set(item["pos"] for item in similarityList[:len(mostUsed)])
    # 
    # # Extract most used sources
    # mostUsedSources = set(source for source, _ in mostUsed)
    # 
    # # --- Metrics ---
    # # Matches
    # matches = len(topSources & mostUsedSources)
    # 
    # # Accuracy
    # accuracy = matches / len(topSources) if topSources else 0
    # 
    # # Precision and Recall
    # precision = matches / len(mostUsedSources) if mostUsedSources else 0
    # recall = matches / len(topSources) if topSources else 0
    # 
    # # F1-Score
    # f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # 
    # # Weighted Accuracy
    # total_weight = sum(count for _, count in mostUsed)
    # weighted_matches = sum(
    #     count for source, count in mostUsed if source in topSources
    # )
    # weighted_accuracy = weighted_matches / total_weight if total_weight else 0
    # 
    # # Kendall's Tau and Spearman's Rho
    # topRanking = [item["pos"] for item in similarityList[:len(mostUsed)]]
    # mostUsedRanking = [source for source, _ in mostUsed]
    # kendall_tau, _ = kendalltau(topRanking, mostUsedRanking)
    # spearman_rho, _ = spearmanr(topRanking, mostUsedRanking)
    # 
    # # Top-k Intersection
    # top_k_intersection = len(topSources & mostUsedSources)
    # 
    # # --- Print Results ---
    # print("\n--- Overall Metrics ---")
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # print(f"Precision: {precision * 100:.2f}%")
    # print(f"Recall: {recall * 100:.2f}%")
    # print(f"F1-Score: {f1_score * 100:.2f}%")
    # print(f"Weighted Accuracy: {weighted_accuracy * 100:.2f}%")
    # print(f"Kendall's Tau: {kendall_tau:.2f}")
    # print(f"Spearman's Rho: {spearman_rho:.2f}")
    # print(f"Top-{len(mostUsed)} Intersection: {top_k_intersection}/{len(mostUsed)}")
    # 
    # # --- Print Overall Similarity Scores ---
    # print("\n--- Overall Similarity Scores ---")
    # print(f"Cosine Similarity (Mean): {aggregate_scores['cosine']:.4f}")
    # print(f"Euclidean Distance (Mean): {aggregate_scores['euclidean']:.4f}")
    # print(f"Manhattan Distance (Mean): {aggregate_scores['manhattan']:.4f}")
    # print(f"Jaccard Similarity (Mean): {aggregate_scores['jaccard']:.4f}" if aggregate_scores["jaccard"] is not None else "Jaccard Similarity: N/A")
    # print(f"Hamming Distance (Mean): {aggregate_scores['hamming']:.4f}")
    # print(f"Pearson Correlation (Mean): {aggregate_scores['pearson']:.4f}" if aggregate_scores["pearson"] is not None else "Pearson Correlation: N/A")

# Global storage (optional)
original_activation_similarity, metrics_activation_similarity, mt_activation_similarity = [], [], []
def blendActivations(name, mostUsed, evaluationActivations, layerNumbersToCheck, store_globally=False):
    totalSources = sum(count for _, count in mostUsed)
    blendedActivations = np.zeros_like(evaluationActivations[layerNumbersToCheck])

    for source, count in mostUsed:
        activationsBySources = RENN.activationsBySources[source]
        for layerIdx, layerNumber in enumerate(layerNumbersToCheck):
            neurons = activationsBySources[layerNumber]
            blendedActivations[layerIdx] += neurons * (count / totalSources)

    # Flatten and reshape for similarity computation
    eval_flat = evaluationActivations[layerNumbersToCheck].flatten().reshape(1, -1).astype(np.float64)
    blend_flat = blendedActivations.flatten().reshape(1, -1).astype(np.float64)

    # --- Compute Metrics ---
    cosine_sim, euclidean_dist, manhattan_dist, jaccard_sim, hamming_dist, pearson_corr = computeSimilarity(eval_flat, blend_flat)
    kendall_tau, _ = kendalltau(eval_flat.squeeze(), blend_flat.squeeze())
    spearman_rho, _ = spearmanr(eval_flat.squeeze(), blend_flat.squeeze())

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

resultDataframe = ""
def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, analyze=False):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, original_image_similarity, metrics_image_similarity, mt_image_similarity, original_activation_similarity, metrics_activation_similarity, mt_activation_similarity, resultDataframe

    original_image_similarity, metrics_image_similarity, mt_image_similarity, original_activation_similarity, metrics_activation_similarity, mt_activation_similarity = [], [], [], [], [], []

    #Make sure to set new dictionaries for the hooks to fill - they are global!
    dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource = RENN.initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model)
    
    if analyze:
        METRICS_COMBINATIONS = RENN.create_global_metric_combinations(3, 3, True)
        #metricsDictionaryForSourceLayerNeuron = createRandomDictionary(metricsDictionaryForSourceLayerNeuron) #Random values per metrics min and max
        #metricsDictionaryForSourceLayerNeuron = np.full(metricsDictionaryForSourceLayerNeuron.shape, 0.5) #Fixed Vector

    mostUsedList = []
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

            sourcesActivation, metricSourcesActivation, mtSourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation, mostUsedMetricSourcesWithActivation, mostUsedMTSourcesWithActivation, mostUsedSourcesPerLayerWithActivation = RENN.getMostUsedSources(sourcesActivation, metricSourcesActivation, mtSourcesActivation, closestSources, "Activation")
            #20 sources only because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation, "Not Weighted")
            blendedMetricSourceImageActivation = blendImagesTogether(mostUsedMetricSourcesWithActivation, "Not Weighted")
            blendedMTSourceImageActivation = blendImagesTogether(mostUsedMTSourcesWithActivation, "Not Weighted")

            showImagesUnweighted("Per Neuron", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
            showImagesUnweighted("Metrics", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedMetricSourceImageActivation, blendedMetricSourceImageSum, mostUsedMetricSourcesWithActivation[:showClosestMostUsedSources], mostUsedMetricSourcesWithSum[:showClosestMostUsedSources])
            showImagesUnweighted("Magnitude Truncation", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedMTSourceImageActivation, blendedMTSourceImageSum, mostUsedMTSourcesWithActivation[:showClosestMostUsedSources], mostUsedMTSourcesWithSum[:showClosestMostUsedSources])
        else:
            sourcesSum, metricSourcesSum, mtSourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum = getClosestSourcesPerNeuronAndLayer(sourcesSum, metricSourcesSum, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Sum")

            sourcesActivation, metricSourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = getClosestSourcesPerNeuronAndLayer(sourcesActivation, metricSourcesActivation, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Activation")
            #RENN.analyzeData(closestSources, dictionaryForSourceLayerNeuron[pos])

        if(analyze):
            #mostUsed, mostUsedMetrics, mostUsedMT = #RENN.getMostUsedSources(sourcesSum, metricSourcesSum, mtSourcesSum, closestSources)
            mostUsedList.append(mostUsedSourcesPerLayerWithSum)
            blendActivations("", mostUsedSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layersToCheck, True)
            evaluateImageSimilarity("", sample, mostUsedSourcesWithSum)
            if metricsEvaluation:
                blendActivations("Metrics", mostUsedMetricSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layersToCheck, True)
                evaluateImageSimilarity("Metrics", sample, mostUsedMetricSourcesWithSum)
            if RENN.mtEvaluation:
                blendActivations("MT", mostUsedMTSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layersToCheck, True)
                evaluateImageSimilarity("MT", sample, mostUsedMTSourcesWithSum)

            #Per sample, not overall
            if paretoEvaluation:
                for metric_combination in METRICS_COMBINATIONS:
                    metricSourcesSum, layerNumbersToCheck = RENN.identifyClosestSourcesByMetricCombination(closestSources, metricsDictionaryForSourceLayerNeuron[pos], metric_combination, mode="Sum")
                    mostUsedMetricSourcesWithSum = RENN.getMostUsedSourcesByMetrics(metricSourcesSum, closestSources, weightedMode="Sum")

                    evaluateImageSimilarityByMetrics("Metrics", metric_combination, sample, mostUsedMetricSourcesWithSum)
    if analyze:
        #Per sample, not overall    
        if weightTuning and paretoEvaluation:
            optimizations_to_run = evaluate_pareto()
            results, combinations = optimize_weights_of_best_combinations(closestSources, optimizations_to_run)
            synthesize_overall_weights(results)
            pareto_df, weight_stats_df, study, union_combination_names = find_overall_weights_via_moo(closestSources, combinations)
            find_best_metric_weights(pareto_df, closestSources)

        resultDataframe = evaluate_metric_combinations_overall(
            mostUsedList,
            hidden_sizes=hidden_sizes,
            closestSources=closestSources,
            all_metric_combinations=METRICS_COMBINATIONS,
            mode="Sum"
        )

# ----------------------------------------------------------------------------
# Helper Functions for Source List Similarities / Distances
# ----------------------------------------------------------------------------

def createRandomDictionary(original):
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
        min_val = np.min(original_array)
        max_val = np.max(original_array)
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

# --- Existing ---
def calculate_source_cosine_similarity(list1, list2):
    """ Calculates standard cosine similarity between source count vectors. """
    counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
    all_source_ids = sorted(list(counts1.keys() | counts2.keys()))
    if not all_source_ids: return 1.0
    vec1 = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float).reshape(1, -1)
    vec2 = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float).reshape(1, -1)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 and norm2 == 0: return 1.0
    if norm1 == 0 or norm2 == 0: return 0.0
    try: return np.clip(cosine_similarity(vec1, vec2)[0, 0], 0.0, 1.0)
    except Exception: return 0.0

def calculate_log_cosine_similarity(list1, list2):
    """ Calculates cosine similarity between log2(count+1) transformed source vectors. """
    counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
    all_source_ids = sorted(list(counts1.keys() | counts2.keys()))
    if not all_source_ids: return 1.0
    vec1 = np.array([math.log2(counts1.get(src_id, 0) + 1) for src_id in all_source_ids], dtype=float).reshape(1, -1)
    vec2 = np.array([math.log2(counts2.get(src_id, 0) + 1) for src_id in all_source_ids], dtype=float).reshape(1, -1)
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 and norm2 == 0: return 1.0
    if norm1 == 0 or norm2 == 0: return 0.0
    try: return np.clip(cosine_similarity(vec1, vec2)[0, 0], 0.0, 1.0)
    except Exception: return 0.0

def calculate_jsd(list1, list2):
    """ Calculates Jensen-Shannon Divergence (base 2) between source count distributions. """
    counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
    all_source_ids = sorted(list(counts1.keys() | counts2.keys()))
    if not all_source_ids: return 0.0
    vec1 = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float)
    vec2 = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float)
    sum1 = np.sum(vec1); sum2 = np.sum(vec2)
    if sum1 == 0 and sum2 == 0: return 0.0
    if sum1 == 0 or sum2 == 0: return 1.0
    p = vec1 / sum1; q = vec2 / sum2
    try:
        jsd_score = distance.jensenshannon(p, q, base=2.0)
        return jsd_score if not np.isnan(jsd_score) else 1.0
    except Exception: return np.nan

def calculate_rank_correlation(list1, list2):
    """ Calculates Spearman and Kendall rank correlation based on counts. """
    if not list1 or not list2: return np.nan, np.nan
    try:
        sorted_list1 = sorted(list1, key=lambda x: x[1], reverse=True)
        sorted_list2 = sorted(list2, key=lambda x: x[1], reverse=True)
        rank_map1 = {id: rank + 1 for rank, (id, count) in enumerate(sorted_list1)}
        rank_map2 = {id: rank + 1 for rank, (id, count) in enumerate(sorted_list2)}
    except IndexError: return np.nan, np.nan
    common_ids = set(rank_map1.keys()) & set(rank_map2.keys())
    if len(common_ids) < 2: return np.nan, np.nan
    ranks1 = [rank_map1[id] for id in common_ids]; ranks2 = [rank_map2[id] for id in common_ids]
    try:
        spearman_corr, _ = spearmanr(ranks1, ranks2); kendall_tau, _ = kendalltau(ranks1, ranks2)
        spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
        kendall_tau = kendall_tau if not np.isnan(kendall_tau) else 0.0
    except Exception: spearman_corr, kendall_tau = np.nan, np.nan
    return spearman_corr, kendall_tau

def calculate_top_k_overlap(list1, list2, k):
    """ Calculates Intersection@k, Precision@k, Recall@k based on counts. """
    if k <= 0 or not list1 or not list2: return 0, 0.0, 0.0
    try:
        sorted_list1 = sorted(list1, key=lambda x: x[1], reverse=True)
        sorted_list2 = sorted(list2, key=lambda x: x[1], reverse=True)
        top_k_ids1 = set(item[0] for item in sorted_list1[:k])
        top_k_ids2 = set(item[0] for item in sorted_list2[:k])
        intersection_set = top_k_ids1 & top_k_ids2
        intersection_count = len(intersection_set)
        precision_at_k = intersection_count / k
        num_relevant_in_top_k1 = len(top_k_ids1) # Actual number of unique items in ref top k
        recall_at_k = intersection_count / num_relevant_in_top_k1 if num_relevant_in_top_k1 > 0 else 0.0
    except Exception: return 0, 0.0, 0.0
    return intersection_count, precision_at_k, recall_at_k

# --- NEW HELPERS ---
def calculate_vector_distances(list1, list2):
    """ Calculates Euclidean (L2) and Manhattan (L1) distances between count vectors. """
    counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
    all_source_ids = sorted(list(counts1.keys() | counts2.keys()))
    if not all_source_ids: return 0.0, 0.0 # Zero distance if both empty

    vec1 = np.array([counts1.get(src_id, 0) for src_id in all_source_ids], dtype=float)
    vec2 = np.array([counts2.get(src_id, 0) for src_id in all_source_ids], dtype=float)

    # Handle case where one list is empty (max possible distance conceptually)
    # Though norm calculation handles this implicitly if one vector is zero
    # if (len(list1) == 0 and len(list2) > 0) or (len(list2) == 0 and len(list1) > 0):
    #     return np.inf, np.inf # Or some large number / NaN?

    try:
        diff = vec1 - vec2
        euclidean_dist = np.linalg.norm(diff, ord=2)
        manhattan_dist = np.linalg.norm(diff, ord=1)
        return euclidean_dist, manhattan_dist
    except Exception:
        return np.nan, np.nan

def calculate_ruzicka_similarity(list1, list2):
    """ Calculates Ruzicka similarity (Generalized Jaccard for counts). """
    counts1 = Counter(dict(list1)); counts2 = Counter(dict(list2))
    all_source_ids = counts1.keys() | counts2.keys() # Union of keys
    if not all_source_ids: return 1.0 # Identical empty lists

    sum_min = 0.0
    sum_max = 0.0
    for id in all_source_ids:
        c1 = counts1.get(id, 0)
        c2 = counts2.get(id, 0)
        sum_min += min(c1, c2)
        sum_max += max(c1, c2)

    return sum_min / sum_max if sum_max > 0 else 1.0

def calculate_symmetric_difference_size(list1, list2):
    """ Calculates the number of items present in one list but not the other. """
    try:
        ids1 = set(item[0] for item in list1)
        ids2 = set(item[0] for item in list2)
        return len(ids1.symmetric_difference(ids2))
    except IndexError: # Handle malformed tuples
        return np.nan
    except Exception:
        return np.nan

# ----------------------------------------------------------------------------
# Constants for Metric Keys (Global Scope) - UPDATED
# ----------------------------------------------------------------------------
ACTIVATION_METRIC_KEYS = ["kendall_tau", "spearman_rho", "cosine_similarity", "euclidean_distance", "manhattan_distance", "jaccard_similarity", "hamming_distance", "pearson_correlation"]
IMAGE_SIM_KEY_MAP = {'Cosine Sim': 'cosine_sim', 'Euclidean Dst': 'euclidean_dst', 'Manhattan Dst': 'manhattan_dst', 'Jaccard Sim': 'jaccard_sim', 'Hamming Dst': 'hamming_dst', 'Pearson Corr': 'pearson_corr', 'Kendall Tau': 'kendall_tau', 'Spearman Rho': 'spearman_rho'}
IMG_SIM_PREFIX = "img_"
IMAGE_METRIC_KEYS_PREFIXED = [f"{IMG_SIM_PREFIX}{v}" for v in IMAGE_SIM_KEY_MAP.values()]

# Source Overlap Metric Keys - UPDATED
SOURCE_COSINE_METRIC = "source_cosine_similarity"
SOURCE_LOG_COSINE_METRIC = "source_log_cosine_similarity"
SOURCE_JSD_METRIC = "source_jsd"
SOURCE_SPEARMAN_METRIC = "source_spearman_rank_corr"
SOURCE_KENDALL_METRIC = "source_kendall_rank_tau"
SOURCE_INTERSECT_K_METRIC = "source_intersect_at_k"
SOURCE_PRECISION_K_METRIC = "source_precision_at_k"
SOURCE_RECALL_K_METRIC = "source_recall_at_k"
SOURCE_EUCLIDEAN_METRIC = "source_euclidean_dist" # New
SOURCE_MANHATTAN_METRIC = "source_manhattan_dist" # New
SOURCE_RUZICKA_METRIC = "source_ruzicka_similarity" # New
SOURCE_SYMM_DIFF_METRIC = "source_symmetric_diff_size" # New

SOURCE_METRIC_KEYS = [
    SOURCE_COSINE_METRIC, SOURCE_LOG_COSINE_METRIC, SOURCE_JSD_METRIC,
    SOURCE_SPEARMAN_METRIC, SOURCE_KENDALL_METRIC, SOURCE_INTERSECT_K_METRIC,
    SOURCE_PRECISION_K_METRIC, SOURCE_RECALL_K_METRIC, SOURCE_EUCLIDEAN_METRIC,
    SOURCE_MANHATTAN_METRIC, SOURCE_RUZICKA_METRIC, SOURCE_SYMM_DIFF_METRIC
]

# Combined list for robust NaN padding and aggregation checks - UPDATED
ALL_METRIC_KEYS_FOR_AGGREGATION = ACTIVATION_METRIC_KEYS + IMAGE_METRIC_KEYS_PREFIXED + SOURCE_METRIC_KEYS

# ----------------------------------------------------------------------------
# Worker Function for Processing a Single Sample - UPDATED
# ----------------------------------------------------------------------------
def process_sample_evaluation(args):
    (pos, sample, originalMostUsedSources, evaluationActivations,
     metricsSampleActivations, all_metric_combinations, closestSources,
     mode
     ) = args

    sample_results = {}
    if metricsSampleActivations is None: return None # Need metric scores
    original_source_ids = set(src_id for src_id, count in originalMostUsedSources)

    for metric_combination in all_metric_combinations:
        combination_str = str(metric_combination)
        current_results = {key: np.nan for key in ALL_METRIC_KEYS_FOR_AGGREGATION}

        try:
            metricSources, layerNumbersToCheck = RENN.identifyClosestSourcesByMetricCombination(
                closestSources, metricsSampleActivations, metric_combination, mode=mode
            )

            if not layerNumbersToCheck:
                for metric_key in ALL_METRIC_KEYS_FOR_AGGREGATION: sample_results[(combination_str, metric_key)] = np.nan
                continue

            mostUsedMetricSources = RENN.getMostUsedSourcesByMetrics(
                metricSources, closestSources, weightedMode=mode
            )

            # Calculate similarities ONLY if mostUsedMetricSources were found
            if mostUsedMetricSources:
                # --- Calculate ALL Source Similarities/Distances ---
                current_results[SOURCE_COSINE_METRIC] = calculate_source_cosine_similarity(originalMostUsedSources, mostUsedMetricSources)
                current_results[SOURCE_LOG_COSINE_METRIC] = calculate_log_cosine_similarity(originalMostUsedSources, mostUsedMetricSources)
                current_results[SOURCE_JSD_METRIC] = calculate_jsd(originalMostUsedSources, mostUsedMetricSources)
                spearman, kendall = calculate_rank_correlation(originalMostUsedSources, mostUsedMetricSources)
                current_results[SOURCE_SPEARMAN_METRIC] = spearman
                current_results[SOURCE_KENDALL_METRIC] = kendall
                intersect_k, precision_k, recall_k = calculate_top_k_overlap(originalMostUsedSources, mostUsedMetricSources, closestSources)
                current_results[SOURCE_INTERSECT_K_METRIC] = intersect_k
                current_results[SOURCE_PRECISION_K_METRIC] = precision_k
                current_results[SOURCE_RECALL_K_METRIC] = recall_k
                # New Metrics
                euclidean, manhattan = calculate_vector_distances(originalMostUsedSources, mostUsedMetricSources)
                current_results[SOURCE_EUCLIDEAN_METRIC] = euclidean
                current_results[SOURCE_MANHATTAN_METRIC] = manhattan
                current_results[SOURCE_RUZICKA_METRIC] = calculate_ruzicka_similarity(originalMostUsedSources, mostUsedMetricSources)
                current_results[SOURCE_SYMM_DIFF_METRIC] = calculate_symmetric_difference_size(originalMostUsedSources, mostUsedMetricSources)


                # --- Calculate Image Similarity ---
                if evaluationActivations is not None:
                    image_sim_dict = evaluateImageSimilarityByMetrics(name="Metrics_MP", combination=combination_str, sample=sample, mostUsed=mostUsedMetricSources, storeGlobally=False)
                    for eval_key, store_suffix in IMAGE_SIM_KEY_MAP.items():
                        if eval_key in image_sim_dict: current_results[f"{IMG_SIM_PREFIX}{store_suffix}"] = image_sim_dict[eval_key]

                    # --- Calculate Activation Similarity ---
                    if layerNumbersToCheck:
                        activation_sim_dict = blendActivations(name=f"metric_combo_{combination_str}", mostUsed=mostUsedMetricSources, evaluationActivations=evaluationActivations, layerNumbersToCheck=layerNumbersToCheck, store_globally=False)
                        for act_key in ACTIVATION_METRIC_KEYS:
                            if act_key in activation_sim_dict: current_results[act_key] = activation_sim_dict[act_key]

            # Store results for this combination (will store NaNs if mostUsedMetricSources was empty)
            for metric_key, value in current_results.items():
                sample_results[(combination_str, metric_key)] = value

        except Exception as e:
            print(f"\n--- WORKER ERROR (PID {os.getpid()}) processing combination {combination_str} for sample {pos} ---")
            for metric_key in ALL_METRIC_KEYS_FOR_AGGREGATION: sample_results[(combination_str, metric_key)] = np.nan
            print(f"--- WORKER (PID {os.getpid()}) continuing ---")

    return sample_results

# ----------------------------------------------------------------------------
# Main Evaluation Function (Multiprocessing Version) - UPDATED
# ----------------------------------------------------------------------------
def evaluate_metric_combinations_overall(mostUsedList, hidden_sizes, closestSources, all_metric_combinations, mode="Sum", max_workers=None):
    start_time = time.time()
    print(f"\nStarting evaluation with multiprocessing (max_workers={max_workers or os.cpu_count()})...")
    print("Metrics: Activation Sim, Image Sim, Source Sims (Cos, LogCos, JSD, Rank, TopK, Dist, Ruzicka, SymmDiff)") # Updated description

    if max_workers is None: max_workers = os.cpu_count(); print(f"Using default max_workers = {max_workers}")

    futures = []; results_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_samples = 0
        # Submit tasks for each sample
        for pos, (sample, _) in enumerate(eval_dataloader):
            if processed_samples >= eval_samples: break
            originalMostUsedSources = mostUsedList[pos] if pos < len(mostUsedList) else []
            evaluationActivations = dictionaryForSourceLayerNeuron[pos]
            metricsSampleActivations = metricsDictionaryForSourceLayerNeuron[pos]

            if metricsSampleActivations is None: continue
            if not isinstance(metricsSampleActivations, np.ndarray):
                try: metricsSampleActivations = np.array(metricsSampleActivations)
                except Exception: print(f"Warning: Skipping sample {pos}, cannot convert metric activations to numpy array."); continue

            try: sample_data = sample.float()
            except Exception as e: print(f"Warning: Sample prep failed {pos}: {e}. Skipping."); continue

            # Update args tuple for the worker - ADDED k_top_overlap
            args = (pos, sample_data, originalMostUsedSources, evaluationActivations,
                    metricsSampleActivations, all_metric_combinations, closestSources, mode
                    )
            # External functions assumed available in worker scope
            futures.append(executor.submit(process_sample_evaluation, args))
            processed_samples += 1

        print(f"Submitted {len(futures)} tasks to {max_workers} workers.")
        # Retrieve results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result();
                if result is not None: results_list.append(result)
            except Exception as e: print(f"\n--- ERROR retrieving result: {e} ---")

    print(f"\nCollected results from {len(results_list)} tasks.")
    if not results_list: print("Error: No results collected."); return None

    # --- Aggregate Results ---
    print("Aggregating results...")
    results_aggregator = defaultdict(lambda: defaultdict(list))
    for sample_result_dict in results_list:
        for (combination_str, metric_key), score in sample_result_dict.items():
            if metric_key in ALL_METRIC_KEYS_FOR_AGGREGATION:
                results_aggregator[combination_str][metric_key].append(score)

    # --- Calculate Averages ---
    final_results = {}
    for combination_str, metric_scores_dict in results_aggregator.items():
        avg_scores = {}
        for metric_key in ALL_METRIC_KEYS_FOR_AGGREGATION:
            scores_list = metric_scores_dict.get(metric_key, [])
            if scores_list: avg_scores[f"avg_{metric_key}"] = np.nanmean([s for s in scores_list if s is not None])
            else: avg_scores[f"avg_{metric_key}"] = np.nan
        final_results[combination_str] = avg_scores

    if not final_results: print("Error: Aggregation failed."); return None

    # --- Create DataFrame ---
    try:
        results_df = pd.DataFrame.from_dict(final_results, orient='index')
        print(f"Aggregated DataFrame shape: {results_df.shape}")
        if results_df.empty: print("Error: DataFrame empty."); return None
    except Exception as e: print(f"Error creating DataFrame: {e}"); return None

    # --- Define Logical Column Order (Updated) ---
    ordered_columns = []
    act_sim_ordered = ['cosine_similarity', 'pearson_correlation', 'kendall_tau', 'spearman_rho', 'jaccard_similarity']
    act_dist_ordered = ['euclidean_distance', 'manhattan_distance', 'hamming_distance']
    img_sim_ordered = ['cosine_sim', 'pearson_corr', 'kendall_tau', 'spearman_rho', 'jaccard_sim']
    img_dist_ordered = ['euclidean_dst', 'manhattan_dst', 'hamming_dst']
    # Source metrics together - UPDATED
    source_sim_ordered = [
        SOURCE_COSINE_METRIC, SOURCE_LOG_COSINE_METRIC, SOURCE_RUZICKA_METRIC, # Count-based Sims
        SOURCE_SPEARMAN_METRIC, SOURCE_KENDALL_METRIC, # Rank Sims
        SOURCE_PRECISION_K_METRIC, SOURCE_RECALL_K_METRIC, SOURCE_INTERSECT_K_METRIC # TopK Sims
    ]
    source_dist_ordered = [
        SOURCE_JSD_METRIC, SOURCE_EUCLIDEAN_METRIC, SOURCE_MANHATTAN_METRIC, # Count/Distro Dists
        SOURCE_SYMM_DIFF_METRIC # Set Dist
    ]
    ordered_columns.extend([f'avg_{m}' for m in act_sim_ordered]); ordered_columns.extend([f'avg_{m}' for m in act_dist_ordered])
    ordered_columns.extend([f'avg_{IMG_SIM_PREFIX}{m}' for m in img_sim_ordered]); ordered_columns.extend([f'avg_{IMG_SIM_PREFIX}{m}' for m in img_dist_ordered])
    ordered_columns.extend([f'avg_{m}' for m in source_sim_ordered]) # Add ALL source similarities
    ordered_columns.extend([f'avg_{m}' for m in source_dist_ordered]) # Add ALL source distances/differences
    remaining_cols = [col for col in results_df.columns if col not in ordered_columns]; ordered_columns.extend(sorted(remaining_cols))
    final_ordered_columns = [col for col in ordered_columns if col in results_df.columns]

    # --- Reindex DataFrame Columns ---
    try: results_df = results_df[final_ordered_columns]; print("Reordered DataFrame columns.")
    except Exception as e: print(f"Warning: Column reordering failed: {e}.")

    # --- Define Row Sorting Priority (Updated) ---
    # Added new source metrics as lower-priority tie-breakers
    metrics_priority_list = [
        'avg_cosine_similarity',
        f'avg_{IMG_SIM_PREFIX}cosine_sim',
        f'avg_{SOURCE_COSINE_METRIC}',
        f'avg_{SOURCE_LOG_COSINE_METRIC}',
        f'avg_{SOURCE_RUZICKA_METRIC}',      # Added Ruzicka (Higher better)
        f'avg_{SOURCE_SPEARMAN_METRIC}',
        f'avg_{SOURCE_PRECISION_K_METRIC}',
        f'avg_{SOURCE_RECALL_K_METRIC}',
        f'avg_{SOURCE_JSD_METRIC}',
        f'avg_{SOURCE_EUCLIDEAN_METRIC}',    # Added Euclidean (Lower better)
        f'avg_{SOURCE_MANHATTAN_METRIC}',    # Added Manhattan (Lower better)
        f'avg_{SOURCE_SYMM_DIFF_METRIC}', # Added SymmDiff (Lower better)
        'avg_euclidean_distance',
        f'avg_{IMG_SIM_PREFIX}euclidean_dst',
        f'avg_{SOURCE_KENDALL_METRIC}',
        f'avg_{SOURCE_INTERSECT_K_METRIC}',
        'avg_kendall_tau', # Activation kendall
    ]
    # Updated keywords for sorting direction inference
    similarity_keywords = ['similarity', 'correlation', 'tau', 'rho', 'spearman', 'kendall', 'precision', 'recall', 'intersect', 'ruzicka']
    distance_keywords = ['distance', 'dst', 'jsd', 'euclidean', 'manhattan', 'diff'] # Added more distance types

    # --- Prepare Row Sorting Parameters (Updated Logic) ---
    sort_by_columns = []; sort_ascending_flags = []; sort_descriptions = []
    for metric_col in metrics_priority_list:
        if metric_col not in results_df.columns: continue
        ascending_order = False; sort_type = " (Desc)"; metric_col_lower = metric_col.lower()
        is_similarity = any(keyword in metric_col_lower for keyword in similarity_keywords)
        is_distance = any(keyword in metric_col_lower for keyword in distance_keywords)

        # Explicit checks first for distances/lower-is-better
        if SOURCE_JSD_METRIC in metric_col_lower \
                or SOURCE_EUCLIDEAN_METRIC in metric_col_lower \
                or SOURCE_MANHATTAN_METRIC in metric_col_lower \
                or SOURCE_SYMM_DIFF_METRIC in metric_col_lower \
                or is_distance:
            ascending_order = True; sort_type = " (Asc)"
        # Otherwise assume higher is better (covers similarities and rank correlations etc.)
        elif is_similarity:
            ascending_order = False; sort_type = " (Desc)"
        # else: keep default (Desc) for anything else

        sort_by_columns.append(metric_col); sort_ascending_flags.append(ascending_order)
        sort_descriptions.append(f"{metric_col.replace('avg_', '')}{sort_type}")

    # --- Perform Final Row Sort ---
    if not sort_by_columns: print("\nWarning: No valid columns for sorting rows."); final_df_to_return = results_df
    else:
        print(f"\n--- Final Ranking Sorted By Rows: {' -> '.join(sort_descriptions)} ---")
        try: final_df_to_return = results_df.sort_values(by=sort_by_columns, ascending=sort_ascending_flags, na_position='last')
        except Exception as e: print(f"\nError during final row sorting: {e}."); final_df_to_return = results_df

    # --- Print Preview and Export ---
    pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', None); pd.set_option('display.width', 2000); pd.set_option('display.max_colwidth', None)
    print("\n--- Top Results Preview (Full results in CSV) ---"); print(final_df_to_return)
    
    # --- Generate Timestamped Filename ---
    local_time_struct = time.localtime()
    formatted_time = time.strftime("%Y%m%d_%H%M%S", local_time_struct) # Format for filename
    # Example: Construct filename with parameters (ensure variables like train_samples exist)
    try:
        filename_params = f"E{eval_samples}_T{train_samples}_L{hidden_sizes[0][1]}"
        output_csv_filename = f"{formatted_time}-{filename_params}_metrics.csv"
    except NameError:
        print("Warning: Could not create detailed filename, using default.")
        output_csv_filename = f"{formatted_time}_metric_evaluation_results.csv" # Fallback


    # --- Export Full Results to CSV ---
    try:
        final_df_to_return.to_csv(output_csv_filename)
        print(f"\nFull results successfully exported to: {output_csv_filename}")
    except Exception as e:
        print(f"\nWarning: Failed to export results to CSV file '{output_csv_filename}': {e}")

    end_time = time.time()
    print(f"\nEvaluation complete in {end_time - start_time:.2f} seconds.")
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
    kendall_tau, spearman_rho = None, None
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

# Weighting Test Area

def evaluate_pareto():
    # --- Indices of metrics in the tuple (must match evaluateImageSimilarityByMetrics) ---
    IDX_COMBO = 0
    IDX_COS_SIM = 1
    IDX_EUC_DIST = 2
    IDX_MAN_DIST = 3
    IDX_JAC_SIM = 4
    IDX_HAM_DIST = 5
    IDX_PEARSON = 6
    IDX_KENDALL = 7
    IDX_SPEARMAN = 8

    # --- Configuration for Pareto Analysis ---

    # 1. Define Objectives: Map metric index to 'higher_better' or 'lower_better'
    #    *** ADJUST THIS DICTIONARY based on the metrics YOU want to consider ***
    objectives = {
        IDX_COS_SIM: 'higher_better',
        IDX_EUC_DIST: 'lower_better',
        IDX_MAN_DIST: 'lower_better',
        # IDX_JAC_SIM: 'higher_better', # Uncomment if relevant & reliable
        # IDX_HAM_DIST: 'lower_better', # Uncomment if relevant & reliable
        IDX_PEARSON: 'higher_better',
        IDX_KENDALL: 'higher_better',
        IDX_SPEARMAN: 'higher_better'
    }
    objective_indices = list(objectives.keys())

    def dominates(result_a, result_b, objectives):
        """Checks if result_a dominates result_b based on the defined objectives."""
        a_is_strictly_better_on_any = False
        for idx, type in objectives.items():
            try:
                val_a = result_a[idx]
                val_b = result_b[idx]
            except IndexError:
                return False # Data malformed

            if val_a is None and val_b is None:
                continue # Equal Nones, proceed
            elif val_a is None: # 'a' has None, 'b' does not. 'a' cannot be >= 'b'.
                return False
            elif val_b is None: # 'b' has None, 'a' does not. Proceed checking others.
                pass

            # Check if 'a' is WORSE than 'b'
            is_worse = False
            if type == 'higher_better' and (val_a is None or (val_b is not None and val_a < val_b)):
                is_worse = True
            elif type == 'lower_better' and (val_a is None or (val_b is not None and val_a > val_b)):
                is_worse = True

            # Handle case where only val_b is None (a is better if higher_better, worse if lower_better)
            if val_b is None and val_a is not None:
                if type == 'higher_better': # a is better than None
                    is_strictly_better = True
                else: # a is worse than None (lower should be better)
                    is_worse = True # Treat non-None as worse than None for lower_better objective

            if is_worse:
                return False

            # Check if 'a' is STRICTLY BETTER than 'b' (only if both not None)
            is_strictly_better = False
            if val_a is not None and val_b is not None:
                if type == 'higher_better' and val_a > val_b:
                    is_strictly_better = True
                elif type == 'lower_better' and val_a < val_b:
                    is_strictly_better = True

            if is_strictly_better:
                a_is_strictly_better_on_any = True

        return a_is_strictly_better_on_any

    # --- Find the Pareto Front Directly from best_image_similarity ---

    def find_pareto_front(results_list, objectives_def):
        # 1. Filter out basic errors if necessary (e.g., if strings were added)
        #    Adjust this check based on how errors might appear in your list.
        valid_results = [res for res in results_list if isinstance(res[1], (int, float, type(None)))]
        # Or if errors are marked differently:
        # valid_results = [res for res in results_list if res[0] != 'ERROR_MARKER']

        if not valid_results:
            print("No valid results found in the input list.")
            return []

        num_results = len(valid_results)
        is_non_dominated = [True] * num_results # Assume all are non-dominated initially

        print(f"\n--- Finding Pareto Front ({len(objectives_def)} objectives) ---")
        print(f"Objectives (Index: Type): {objectives_def}")
        print(f"Processing {num_results} valid combinations...")

        for i in range(num_results):
            # Optimization: if i is already known to be dominated, it can't dominate others effectively
            # if not is_non_dominated[i]:
            #     continue

            for j in range(num_results):
                if i == j:
                    continue

                # Check if solution j dominates solution i
                try:
                    if dominates(valid_results[j], valid_results[i], objectives_def):
                        is_non_dominated[i] = False # Solution i is dominated
                        break # No need to check other potential dominators for i
                except Exception as e:
                    print(f"Error during dominance check between {valid_results[j][0]} and {valid_results[i][0]}: {e}")
                    # Decide how to handle comparison errors - e.g., assume i is not dominated by j here
                    pass


        # Collect the results that are non-dominated
        pareto_front_results = [valid_results[i] for i in range(num_results) if is_non_dominated[i]]
        return pareto_front_results

    # Make sure best_image_similarity is populated before calling this
    if 'best_image_similarity' in globals() and best_image_similarity:
        pareto_front = find_pareto_front(best_image_similarity, objectives)

        # --- Display the Pareto Front ---
        print(f"\n--- Pareto Optimal Set ({len(pareto_front)} combinations) ---")
        if not pareto_front:
            print("No non-dominated solutions found.")
        else:
            print("These combinations represent the best trade-offs based on the selected objectives:")
            # Sort alphabetically by combo name for consistent output (optional)
            # Prepare mappings needed (assuming constants and RENN.POTENTIAL_METRICS are available globally)
        metric_keys_list = list(RENN.POTENTIAL_METRICS.keys())

        # Create a reverse map from index to name (needed for output dict)
        # Also map index to simple direction string
        index_to_metric_name = {}
        index_to_direction = {}

        # Use objectives dictionary as the source of targets
        for idx, direction_type in objectives.items():
            # Find the name corresponding to the index (requires checking all IDX_* constants)
            # This is a bit verbose but necessary without a predefined reverse map
            name = f"Unknown Metric {idx}" # Default
            if idx == IDX_COS_SIM: name = "Cosine Sim"
            elif idx == IDX_EUC_DIST: name = "Euclidean Dst"
            elif idx == IDX_MAN_DIST: name = "Manhattan Dst"
            elif idx == IDX_JAC_SIM: name = "Jaccard Sim"
            elif idx == IDX_HAM_DIST: name = "Hamming Dst"
            elif idx == IDX_PEARSON: name = "Pearson Corr"
            elif idx == IDX_KENDALL: name = "Kendall Tau"
            elif idx == IDX_SPEARMAN: name = "Spearman Rho"
            # Add more elif branches if you have other IDX_* constants

            index_to_metric_name[idx] = name

            # Convert 'higher_better'/'lower_better' to 'maximize'/'minimize'
            if direction_type == 'higher_better':
                index_to_direction[idx] = 'maximize'
            elif direction_type == 'lower_better':
                index_to_direction[idx] = 'minimize'
            else:
                index_to_direction[idx] = 'unknown' # Handle unexpected values

        # --- Generate New Optimizations List ---
        # This list will contain the newly generated optimization configs
        generated_optimizations_to_run = []

        # Iterate through each objective defined for the Pareto analysis
        for target_idx, direction_type in objectives.items():

            target_metric_name = index_to_metric_name.get(target_idx, f"Unknown Metric {target_idx}")
            direction = index_to_direction.get(target_idx, 'unknown')

            if direction == 'unknown':
                print(f"Warning: Skipping objective index {target_idx} due to unknown direction type '{direction_type}'")
                continue

            best_solution_for_target = None
            best_value = None

            # Set initial best value based on optimization direction
            if direction == "maximize":
                best_value = -float('inf')
            else: # Minimize
                best_value = float('inf')

            # Find the best solution IN THE PARETO FRONT for this specific objective index
            for solution in pareto_front:
                # Direct access, assuming valid solution structure and index
                # Ignoring detailed checks as requested
                try:
                    current_value = solution[target_idx]

                    # Determine if current solution is better
                    is_better = False
                    if direction == "maximize" and current_value > best_value:
                        is_better = True
                    elif direction == "minimize" and current_value < best_value:
                        is_better = True

                    # Update best if current is better
                    if is_better:
                        best_value = current_value
                        best_solution_for_target = solution
                except (IndexError, TypeError):
                    # Minimal handling if direct access fails
                    pass # Continue to next solution

            # --- Format the result dictionary if a best solution was found ---
            if best_solution_for_target:
                # Extract combination indices from the best found solution
                combo_indices = best_solution_for_target[IDX_COMBO]
                # Map indices to names (assuming indices are valid)
                try:
                    found_combination_names = [metric_keys_list[i] for i in combo_indices]
                except IndexError:
                    # Basic error handling for invalid indices in the combo
                    print(f"Warning: Invalid index found in combination {combo_indices} for target {target_metric_name}. Skipping this target.")
                    continue # Skip appending this flawed entry


                # Create the dictionary in the specified format
                # Using the metric index (target_idx) as the 'id' for traceability
                optimization_dict = {
                    "id": target_idx,
                    "target_metric": target_metric_name,
                    "direction": direction, # 'maximize' or 'minimize'
                    "combination_names": found_combination_names
                }
                # Add the generated optimization config to the list
                generated_optimizations_to_run.append(optimization_dict)
            # else: If no best solution found (e.g., errors, empty pareto_front),
            # simply no dictionary is added for this objective index.

        # The variable 'generated_optimizations_to_run' now holds the list
        # of dictionaries, formatted as requested, based on the best performers
        # from the Pareto front for each objective.
        # You can now use this list for your next steps.

        # Example: Print the generated list
        print("\n--- Generated Optimizations based on Best Pareto Solutions ---")
        print(generated_optimizations_to_run)

        return generated_optimizations_to_run

    else:
        print("The global list 'best_image_similarity' is not defined or is empty.")
        print("Please run 'evaluateImageSimilarityByMetrics' first to populate it.")

def optimize_weights_of_best_combinations(closestSources, optimizations_to_run):
    # Get the ordered list of all metric names once
    if not hasattr(RENN, 'POTENTIAL_METRICS') or not isinstance(RENN.POTENTIAL_METRICS, dict):
        print("Error: RENN.POTENTIAL_METRICS is not defined or not a dictionary.")
        return None # Or raise an error
    all_metric_names_list = list(RENN.POTENTIAL_METRICS.keys())

    # Convert combination names to indices
    for config in optimizations_to_run:
        try:
            config["combination_indices"] = tuple(all_metric_names_list.index(name) for name in config["combination_names"])
        except ValueError as e:
            print(f"Error: Metric name '{e}' in combination ID {config['id']} not found in global METRICS keys.")
            config["combination_indices"] = None

    def objective(trial, combination_indices, metric_names_to_weight, target_metric):
        # --- This objective function remains the same as your corrected version ---
        # --- It includes the checks for inputs and return values ---
        if not combination_indices:
            print("Skipping trial due to invalid combination indices.")
            raise optuna.exceptions.TrialPruned()

        metric_weights = {}
        for metric_name in metric_names_to_weight:
            metric_weights[metric_name] = trial.suggest_float(f"weight_{metric_name}", 0.0, 2.0)

        all_final_scores = []
        # Ensure eval_dataloader is defined and accessible
        if 'eval_dataloader' not in globals() and 'eval_dataloader' not in locals():
            print("Error: eval_dataloader not found.")
            raise optuna.exceptions.TrialPruned("eval_dataloader is missing.")

        for pos, data_batch in enumerate(eval_dataloader):
            if isinstance(data_batch, (list, tuple)) and len(data_batch) >= 1:
                sample = data_batch[0]
            else:
                # print(f"Warning: Unexpected data format in eval_dataloader at pos {pos}. Skipping batch.")
                continue

            try:
                if pos not in metricsDictionaryForSourceLayerNeuron:
                    # print(f"Warning: No pre-computed metrics found for sample index {pos}. Skipping sample.")
                    continue

                metricsOutputs = metricsDictionaryForSourceLayerNeuron[pos]

                metricSourcesSum, layerNumbersToCheck = RENN.identifyClosestSourcesByMetricCombination(
                    closestSources,
                    metricsOutputs,
                    combination_indices,
                    metric_weights,
                    "Sum"
                )
                mostUsedMetricSourcesWithSum = RENN.getMostUsedSourcesByMetrics(
                    metricSourcesSum,
                    closestSources,
                    weightedMode="Sum",
                    info=False
                )

                if not mostUsedMetricSourcesWithSum:
                    # print(f"Warning: No 'most used sources' found for sample {pos}. Skipping evaluation call.")
                    continue

                final_eval_scores = evaluateImageSimilarityByMetrics(
                    "Metrics",
                    combination_indices,
                    sample,
                    mostUsedMetricSourcesWithSum
                )

                if final_eval_scores is None:
                    # print(f"Warning: evaluateImageSimilarityByMetrics returned None for sample {pos}. Skipping score extraction.")
                    continue

                score = final_eval_scores.get(target_metric)

                if score is None:
                    # print(f"Warning: Target metric '{target_metric}' not found in evaluateImageSimilarityByMetrics output dictionary for sample {pos}.")
                    continue
                if not isinstance(score, (int, float)):
                    # print(f"Warning: Score for {target_metric} is not numeric: {score}. Skipping sample.")
                    continue
                if np.isnan(score):
                    # print(f"Warning: Score for {target_metric} is NaN. Skipping sample.")
                    continue

                all_final_scores.append(score)

            except Exception as e:
                print(f"Error during evaluation processing for sample {pos} in trial {trial.number}: {e}")
                # print(traceback.format_exc()) # Uncomment for detailed traceback inside objective
                continue

        if not all_final_scores:
            # print(f"Warning: No valid scores collected for target '{target_metric}' in trial {trial.number}. Pruning.")
            raise optuna.exceptions.TrialPruned("No valid scores obtained across samples.")

        average_performance = np.mean(all_final_scores)

        if np.isnan(average_performance):
            # print(f"Warning: Average performance is NaN for target '{target_metric}' in trial {trial.number}. Pruning.")
            raise optuna.exceptions.TrialPruned("Average performance is NaN.")

        return average_performance


    # --- 3. Run the Optimization Loop ---
    N_TRIALS = 100
    all_best_results = {}
    #print(f"\nStarting optimization runs ({N_TRIALS} trials each)...")

    for config in optimizations_to_run:
        if config["combination_indices"] is None:
            print(f"\nSkipping optimization for Combination ID {config['id']} due to invalid indices.")
            continue

        print(f"\n--- Optimizing for: {config['target_metric']} (ID: {config['id']}, Direction: {config['direction']}) ---")
        print(f"Combination: {config['combination_names']}")
        start_time = time.time()
        study = optuna.create_study(direction=config['direction'])

        try:
            # --- Run the optimization ---
            study.optimize(
                lambda trial: objective(
                    trial,
                    config["combination_indices"],
                    config["combination_names"],
                    config["target_metric"]
                ),
                n_trials=N_TRIALS,
                # catch=(Exception,) # Consider using this to catch unexpected errors during optimize
            )

            # --- Process results IF optimization didn't raise an exception ---
            end_time = time.time()
            duration = end_time - start_time

            best_trial = study.best_trial # Get the best trial object

            best_weights_raw = {k: v for k, v in sorted(best_trial.params.items())}
            best_weights_clean = {k.replace('weight_', ''): v for k, v in best_weights_raw.items()}

            # === V V V === CORRECTED TRIAL COUNTING === V V V ===
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            # === ^ ^ ^ === CORRECTED TRIAL COUNTING === ^ ^ ^ ===

            all_best_results[f"ID_{config['id']}_{config['target_metric']}"] = {
                "status": "Success", # Mark as Success
                "target_metric": config['target_metric'],
                "direction": config['direction'],
                "best_score": best_trial.value,
                "best_weights": best_weights_clean,
                "combination_names": config['combination_names'],
                "duration_seconds": duration,
                # === V V V === USE CORRECTED COUNTS === V V V ===
                "n_trials_completed": len(completed_trials),
                "n_trials_failed": len(failed_trials),
                "n_trials_pruned": len(pruned_trials),
                # === ^ ^ ^ === USE CORRECTED COUNTS === ^ ^ ^ ===
            }

            print(f"Optimization completed in {duration:.2f} seconds.")
            print(f"Best Score ({config['target_metric']}): {best_trial.value:.6f}")
            print("Best Weights:")
            for name, weight in best_weights_clean.items():
                print(f"  {name}: {weight:.4f}")
            # === V V V === PRINT CORRECTED COUNTS === V V V ===
            #print(f"Trials: {len(completed_trials)} Complete, {len(pruned_trials)} Pruned, {len(failed_trials)} Failed.")
            # === ^ ^ ^ === PRINT CORRECTED COUNTS === ^ ^ ^ ===

        except Exception as e:
            # --- Handle errors DURING study.optimize or result processing ---
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n!!! Optimization FAILED for ID {config['id']} ({config['target_metric']}) after {duration:.2f}s: {e} !!!")

            all_best_results[f"ID_{config['id']}_{config['target_metric']}"] = {
                "status": "Failed",
                "error": str(e),
                "target_metric": config['target_metric'],
                "direction": config['direction'],
                "combination_names": config['combination_names'],
                "duration_seconds": duration,
                # Store partial trial counts if study object exists
                "n_trials_run": len(study.trials) if study else 0,
            }


    # --- 4. Display Final Summary ---
    print("\n\n--- === Overall Optimization Summary === ---")
    for key, result in all_best_results.items():
        print(f"\nResult for: {key}")
        if result.get("status") == "Failed":
            print(f"  Status: FAILED")
            print(f"  Error: {result['error']}")
            print(f"  Target Metric: {result.get('target_metric', 'N/A')} ({result.get('direction', 'N/A')})")
            print(f"  Combination: {result.get('combination_names', 'N/A')}")
            print(f"  Duration: {result.get('duration_seconds', -1):.2f}s")
            print(f"  Trials Run Attempted: {result.get('n_trials_run', 'N/A')}") # Show attempted trials if failed
        elif result.get("status") == "Success":
            print(f"  Status: Success")
            print(f"  Target Metric: {result['target_metric']} ({result['direction']})")
            print(f"  Best Score Achieved: {result['best_score']:.6f}")
            print(f"  Combination: {result['combination_names']}")
            print(f"  Optimal Weights Found:")
            if 'best_weights' in result:
                for name, weight in result['best_weights'].items():
                    print(f"    {name}: {weight:.4f}")
            else:
                print("    (No weights available)") # Should not happen on success
            print(f"  Duration: {result['duration_seconds']:.2f}s")
            # === V V V === USE CORRECTED COUNT KEYS === V V V ===
            print(f"  Trials: {result.get('n_trials_completed', 'N/A')} Complete, {result.get('n_trials_pruned', 'N/A')} Pruned, {result.get('n_trials_failed', 'N/A')} Failed.")
            # === ^ ^ ^ === USE CORRECTED COUNT KEYS === ^ ^ ^ ===
        else:
            print(f"  Status: {result.get('status', 'Unknown')}")

    return all_best_results, optimizations_to_run

def synthesize_overall_weights(all_best_results):
    print("\n--- Synthesizing Overall Weights from Optimization Results ---")

    # Dictionary to store lists of weights for each metric name
    metric_weight_data = defaultdict(list)
    successful_runs = 0

    # --- 1. Aggregate Weights from Successful Runs ---
    for key, result in all_best_results.items():
        if result.get("status") == "Success" and 'best_weights' in result:
            successful_runs += 1
            # Iterate through the weights found in this successful run
            for metric_name, weight in result['best_weights'].items():
                # Only aggregate weights for metrics that were actually optimized (non-zero suggested range)
                # In our case, all metrics in the combination got weights suggested between 0.0 and 2.0
                metric_weight_data[metric_name].append(weight)
        else:
            print(f"Skipping result '{key}' as it did not succeed or lacks weights.")

    if successful_runs == 0:
        print("Error: No successful optimization runs found in the input results.")
        return None, None, None, None

    # --- 2. Calculate Weight Statistics ---
    weight_stats = []
    all_metrics_encountered = sorted(metric_weight_data.keys())

    for metric_name in all_metrics_encountered:
        weights = metric_weight_data[metric_name]
        if weights: # Ensure list is not empty
            stats = {
                'Metric': metric_name,
                'Count': len(weights), # How many successful runs included this metric's weights
                'Mean Weight': np.mean(weights),
                'Std Dev': np.std(weights),
                'Min Weight': np.min(weights),
                'Max Weight': np.max(weights)
            }
            weight_stats.append(stats)

    if not weight_stats:
        print("Error: No valid weight data collected.")
        return None, None, None, None

    stats_df = pd.DataFrame(weight_stats).set_index('Metric')
    stats_df = stats_df.sort_values(by='Mean Weight', ascending=False) # Sort by average importance

    print("\n--- Weight Statistics Across Successful Runs ---")
    print(stats_df.to_string(float_format="%.4f"))

    # --- 3. Identify Potentially Less Important Metrics ---
    # Heuristic: Flag metrics where the *maximum* weight assigned was also low,
    # suggesting the optimizer never found it very useful across objectives.
    # Also flag metrics with very low average weight.
    # Adjust thresholds as needed.
    max_weight_threshold = 0.20 # e.g., never received weight > 0.2
    avg_weight_threshold = 0.10 # e.g., average weight < 0.1

    flagged_metrics = []
    for index, row in stats_df.iterrows():
        metric_name = index # Get metric name from index
        if row['Max Weight'] < max_weight_threshold:
            print(f"Flagging '{metric_name}': Maximum weight ({row['Max Weight']:.4f}) was below threshold {max_weight_threshold}.")
            flagged_metrics.append(metric_name)
        elif row['Mean Weight'] < avg_weight_threshold and metric_name not in flagged_metrics:
            print(f"Flagging '{metric_name}': Mean weight ({row['Mean Weight']:.4f}) was below threshold {avg_weight_threshold}.")
            flagged_metrics.append(metric_name)


    # --- 4. Propose Overall Weighting Scheme (Simple Average) ---
    # Create a dictionary using the calculated mean weights
    overall_weights_avg = stats_df['Mean Weight'].to_dict()

    print("\n--- Suggested Overall Weighting (Based on Average) ---")
    # Print sorted by weight
    for metric, weight in sorted(overall_weights_avg.items(), key=lambda item: item[1], reverse=True):
        print(f"  {metric}: {weight:.4f}")

    # Optional: Create a version where flagged metrics are set to 0
    # overall_weights_avg_filtered = overall_weights_avg.copy()
    # for metric in flagged_metrics:
    #     overall_weights_avg_filtered[metric] = 0.0
    # print("\n--- Suggested Overall Weighting (Flagged Metrics Zeroed) ---")
    # # Print sorted by weight
    # for metric, weight in sorted(overall_weights_avg_filtered.items(), key=lambda item: item[1], reverse=True):
    #     print(f"  {metric}: {weight:.4f}")


    print(f"\n--- Metrics Flagged as Potentially Less Important (Low Max/Avg Weight) ---")
    if flagged_metrics:
        for fm in flagged_metrics:
            print(f"  - {fm}")
    else:
        print("  None based on current thresholds.")

    print("\nNote: 'Less Important' is based on consistently low weights assigned by the optimizer across different objectives.")
    print("      It does not strictly prove redundancy. Consider domain knowledge and metric correlations.")

    return stats_df, overall_weights_avg, flagged_metrics, metric_weight_data

def find_overall_weights_via_moo(
        closestSources,
        optimizations_to_consider, # Pass the list of dictionaries defining the scenarios
        n_trials=300 # Might need more trials for a larger combination
):
    print(f"\n--- Finding Overall Weights via MOO on Union Combination ---")
    if not optimizations_to_consider or not isinstance(optimizations_to_consider, list):
        print("Error: Please provide the list of optimization scenario dictionaries.")
        return None, None, None, None

    # --- 1. Create the Union Combination ---
    unique_metric_names = OrderedDict() # Use OrderedDict to preserve insertion order somewhat
    print("Aggregating metrics from provided scenarios:")
    for config in optimizations_to_consider:
        combo_id = config.get('id', 'N/A')
        if "combination_names" in config and isinstance(config["combination_names"], list):
            # print(f"  Processing Combo ID: {combo_id}") # Optional debug print
            for name in config["combination_names"]:
                unique_metric_names[name] = None # Use dict keys for uniqueness
        else:
            print(f"  Warning: Skipping config ID {combo_id} due to missing/invalid 'combination_names'.")

    union_combination_names = list(unique_metric_names.keys())

    if not union_combination_names:
        print("Error: No valid metric names found to create a union combination.")
        return None, None, None, None

    print(f"\nCreated union combination with {len(union_combination_names)} unique metrics:")
    print(f"{union_combination_names}")
    print(f"Number of trials for MOO: {n_trials}")


    # --- 2. Define Objectives and Directions ---
    target_metrics = [
        'Cosine Sim', 'Euclidean Dst', 'Manhattan Dst',
        'Pearson Corr', 'Kendall Tau', 'Spearman Rho'
    ]
    directions = [
        'maximize', 'minimize', 'minimize', 'maximize', 'maximize', 'maximize'
    ]
    n_objectives = len(target_metrics)
    print(f"\nOptimizing for {n_objectives} objectives: {target_metrics}")
    print(f"Respective directions: {directions}")

    # --- 3. Prepare Combination Indices for the Union Combo ---
    study = None
    if not hasattr(RENN, 'POTENTIAL_METRICS') or not isinstance(RENN.POTENTIAL_METRICS, dict):
        print("Error: RENN.POTENTIAL_METRICS is not defined or not a dictionary.")
        return None, None, None, None
    all_metric_names_list = list(RENN.POTENTIAL_METRICS.keys())
    try:
        union_combination_indices = tuple(all_metric_names_list.index(name) for name in union_combination_names)
    except ValueError as e:
        print(f"Error: Metric name '{e}' (from union combination) not found in RENN.POTENTIAL_METRICS keys.")
        return None, None, None, None

    metric_names_to_weight = union_combination_names

    # --- 4. Define MOO Objective Function (moo_objective_union) ---
    def moo_objective_union(trial):
        metric_weights = {name: trial.suggest_float(f"weight_{name}", 0.0, 2.0)
                          for name in metric_names_to_weight}
        batch_scores = defaultdict(list)
        if 'eval_dataloader' not in globals() and 'eval_dataloader' not in locals():
            raise optuna.exceptions.TrialPruned("eval_dataloader is missing.")

        for pos, data_batch in enumerate(eval_dataloader):
            if isinstance(data_batch, (list, tuple)) and len(data_batch) >= 1:
                sample = data_batch[0]
            else: continue

            try:
                if pos not in metricsDictionaryForSourceLayerNeuron: continue
                metricsOutputs = metricsDictionaryForSourceLayerNeuron[pos]

                # Use UNION indices and suggested weights
                metricSourcesSum, _ = RENN.identifyClosestSourcesByMetricCombination(
                    closestSources, metricsOutputs, union_combination_indices, metric_weights, "Sum"
                )
                mostUsed = RENN.getMostUsedSourcesByMetrics(
                    metricSourcesSum, closestSources, weightedMode="Sum", info=False
                )
                if not mostUsed: continue

                # Evaluate using UNION indices
                eval_scores_dict = evaluateImageSimilarityByMetrics(
                    "Metrics-MOO-Union", union_combination_indices, sample, mostUsed
                )
                if eval_scores_dict is None: continue

                valid_sample = True; temp_scores = []
                for metric_name in target_metrics:
                    score = eval_scores_dict.get(metric_name)
                    if score is None or not isinstance(score, (int, float)) or np.isnan(score):
                        valid_sample = False; break
                    temp_scores.append(score)
                if valid_sample:
                    for i, metric_name in enumerate(target_metrics):
                        batch_scores[metric_name].append(temp_scores[i])
            except Exception as e:
                # Be less verbose during optimization, only print if needed for debug
                # print(f"Error in objective sample {pos} trial {trial.number}: {e}")
                pass # Continue silently on sample error

        average_scores = []
        valid_trial = True
        for metric_name in target_metrics:
            scores_list = batch_scores[metric_name]
            if not scores_list: valid_trial = False; break
            avg_score = np.mean(scores_list)
            if np.isnan(avg_score): valid_trial = False; break
            average_scores.append(avg_score)
        if not valid_trial:
            raise optuna.exceptions.TrialPruned("Failed valid averages.")
        return tuple(average_scores)

    # --- 5. Create and Run MOO Study ---
    start_time = time.time()
    study_name = f"MOO_Study_UnionCombo_{int(time.time())}" # Add timestamp for uniqueness
    try:
        sampler = optuna.samplers.NSGAIISampler()
        study = optuna.create_study(study_name=study_name, directions=directions, sampler=sampler, load_if_exists=False)
        study.optimize(moo_objective_union, n_trials=n_trials)
        status = "Success"
    except Exception as e:
        end_time = time.time(); duration = end_time - start_time
        print(f"\n!!! MOO Optimization FAILED after {duration:.2f}s: {e} !!!")
        return None, None, study, union_combination_names

    # --- 6. Process Results ---
    end_time = time.time(); duration = end_time - start_time
    print(f"\nMOO study completed in {duration:.2f} seconds.")
    pareto_df = None; weight_stats_df = None

    if status == "Success" and study.best_trials:
        print(f"\nProcessing {len(study.best_trials)} Pareto-optimal solutions for the UNION combination...")
        solution_data = []; weights_data = defaultdict(list)
        try: # Add try-except around accessing potentially non-existent params/values
            metric_names_from_params = [p.replace('weight_', '') for p in study.best_trials[0].params.keys()]
            for trial in study.best_trials:
                solution = {'Trial': trial.number}
                for i, metric_name in enumerate(target_metrics):
                    solution[f'Score_{metric_name}'] = trial.values[i]
                for param_name, weight in trial.params.items():
                    metric_name = param_name.replace('weight_', '')
                    solution[f'Weight_{metric_name}'] = weight
                    weights_data[metric_name].append(weight)
                solution_data.append(solution)

            pareto_df = pd.DataFrame(solution_data)
            print("\n--- Pareto Optimal Solutions (Subset Shown) for UNION Combination ---")
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
                print(pareto_df)

            weight_stats_list = []
            for metric_name in metric_names_from_params:
                weights = weights_data[metric_name]
                if weights:
                    stats = {'Metric': metric_name, 'Mean Weight': np.mean(weights), 'Std Dev': np.std(weights),
                             'Min Weight': np.min(weights), 'Max Weight': np.max(weights)}
                    weight_stats_list.append(stats)
            if weight_stats_list:
                weight_stats_df = pd.DataFrame(weight_stats_list).set_index('Metric')
                weight_stats_df = weight_stats_df.sort_values(by='Mean Weight', ascending=False)
                print("\n--- Weight Statistics Across Pareto Front (UNION Combination) ---")
                print(weight_stats_df.to_string(float_format="%.4f"))
            else: print("\nWarning: Could not generate weight statistics.")
        except Exception as e:
            print(f"Error during results processing: {e}")

    elif status == "Success":
        print("\nWarning: Optimization reported success but no best trials found.")

    # --- 7. Return Results ---
    print("\n--- MOO Function Finished ---")
    return pareto_df, weight_stats_df, study, union_combination_names

def select_balanced_solution_from_pareto(pareto_df, target_metrics, directions):
    if pareto_df is None or pareto_df.empty:
        print("Error: Input Pareto DataFrame is empty or None.")
        return None, None
    if len(target_metrics) != len(directions):
        print("Error: target_metrics and directions lists must have the same length.")
        return None, None

    print("\n--- Selecting Balanced Solution (Closest to Ideal Point) ---")
    normalized_scores = pd.DataFrame(index=pareto_df.index)
    ideal_point = {} # Stores best possible value for each objective

    # --- 1 & 2: Find Ideal and Normalize ---
    print("Normalizing objective scores...")
    for i, metric in enumerate(target_metrics):
        score_col = f'Score_{metric}'
        if score_col not in pareto_df.columns:
            print(f"Error: Score column '{score_col}' not found in Pareto DataFrame.")
            return None, None

        scores = pareto_df[score_col]
        min_val = scores.min()
        max_val = scores.max()

        # Handle case where all scores are the same for an objective
        if np.isclose(min_val, max_val):
            # If min==max, objective doesn't distinguish solutions.
            # Assign a neutral normalized value (e.g., 0.5 or 1 if max is better)
            if directions[i] == 'maximize':
                normalized_scores[metric] = 1.0 if not np.isnan(min_val) else 0.0
                ideal_point[metric] = 1.0
            else: # minimize
                normalized_scores[metric] = 1.0 if not np.isnan(min_val) else 0.0 # Normalized best is 1
                ideal_point[metric] = 1.0
            print(f"  Metric '{metric}': All values identical ({min_val:.4f}). Normalized to 1.0.")
            continue


        if directions[i] == 'maximize':
            # Normalize so 1 is best (max_val)
            normalized_scores[metric] = (scores - min_val) / (max_val - min_val)
            ideal_point[metric] = 1.0 # Ideal is max (normalized to 1)
        else: # minimize
            # Normalize and invert so 1 is best (min_val)
            normalized_scores[metric] = (max_val - scores) / (max_val - min_val)
            ideal_point[metric] = 1.0 # Ideal is min (normalized to 1)

        # Replace potential NaN from division by zero if any score was NaN initially
        normalized_scores[metric] = normalized_scores[metric].fillna(0)


    # --- 3. Calculate Distance from Ideal ---
    # Ideal point in normalized space is [1, 1, ..., 1]
    distances = np.sqrt(np.sum((1.0 - normalized_scores[target_metrics])**2, axis=1))

    # --- 4. Select Closest ---
    closest_idx = distances.idxmin() # Get the index of the row with the minimum distance
    selected_solution_row = pareto_df.loc[closest_idx]

    print(f"\nSelected solution at index {closest_idx} (minimum distance {distances.min():.4f} to ideal point).")
    print("Selected Solution Scores & Weights:")
    # Select only Score and Weight columns for clarity
    score_cols = [f'Score_{m}' for m in target_metrics]
    weight_cols = [col for col in pareto_df.columns if col.startswith('Weight_')]
    print(selected_solution_row[score_cols + weight_cols])


    # --- Extract Final Weights ---
    final_weights = {col.replace('Weight_', ''): selected_solution_row[col]
                     for col in pareto_df.columns if col.startswith('Weight_')}

    print("\n--- Final Overall Weights (Closest to Ideal) ---")
    # Print sorted by weight
    for metric, weight in sorted(final_weights.items(), key=lambda item: item[1], reverse=True):
        print(f"  {metric}: {weight:.4f}")

    return final_weights, closest_idx

def evaluate_metric_importance_via_ablation(
        closestSources,
        base_combination_names, # List of metric names in the starting combination
):
    print(f"\n--- Running Ablation Study for Combination: {base_combination_names} ---")

    # --- 1. Define Target Metrics for Evaluation ---
    target_metrics = [
        'Cosine Sim', 'Euclidean Dst', 'Manhattan Dst',
        'Pearson Corr', 'Kendall Tau', 'Spearman Rho'
    ]
    # Define which direction is better (True=Higher, False=Lower)
    higher_is_better = {
        'Cosine Sim': True, 'Euclidean Dst': False, 'Manhattan Dst': False,
        'Pearson Corr': True, 'Kendall Tau': True, 'Spearman Rho': True
    }

    # --- 2. Helper Function to Evaluate a Combination ---
    # This internal function runs the evaluation pipeline for a given combo & weights
    def _evaluate_combination(combo_names, metric_weights):
        if not combo_names:
            print("Warning: Empty combination provided for evaluation.")
            return {metric: np.nan for metric in target_metrics} # Return NaNs if combo is empty

        # Get indices for the current combination
        try:
            combo_indices = tuple(all_metric_names_list.index(name) for name in combo_names)
        except ValueError as e:
            print(f"Error getting indices during evaluation: {e}")
            return {metric: np.nan for metric in target_metrics} # Return NaNs on index error

        batch_scores = defaultdict(list)
        if 'eval_dataloader' not in globals() and 'eval_dataloader' not in locals():
            print("Error: eval_dataloader not found during evaluation.")
            return {metric: np.nan for metric in target_metrics}

        for pos, data_batch in enumerate(eval_dataloader):
            if isinstance(data_batch, (list, tuple)) and len(data_batch) >= 1:
                sample = data_batch[0]
            else: continue

            try:
                if pos not in metricsDictionaryForSourceLayerNeuron: continue
                metricsOutputs = metricsDictionaryForSourceLayerNeuron[pos]

                metricSourcesSum, _ = RENN.identifyClosestSourcesByMetricCombination(
                    closestSources, metricsOutputs, combo_indices, metric_weights, "Sum"
                )
                mostUsed = RENN.getMostUsedSourcesByMetrics(
                    metricSourcesSum, closestSources, weightedMode="Sum", info=False
                )
                if not mostUsed: continue

                eval_scores_dict = evaluateImageSimilarityByMetrics(
                    "Metrics-Ablation", combo_indices, sample, mostUsed
                )
                if eval_scores_dict is None: continue

                valid_sample = True; temp_scores = []
                for metric_name in target_metrics:
                    score = eval_scores_dict.get(metric_name)
                    if score is None or not isinstance(score, (int, float)) or np.isnan(score):
                        valid_sample = False; break
                    temp_scores.append(score)
                if valid_sample:
                    for i, metric_name in enumerate(target_metrics):
                        batch_scores[metric_name].append(temp_scores[i])
            except Exception as e:
                print(f"Error evaluating sample {pos}: {e}")
                continue # Skip sample on error

        # Calculate average scores
        average_scores = {}
        for metric_name in target_metrics:
            scores_list = batch_scores[metric_name]
            if not scores_list:
                print(f"Warning: No valid scores for {metric_name} during evaluation.")
                average_scores[metric_name] = np.nan # Use NaN if no scores
            else:
                average_scores[metric_name] = np.mean(scores_list)

        return average_scores

    # --- 3. Check RENN.POTENTIAL_METRICS and Get All Names ---
    if not hasattr(RENN, 'POTENTIAL_METRICS') or not isinstance(RENN.POTENTIAL_METRICS, dict):
        print("Error: RENN.POTENTIAL_METRICS is not defined or not a dictionary.")
        return None, None, None
    all_metric_names_list = list(RENN.POTENTIAL_METRICS.keys())

    # --- 4. Evaluate Baseline Performance ---
    print(f"\nCalculating baseline performance for full combination ({len(base_combination_names)} metrics)...")
    # Use equal weights for fair comparison in ablation
    baseline_weights = {name: 1.0 for name in base_combination_names}
    baseline_scores = _evaluate_combination(base_combination_names, baseline_weights)

    if any(np.isnan(score) for score in baseline_scores.values()):
        print("Error: Failed to calculate valid baseline scores. Aborting ablation study.")
        print("Baseline Scores:", baseline_scores)
        return baseline_scores, None, None

    print("Baseline Scores:")
    for metric, score in baseline_scores.items():
        print(f"  {metric}: {score:.6f}")


    # --- 5. Perform Ablation Runs ---
    ablation_results = []
    print("\nPerforming ablation runs (removing one metric at a time)...")
    for metric_to_remove in base_combination_names:
        print(f"  Removing: {metric_to_remove}")
        # Create the temporary pruned combination
        pruned_combo_names = [m for m in base_combination_names if m != metric_to_remove]

        # Use equal weights for the pruned combo
        pruned_weights = {name: 1.0 for name in pruned_combo_names}
        pruned_scores = _evaluate_combination(pruned_combo_names, pruned_weights)

        result_row = {'Metric Removed': metric_to_remove}
        result_row.update(pruned_scores) # Add scores for this run
        ablation_results.append(result_row)

    if not ablation_results:
        print("Error: No ablation results generated.")
        return baseline_scores, None, None

    ablation_df = pd.DataFrame(ablation_results).set_index('Metric Removed')

    # --- 6. Calculate Performance Change ---
    change_results = []
    for metric_removed, pruned_scores_row in ablation_df.iterrows():
        change_row = {'Metric Removed': metric_removed}
        for metric_name in target_metrics:
            baseline = baseline_scores[metric_name]
            pruned = pruned_scores_row[metric_name]
            if np.isnan(baseline) or np.isnan(pruned):
                change = np.nan
            else:
                change = pruned - baseline # Calculate raw change

                # Optional: Calculate percentage change or normalize impact later
                # if higher_is_better[metric_name]:
                #     # Higher score is better, negative change is bad
                #     pass
                # else:
                #     # Lower score is better, positive change is bad
                #     change = -change # Invert change so negative is always bad

            change_row[f'Change_{metric_name}'] = change
        change_results.append(change_row)

    change_df = pd.DataFrame(change_results).set_index('Metric Removed')

    print("\n--- Ablation Results (Scores after removing metric) ---")
    print(ablation_df.to_string(float_format="%.4f"))
    print("\n--- Performance Change (vs Baseline) when Metric Removed ---")
    print(change_df.to_string(float_format="%.4f"))


    print("\n--- Interpreting Change ---")
    print("For MAXIMIZE objectives (Cosine Sim, Pearson, Kendall, Spearman):")
    print("  Negative change = Performance got WORSE when metric was removed (Metric likely useful)")
    print("  Positive change = Performance got BETTER when metric was removed (Metric potentially redundant/harmful)")
    print("For MINIMIZE objectives (Euclidean Dst, Manhattan Dst):")
    print("  Positive change = Performance got WORSE when metric was removed (Metric likely useful)")
    print("  Negative change = Performance got BETTER when metric was removed (Metric potentially redundant/harmful)")
    print("Small changes (close to zero) indicate less impact.")

    return baseline_scores, ablation_df, change_df

def run_multi_objective_weight_optimization(
        closestSources,
        combination_names_to_optimize, # REQUIRED: Pass the list of metric names for the single combo you want to optimize
        n_trials=200 # MOO often benefits from more trials
):
    print(f"\n--- Running Multi-Objective Optimization (MOO) for Weights ---")
    # --- Input Validation ---
    if not combination_names_to_optimize or not isinstance(combination_names_to_optimize, list):
        print("Error: Please provide a valid list of metric names for 'combination_names_to_optimize'.")
        return None, None, None
    if not all(isinstance(name, str) and name for name in combination_names_to_optimize):
        print("Error: 'combination_names_to_optimize' must contain non-empty strings.")
        return None, None, None

    print(f"Optimizing weights within combination: {combination_names_to_optimize}")
    print(f"Number of trials: {n_trials}")

    # --- 1. Define Objectives and Directions ---
    target_metrics = [
        'Cosine Sim', 'Euclidean Dst', 'Manhattan Dst',
        'Pearson Corr', 'Kendall Tau', 'Spearman Rho'
    ]
    directions = [
        'maximize', 'minimize', 'minimize', 'maximize', 'maximize', 'maximize'
    ]
    n_objectives = len(target_metrics)
    print(f"Optimizing for {n_objectives} objectives: {target_metrics}")
    print(f"Respective directions: {directions}")

    # --- 2. Prepare Combination Indices ---
    study = None
    if not hasattr(RENN, 'POTENTIAL_METRICS') or not isinstance(RENN.POTENTIAL_METRICS, dict):
        print("Error: RENN.POTENTIAL_METRICS is not defined or not a dictionary.")
        return None, None, None
    all_metric_names_list = list(RENN.POTENTIAL_METRICS.keys())
    try:
        combination_indices = tuple(all_metric_names_list.index(name) for name in combination_names_to_optimize)
    except ValueError as e:
        print(f"Error: Metric name '{e}' (from combination_names_to_optimize) not found in RENN.POTENTIAL_METRICS keys.")
        return None, None, None

    metric_names_to_weight = combination_names_to_optimize

    # --- 3. Define MOO Objective Function ---
    def moo_objective(trial):
        metric_weights = {name: trial.suggest_float(f"weight_{name}", 0.0, 2.0)
                          for name in metric_names_to_weight}
        batch_scores = defaultdict(list)
        if 'eval_dataloader' not in globals() and 'eval_dataloader' not in locals():
            raise optuna.exceptions.TrialPruned("eval_dataloader is missing.")

        for pos, data_batch in enumerate(eval_dataloader):
            if isinstance(data_batch, (list, tuple)) and len(data_batch) >= 1:
                sample = data_batch[0]
            else: continue

            try:
                if pos not in metricsDictionaryForSourceLayerNeuron: continue
                metricsOutputs = metricsDictionaryForSourceLayerNeuron[pos]
                metricSourcesSum, _ = RENN.identifyClosestSourcesByMetricCombination(
                    closestSources, metricsOutputs, combination_indices, metric_weights, "Sum"
                )
                mostUsed = RENN.getMostUsedSourcesByMetrics(
                    metricSourcesSum, closestSources, weightedMode="Sum", info=False
                )
                if not mostUsed: continue
                eval_scores_dict = evaluateImageSimilarityByMetrics(
                    "Metrics-MOO", combination_indices, sample, mostUsed
                )
                if eval_scores_dict is None: continue

                valid_sample = True; temp_scores = []
                for metric_name in target_metrics:
                    score = eval_scores_dict.get(metric_name)
                    if score is None or not isinstance(score, (int, float)) or np.isnan(score):
                        valid_sample = False; break
                    temp_scores.append(score)
                if valid_sample:
                    for i, metric_name in enumerate(target_metrics):
                        batch_scores[metric_name].append(temp_scores[i])
            except Exception as e:
                # Log less verbosely during runs
                # print(f"Error in objective sample {pos} trial {trial.number}: {e}")
                pass # Continue silently

        average_scores = []
        valid_trial = True
        for metric_name in target_metrics:
            scores_list = batch_scores[metric_name]
            if not scores_list: valid_trial = False; break
            avg_score = np.mean(scores_list)
            if np.isnan(avg_score): valid_trial = False; break
            average_scores.append(avg_score)
        if not valid_trial:
            raise optuna.exceptions.TrialPruned("Failed valid averages.")
        return tuple(average_scores)

    # --- 4. Create and Run MOO Study ---
    start_time = time.time()
    combo_name_short = '_'.join(name.split()[0] for name in combination_names_to_optimize[:3])
    study_name = f"MOO_Study_{combo_name_short}_{int(time.time())}"
    try:
        sampler = optuna.samplers.NSGAIISampler()
        study = optuna.create_study(study_name=study_name, directions=directions, sampler=sampler, load_if_exists=False)
        study.optimize(moo_objective, n_trials=n_trials)
        status = "Success"
    except Exception as e:
        end_time = time.time(); duration = end_time - start_time
        print(f"\n!!! MOO Optimization FAILED after {duration:.2f}s: {e} !!!")
        return None, None, study

    # --- 5. Process Results (Pareto Front) if Successful ---
    end_time = time.time(); duration = end_time - start_time
    print(f"\nMOO study completed in {duration:.2f} seconds.")
    pareto_df = None; weight_stats_df = None

    if status == "Success" and study.best_trials:
        print(f"\nProcessing {len(study.best_trials)} Pareto-optimal solutions...")
        solution_data = []; weights_data = defaultdict(list)

        # Check if trials have parameters before proceeding
        if not study.best_trials[0].params:
            print("Warning: Best trial found has no parameters (weights). Cannot process results.")
            return None, None, study

        metric_names_from_params = sorted([p.replace('weight_', '') for p in study.best_trials[0].params.keys()])

        for trial in study.best_trials:
            if trial.values is None or trial.params is None:
                print(f"Warning: Skipping Trial {trial.number} due to missing values or params.")
                continue
            solution = {'Trial': trial.number}
            for i, metric_name in enumerate(target_metrics):
                try: solution[f'Score_{metric_name}'] = trial.values[i]
                except IndexError: solution[f'Score_{metric_name}'] = np.nan
            # === V V V === CORRECTED WEIGHT KEY === V V V ===
            for param_name, weight in trial.params.items():
                metric_name = param_name.replace('weight_', '')
                # Use the CORRECT key format 'Weight_METRICNAME' for the DataFrame column
                solution[f'Weight_{metric_name}'] = weight
                if not np.isnan(weight):
                    weights_data[metric_name].append(weight) # Collect for stats
            # === ^ ^ ^ === CORRECTED WEIGHT KEY === ^ ^ ^ ===
            solution_data.append(solution)

        if not solution_data:
            print("Warning: No valid solutions found to create Pareto DataFrame.")
            return None, None, study
        pareto_df = pd.DataFrame(solution_data)
        print("\n--- Pareto Optimal Solutions (Subset Shown) ---")
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
            print(pareto_df)

        # --- 6. Analyze Weights Across Pareto Front ---
        weight_stats_list = []
        for metric_name in metric_names_from_params:
            weights = weights_data[metric_name]
            if weights:
                stats = {'Metric': metric_name, 'Mean Weight': np.mean(weights), 'Std Dev': np.std(weights),
                         'Min Weight': np.min(weights), 'Max Weight': np.max(weights)}
                weight_stats_list.append(stats)
        if weight_stats_list:
            weight_stats_df = pd.DataFrame(weight_stats_list).set_index('Metric')
            weight_stats_df = weight_stats_df.loc[metric_names_from_params].sort_values(by='Mean Weight', ascending=False)
            print("\n--- Weight Statistics Across Pareto Front Solutions ---")
            print(weight_stats_df.to_string(float_format="%.4f"))
        else: print("\nWarning: Could not generate weight statistics.")

    elif status == "Success":
        print("\nWarning: Optimization reported success but no best trials found.")

    # --- 7. Return Results ---
    print("\n--- MOO Function Finished ---")
    return pareto_df, weight_stats_df, study

def find_best_metric_weights(pareto_df, closestSources):
    target_metrics_used = ['Cosine Sim', 'Euclidean Dst', 'Manhattan Dst', 'Pearson Corr', 'Kendall Tau', 'Spearman Rho']
    directions_used = ['maximize', 'minimize', 'minimize', 'maximize', 'maximize', 'maximize']
    select_balanced_solution_from_pareto(pareto_df, target_metrics_used, directions_used)
    evaluate_metric_importance_via_ablation(closestSources,['Mahalanobis', 'Pearson Correlation', 'Chi-square',
                                                            'L2 norm (Euclidean)', 'Standardized Euclidean', 'Peak-to-Peak Range', 'Cosine Similarity', 'L1 norm (Manhattan)', 'Median',
                                                            'Spearman Correlation', 'Variance', 'Lp norm (Minkowski p=3)', 'L∞ norm (Chebyshev)'])

    # 1. Define the final pruned combination based on ablation analysis
    pruned_combination_names = ['Pearson Correlation', 'L2 norm (Euclidean)', 'Peak-to-Peak Range', 'Cosine Similarity',
                                'L1 norm (Manhattan)', 'Median', 'Spearman Correlation', 'L∞ norm (Chebyshev)', 'Variance']

    # 2. Define other necessary parameters (ensure these are correctly set for your environment)
    num_trials_for_final_opt = 200 # Adjust as needed (e.g., 200-500)

    print(f"\n--- Starting MOO Weight Optimization for FINAL PRUNED Combination ---")
    print(f"Pruned Combination: {pruned_combination_names}")

    # 3. Call the MOO function with the pruned combination
    # Note: Current date is Sunday, April 13, 2025. This doesn't affect the code logic itself.
    final_pareto_df, final_weight_stats_df, final_study = run_multi_objective_weight_optimization(
        closestSources=closestSources,
        combination_names_to_optimize=pruned_combination_names, # Pass the pruned list
        n_trials=num_trials_for_final_opt
    )

    # 4. Analyze the results and select the ultimate weights
    if final_pareto_df is not None:
        print("\n--- Analyzing FINAL MOO Results for PRUNED Combination ---")

        # Use the 'select_balanced_solution_from_pareto' function (defined previously)
        # to automatically pick a balanced solution from this new Pareto front.

        # Define objectives and directions (must match the ones inside the MOO function)
        target_metrics_used = [
            'Cosine Sim', 'Euclidean Dst', 'Manhattan Dst',
            'Pearson Corr', 'Kendall Tau', 'Spearman Rho'
        ]
        directions_used = [
            'maximize', 'minimize', 'minimize', 'maximize', 'maximize', 'maximize'
        ]

        # Assume select_balanced_solution_from_pareto function is defined
        try:
            # from your_module import select_balanced_solution_from_pareto # Import if needed
            ultimate_final_weights, selected_idx = select_balanced_solution_from_pareto(
                pareto_df=final_pareto_df,
                target_metrics=target_metrics_used,
                directions=directions_used
            )

            if ultimate_final_weights:
                print(f"\n--- === ULTIMATE FINAL WEIGHTING (Balanced Solution for Pruned Combo) === ---")
                # Print sorted by weight
                for metric, weight in sorted(ultimate_final_weights.items(), key=lambda item: item[1], reverse=True):
                    print(f"  {metric}: {weight:.4f}")

                # You can now use 'ultimate_final_weights' as THE final dictionary
                # for identifyClosestSourcesByMetricCombination, making sure to also
                # use the combination_indices corresponding to 'pruned_combination_names'.

                # Example: Get indices for the pruned combo
                # all_metric_names_list = list(RENN.POTENTIAL_METRICS.keys())
                # pruned_indices = tuple(all_metric_names_list.index(name) for name in pruned_combination_names)
                # print("\nIndices for pruned combination:", pruned_indices)

            else:
                print("\nCould not automatically select a balanced solution from the final Pareto front.")
                print("Manual analysis of 'final_pareto_df' is recommended.")

        except NameError:
            print("\nError: 'select_balanced_solution_from_pareto' function not defined.")
            print("Please define or import it. Manual analysis of 'final_pareto_df' required.")
        except Exception as e:
            print(f"\nError during final selection: {e}")
            print("Manual analysis of 'final_pareto_df' required.")

    else:
        print("\nMOO for the pruned combination did not produce a Pareto DataFrame.")