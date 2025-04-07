import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
import numpy as np
import Customizable_RENN as RENN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import spearmanr, kendalltau, pearsonr
from IPython.display import clear_output
import time
import math

mnist, to_categorical, nn, DataLoader, device, metricsEvaluation = "", "", "", "", "", True
train_dataloader, test_dataloader, eval_dataloader, trainDataSet, testDataSet, trainSubset, testSubset, x_train, y_train, x_test, y_test, x_eval, y_eval = "", "", "", "", "", "", "", "", "", "", "", "", ""
model, criterion_class, chosen_optimizer, layers = "", "", "", ""
train_samples, eval_samples, test_samples = 1, 1, 1
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource = [], [], [], [], [], []

def initializePackages(mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, devicePackage):
    global mnist, to_categorical, nn, DataLoader, device

    mnist, to_categorical, nn, DataLoader, device = mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, devicePackage

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

"""# Evaluation: Output"""

import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

import plotly.graph_objects as go
import plotly.subplots as sp

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
    image = Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA")
    weights = []
    total = 0

    if mode == "Not Weighted":
        for sourceNumber, counter in mostUsedSources:
            total += counter

        for sourceNumber, counter in mostUsedSources:
            image = Image.blend(image, Image.fromarray(x_train[sourceNumber].numpy()*255).convert("RGBA"), (counter/total))
            weights.append(counter/total)

    return (image, weights)

def blendIndividualImagesTogether(mostUsedSources, closestSources, layer=False):
    image = Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA")

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
                    image = Image.blend(image, Image.fromarray(x_train[wSource[0]].numpy()*255).convert("RGBA"), 1)
                else:
                    image = Image.blend(image, Image.fromarray(x_train[wSource.source].numpy()*255).convert("RGBA"), 1)
            else:
                if(layer):
                    image = Image.blend(image, Image.fromarray(x_train[wSource[0]].numpy()*255).convert("RGBA"), wSource[1] / total)
                else:
                    #print(f"Diff: {wSource.difference}, Total: {total}, Calculation: {(1 - (wSource.difference / total)) / closestSources}")
                    image = Image.blend(image, Image.fromarray(x_train[wSource.source].numpy()*255).convert("RGBA"), (1 - (wSource.difference / total)) / closestSources)

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

def evaluateImageSimilarity(name, sample, mostUsed):
    similarityList = []

    # Flatten and reshape the sample
    sample = np.asarray(sample.flatten().reshape(1, -1))

    blended_image = blendIndividualImagesTogether(mostUsed, len(mostUsed), layer=True)
    # Compute similarity for the blended image with sample
    blended_image_flat = np.asarray(blended_image.convert('L')).flatten() / 255.0
    blended_image_flat = blended_image_flat.reshape(1, -1)

    cosine_similarity, euclidean_distance, manhattan_distance, jaccard_similarity, hamming_distance, pearson_correlation = computeSimilarity(sample, blended_image_flat)

    kendall_tau, _ = kendalltau(sample, blended_image_flat)
    spearman_rho, _ = spearmanr(sample, blended_image_flat)

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
all_similarity_results = []

def blendActivations(mostUsed, evaluationActivations, layerNumbersToCheck, store_globally=False):
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
        all_similarity_results.append(results)

    # --- Print Results ---
    print("\n--- Blended Activation Similarity Scores ---")
    for metric, value in results.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return results  # Return for immediate use

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, analyze=False):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

    #Make sure to set new dictionaries for the hooks to fill - they are global!
    dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource = RENN.initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model)
    
    mostUsedList = []
    mostUsedMetricsList = []
    
    for pos, (sample, true) in enumerate(eval_dataloader):
        sample = sample.float()
        prediction = predict(sample)
        mostUsedSourcesWithSum = ""
        layersToCheck = []

        if(visualizationChoice == "Weighted"):
            sourcesSum, metricSourcesSum, mtSourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum, mostUsedMetricSourcesWithSum, mostUsedMMSourcesWithSum = RENN.getMostUsedSources(sourcesSum, metricSourcesSum, mtSourcesSum, closestSources, "Sum")
            #20 because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageSum = blendImagesTogether(mostUsedSourcesWithSum[:20], "Not Weighted")
            blendedMetricSourceImageSum = blendImagesTogether(mostUsedMetricSourcesWithSum[:20], "Not Weighted")
            blendedMMSourceImageSum = blendImagesTogether(mostUsedMMSourcesWithSum[:20], "Not Weighted")
            layersToCheck = layerNumbersToCheck # Switch to another variable to use correct layers for analyzation

            sourcesActivation, metricSourcesActivation, mtSourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation, mostUsedMetricSourcesWithActivation, mostUsedMMSourcesWithActivation = RENN.getMostUsedSources(sourcesActivation, metricSourcesActivation, mtSourcesActivation, closestSources, "Activation")
            #20 sources only because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation[:20], "Not Weighted")
            blendedMetricSourceImageActivation = blendImagesTogether(mostUsedMetricSourcesWithActivation[:20], "Not Weighted")
            blendedMMSourceImageActivation = blendImagesTogether(mostUsedMMSourcesWithActivation[:20], "Not Weighted")

            #showImagesUnweighted("Per Neuron", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
            #showImagesUnweighted("Metrics", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedMetricSourceImageActivation, blendedMetricSourceImageSum, mostUsedMetricSourcesWithActivation[:showClosestMostUsedSources], mostUsedMetricSourcesWithSum[:showClosestMostUsedSources])
            #showImagesUnweighted("Magnitude Truncation", createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedMMSourceImageActivation, blendedMMSourceImageSum, mostUsedMMSourcesWithActivation[:showClosestMostUsedSources], mostUsedMMSourcesWithSum[:showClosestMostUsedSources])
        else:
            sourcesSum, metricSourcesSum, mtSourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum = getClosestSourcesPerNeuronAndLayer(sourcesSum, metricSourcesSum, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Sum")

            sourcesActivation, metricSourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], metricsDictionaryForSourceLayerNeuron[pos], mtDictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = getClosestSourcesPerNeuronAndLayer(sourcesActivation, metricSourcesActivation, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Activation")
            #RENN.analyzeData(closestSources, dictionaryForSourceLayerNeuron[pos])
    
        if(analyze):
            #mostUsed, mostUsedMetrics, mostUsedMM = #RENN.getMostUsedSources(sourcesSum, metricSourcesSum, mtSourcesSum, closestSources)
            mostUsedList.append(mostUsedSourcesWithSum)
            blendActivations(mostUsedSourcesWithSum, dictionaryForSourceLayerNeuron[pos], layersToCheck, True)
            #evaluateImageSimilarity("", sample, mostUsedSourcesWithSum)
            #evaluateImageSimilarity("Metrics", sample, mostUsedMetricSourcesWithSum)
            #evaluateImageSimilarity("MT", sample, mostUsedMMSourcesWithSum)

        #if pos % 10 == 0:  # Clear every 10 samples
        #    clear_output(wait=True)  # Keeps the last output visible
        #    time.sleep(1)  # Prevents UI freezing
    
    #print(f"Time passed since start: {time_since_start(startTime)}")