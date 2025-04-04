import torch
from torch import nn
import numpy as np
import concurrent.futures
import threading
import sys
from collections import Counter
from scipy.spatial import distance
from scipy.stats import spearmanr, kendalltau, pearsonr
from time import perf_counter
from collections import defaultdict
import heapq
import LLM_Small1x1 as Small1x1
import LLM_GPT2 as GPT2
import LLM_LSTM as LSTM
import scipy.sparse as sp
import shutil
import os


mtEvaluation, NumberOfComponents = True, 20

layer, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, totalLayers = 0, 0, [], [], [], [], 0
llm, metricsEvaluation, useBitNet, layerSizes, device, hidden_sizes, layers, currentLayer, relevantLayerIndices = False, False, False, [], "", [], 0, [], []
sourceArray, fileName, contextLength, io, pd, pa, pq, zstd, levenshtein, chosenDataSet, baseDirectory = "", "", 1, "", "", "", "", "", "", "", "./LookUp"
metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, metricsActivationsBySources, metricsActivationsByLayers, layersToCheck = [], [], [], [], []
mmDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, mtActivationsBySources, mtActivationsByLayers = [], [], [], []

METRICS = {
    # 1. L-family distances (vectorized)
    'L2 norm (Euclidean)': lambda d, c: np.sqrt(np.sum((d - c)**2)),
    'Squared Euclidean': lambda d, c: np.sum((d - c)**2),
    'L1 norm (Manhattan)': lambda d, c: np.sum(np.abs(d - c)),
    'Canberra': lambda d, c: np.sum(np.abs(d - c) / (np.abs(d) + np.abs(c) + 1e-10)),
    'L∞ norm (Chebyshev)': lambda d, c: np.max(np.abs(d - c)),
    'Lp norm (Minkowski p=3)': lambda d, c: np.sum(np.abs(d - c)**3)**(1/3),

    # 2. Correlation measures (precomputed reference)
    'Cosine Similarity': lambda d, c: (1 - distance.cosine(d, c) if (np.linalg.norm(d) > 1e-9 and np.linalg.norm(c) > 1e-9) else 0.0),
    'Pearson Correlation': lambda d, ref: np.corrcoef(d, ref)[0, 1] if np.std(d) > 1e-9 else 0.0,
    'Spearman Correlation': lambda d, ref: spearmanr(d, ref).correlation if np.std(d) > 1e-9 else 0.0,

    # 3. Statistical distances (precomputed variances)
    'Mahalanobis': lambda d, c, v: np.sqrt(np.sum((d - c)**2 / v)),
    'Standardized Euclidean': lambda d, c, v: np.sqrt(np.sum((d - c)**2 / v)),
    'Chi-square': lambda d, c: np.sum(np.where((d + c) > 0, (d - c)**2 / (d + c + 1e-10), 0)),
    'Jensen-Shannon': lambda d, c: distance.jensenshannon(
        (d - d.min() + 1e-10) / (d.sum() - d.min() + 1e-10),
        (c - c.min() + 1e-10) / (c.sum() - c.min() + 1e-10)
    ),

    # 4. Discrete metrics (precomputed values)
    'Levenshtein': lambda s1, s2: levenshtein(s1, s2),
    'Hamming': lambda d, c: np.count_nonzero(np.round(d, 2) != np.round(c, 2)),
    'Jaccard/Tanimoto': lambda s1, s2: len(s1 & s2) / max(len(s1 | s2), 1),
    'Sørensen–Dice': lambda s1, s2: 2 * len(s1 & s2) / max((len(s1) + len(s2)), 1)
}

def initializePackages(devicePackage, ioPackage, pdPackage, paPackage, pqPackage, zstdPackage, levenshteinPackage, chosenDataSetPackage, seed="", useBitLinear=False):
    global device, useBitNet, io, pd, pa, pq, zstd, levenshtein, chosenDataSet

    device, io, pd, pa, pq, zstd, levenshtein, chosenDataSet = devicePackage, ioPackage, pdPackage, paPackage, pqPackage, zstdPackage, levenshteinPackage, chosenDataSetPackage
    useBitNet = useBitLinear

    if(seed != ""):
        torch.manual_seed(seed)
        np.random.seed(seed)

    shutil.rmtree(baseDirectory, ignore_errors=True)
    print("Initialized Packages for Customizable RENN")

#Bitnet-1.58b
def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.type(dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, weight_bits=1, input_bits=8, **kwargs):
        super(BitLinear, self).__init__(in_features, out_features, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, input):
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()
        out = nn.functional.linear(quant_input, quant_weight)

        if self.bias is not None:
            if out.dim() == 2:  # Batch of outputs, 2D tensor
                out += self.bias.view(1, -1).expand_as(out)
            else:  # Single output, 1D tensor
                out += self.bias

        return out

def getLayer(hidden_layers, layerNumber, input_size=10, output_size=10):
    if(hidden_layers[layerNumber][0] == "Linear"):
        if useBitNet:
            return BitLinear(input_size, output_size)
        return nn.Linear(input_size, output_size)
    elif(hidden_layers[layerNumber][0] == "Conv2d"):
        return nn.Conv2d(input_size, output_size)
    return False

def getActivation(hidden_layers, layerNumber):
    if(hidden_layers[layerNumber][2] == "ReLU"):
        return nn.ReLU()
    elif(hidden_layers[layerNumber][2] == "Sigmoid"):
        return nn.Sigmoid()
    elif(hidden_layers[layerNumber][2] == "Tanh"):
        return nn.Tanh()
    return False

def checkIfActivationLayerExists(hidden_layers, layerNumber):
    if hidden_layers[layerNumber][2] != "None":
        return True
    return False

def createLayers(layerName, layerType, activationLayerType):
    global currentLayer, layers

    layers.append((layerName, layerType, activationLayerType))
    relevantLayerIndices.append(currentLayer*2)
    if(activationLayerType != "None"):
        relevantLayerIndices.append((currentLayer*2)+1)
    currentLayer += 1
    return layerName, layerType

class CustomizableRENN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(CustomizableRENN, self).__init__()

        #Add input and output layer to the hidden_layers
        self.num_layers = (len(hidden_layers))
        self.hidden_layers = hidden_layers
        global layers, currentLayer, relevantLayerIndices

        layers = []
        currentLayer = 0
        relevantLayerIndices = []
        for layer in range(self.num_layers):
            if layer == 0:
                setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, input_size, hidden_layers[layer][1]), self.hidden_layers[layer][2]))
            elif layer == (self.num_layers - 1):
                setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, hidden_layers[layer-1][1], output_size), self.hidden_layers[layer][2]))
            else:
                #print(layer, layer-1, self.num_layers, hidden_sizes[layer-1], hidden_sizes[layer])
                setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, hidden_layers[layer-1][1], hidden_layers[layer][1]), self.hidden_layers[layer][2]))

            if checkIfActivationLayerExists(self.hidden_layers, layer):
                setattr(self, f'activation{layer}', getActivation(self.hidden_layers, layer))

        # Initialize weights with only -1, 0, and 1 values
        #if useBitNet:
            #self.apply(self.custom_weight_init)

    def custom_weight_init(self, module):
        if isinstance(module, nn.Linear):
            # Generate random values of -1, 0, or 1 for each weight element
            with torch.no_grad():
                module.weight.data = torch.tensor(np.random.choice([-1, 0, 1], module.weight.shape)).float()
            if module.bias is not None:
                module.bias.data = torch.tensor(np.random.choice([-1, 0, 1], module.bias.shape)).float()

    def forward(self, x):
        for layer in range(self.num_layers):
            x = getattr(self, f'fc{layer}')(x)

            if (checkIfActivationLayerExists(self.hidden_layers, layer)):
                x = getattr(self, f'activation{layer}')(x)
        return x

# Forward hook
def forward_hook(module, input, output):
    global layer, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, \
        metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, \
        sourceArray, hidden_sizes, llm, fileName, layersToCheck
    
    #if not (isinstance(module, nn.Sequential) or isinstance(module, Small1x1.FeedForward) or isinstance(module, Small1x1.TransformerBlock) or isinstance(module, nn.Dropout) or isinstance(module, GPT2.FeedForward) or isinstance(module, GPT2.TransformerBlock)):
    if (llm):
        actualLayer = layer
        layerNeurons = layers[actualLayer][1]
        #if(source >= dictionaryForSourceLayerNeuron.shape[0]):
        #    return
    else:
        actualLayer = int(layer/2)
        layerNeurons = layers[actualLayer][1].out_features

    correctTypes = False
    if not llm:
        activation_type = type(getActivation(hidden_sizes, actualLayer)) 
        layer_type = type(getLayer(hidden_sizes, actualLayer))
        if (type(module) == activation_type or type(module) == layer_type):
            correctTypes = True

    relevantOutput = output[0].cpu().numpy()      

    #print(layer, layers[layer], relevantOutput.shape)

    if(correctTypes or llm):
        #Use for array structure like: [source, layer, neuron]
        if(len(relevantOutput.shape) > 1):
            if(relevantOutput.shape[1] != layerNeurons):
                layerNeurons = relevantOutput.shape[1]
                #layers[actualLayer] = (layers[actualLayer][0], relevantOutput.shape[1], layers[layer][2:])
        if(correctTypes):
            dictionaryForSourceLayerNeuron[source][layer,:layerNeurons] = relevantOutput
        # if(source == 0):
        #   print(relevantOutput, dictionaryForSourceLayerNeuron[source][layer,:layerNeurons])

        #Use for array structure like: [layer, neuron, source]
        output = relevantOutput if len(relevantOutput.shape) == 1 else relevantOutput[0]
        if metricsEvaluation:
            metricsArray = createMetricsArray(output) 
            metricsDictionaryForSourceLayerNeuron[source][layer] = metricsArray
            metricsDictionaryForLayerNeuronSource[layer][source] = metricsArray
        if mtEvaluation:
            reduced = np.argsort(-np.abs(output))[:min(NumberOfComponents, output.shape[0])]
            mtDictionaryForSourceLayerNeuron[source][layer,:len(reduced)] = reduced
            mtDictionaryForLayerNeuronSource[layer][source,:len(reduced)] = reduced
        
        if(llm):
            if(actualLayer in layersToCheck or layersToCheck == []):
                sourceNumber, sentenceNumber = chosenDataSet.getSourceAndSentenceIndex(source, fileName)
                if sourceNumber is not None and sentenceNumber is not None:
                    #print(f"Create File: LookUp/{fileName}/Layer{layer}/Source={result[0]}/Sentence{result[1]}-0")
                    append_structured_sparse(output[:layerNeurons], actualLayer, sourceNumber, sentenceNumber)
        else:
            for neuronNumber, neuron in enumerate(output):
                if neuronNumber < layerNeurons:
                    dictionaryForLayerNeuronSource[layer][neuronNumber][source] = neuron
                else:
                    break

        if(layer % 2 == 0 and not llm):
            if(checkIfActivationLayerExists(hidden_sizes, actualLayer)):
                layer += 1
            elif(layer == (len(layers)*2)-2):
                layer = 0
            else:
                layer += 2
        else:
            if((layer == (len(layers)*2)-1 and not llm) or (layer == (len(layers))-1 and llm)):
                layer = 0
            else:
                layer += 1

def attachHooks(hookLoader, model, llmType = False, filename = "", sourceOffset=0, lstm = False):
    global source, layer, sourceArray, fileName

    fileName = filename
    hooks = []  # Store the handles for each hook
    outputs = np.array([])

    for name, module in model.named_modules():
        if not isinstance(module, CustomizableRENN) and not isinstance(module, GPT2.GPTModel)\
                and not isinstance(module, Small1x1.GPTModel):
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    with torch.no_grad():
        # Forward Pass
        for tempSource, (inputs, labels) in enumerate(hookLoader):
            source = tempSource + sourceOffset
            layer = 0
            if not llmType:
                inputs = inputs.float()
            else:
                actualSource, actualSentenceNumber = chosenDataSet.getSourceAndSentenceIndex(source, fileName)
                print(f"Saving all Activations for {fileName}-Source {tempSource} (Actual {fileName}-Source: {actualSource}:{actualSentenceNumber})")
            inputs = inputs.to(device)
            if lstm:
                state_h, state_c = model.init_hidden(1)
                state_h, state_c = state_h.to(device), state_c.to(device)
                _, _ = model(inputs, state_h, state_c)
            else:
                _ = model(inputs)

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

def createDictionaries(hidden_sizes, totalLayersParameter, train_samples, llmType = False):
    global activationsBySources, activationsByLayers, metricsActivationsBySources, metricsActivationsByLayers, mtActivationsBySources, mtActivationsByLayers, totalLayers, layerSizes
    
    totalLayers = totalLayersParameter
    layerSizes = [size[1] for size in hidden_sizes[:]]
    if not llmType:
        if useBitNet:
            activationsBySources = np.zeros((train_samples, totalLayers, np.max(layerSizes)), dtype=int)
            activationsByLayers = np.zeros((totalLayers, np.max(layerSizes), train_samples), dtype=int)
        else:
            activationsBySources = np.zeros((train_samples, totalLayers, np.max(layerSizes)), dtype=np.float128)
            activationsByLayers = np.zeros((totalLayers, np.max(layerSizes), train_samples), dtype=np.float128)
        
        if metricsEvaluation:
            metricsActivationsBySources = np.zeros((train_samples, totalLayers, len(METRICS)), dtype=np.float128)
            metricsActivationsByLayers = np.zeros((totalLayers, train_samples, len(METRICS)), dtype=np.float128)

        if mtEvaluation:
            mtActivationsBySources = np.zeros((train_samples, totalLayers, NumberOfComponents), dtype=np.float128)
            mtActivationsByLayers = np.zeros((totalLayers, train_samples, NumberOfComponents), dtype=np.float128)
        
    print("Hook-Dictionaries created")

def runHooks(train_dataloader, model, layersParameter=layers, llmType=False, context_length=1, lstm=False, layersToCheckParameter=[]):
    global layers, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers,\
        metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, metricsActivationsBySources, metricsActivationsByLayers,\
        mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, mtActivationsBySources, mtActivationsByLayers, llm, contextLength, layersToCheck

    #Variables for usage within the hook
    llm = llmType
    layers = layersParameter
    contextLength = context_length
    layersToCheck = layersToCheckParameter

    if not llm:
        dictionaryForSourceLayerNeuron = activationsBySources
        dictionaryForLayerNeuronSource = activationsByLayers
        metricsDictionaryForSourceLayerNeuron = metricsActivationsBySources
        metricsDictionaryForLayerNeuronSource = metricsActivationsByLayers
        mtDictionaryForSourceLayerNeuron = mtActivationsBySources
        mtDictionaryForLayerNeuronSource = mtActivationsByLayers

    attachHooks(train_dataloader, model, llmType, filename="Training", sourceOffset=0, lstm=lstm)

    if not llm:
        activationsBySources = dictionaryForSourceLayerNeuron
        activationsByLayers = dictionaryForLayerNeuronSource
        metricsActivationsBySources = metricsDictionaryForSourceLayerNeuron
        metricsActivationsByLayers = metricsDictionaryForLayerNeuronSource
        mtActivationsBySources = mtDictionaryForSourceLayerNeuron
        mtActivationsByLayers = mtDictionaryForLayerNeuronSource
        
    print("Hooks finished successfully")

def initializeHook(train_dataloader, model, hidden_sizesParameter, train_samples, metricsEvaluationParameter=False):
    global totalLayers, hidden_sizes, metricsEvaluation
    
    metricsEvaluation = metricsEvaluationParameter
    print("Initializing Hooks")
    hidden_sizes = hidden_sizesParameter
    totalLayers = len(layers)*2
    createDictionaries(hidden_sizes, totalLayers, train_samples)
    runHooks(train_dataloader, model, layers)

def initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model, filename="Evaluation", llmType = False, sourceOffset=0, lstm=False):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

    if not llm:
        dictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, np.max(layerSizes)), dtype=np.float128)
        dictionaryForLayerNeuronSource = np.zeros((totalLayers, np.max(layerSizes), eval_samples), dtype=np.float128)
        metricsDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, len(METRICS)), dtype=np.float128)
        metricsDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, len(METRICS)), dtype=np.float128)
        mtDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, NumberOfComponents), dtype=np.float128)
        mtDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, NumberOfComponents), dtype=np.float128)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader, model, llmType, filename, sourceOffset, lstm)

    return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

# Global configuration
METRIC_WEIGHTS = {name: 1.0 for name in METRICS.keys()}
metrics_optimizer = None  # Will be initialized on first call
# Add to global initialization
mt_component_optimizer = None
optimal_components_overall = 0

def identifyClosestSources(closestSources, outputs, metricsOutputs, mtOutputs, mode=""):
    global layers, METRIC_WEIGHTS, metrics_optimizer, mt_component_optimizer, optimal_components_overall

    # Initialize optimizer on first call
    if metricsEvaluation and metrics_optimizer is None:
        metrics_optimizer = MetricWeightOptimizer(list(METRICS.keys()), top_k=closestSources)
    if mtEvaluation and mt_component_optimizer is None:
        mt_component_optimizer = ComponentOptimizer()

    dictionary = activationsByLayers
    metricsDictionary = metricsActivationsByLayers
    mtDictionary = mtActivationsByLayers
    
    # Layer selection logic
    if mode == "Sum":
        layerNumbersToCheck = [idx * 2 for idx, (name, layerNumber, activation) in enumerate(layers)]
    elif mode == "Activation":
        layerNumbersToCheck = [
            (idx * 2) + 1 for idx, (name, layerNumber, activation) in enumerate(layers)
            if getActivation(hidden_sizes, idx) != False
        ]
    else:
        layerNumbersToCheck = [idx for idx, _ in enumerate(layers)]

    layersToCheck = dictionary[layerNumbersToCheck]
    outputsToCheck = outputs[layerNumbersToCheck]
    identifiedClosestSources = np.empty((len(layersToCheck), np.max(layerSizes), closestSources), dtype=tuple)
    metricsLayersToCheck = metricsDictionary[layerNumbersToCheck]
    metricsOutputsToCheck = metricsOutputs[layerNumbersToCheck]
    identifiedClosestMetricSources = np.empty((len(layersToCheck), closestSources), dtype=tuple)
    mtLayersToCheck = mtDictionary[layerNumbersToCheck]
    mtOutputsToCheck = mtOutputs[layerNumbersToCheck]
    identifiedClosestMTSources = np.empty((len(layersToCheck), closestSources), dtype=tuple)

    if mtEvaluation:
        # Update component optimizer with current sample
        for layer_idx in range(len(layersToCheck)):
            mt_component_optimizer.update(outputsToCheck[layer_idx], mtLayersToCheck[layer_idx])

        # Get current best component count
        optimal_components = mt_component_optimizer.get_optimal_components()
        if optimal_components != optimal_components_overall:
            optimal_components_overall = optimal_components
            print("Adjusted Optimal Components to: " + str(optimal_components))

    for currentLayer, (layer, currentMetricsLayer, currentMTLayer) in enumerate(zip(layersToCheck, metricsLayersToCheck, mtLayersToCheck)):
        # 1. Collect neuron-based indices and differences
        layer_neuron_indices = []
        neuron_differences = []

        for currentNeuron, neuron in enumerate(layer):
            maxNeurons = layers[currentLayer][1]
            if not isinstance(maxNeurons, int):
                maxNeurons = maxNeurons.out_features
            if currentNeuron < maxNeurons:
                differences = np.abs(neuron - np.full(len(neuron), outputsToCheck[currentLayer][currentNeuron]))
                sorted_indices = np.argsort(differences)
                closest_indices = sorted_indices[:closestSources]
                layer_neuron_indices.append(sorted_indices)
                neuron_differences.append(differences)

                # Store neuron results
                tuples = tuple(
                    (closest_indices[i], neuron[closest_indices[i]],
                     differences[closest_indices[i]])
                    for i in range(closestSources)
                )
                identifiedClosestSources[currentLayer][currentNeuron] = tuples

        # Get target indices from neuron-based approach        
        target_indices = np.concatenate(
            [indices for indices in layer_neuron_indices if len(indices) > 0]
        )
        
        # 2. Process metrics for this layer
        if metricsEvaluation:
            # Calculate individual metric scores
            metric_scores = {}
            raw_diffs = np.abs(currentMetricsLayer - metricsOutputsToCheck[currentLayer][np.newaxis, :])

            # Normalize per metric
            min_vals = currentMetricsLayer.min(axis=0)
            max_vals = currentMetricsLayer.max(axis=0)
            norm_samples = (currentMetricsLayer - min_vals) / (max_vals - min_vals + 1e-10)
            norm_ref = (metricsOutputsToCheck[currentLayer] - min_vals) / (max_vals - min_vals + 1e-10)

            # Handle similarity metrics
            similarity_indices = [i for i, name in enumerate(METRICS.keys())
                                  if name in {'Cosine Similarity', 'Pearson Correlation',
                                              'Spearman Correlation', 'Jaccard/Tanimoto', 'Sørensen–Dice'}]
            norm_samples[:, similarity_indices] = 1 - norm_samples[:, similarity_indices]
            norm_ref[similarity_indices] = 1 - norm_ref[similarity_indices]

            # Store normalized scores per metric
            for i, name in enumerate(METRICS.keys()):
                metric_scores[name] = np.abs(norm_samples[:, i] - norm_ref[i])

            # Update weights using this sample
            metrics_optimizer.update_weights(metric_scores, target_indices)
            METRIC_WEIGHTS = metrics_optimizer.get_weights()

            # Combine scores using optimized weights
            combined_scores = np.sum([
                metric_scores[name] * METRIC_WEIGHTS[name]
                for name in METRICS.keys()
            ], axis=0)

            sorted_metric_indices = np.argsort(combined_scores)
            closest_metric_indices = sorted_metric_indices[:closestSources]

            # Create output tuples
            tuples = tuple(
                (closest_metric_indices[i],
                 currentMetricsLayer[closest_metric_indices[i]],
                 raw_diffs[closest_metric_indices[i]])
                for i in range(closestSources)
            )
            identifiedClosestMetricSources[currentLayer] = tuples

        if mtEvaluation:
            mt_differences = np.sum(np.abs(currentMTLayer - mtOutputsToCheck[currentLayer][np.newaxis, :]), axis=1)
            mt_sorted_indices = np.argsort(mt_differences)
            mt_closest_indices = mt_sorted_indices[:closestSources]

            # Store results
            tuples = tuple(
                (mt_closest_indices[i],
                 currentMTLayer[mt_closest_indices[i]],
                 mt_differences[mt_closest_indices[i]])
                for i in range(closestSources)
            )
            identifiedClosestMTSources[currentLayer] = tuples

    return identifiedClosestSources, identifiedClosestMetricSources, identifiedClosestMTSources, outputsToCheck, layerNumbersToCheck
    
def getMostUsed(sources, mode="", evaluation=""):
    mostUsed = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        if evaluation == "Metrics" or evaluation == "Magnitude Truncation":
            for sourceNumber, value, difference in layer:
                if(sourceNumber != 'None'):
                    mostUsed.append(sourceNumber)
                    sourceCounter += 1
        else:
            for currentNeuron, neuron in enumerate(layer):
                maxNeurons = layers[currentLayer][1] if mode == "" else layers[currentLayer][1].out_features
                if not isinstance(maxNeurons, int):  # Ensure maxNeurons is an integer
                    maxNeurons = maxNeurons.out_features
                if(currentNeuron < maxNeurons):
                    for sourceNumber, value, difference in neuron:
                        if(sourceNumber != 'None'):
                            mostUsed.append(sourceNumber)
                            sourceCounter += 1
    return sourceCounter, mostUsed

def getMostUsedFromDataFrame(df, evalSample, closestSources, weightedMode=""):
    # Filter entries for the specific evalSample
    relevant_entries = df[df.index.get_level_values('evalSample') == evalSample]

    # Use value_counts to count occurrences of each source directly
    sources = relevant_entries['source']

    # Filter out invalid sources ('None')
    valid_entries = relevant_entries[sources != 'None']
    ascending_order = True  # Sort by ascending for lowest total weights

    if weightedMode == "Sum":
        # Group by 'source' and sum the 'difference' column as weights
        weighted_counts = valid_entries.groupby('source')['difference'].sum()
    elif weightedMode == "Mean":
        # Group by 'source' and calculate the average of 'difference'
        weighted_counts = valid_entries.groupby('source')['difference'].mean()
    else:
        # Default behavior: Count occurrences
        weighted_counts = valid_entries['source'].value_counts()
        ascending_order = False  # Sort by descending for highest counts

    # Sort weighted sources by the determined order
    sorted_sources = weighted_counts.sort_values(ascending=ascending_order).head(closestSources)
    # Total weight (sum or mean) or total count for closest sources
    total_weight = sorted_sources.sum()

    # Print the total weight (sum, mean, or total count depending on the mode)
    #print(f"Total Weight for Weighted Mode={weightedMode}: {total_weight}")

    # Convert to a Counter-like output (sorted already by the determined order)
    counter = [(source, weight) for source, weight in sorted_sources.items()]

    #print(f"Total closest Sources (Weighted Mode={weightedMode}):", total_weight,
    #      "|", closestSources, "closest Sources in format [SourceNumber, Weight]:", counter)

    # Fix: Convert the 'source' column of valid_entries to a list
    sourceCounter = valid_entries['source'].value_counts().sum()  # Total count of valid sources
    mostUsed = valid_entries['source'].tolist()  # Extract 'source' as a list

    return sourceCounter, mostUsed

def getMostUsedSources(sources, metricsSources, mtSources, closestSources, evalSample=0, weightedMode="", info=True):
    metricsMostUsed, metricsSourceCounter = [], []
    mtMostUsed, mtSourceCounter = [], []
    if llm:
        sourceCounter, mostUsed = getMostUsedFromDataFrame(sources, evalSample, closestSources, weightedMode)
    else:
        sourceCounter, mostUsed = getMostUsed(sources, weightedMode)
        if metricsEvaluation:
            metricsSourceCounter, metricsMostUsed = getMostUsed(metricsSources, weightedMode, evaluation="Metrics")
        if mtEvaluation:
            mtSourceCounter, mtMostUsed = getMostUsed(mtSources, weightedMode, evaluation="Magnitude Truncation")
    counter = Counter(mostUsed)
    metricsCounter = Counter(metricsMostUsed)
    mtCounter = Counter(mtMostUsed)
    
    if(info):
        print("Total closest Sources (Per Neuron):", sourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", counter.most_common()[:closestSources])
        if metricsEvaluation:
            print("Total closest Sources (Metrics):", metricsSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", metricsCounter.most_common()[:closestSources])
        if mtEvaluation:
            print("Total closest Sources (MT):", mtSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", mtCounter.most_common()[:closestSources])
    return counter.most_common()[:closestSources], metricsCounter.most_common()[:closestSources], mtCounter.most_common()[:closestSources]

# Normalize function to convert to integer range for sparse arrays
def normalize_to_integer_sparse(sparse_data, min_val, max_val):
    min_int, max_int = 0, 4294967295

    if min_val == max_val:
        # If all elements are identical, return sparse zero array
        return sp.coo_matrix(sparse_data.shape, dtype=np.int64)

    # Compute scaling factors for normalization
    scale_factor = (max_int - min_int) / (max_val - min_val)
    shift_factor = min_int - min_val * scale_factor

    # Convert sparse data to COO format for element-wise multiplication
    sparse_data = sparse_data.tocoo()

    # Apply normalization directly: sparse element-wise multiplication and shift
    normalized_data = sparse_data.multiply(scale_factor)
    normalized_data.data += shift_factor

    # Convert data to int64 and ensure values are within the correct range
    normalized_data.data = np.clip(np.round(normalized_data.data), min_int, max_int).astype(np.int64)

    return normalized_data

# Function to compress DataFrame using ZSTD
def compress_dataframe_zstd(filepath, df, source_name, sentence_number):

    # Convert to PyArrow Table
    table = pa.Table.from_pandas(df)

    # Ensure the layer directory exists
    os.makedirs(filepath, exist_ok=True)

    # Write to partitioned dataset, creating partitions if necessary
    pq.write_to_dataset(
        table,
        root_path=f"{filepath}",
        partition_cols=['Source'],
        basename_template=f"Sentence{sentence_number}-{{i}}.parquet",
        compression='zstd'
    )

# Function to append structured data to Parquet file without loading entire file
def append_structured_sparse(array, filename, source_name, sentence_number):
    os.makedirs(baseDirectory, exist_ok=True)
    filepath = os.path.join(baseDirectory, fileName, f"Layer{filename}")

    # Convert to sparse COO format and normalize
    sparse_array = sp.coo_matrix(array) if not sp.issparse(array) else array
    del array  # Free original array memory

    # Normalize the sparse array directly (still in COO format)
    min_val, max_val = sparse_array.min(), sparse_array.max()
    normalized_sparse_array = normalize_to_integer_sparse(sparse_array, min_val, max_val)
    del sparse_array  # Free the sparse array memory

    if normalized_sparse_array.nnz != 0:
        # Prepare sparse row dictionary with normalization
        row_dict = {
            'Source': source_name,
            'Sentence': sentence_number,
            'Min': float(min_val),
            'Max': float(max_val),
            **{f"Neuron {idx}": val for idx, val in zip(normalized_sparse_array.indices, normalized_sparse_array.data)},
        }

        # Convert to DataFrame and pass into the compression function
        new_row = pd.DataFrame([row_dict])
        compress_dataframe_zstd(filepath, new_row, source_name, sentence_number)
        del new_row

    #print(f"Data for {source_name} appended to {filename}.")

# Split the path based on "/" to get the relevant parts
def getInformationFromFileName(filepath):
    path_parts = filepath.split('/')

    # Extract Layer, Source, and Sentence
    layer = int(path_parts[3].replace('Layer', ''))  # Layer part: "Layer0" -> 0
    source = int(path_parts[4].replace('Source=', ''))  # Source part: "Source=0" -> 0
    sentence = int(path_parts[5].split('-')[0].replace('Sentence', ''))  # Sentence part: "Sentence0-Sequence0-0" -> 0
    #sequence = int(path_parts[5].split('-')[1].replace('Sequence', '')) # Sequence part: "Sentence0-Sequence0-0" -> 0
    return layer, source, sentence

def reconstruct_from_normalized(sparse_array, min_val, max_val):
    # Inverse normalization scaling factors
    scale_factor = (max_val - min_val) / 4294967295
    shift_factor = min_val

    # Apply normalization directly on sparse matrix (element-wise multiplication)
    normalized_data = sparse_array.multiply(scale_factor)
    normalized_data.data += shift_factor

    # Return the updated sparse matrix with normalized data
    return normalized_data

def getNormalizedValues(full_path, evalSample):
    # Duplicate files to preserve the originals
    copy_path = get_safe_copy_path(full_path, evalSample)
    # Duplicate files to preserve the originals
    shutil.copy(full_path, copy_path)
    # Read sparse arrays directly from the duplicated parquet files
    sentenceDf = pq.read_table(copy_path).to_pandas(safe=False)
    # Remove copied file after reading
    safe_remove(copy_path)
    # Extract scalar min and max values from the first row# Extract scalar min and max values from the first row
    min, max = sentenceDf['Min'].iloc[0], sentenceDf['Max'].iloc[0]
    # Drop non-neuron columns
    sentenceDf = sentenceDf.drop(columns=['Source', 'Sentence', 'Min', 'Max'])
    # Convert DataFrames to sparse COO matrices
    neurons = sp.coo_matrix(np.asarray(sentenceDf.to_numpy()))

    return reconstruct_from_normalized(neurons, min, max).tocoo()

# Global lock for thread-safe access to shared data
file_lock = threading.Lock()
# Global lock for thread-safe access to print statements
print_lock = threading.Lock()
# Thread-safe lock for concurrent updates
data_lock = threading.Lock()

def get_safe_copy_path(full_path, eval_sample):
    with file_lock:  # Ensure filename manipulation is thread-safe
        # Split the file path into base and extension
        file_base, file_ext = os.path.splitext(full_path)

        # Parse the existing evaluation suffixes to prevent duplication
        suffix = f"-E{eval_sample}"
        if not file_base.endswith(suffix):  # Only append if not already there
            file_base += suffix

        # Construct the final path
        copy_path = f"{file_base}{file_ext}"
        return copy_path

def thread_safe_print(message):
    with print_lock:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()

# Helper function for I/O-bound tasks (file copying)
def safe_copy_file(src, dest):
    with file_lock:
        shutil.copy(src, dest)

def safe_remove(file_path):
    with file_lock:
        if os.path.exists(file_path):  # Check if the file exists before attempting to delete
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")

# Helper function for CPU-bound tasks (matrix manipulations)
def process_sample_cpu(evalSample, evalOffset, trainPath, evalPath, generatedEvalPath, closestSources, info):
    local_eval_data = []
    local_generated_eval_data = []

    evalSource, eval_sentenceNumber = chosenDataSet.getSourceAndSentenceIndex(evalOffset + evalSample, "Evaluation")
    if(info):
        thread_safe_print(f"Starting Evaluation for Evaluation-Sample {evalSample} (Actual Evaluation-Source: {evalSource}:{eval_sentenceNumber})")

    for (train_dirpath, _, train_filenames) in os.walk(trainPath):
        for train_filename in train_filenames:
            train_full_path = os.path.join(train_dirpath, train_filename)

            if os.path.exists(train_full_path):

                layerNumber, sourceNumber, train_sentenceNumber = getInformationFromFileName(train_full_path)
                eval_full_path = os.path.join(evalPath, f"Layer{layerNumber}", f"Source={evalSource}", f"Sentence{eval_sentenceNumber}-0.parquet")
                generated_eval_full_path = os.path.join(generatedEvalPath, f"Layer{layerNumber}", f"Source={evalSource}", f"Sentence{eval_sentenceNumber}-0.parquet")

                evalPathExists, generatedEvalPathExists = os.path.exists(eval_full_path), os.path.exists(generated_eval_full_path)
                toCheck = []
                normalizedTrainNeurons = getNormalizedValues(train_full_path, evalSample)

                if evalPathExists:
                    normalizedEvalNeurons = getNormalizedValues(eval_full_path, evalSample)
                    max_eval_rows = max(normalizedTrainNeurons.shape[0], normalizedEvalNeurons.shape[0])
                    max_eval_cols = max(normalizedTrainNeurons.shape[1], normalizedEvalNeurons.shape[1])
                    toCheck.append((local_eval_data, normalizedEvalNeurons, max_eval_rows, max_eval_cols))

                if generatedEvalPathExists:
                    normalizedGeneratedEvalNeurons = getNormalizedValues(generated_eval_full_path, evalSample)
                    max_generated_eval_rows = max(normalizedTrainNeurons.shape[0], normalizedGeneratedEvalNeurons.shape[0])
                    max_generated_eval_cols = max(normalizedTrainNeurons.shape[1], normalizedGeneratedEvalNeurons.shape[1])
                    toCheck.append((local_generated_eval_data, normalizedGeneratedEvalNeurons, max_generated_eval_rows, max_generated_eval_cols))

                if len(toCheck) > 0:
                    for (currentData, currentNeurons, max_rows, max_cols) in toCheck:
                        alignedTrain = sp.coo_matrix((normalizedTrainNeurons.data,
                                                      (normalizedTrainNeurons.row, normalizedTrainNeurons.col)),
                                                     shape=(max_rows, max_cols))
                        alignedEval = sp.coo_matrix((currentNeurons.data,
                                                     (currentNeurons.row, currentNeurons.col)),
                                                    shape=(max_rows, max_cols))
                        common_mask = alignedTrain.multiply(alignedEval)
                        differencesBetweenSources = sp.coo_matrix(np.abs(alignedTrain - alignedEval).multiply(common_mask))

                        for neuron_idx, difference in zip(differencesBetweenSources.col, differencesBetweenSources.data):
                            sparse_traincol = normalizedTrainNeurons.getcol(neuron_idx)
                            sparse_evalcol = currentNeurons.getcol(neuron_idx)
                            if sparse_traincol.nnz > 0:
                                neuron_value = sparse_traincol.data[0]
                                eval_neuron_value = sparse_evalcol.data[0] if sparse_evalcol.nnz > 0 else 0
                                current_source = f"{sourceNumber}:{train_sentenceNumber}"
                                currentData.append({'evalSample': evalSample, 'layer': layerNumber, 'neuron': neuron_idx,
                                                    'source': current_source, 'eval_neuron_value': eval_neuron_value, 'neuron_value': neuron_value, 'difference': difference})

    return local_eval_data, local_generated_eval_data

def process_sample_io(evalSample, evalOffset, trainPath, evalPath, generatedEvalPath, layersToCheck, info):
    # I/O-bound operations such as file copying and reading parquet files
    evalSource, eval_sentenceNumber = chosenDataSet.getSourceAndSentenceIndex(evalOffset + evalSample, "Evaluation")
    if info:
        thread_safe_print(f"Starting I/O-bound tasks for Evaluation-Sample {evalSample} (Actual Evaluation-Source: {evalSource}:{eval_sentenceNumber})")

    to_copy = []
    for (train_dirpath, _, train_filenames) in os.walk(trainPath):
        for train_filename in train_filenames:
            train_full_path = os.path.join(train_dirpath, train_filename)
            if not os.path.exists(train_full_path):
                continue

            layerNumber, sourceNumber, train_sentenceNumber = getInformationFromFileName(train_full_path)
            if (layerNumber in layersToCheck or layersToCheck == []):
                eval_full_path = os.path.join(evalPath, f"Layer{layerNumber}", f"Source={evalSource}", f"Sentence{eval_sentenceNumber}-0.parquet")
                generated_eval_full_path = os.path.join(generatedEvalPath, f"Layer{layerNumber}", f"Source={evalSource}", f"Sentence{eval_sentenceNumber}-0.parquet")

                # Queue up file copies (avoiding actual copying until all paths are gathered)
                to_copy.append((train_full_path, eval_full_path, generated_eval_full_path))

    # Perform file copying concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(safe_copy_file, src, dest) for (src, dest, _) in to_copy
        ]
        # Ensure all copying is done
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return to_copy  # Return file paths for CPU-bound processing

def getClosestSourcesFromDf(df, closestSources):
    # Sort by evalSample, layer, neuron, and difference
    closest_sources = (
        df.sort_values(by=['evalSample', 'layer', 'neuron', 'difference'])  # Sort by evalSample, layer, neuron, then difference
        .groupby(['evalSample', 'layer', 'neuron'], group_keys=False)  # Group by evalSample, layer, and neuron
        .head(closestSources)  # Select top N rows per group
    )
    return closest_sources

def identifyClosestLLMSources(evalSamples, evalOffset, closestSources, onlyOneEvaluation=False, layersToCheck=[], info=True):
    global layers, layerSizes, fileName

    trainPath = os.path.join(baseDirectory, "Training")
    evalPath = os.path.join(baseDirectory, "Evaluation", "Sample")
    generatedEvalPath = os.path.join(baseDirectory, "Evaluation", "Generated")

    # Initialize lists for direct usage
    eval_data = []
    generated_eval_data = []

    # Step 1: Handle I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        io_futures = [
            executor.submit(process_sample_io, sampleNumber, evalOffset, trainPath, evalPath, generatedEvalPath, layersToCheck, info)
            for sampleNumber in range(evalSamples)
        ]
        for future in concurrent.futures.as_completed(io_futures):
            try:
                future.result()  # Ensure I/O tasks complete
            except Exception as e:
                if not onlyOneEvaluation:
                    print(f"I/O Task Exception for sample: {e}")

    # Step 2: Handle CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        cpu_futures = [
            executor.submit(process_sample_cpu, sampleNumber, evalOffset, trainPath, evalPath, generatedEvalPath, closestSources, info)
            for sampleNumber in range(evalSamples)
        ]

        for future in concurrent.futures.as_completed(cpu_futures):
            try:
                local_eval_data, local_generated_eval_data = future.result()

                # Thread-safe addition to shared data
                with data_lock:
                    eval_data.extend(local_eval_data)
                    generated_eval_data.extend(local_generated_eval_data)
            except Exception as e:
                if not onlyOneEvaluation:
                    print(f"CPU Task Exception for sample: {e}")

    # Create the DataFrame from the collected data
    eval_df = pd.DataFrame(eval_data)
    generated_eval_df = pd.DataFrame(generated_eval_data)

    # Extract closest sources from DataFrame
    #eval_df = getClosestSourcesFromDf(eval_df, closestSources)
    generated_eval_df = getClosestSourcesFromDf(generated_eval_df, closestSources)

    # Set index for easier retrieval
    #eval_df.set_index(['evalSample'], inplace=True)
    generated_eval_df.set_index(['evalSample'], inplace=True)

    # Save the DataFrame as Parquet
    #eval_df.to_parquet('identifiedClosestEvalSources.parquet', compression='zstd')
    generated_eval_df.to_parquet('identifiedClosestGeneratedEvalSources.parquet', compression='zstd')

    return eval_df, generated_eval_df

#-------------------------------------------Debug----------------------------
def save_sparse_3d_array(array, filename):
    # Flatten the 3D array to a 2D array where each row represents one element
    shape = array.shape
    flat_array = array.reshape(-1)

    # Convert flattened array to COO format (ignoring zeros)
    sparse_array = sp.coo_matrix(flat_array)

    # Save row indices, column indices (which we interpret as flat indices), and data
    np.savetxt(filename, np.column_stack((sparse_array.col, sparse_array.data)), fmt='%d %.6f')

    # Save the shape for reconstruction
    with open(filename, 'a') as f:
        f.write(f"# shape: {shape}\n")

def getValuesCount(dictionary):
    # Get unique values and their counts
    unique_values, counts = np.unique(dictionary, return_counts=True)

    # Calculate the total number of elements in the dictionary
    total_elements = np.prod(dictionary.shape)  # Total elements in the 3D array

    # Flatten the dictionary to work with all values easily
    flat_values = dictionary.flatten()

    # Use Counter to count occurrences of each value
    counts = Counter(flat_values)

    # Calculate the percentage of unique values
    unique_count = len(unique_values)
    unique_percentage = (unique_count / total_elements) * 100

    # Print unique counts and percentage
    print("Unique counts: ", unique_count, ", Total elements: ", total_elements, ", Unique Percentage: ", unique_percentage)

    # Collect non-unique values
    non_unique_list = [value for value, count in counts.items() if count > 1 for _ in range(count)]

    # Use Counter on the non-unique list to get counts
    non_unique_counts = Counter(non_unique_list)

    print(non_unique_counts.most_common()[:100])

    return unique_values

def getValueClusters(dictionary):
    # Create a dictionary to hold lists of neurons by their values
    value_clusters = {}

    # Define cluster ranges (you can customize these based on your needs)
    clusters = {
        "(-1000, -1)": (-1000, -1),
        "(-1, 0)": (-1, 0),
        "(0, 1)": (0, 1),
        "(1, 1000)": (1, 1000)
    }

    # Populate the dictionary
    for source in range(dictionary.shape[0]):
        for layer in range(dictionary.shape[1]):
            for neuron in range(dictionary.shape[2]):
                current = dictionary[source][layer][neuron]

                if (current != 0.0):
                    # Identify which cluster the current value falls into
                    for cluster_name, (lower, upper) in clusters.items():
                        if lower <= current < upper:
                            if cluster_name not in value_clusters:
                                value_clusters[cluster_name] = []
                            value_clusters[cluster_name].append((source, layer, neuron, current))
                            break  # Stop checking once the value is added to a cluste

    # Display the results
    for cluster_name, neurons in value_clusters.items():
        print(f"{cluster_name}: {len(neurons)} activations, Min: {min(neuron[-1] for neuron in neurons)}, Max: {max(neuron[-1] for neuron in neurons)}")
        #for source, layer, neuron, value in neurons:
            #print(f"  Source: {source}, Layer: {layer}, Neuron: {neuron}, Value: {value}")

        # Function to find the first differing position of two float values
def find_differing_position(val1, val2):
    str_val1 = f"{val1:.10f}"  # Convert to string with fixed precision
    str_val2 = f"{val2:.10f}"  # Convert to string with fixed precision

    # Compare the two strings character by character
    for i in range(min(len(str_val1), len(str_val2))):
        if str_val1[i] != str_val2[i]:
            return i  # Return the position where they differ
    return min(len(str_val1), len(str_val2))  # If they are identical up to the length of the shorter one


def getMinimumPrecision(unique_values):
    # Define the valid float precisions and their approximate decimal places
    float_precisions = {
        np.float128: 33,  # 33 to 34 decimal places (reference for float128 comparison)
        np.float64: 15,   # Reference precision (original)
        np.float32: 7,    # 7 to 8 decimal places
        np.float16: 3     # 3 to 4 decimal places
    }

    # Add custom precisions for 1 to 8 decimal places
    for i in range(1, 9):
        float_precisions[f'float_{9-i}'] = 9-i

    def calculate_mse(original_values, rounded_values):
        # Calculate the Mean Squared Error
        mse = np.mean((original_values - rounded_values) ** 2)
        return mse

    mse_results = {}
    loss_results = {}

    # Step 1: Compare each precision with np.float128 (highest precision reference)
    print("Comparing float128 (highest precision) with other float types:")
    for float_type, precision in float_precisions.items():
        if isinstance(float_type, str):  # Custom precision (float_1 to float_8)
            rounded_values = np.round(unique_values, decimals=precision)
            mse = calculate_mse(unique_values.astype(np.float128), rounded_values)
        else:  # Standard float types (float128, float64, etc.)
            rounded_values = np.round(unique_values.astype(float_type), decimals=precision)
            mse = calculate_mse(unique_values.astype(np.float128), rounded_values)

        mse_results[float_type] = mse
        loss_percentage = mse / np.mean(unique_values.astype(np.float128)**2) * 100
        loss_results[float_type] = loss_percentage

        precision_name = float_type if isinstance(float_type, str) else float_type.__name__
        precision_bits = precision if isinstance(float_type, str) else np.dtype(float_type).itemsize * 8
        print(f"Float type: {precision_name}, Precision: {precision_bits} bits, "
              f"MSE: {mse}, Loss of Information: {loss_percentage}%")

    return mse_results, loss_results

def mse(true_values, predicted_values):
    return np.mean((true_values - predicted_values) ** 2)

def compare_precision_results(closestSources, outputs):
    # Define the precision levels
    float_precisions = {
        np.float128: 33,  # 33 to 34 decimal places (reference for float128 comparison)
        np.float64: 15,   # Reference precision (original)
        np.float32: 7,    # 7 to 8 decimal places
        np.float16: 3     # 3 to 4 decimal places
    }

    # Add custom precisions for 1 to 8 decimal places
    for i in range(1, 9):
        float_precisions[f'float_{9-i}'] = 9 - i

    # Use the highest precision (np.float128) as the base reference
    base_results, _, _ = identifyClosestSources(closestSources, outputs)  # Using np.float128

    # Run the RENN.getMostUsedSources method with base precision
    mostUsedBase = getMostUsedSources(base_results, closestSources)

    # Store the precision levels and their corresponding differences
    precision_results = {}

    for float_type, precision in float_precisions.items():
        # Handle custom float precisions differently
        if isinstance(float_type, str):  # Custom precision (like 'float_8')
            # Round values in outputs and dictionary to the specified decimal places
            rounded_outputs = np.round(outputs, decimals=precision)
            rounded_dictionary = np.round(activationsByLayers, decimals=precision)
        else:  # Standard numpy float types
            rounded_outputs = outputs.astype(float_type)
            rounded_dictionary = activationsByLayers.astype(float_type)

        # Run identifyClosestSources with the rounded or cast values
        results, _, _ = identifyClosestSources(closestSources, rounded_outputs)

        # Run RENN.getMostUsedSources with the new results
        mostUsed = getMostUsedSources(results, closestSources)

        # Compare the results from getMostUsedSources (this is where you compare)
        if mostUsed == mostUsedBase:
            precision_results[float_type] = precision

    # Find the precision with the least decimals but still producing identical results
    if precision_results:
        best_precision = min(precision_results, key=precision_results.get)
    else:
        best_precision = np.float128  # Fallback to np.float128 if no match was found

    print(f"Best Precision with least decimals: {best_precision}")
    return best_precision, precision_results

def analyzeData(closestSources, outputs, analyzeActivationsBySources = True):
    dictionary = activationsBySources
    if(analyzeActivationsBySources):
        dictionary = activationsByLayers

    #save_sparse_3d_array(dictionary, 'SparseArray.txt')
    #unique_values = getValuesCount(dictionary)
    compare_precision_results(closestSources, outputs)

    #getValueClusters(dictionary)
    #getMinimumPrecision(unique_values)
    print("\n")

def createMetricsArray(data):
    processor = MetricProcessor()
    processor.preprocess(data)
    results = processor.calculate(data)

    return list(results.values())

class MetricProcessor:
    def __init__(self, comparison_value=0.5):
        self.comparison = None
        self.reference = None
        self.variances = None
        self.round_cache = {}

    def preprocess(self, data):
        """One-time preprocessing for all metrics"""
        # Base arrays
        self.comparison = np.full_like(data, 0.5)
        self.reference = np.linspace(0, 1, len(data))

        # Variance cache
        self.variances = np.var(np.vstack([data, self.comparison]), axis=0) + 1e-10

        # Discrete metric caches
        self.round_cache = {
            'lev1_d': [f"{x:.1f}" for x in np.round(data, 1)],
            'lev1_c': [f"{x:.1f}" for x in np.round(self.comparison, 1)],
            'round2_d': set(np.round(data, 2)),
            'round2_c': set(np.round(self.comparison, 2))
        }

    def calculate(self, data):
        """Calculate all metrics with preprocessed data"""
        return {
            # L-family
            'L2 norm (Euclidean)': METRICS['L2 norm (Euclidean)'](data, self.comparison),
            'Squared Euclidean': METRICS['Squared Euclidean'](data, self.comparison),
            'L1 norm (Manhattan)': METRICS['L1 norm (Manhattan)'](data, self.comparison),
            'Canberra': METRICS['Canberra'](data, self.comparison),
            'L∞ norm (Chebyshev)': METRICS['L∞ norm (Chebyshev)'](data, self.comparison),
            'Lp norm (Minkowski p=3)': METRICS['Lp norm (Minkowski p=3)'](data, self.comparison),

            # Correlations
            'Cosine Similarity': METRICS['Cosine Similarity'](data, self.comparison),
            'Pearson Correlation': METRICS['Pearson Correlation'](data, self.reference),
            'Spearman Correlation': METRICS['Spearman Correlation'](data, self.reference),

            # Statistical
            'Mahalanobis': METRICS['Mahalanobis'](data, self.comparison, self.variances),
            'Standardized Euclidean': METRICS['Standardized Euclidean'](data, self.comparison, self.variances),
            'Chi-square': METRICS['Chi-square'](data, self.comparison),
            'Jensen-Shannon': METRICS['Jensen-Shannon'](data, self.comparison),

            # Discrete
            'Levenshtein': METRICS['Levenshtein'](''.join(self.round_cache['lev1_d']), ''.join(self.round_cache['lev1_c'])),
            'Hamming': METRICS['Hamming'](data, self.comparison),
            'Jaccard/Tanimoto': METRICS['Jaccard/Tanimoto'](self.round_cache['round2_d'], self.round_cache['round2_c']),
            'Sørensen–Dice': METRICS['Sørensen–Dice'](self.round_cache['round2_d'], self.round_cache['round2_c'])
        }

class MetricWeightOptimizer:
    def __init__(self, metric_names, top_k=10, learning_rate=0.2, reg_strength=0.01):
        self.weights = {name: 1.0 for name in metric_names}
        self.top_k = top_k
        self.base_lr = learning_rate
        self.reg_strength = reg_strength
        self.iteration = 0
        self.margin = 0.05  # Minimum score difference margin
        self.best_weights = None
        self.best_score = -np.inf
        self.gradient_history = {name: [] for name in metric_names}

    def _calculate_gradient(self, metric_scores, target_indices):
        gradients = {name: 0.0 for name in self.weights}
        combined = np.sum([self.weights[name] * metric_scores[name]
                           for name in self.weights], axis=0)

        predicted_ranking = np.argsort(combined)[:self.top_k]
        target_set = set(target_indices[:self.top_k])

        # Calculate pairwise gradients
        for target_idx in target_set:
            for pred_idx in predicted_ranking:
                if pred_idx == target_idx:
                    continue

                # Margin-based gradient calculation
                score_diff = combined[pred_idx] - combined[target_idx]
                if score_diff > -self.margin:
                    for name in self.weights:
                        feature_diff = metric_scores[name][pred_idx] - metric_scores[name][target_idx]
                        gradients[name] += feature_diff * (1 if pred_idx not in target_set else -1)

        # Apply regularization
        for name in self.weights:
            gradients[name] -= 2 * self.reg_strength * self.weights[name]

        return gradients

    def update_weights(self, metric_scores, target_indices):
        gradients = self._calculate_gradient(metric_scores, target_indices)

        # Adaptive learning rate with decay
        lr = self.base_lr / (1 + 0.001 * self.iteration)

        # Update weights with gradient clipping
        for name in self.weights:
            grad = np.clip(gradients[name], -1.0, 1.0)
            self.weights[name] += lr * grad
            self.gradient_history[name].append(grad)

        # Apply constraints
        for name in self.weights:
            self.weights[name] = np.clip(self.weights[name], 0.5, 3.0)

        self.iteration += 1

        # Track best weights
        current_score = self._evaluate(metric_scores, target_indices)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_weights = self.weights.copy()

    def _evaluate(self, metric_scores, target_indices):
        combined = np.sum([self.weights[name] * metric_scores[name]
                           for name in self.weights], axis=0)
        predicted = set(np.argsort(combined)[:self.top_k])
        target = set(target_indices[:self.top_k])
        return len(predicted & target) / self.top_k

    def get_weights(self):
        return self.best_weights if self.best_weights else self.weights.copy()

class ComponentOptimizer:
    global optimal_components
    def __init__(self, max_components=50, history_size=100):
        self.max_components = max_components
        self.history = []
        self.component_errors = {}
        self.best_components = None
        self.history_size = history_size

    def update(self, original, truncated_versions):
        """Update statistics with new data sample"""
        errors = []
        for n in range(1, min(self.max_components + 1, len(original))):
            # Calculate reconstruction error for this component count
            reconstructed = np.zeros_like(original)
            indices = np.argsort(-np.abs(original))[:n]
            reconstructed[indices] = original[indices]
            error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
            errors.append((n, error))

        self.history.append(errors)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_optimal_components(self):
        """Determine optimal component count based on accumulated data"""
        if not self.history:
            return self.max_components  # Default to max

        # Aggregate errors across all samples
        error_sums = defaultdict(float)
        counts = defaultdict(int)

        for sample in self.history:
            for n, err in sample:
                error_sums[n] += err
                counts[n] += 1

        # Calculate mean errors and find optimal
        mean_errors = {n: error_sums[n]/counts[n] for n in error_sums}
        sorted_components = sorted(mean_errors.items(), key=lambda x: (x[1], x[0]))

        # Find the knee point (best trade-off)
        best_n = sorted_components[0][0]
        for n, err in sorted_components[1:]:
            if err > 1.1 * sorted_components[0][1]:
                break
            if n < best_n:
                best_n = n
        
        return best_n