import torch
from torch import nn
import numpy as np
import concurrent.futures
import threading
import sys
import itertools
from collections import Counter
import joblib
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.stats import spearmanr, kendalltau, pearsonr, skew, kurtosis, entropy, wasserstein_distance
from time import perf_counter
from collections import defaultdict
import heapq
import math
import copy
import random
import LLM_Small1x1 as Small1x1
import LLM_GPT2 as GPT2
import LLM_LSTM as LSTM
import scipy.sparse as sp
import shutil
import os


mtEvaluation, NumberOfComponents = True, 20
comparison_values = [0.1, 0.5, 0.9]

layer, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, totalLayers = 0, 0, [], [], [], [], 0
llm, metricsEvaluation, useBitNet, layerSizes, device, hidden_sizes, layers, currentLayer, relevantLayerIndices = False, False, False, [], "", [], 0, [], []
sourceArray, fileName, contextLength, io, pd, pa, pq, zstd, levenshtein, cma, chosenDataSet, baseDirectory = "", "", 1, "", "", "", "", "", "", "", "", "./LookUp"
metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, metricsActivationsBySources, metricsActivationsByLayers, layersToCheck = [], [], [], [], []
mmDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, mtActivationsBySources, mtActivationsByLayers = [], [], [], []


# ================== Helper Functions (Standalone) ==================

def _normalize_safe(arr):
    """Normalizes array to sum to 1, handling edge cases."""
    # Ensure input is a NumPy array of floats
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=float)
    elif arr.dtype != float:
        arr = arr.astype(float)

    if arr.size == 0:
        return np.array([], dtype=float)

    min_val = np.min(arr)
    # Shift to be non-negative, add epsilon for stability
    shifted_arr = arr - min_val + 1e-10
    arr_sum = np.sum(shifted_arr)

    # Check if sum is effectively zero (handles all same value or all zero cases)
    if arr_sum < 1e-10:
        return np.full(arr.shape, 1.0 / arr.size, dtype=float) if arr.size > 0 else np.array([], dtype=float)
    else:
        return shifted_arr / arr_sum

# Optimization: Added epsilon inside sqrt for robustness in Mahalanobis/Std Euclidean.
# Combined L2/L1/Linf norms for data vector into one section.
METRICS = {
    # === 1. L-family distances (vs comparison vector `c`) ===
    'L2 norm (Euclidean)': lambda d, c, **_: np.sqrt(np.sum((d - c)**2)),
    'Squared Euclidean': lambda d, c, **_: np.sum((d - c)**2),
    'L1 norm (Manhattan)': lambda d, c, **_: np.sum(np.abs(d - c)),
    'Canberra': lambda d, c, **_: np.sum(np.abs(d - c) / (np.abs(d) + np.abs(c) + 1e-10)),
    'L∞ norm (Chebyshev)': lambda d, c, **_: np.max(np.abs(d - c)),
    'Lp norm (Minkowski p=3)': lambda d, c, **_: np.sum(np.abs(d - c)**3)**(1/3),

    # === 2. Correlation measures (vs comparison `c` or reference `ref`) ===
    'Cosine Similarity': lambda d, c, **_: (1 - distance.cosine(d, c) if (np.linalg.norm(d) > 1e-9 and np.linalg.norm(c) > 1e-9) else 0.0),
    'Pearson Correlation': lambda d, ref, **_: np.corrcoef(d, ref)[0, 1] if (np.std(d) > 1e-9 and np.std(ref) > 1e-9) else 0.0, # Check std dev for both
    'Spearman Correlation': lambda d, ref, **_: spearmanr(d, ref).correlation if (np.std(d) > 1e-9 and np.std(ref) > 1e-9) else 0.0, # Check std dev for both

    # === 3. Statistical distances (vs comparison vector `c`) ===
    'Mahalanobis': lambda d, c, v, **_: np.sqrt(np.sum((d - c)**2 / v)), # Variance `v` precomputed
    'Standardized Euclidean': lambda d, c, v, **_: np.sqrt(np.sum((d - c)**2 / v)), # Variance `v` precomputed
    'Chi-square': lambda d, c, **_: np.sum(np.where((d + c) > 1e-10, (d - c)**2 / (d + c + 1e-10), 0)), # Added epsilon check
    # Note: Jensen-Shannon & KL require normalized inputs
    'Jensen-Shannon': lambda d, c, **_: distance.jensenshannon(_normalize_safe(d), _normalize_safe(c)),
    'KL Divergence': lambda d, c, **_: entropy(_normalize_safe(d), _normalize_safe(c)), # P=data, Q=baseline
    'KL Divergence Reversed': lambda d, c, **_: entropy(_normalize_safe(c), _normalize_safe(d)), # P=baseline, Q=data
    'Wasserstein': lambda d, c, **_: wasserstein_distance(d, c),

    # === 4. Discrete metrics (comparing processed data vs processed baseline) ===
    # Operate on pre-processed strings/sets/arrays from cache
    'Levenshtein (Rounded Strings)': lambda lev1_d_str, lev1_c_str, **_: levenshtein(lev1_d_str, lev1_c_str),
    'Hamming (Rounded Values)': lambda round2_d_arr, round2_c_arr, **_: np.count_nonzero(round2_d_arr != round2_c_arr),
    'Jaccard (Rounded Sets)': lambda round2_d_set, round2_c_set, **_: len(round2_d_set & round2_c_set) / max(len(round2_d_set | round2_c_set), 1),
    'Sørensen–Dice (Rounded Sets)': lambda round2_d_set, round2_c_set, **_: 2 * len(round2_d_set & round2_c_set) / max((len(round2_d_set) + len(round2_c_set)), 1),

    # === 5. Intrinsic Statistical Properties (of data vector `d`) ===
    'Mean': lambda d, **_: np.mean(d),
    'Median': lambda d, **_: np.median(d),
    'Variance': lambda d, **_: np.var(d),
    'Standard Deviation': lambda d, **_: np.std(d),
    'Skewness': lambda d, **_: skew(d),
    'Kurtosis': lambda d, **_: kurtosis(d), # Fisher (normal=0)
    'Min': lambda d, **_: np.min(d),
    'Max': lambda d, **_: np.max(d),
    'Peak-to-Peak Range': lambda d, **_: np.ptp(d),
    'Shannon Entropy': lambda d, **_: entropy(_normalize_safe(d)), # Requires normalized input

    # === 6. Intrinsic Norms & Sparsity (of data vector `d`) ===
    'L2 Norm': lambda d, **_: np.linalg.norm(d, ord=2),
    'L1 Norm': lambda d, **_: np.linalg.norm(d, ord=1),
    'L_inf Norm': lambda d, **_: np.linalg.norm(d, ord=np.inf),
    'L0 Norm (eps=1e-6)': lambda d, **_: np.count_nonzero(np.abs(d) > 1e-6),
    'L1/L2 Ratio': lambda d, **_: np.linalg.norm(d, ord=1) / (np.linalg.norm(d, ord=2) + 1e-10), # Sparsity measure
}

def initializePackages(devicePackage, ioPackage, pdPackage, paPackage, pqPackage, zstdPackage, levenshteinPackage, cmaPackage, chosenDataSetPackage, seed="", useBitLinear=False):
    global device, useBitNet, io, pd, pa, pq, zstd, levenshtein, cma, chosenDataSet

    device, io, pd, pa, pq, zstd, levenshtein, cma, chosenDataSet = devicePackage, ioPackage, pdPackage, paPackage, pqPackage, zstdPackage, levenshteinPackage, cmaPackage, chosenDataSetPackage
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
        if not isinstance(module, CustomizableRENN) and not isinstance(module, GPT2.GPTModel) \
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
            metricsActivationsBySources = np.zeros((train_samples, totalLayers, len(METRICS) * len(comparison_values)), dtype=np.float128)
            metricsActivationsByLayers = np.zeros((totalLayers, train_samples, len(METRICS) * len(comparison_values)), dtype=np.float128)

        if mtEvaluation:
            mtActivationsBySources = np.zeros((train_samples, totalLayers, NumberOfComponents), dtype=np.float128)
            mtActivationsByLayers = np.zeros((totalLayers, train_samples, NumberOfComponents), dtype=np.float128)

    print("Hook-Dictionaries created")

def runHooks(train_dataloader, model, layersParameter=layers, llmType=False, context_length=1, lstm=False, layersToCheckParameter=[]):
    global layers, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, \
        metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, metricsActivationsBySources, metricsActivationsByLayers, \
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
        metricsDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers,len(METRICS) * len(comparison_values)), dtype=np.float128)
        metricsDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, len(METRICS) * len(comparison_values)), dtype=np.float128)
        mtDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, NumberOfComponents), dtype=np.float128)
        mtDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, NumberOfComponents), dtype=np.float128)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader, model, llmType, filename, sourceOffset, lstm)
    
        create_global_metric_combinations(4, 4)
    return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

OPTIMIZER_EVAL_DATA_CACHE = []
METRIC_WEIGHTS = defaultdict(lambda: 1.0)
mt_component_optimizer = None
optimal_components_overall = 0

GLOBAL_COMBINATION_PERFORMANCE = {}
ALL_METRIC_COMBINATIONS_TO_TEST = []

def identifyClosestSources(closestSources, outputs, metricsOutputs, mtOutputs, mode=""):
    global layers, METRIC_WEIGHTS, metrics_optimizer, mt_component_optimizer, optimal_components_overall

    # Initialize optimizer on first call
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
                differences = np.abs(neuron - outputsToCheck[currentLayer][currentNeuron])
                sorted_indices = np.argsort(differences)
                closest_indices = sorted_indices[:closestSources]
                layer_neuron_indices.append(closest_indices)
                neuron_differences.append(differences)

                # Store neuron results
                tuples = tuple(
                    (closest_indices[i], neuron[closest_indices[i]],
                     differences[closest_indices[i]])
                    for i in range(closestSources)
                )
                identifiedClosestSources[currentLayer][currentNeuron] = tuples

        # Get target indices from neuron-based approach        
        target_indices = np.concatenate([indices for indices in layer_neuron_indices if len(indices) > 0])

        if metricsEvaluation:
            identifiedClosestMetricSources[currentLayer] = evaluate_metrics(target_indices, currentMetricsLayer, metricsOutputsToCheck, currentLayer, mode, closestSources)

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
    differences = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        if evaluation == "Metrics" or evaluation == "Magnitude Truncation":
            for sourceNumber, value, difference in layer:
                if(sourceNumber != 'None'):
                    mostUsed.append(sourceNumber)
                    sourceCounter += 1
                    differences.append(difference)
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
                            differences.append(difference)
    return sourceCounter, mostUsed, differences

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

def weighted_counter(mostUsed, sourceDifferences,
                     freq_weight=0.5, prox_weight=0.5,
                     prox_power=1.0, eps=1e-9):
    sources = np.array(mostUsed)
    diffs = np.array(sourceDifferences)

    # Normalize frequencies (softmax-style)
    freq_counts = Counter(sources)
    max_freq = max(freq_counts.values()) if freq_counts else 1
    freq_scores = np.array([freq_counts[s] / max_freq for s in sources])

    # Normalized proximity scores (smaller diffs = higher scores)
    max_diff = np.max(diffs) if len(diffs) > 0 else 1
    prox_scores = 1 - (diffs / (max_diff + eps)) ** prox_power

    # Combined weighted score
    combined_scores = (freq_weight * freq_scores +
                       prox_weight * prox_scores)

    # Build weighted Counter
    counter = Counter()
    for src, score in zip(sources, combined_scores):
        counter[src] += score

    return counter

def getMostUsedSources(sources, metricsSources, mtSources, closestSources, evalSample=0, weightedMode="", info=True):
    metricsMostUsed, metricsSourceCounter = [], []
    mtMostUsed, mtSourceCounter = [], []
    if llm:
        sourceCounter, mostUsed = getMostUsedFromDataFrame(sources, evalSample, closestSources, weightedMode)
    else:
        sourceCounter, mostUsed, sourceDifferences = getMostUsed(sources, weightedMode)
        if metricsEvaluation:
            metricsSourceCounter, metricsMostUsed, metricsDifferences = getMostUsed(metricsSources, weightedMode, evaluation="Metrics")
        if mtEvaluation:
            mtSourceCounter, mtMostUsed, mtDifferences = getMostUsed(mtSources, weightedMode, evaluation="Magnitude Truncation")
    counter = weighted_counter(mostUsed, sourceDifferences)
    metricsCounter = Counter(metricsMostUsed)
    mtCounter = Counter(mtMostUsed)

    #if(info):
        #print("Total closest Sources (Per Neuron):", sourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", counter.most_common()[:closestSources])
        #if metricsEvaluation:
            #print("Total closest Sources (Metrics):", metricsSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", metricsCounter.most_common()[:closestSources])
        #if mtEvaluation:
            #print("Total closest Sources (MT):", mtSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", mtCounter.most_common()[:closestSources])
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

# === Metrics Evaluation ===
def createMetricsArray(data):
    """Creates a flat list of all metric results for given data and comparison_values."""
    resultList = []
    for comparison_value in comparison_values:
        # Optimization: Create processor once per comparison value
        processor = MetricProcessor(comparison_value=comparison_value)
        # Preprocess is called implicitly by calculate if needed
        results = processor.calculate(data)
        resultList.extend(results.values())

    # Optimization: Convert final list to NumPy array for potential downstream efficiency
    return np.array(resultList, dtype=float)

class MetricProcessor:
    def __init__(self, comparison_value=0.5, round1_prec=1, round2_prec=2):
        self.comparison_value = comparison_value
        self.round1_prec = round1_prec
        self.round2_prec = round2_prec
        self.data_len = -1 # Cache data length

        # Pre-calculated values - initialized in preprocess
        self.comparison = None
        self.reference = None
        self.variances = None
        self.round_cache = {}

    def preprocess(self, data):
        """Preprocesses data once for all metric calculations."""
        # Optimization: Convert to float array immediately and store length
        if not isinstance(data, np.ndarray) or data.dtype != float:
            data = np.array(data, dtype=float)
        self.data_len = len(data)

        # Optimization: Generate base arrays directly with correct dtype
        self.comparison = np.full(self.data_len, self.comparison_value, dtype=float)
        self.reference = np.linspace(0, 1, self.data_len, dtype=float)

        # Variance cache (stable calculation)
        # Stack data and comparison, calculate variance along axis 0 (per feature/dimension)
        # Add epsilon for numerical stability, especially if data[i] == comparison[i]
        stacked_data = np.vstack([data, self.comparison])
        self.variances = np.var(stacked_data, axis=0) + 1e-10 # Ensure non-zero variance

        # Discrete metric caches
        # Optimization: Use list comprehensions and direct joining/set creation
        fmt1 = f"{{:.{self.round1_prec}f}}"
        data_rounded1_str = [fmt1.format(x) for x in data]
        comp_rounded1_str = [fmt1.format(x) for x in self.comparison]

        data_rounded2 = np.round(data, self.round2_prec)
        comp_rounded2 = np.round(self.comparison, self.round2_prec)

        self.round_cache = {
            'lev1_d_str': "".join(data_rounded1_str),
            'lev1_c_str': "".join(comp_rounded1_str),
            'round2_d_set': set(data_rounded2),
            'round2_c_set': set(comp_rounded2),
            'round2_d_arr': data_rounded2,
            'round2_c_arr': comp_rounded2,
        }

    def calculate(self, data):
        """Calculate all metrics using preprocessed data."""
        # Ensure data is float array (might be called without preprocess)
        if not isinstance(data, np.ndarray) or data.dtype != float:
            data = np.array(data, dtype=float)

        # Preprocess if not already done or if data length changed
        # Optimization: Check if length matches to avoid reprocessing identical shapes
        if self.comparison is None or len(data) != self.data_len:
            self.preprocess(data)
            # Re-assign data in case preprocess converted it (though it shouldn't if check passed)
            if not isinstance(data, np.ndarray) or data.dtype != float:
                data = np.array(data, dtype=float)


        # Optimization: Pass all necessary precomputed values via kwargs
        # This makes the METRICS lambdas cleaner and avoids relying on `self` implicitly.
        kwargs = {
            'd': data,
            'c': self.comparison,
            'ref': self.reference,
            'v': self.variances,
            **self.round_cache # Unpack rounding cache directly
        }

        results = {}
        for name, func in METRICS.items():
            try:
                # Pass relevant args using the lambda signature (d, c, ref, v, etc.)
                # The **_ in lambda definitions ignores extra kwargs
                results[name] = func(**kwargs)
            except Exception as e:
                print(f"Warning: Error calculating metric '{name}': {type(e).__name__} - {e}. Assigning NaN.")
                results[name] = np.nan # Assign NaN on error

        return results

def evaluate_metrics(target_indices, currentMetricsLayer, metricsOutputsToCheck, currentLayer, mode, closestSources):
    """
    Uses logic similar to the original block (adapted for N*M columns)
    and returns the calculated Python tuple `tuples`.

    WARNING: Assigning the returned Python tuple (containing NumPy arrays)
    to a np.empty(..., dtype=tuple) slice in the caller is potentially unstable
    and might be the source of downstream TypeErrors. Consider changing
    the dtype of the destination array in the caller to dtype=object.
    """
    # Default empty tuple if calculations fail
    tuples = ()

    try:
        # --- Setup for N*M ---
        # Use 'globals().get' for safer access to globals, provide defaults
        global comparison_values, METRICS, METRIC_WEIGHTS, OPTIMIZER_EVAL_DATA_CACHE
        comp_vals = globals().get('comparison_values')
        metrics_dict = globals().get('METRICS')
        metric_weights = globals().get('METRIC_WEIGHTS', defaultdict(lambda: 1.0)) # Use default if missing
        cache = globals().get('OPTIMIZER_EVAL_DATA_CACHE') # Check if list later

        if not isinstance(comp_vals, (list, tuple)) or not comp_vals: return ()
        if not isinstance(metrics_dict, dict) or not metrics_dict: return ()

        num_comp_vals = len(comp_vals)
        num_metrics = len(metrics_dict)
        if num_metrics == 0: return () # Return empty if no metrics defined
        num_total_cols = num_comp_vals * num_metrics
        metric_name_list = list(metrics_dict.keys())

        # --- Input Validation and Indexing ---
        if not isinstance(metricsOutputsToCheck, (list, tuple, np.ndarray)) or currentLayer >= len(metricsOutputsToCheck):
            print(f"Error (evaluate_metrics): Check failed. currentLayer={currentLayer}, len={len(metricsOutputsToCheck)}")
            return () # Return empty tuple
        target_metrics = metricsOutputsToCheck[currentLayer]

        # --- Shape Checks ---
        if not isinstance(currentMetricsLayer, np.ndarray) or currentMetricsLayer.ndim != 2: return ()
        if currentMetricsLayer.shape[1] != num_total_cols: print(f"Warning: currentMetricsLayer shape mismatch {currentMetricsLayer.shape[1]} vs {num_total_cols}");
        if not isinstance(target_metrics, np.ndarray) or target_metrics.ndim != 1: return ()
        if target_metrics.shape[0] != num_total_cols: print(f"Warning: target_metrics shape mismatch {target_metrics.shape[0]} vs {num_total_cols}");
        num_samples = currentMetricsLayer.shape[0]
        if num_samples == 0: return ()

        # --- Calculation Logic (Adapted for N*M) ---
        raw_diffs = np.abs(currentMetricsLayer - target_metrics.reshape(1, -1))
        min_vals = np.min(currentMetricsLayer, axis=0, keepdims=True); max_vals = np.max(currentMetricsLayer, axis=0, keepdims=True)
        range_vals = max_vals - min_vals; epsilon = 1e-10; safe_range = np.where(range_vals < epsilon, epsilon, range_vals)
        norm_samples = (currentMetricsLayer - min_vals) / safe_range; norm_ref = (target_metrics.reshape(1, -1) - min_vals) / safe_range
        similarity_metric_names = {'Cosine Similarity', 'Pearson Correlation', 'Spearman Correlation', 'Jaccard (Rounded Sets)', 'Sørensen–Dice (Rounded Sets)'}

        for k in range(num_comp_vals):
            for j, name in enumerate(metric_name_list):
                col_idx = k * num_metrics + j
                if col_idx >= norm_samples.shape[1]: continue
                if name in similarity_metric_names: norm_samples[:, col_idx] = 1.0 - norm_samples[:, col_idx]; norm_ref[0, col_idx] = 1.0 - norm_ref[0, col_idx]
        score_diffs = np.abs(norm_samples - norm_ref)

        # --- Caching ---
        if mode == "Sum":
            if isinstance(target_indices, np.ndarray):
                # Create cache dict using scores from the FIRST comparison value block (k=0)
                metric_scores_for_cache = {}
                k = 0 # Index for the first comparison value
                for j, name in enumerate(metric_name_list): # Iterate base metric names (M times)
                    col_idx = k * num_metrics + j # Column index for this metric in the first block
                    if col_idx < score_diffs.shape[1]: # Check if column exists
                        # Get the score column directly, handle NaNs
                        score_col = score_diffs[:, col_idx]
                        if np.any(np.isnan(score_col)):
                            if np.all(np.isnan(score_col)): score_col = np.ones_like(score_col) * 1.0
                            else: mean_val = np.nanmean(score_col); fill_value = mean_val if np.isfinite(mean_val) else 1.0; score_col = np.nan_to_num(score_col, nan=fill_value)
                        metric_scores_for_cache[name] = score_col # Store scores from first block
                    else:
                        # Should not happen if num_comp_vals >= 1
                        metric_scores_for_cache[name] = np.ones(num_samples) * 1.0 # Assign penalty array

                # Append the structured dictionary (now with only first comp_val scores) and targets
                OPTIMIZER_EVAL_DATA_CACHE.append((copy.deepcopy(metric_scores_for_cache), copy.deepcopy(target_indices)))
                print(f"--- DEBUG: Caching scores for FIRST comparison value only ---") # Add print to confirm
            else:
                print(f"Warning (evaluate_metrics): Skipping caching for Sum mode, target_indices invalid type ({type(target_indices)}).")

        # --- Combined Score ---
        weighted_scores_sum = np.zeros(num_samples, dtype=float)
        total_weight = 0.0
        cleaned_scores_cols = {}
        for col_idx in range(num_total_cols):
            if col_idx >= score_diffs.shape[1]: continue
            score_array_col = score_diffs[:, col_idx]
            if np.any(np.isnan(score_array_col)):
                if np.all(np.isnan(score_array_col)): score_array_col = np.ones_like(score_array_col) * 1.0
                else: mean_val = np.nanmean(score_array_col); fill_value = mean_val if np.isfinite(mean_val) else 1.0; score_array_col = np.nan_to_num(score_array_col, nan=fill_value)
            cleaned_scores_cols[col_idx] = score_array_col

        for k in range(num_comp_vals):
            for j, name in enumerate(metric_name_list):
                col_idx = k * num_metrics + j
                if col_idx >= num_total_cols: continue
                # Use direct access or get depending on METRIC_WEIGHTS type
                weight = metric_weights[name] if isinstance(metric_weights, defaultdict) else metric_weights.get(name, 0.0)
                if weight > 1e-10 and col_idx in cleaned_scores_cols:
                    weighted_scores_sum += cleaned_scores_cols[col_idx] * weight
                    total_weight += weight

        # --- Sorting and Tuple Creation ---
        if total_weight < 1e-10:
            print(f"Warning (evaluate_metrics): Total weight zero for layer {currentLayer}.")
            tuples = () # Keep default empty tuple
        else:
            combined_scores = weighted_scores_sum / total_weight
            if combined_scores.ndim != 1:
                print(f"Error: combined_scores shape {combined_scores.shape}")
                tuples = ()
            else:
                if not np.all(np.isfinite(combined_scores)):
                    max_finite = np.max(combined_scores[np.isfinite(combined_scores)], initial=-np.inf); fill_value = max_finite + 1 if np.isfinite(max_finite) else 1e9
                    combined_scores = np.nan_to_num(combined_scores, nan=fill_value, posinf=fill_value, neginf=fill_value)

                sorted_metric_indices = np.argsort(combined_scores)
                num_indices_to_take = min(closestSources, len(sorted_metric_indices))
                closest_metric_indices = sorted_metric_indices[:num_indices_to_take]

                safe_tuples = [] # Use a list first
                max_layer_idx = currentMetricsLayer.shape[0] - 1
                max_raw_diffs_idx = raw_diffs.shape[0] - 1
                for i in range(num_indices_to_take):
                    idx = closest_metric_indices[i]
                    if 0 <= idx <= max_layer_idx and 0 <= idx <= max_raw_diffs_idx:
                        safe_tuples.append((
                            idx,                     # int
                            currentMetricsLayer[idx], # array[C]
                            raw_diffs[idx]            # array[C]
                        ))
                    # else: print("Warning: Index out of bounds...")
                tuples = tuple(safe_tuples) # Convert final list to tuple

    except Exception as e:
        print(f"Error (evaluate_metrics): Unexpected exception during evaluation for layer index {currentLayer}: {type(e).__name__} - {e}")
        tuples = () # Ensure return default on error

    # --- Return the calculated Python tuple directly ---
    return tuples

def create_global_metric_combinations(max_metrics_to_add, max_metrics_to_remove):
    """Generates unique metric combinations based on adding/removing from a baseline."""
    global ALL_METRIC_COMBINATIONS_TO_TEST

    all_available_metrics_set = set(METRICS.keys())
    print(f"Total available metrics: {len(all_available_metrics_set)}")

    # Baseline combination (example, should be configurable)
    current_best_metrics_tuple = ('Canberra', 'Cosine Similarity', 'L1 norm (Manhattan)', 'L2 norm (Euclidean)', 'Lp norm (Minkowski p=3)', 'L∞ norm (Chebyshev)', 'Pearson Correlation', 'Spearman Correlation', 'L0 Norm (eps=1e-6)', 'L1/L2 Ratio', 'Variance')
    baseline_metrics_set = set(current_best_metrics_tuple)

    # Verify baseline
    valid_baseline_set = baseline_metrics_set.intersection(all_available_metrics_set)
    if len(valid_baseline_set) != len(baseline_metrics_set):
        missing = baseline_metrics_set - all_available_metrics_set
        print(f"\n*** Warning: Baseline metrics not found in METRICS keys: {missing} ***")
        print(f"Using the valid subset as baseline: {sorted(list(valid_baseline_set))}\n")
        if not valid_baseline_set:
            print("Error: No valid baseline metrics remain. Stopping combination generation.")
            ALL_METRIC_COMBINATIONS_TO_TEST = []
            return
        baseline_metrics_set = valid_baseline_set # Use the valid subset

    other_metrics_set = all_available_metrics_set - baseline_metrics_set
    print(f"Defined baseline set with {len(baseline_metrics_set)} metrics.")
    print(f"Defined 'other' set with {len(other_metrics_set)} metrics available to add.")

    combinations_to_evaluate = set()
    baseline_tuple = tuple(sorted(list(baseline_metrics_set)))
    if baseline_tuple:
        combinations_to_evaluate.add(baseline_tuple)
        print(f"\nAdded baseline combination: {baseline_tuple}")

    # Add metrics
    print(f"\nGenerating combinations by adding up to {max_metrics_to_add} metrics...")
    for k in range(1, max_metrics_to_add + 1):
        if k > len(other_metrics_set): break
        added_count = 0
        for metrics_to_add in itertools.combinations(other_metrics_set, k):
            new_combo_set = baseline_metrics_set.union(set(metrics_to_add))
            new_tuple = tuple(sorted(list(new_combo_set)))
            if new_tuple not in combinations_to_evaluate:
                combinations_to_evaluate.add(new_tuple)
                added_count +=1
        if added_count > 0: print(f"  Added {added_count} new combinations by adding {k} metric(s). Total: {len(combinations_to_evaluate)}")

    # Remove metrics
    print(f"\nGenerating combinations by removing up to {max_metrics_to_remove} metrics...")
    for k in range(1, max_metrics_to_remove + 1):
        # Prevent removing too many, ensure at least 1 metric remains typically
        if k >= len(baseline_metrics_set): continue
        removed_count = 0
        for metrics_to_remove in itertools.combinations(baseline_metrics_set, k):
            new_combo_set = baseline_metrics_set.difference(set(metrics_to_remove))
            if new_combo_set: # Ensure not empty
                new_tuple = tuple(sorted(list(new_combo_set)))
                if new_tuple not in combinations_to_evaluate:
                    combinations_to_evaluate.add(new_tuple)
                    removed_count += 1
        if removed_count > 0: print(f"  Added {removed_count} new combinations by removing {k} metric(s). Total: {len(combinations_to_evaluate)}")

    ALL_METRIC_COMBINATIONS_TO_TEST = list(combinations_to_evaluate)
    print(f"\n--- Generated {len(ALL_METRIC_COMBINATIONS_TO_TEST)} unique metric combinations ---")


# --- NDCG Calculation (Logic Unchanged, Seems Efficient) ---

# --- NDCG Calculation (Unchanged) ---
def calculate_ndcg(retrieved_indices, target_set, k):
    # ... (function code as provided before) ...
    k = min(k, len(retrieved_indices))
    if k <= 0 or not target_set: return 0.0
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k]):
        if idx in target_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = 0.0
    num_targets = len(target_set)
    for i in range(min(k, num_targets)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

# --- Combined Score Helper (Unchanged) ---
def calculate_combined_score_for_sample(metric_scores_sample, weights_dict, active_metrics):
    # ... (function code as provided before) ...
    try:
        weighted_scores_sum = 0.0; total_weight = 0.0; valid_metrics_count = 0
        for name in active_metrics:
            weight = weights_dict.get(name, 0.0)
            score = metric_scores_sample.get(name)
            if score is not None and np.isfinite(score) and weight > 1e-10:
                weighted_scores_sum += score * weight; total_weight += weight; valid_metrics_count += 1
        if valid_metrics_count == 0 or total_weight < 1e-10: return None
        return weighted_scores_sum / total_weight
    except Exception: return None

# --- Helper function for parallel NDCG evaluation ---
def _process_single_sample_ndcg(sample_data, current_weights, active_metric_names, k):
    """Processes one sample from the eval_dataset to calculate NDCG."""
    try:
        if not isinstance(sample_data, tuple) or len(sample_data) != 2: return None
        metric_scores_sample, target_indices_sample = sample_data
        if not isinstance(metric_scores_sample, dict) or not hasattr(target_indices_sample, '__iter__'): return None

        target_set = set(target_indices_sample)
        if not target_set: return None # No targets to rank against

        num_source_items = -1
        source_indices = []
        # Use try-except for safer dictionary iteration/access
        try:
            first_metric_key = next(iter(metric_scores_sample))
            first_scores_array = metric_scores_sample[first_metric_key]
            if isinstance(first_scores_array, np.ndarray):
                num_source_items = len(first_scores_array)
                source_indices = list(range(num_source_items))
            else: return None # Skip sample if scores aren't arrays
        except (StopIteration, KeyError): return None # Skip if dict empty or key missing unexpectedly

        if num_source_items <= 0: return None

        # Structure scores per source item more efficiently
        # Pre-allocate lists for combined scores and valid indices
        combined_scores_for_sources = [np.nan] * num_source_items
        valid_sources_mask = np.zeros(num_source_items, dtype=bool)

        # Efficiently calculate combined scores if possible (vectorization attempt removed for clarity, use loop)
        temp_scores_per_source = defaultdict(dict)
        metric_names_in_sample = list(metric_scores_sample.keys())
        consistent_len = True
        for metric_name in metric_names_in_sample:
            scores_array = metric_scores_sample[metric_name]
            if not isinstance(scores_array, np.ndarray) or len(scores_array) != num_source_items:
                consistent_len = False
                break
            for src_idx in range(num_source_items):
                temp_scores_per_source[src_idx][metric_name] = scores_array[src_idx]
        if not consistent_len: return None # Skip if arrays have inconsistent lengths

        valid_source_indices = []
        calculated_scores_list = []
        for src_idx in source_indices:
            score = calculate_combined_score_for_sample(
                temp_scores_per_source[src_idx], current_weights, active_metric_names
            )
            if score is not None and np.isfinite(score):
                calculated_scores_list.append(score)
                valid_source_indices.append(src_idx)

        if not valid_source_indices: return None # Skip if no valid combined scores calculated

        # Get ranking (lower combined score is better)
        sorted_indices_within_valid = np.argsort(calculated_scores_list)
        retrieved_original_indices = [valid_source_indices[i] for i in sorted_indices_within_valid]

        # Calculate NDCG
        ndcg = calculate_ndcg(retrieved_original_indices, target_set, k)
        return ndcg if np.isfinite(ndcg) else None

    except Exception as e:
        # print(f"Error processing sample in parallel: {e}") # Optional debug
        return None # Return None on any error within sample processing


# --- Objective Function (OPTIMIZED with Parallelization) ---
def evaluate_average_ndcg(weights_array, active_metric_names, eval_dataset, k, n_jobs=-1):
    """
    Objective function for optimizer. Calculates -1 * Avg NDCG@k using parallel processing.
    Lower is better for minimization. n_jobs=-1 uses all available CPU cores.
    """
    # We assume eval_dataset is already sampled if desired (done in the caller)
    final_ndcg_scores = []
    try:
        # --- Input Validation ---
        if not isinstance(weights_array, np.ndarray): weights_array = np.array(weights_array, dtype=float)
        if weights_array.ndim != 1 or len(weights_array) != len(active_metric_names): return 1.0
        if not eval_dataset: return 1.0

        # --- Create weights dict ---
        current_weights = {name: max(0.0, weight) for name, weight in zip(active_metric_names, weights_array)}

        # --- Parallel Processing over eval_dataset ---
        # Use joblib for parallel execution. Prefer 'threads' if tasks release GIL (NumPy often does).
        # If tasks are heavily Python-based, 'processes' might be better but has more overhead.
        # Start with 'threads' as it has lower overhead.
        results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib.delayed(_process_single_sample_ndcg)(sample_data, current_weights, active_metric_names, k)
            for sample_data in eval_dataset
        )

        # Filter out None results (errors or skipped samples) and calculate average
        final_ndcg_scores = [score for score in results if score is not None]

        # --- Calculate final average NDCG ---
        if not final_ndcg_scores: return 1.0 # Max penalty if no samples evaluated successfully
        average_ndcg = np.mean(final_ndcg_scores)
        # print(f"Avg NDCG: {average_ndcg:.4f} from {len(final_ndcg_scores)} samples (Weights: {weights_array[:3]}...)") # Debug
        return -average_ndcg # Return negative for minimization

    except Exception as e:
        # print(f"CRITICAL ERROR evaluate_average_ndcg: Uncaught exception! Error: {type(e).__name__} - {e}. Returning 1.0")
        return 1.0

# --- Optimization Function (OPTIMIZED with Sampling and Tunable Minimize) ---
def optimize_metrics_and_weights_scipy(
        k,                                  # Int: k for NDCG@k
        min_metrics=5,                      # Int: Minimum number of metrics to keep
        optimizer_method='L-BFGS-B',        # String: Recommend 'L-BFGS-B' or 'SLSQP'
        optimizer_options=None,             # Dict: Options, e.g., {'maxiter': 100, 'ftol': 1e-5}
        eval_dataset_sample_size=None,      # Int or None: Number of samples to use from eval_dataset for optimization steps
        n_jobs=8,                          # Int: Number of parallel jobs for evaluate_average_ndcg (-1 for all cores)
        initial_weights=None,               # Dict: Optional initial weights {metric_name: weight}
        verbose=True                        # Bool: Print progress details
):
    """
    Optimizes metric selection and weights using iterative ABLATION-BASED pruning.
    Evaluates objective function in parallel and allows sampling/optimizer tuning.
    """
    if not OPTIMIZER_EVAL_DATA_CACHE: print("Error: eval_dataset is empty."); return None, None, -np.inf
    if not METRICS: print("Error: METRICS dictionary is empty."); return None, None, -np.inf

    all_metric_names = sorted(list(METRICS.keys()))
    current_metrics = all_metric_names[:]

    # Initialize weights
    if initial_weights: current_weights_dict = {name: max(0.0, initial_weights.get(name, 0.0)) for name in all_metric_names}
    else: current_weights_dict = {name: 1.0 for name in all_metric_names}
    initial_sum = sum(current_weights_dict.values())
    if initial_sum > 1e-10: current_weights_dict = {name: w / initial_sum for name, w in current_weights_dict.items()}
    else: num_metrics = len(all_metric_names); current_weights_dict = {name: 1.0 / num_metrics if num_metrics > 0 else 0.0 for name in all_metric_names}

    best_overall_ndcg = -np.inf # Initialize low
    best_overall_metrics = current_metrics[:]
    best_overall_weights = current_weights_dict.copy()

    # Prepare Optimizer Options
    default_options = {'maxiter': 100, 'disp': False, 'ftol': 1e-6, 'gtol': 1e-5}
    if optimizer_method in ['Nelder-Mead']: default_options.pop('ftol', None); default_options.pop('gtol', None); default_options['adaptive'] = True
    if optimizer_options: default_options.update(optimizer_options)

    iteration = 0
    max_iterations = len(all_metric_names) - min_metrics + 1

    # Prepare eval dataset (full or sampled) for consistent use in optimization steps
    if eval_dataset_sample_size and eval_dataset_sample_size < len(OPTIMIZER_EVAL_DATA_CACHE):
        if verbose: print(f"Sampling {eval_dataset_sample_size} entries from eval_dataset (size {len(OPTIMIZER_EVAL_DATA_CACHE)}) for optimization.")
        current_eval_dataset = random.sample(OPTIMIZER_EVAL_DATA_CACHE, eval_dataset_sample_size)
    else:
        current_eval_dataset = OPTIMIZER_EVAL_DATA_CACHE
        if verbose and eval_dataset_sample_size: print(f"Sample size >= dataset size. Using full dataset.")
    if not current_eval_dataset: print("Error: Eval dataset empty."); return None, None, -np.inf

    # --- Main Optimization Loop ---
    while len(current_metrics) > min_metrics:
        iteration += 1
        if iteration > max_iterations + 5: print("Warning: Exceeded max iterations."); break
        if verbose: print(f"\n--- Iteration {iteration}/{max_iterations}: Optimizing {len(current_metrics)} metrics ---")

        # 1. Optimize weights for the current set of metrics
        active_weights_array = np.array([current_weights_dict.get(name, 1.0 / len(current_metrics) if len(current_metrics)>0 else 0.0) for name in current_metrics]) # Use last known or default guess
        bounds = [(0, None) for _ in current_metrics] if optimizer_method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None
        constraints = ()

        optimized_ndcg_this_iter = -np.inf
        optimized_weights_dict = None
        optimization_successful = False
        final_weights_array_this_iter = active_weights_array # Keep previous weights if optim fails

        try:
            if verbose: print(f"  Running scipy.optimize.minimize (method={optimizer_method}, options={default_options})...")
            opt_result = minimize(
                evaluate_average_ndcg, x0=active_weights_array,
                args=(current_metrics, current_eval_dataset, k, n_jobs), # Pass dataset & n_jobs
                method=optimizer_method, bounds=bounds, constraints=constraints, options=default_options
            )

            if opt_result.success or 'tolerance' in opt_result.message.lower() or 'iteration' in opt_result.message.lower():
                temp_final_weights = np.maximum(0.0, opt_result.x); temp_sum = np.sum(temp_final_weights)
                if temp_sum > 1e-10: temp_final_weights /= temp_sum
                else: num_curr = len(current_metrics); temp_final_weights = np.full(num_curr, 1.0/num_curr if num_curr > 0 else 0.0)

                # Use the weights returned by the optimizer for this iteration
                final_weights_array_this_iter = temp_final_weights
                optimized_weights_dict = {name: w for name, w in zip(current_metrics, final_weights_array_this_iter)}

                # Re-evaluate score with these weights
                final_score_check = evaluate_average_ndcg(final_weights_array_this_iter, current_metrics, current_eval_dataset, k, n_jobs)

                if np.isfinite(final_score_check):
                    optimized_ndcg_this_iter = -final_score_check
                    # Update current_weights_dict *only* if optimization seemed successful and score is finite
                    current_weights_dict.update(optimized_weights_dict)
                    if verbose: print(f"  Optimization Result: Success={opt_result.success}, Status='{opt_result.message}', Optimized Avg NDCG = {optimized_ndcg_this_iter:.6f}")
                    optimization_successful = True # Mark as successful for pruning purposes
                else: print(f"  Warning Iter {iteration}: Optimizer finished but re-eval non-finite ({final_score_check}). Using previous weights for ablation.")
            else: 
                if verbose: print(f"  Optimization Failed Iter {iteration}. Message: {opt_result.message}. Using previous weights for ablation.")

        except Exception as e: print(f"  CRITICAL ERROR Iter {iteration}: Exception during minimize call: {type(e).__name__} - {e}. Using previous weights for ablation.")

        # --- Update Best Overall Result (using score from this iter if successful) ---
        current_score_to_compare = optimized_ndcg_this_iter if optimization_successful else -np.inf
        if current_score_to_compare > best_overall_ndcg:
            best_overall_ndcg = current_score_to_compare
            best_overall_metrics = current_metrics[:]
            # Store the weights that produced this best score
            best_overall_weights = {name: w for name, w in zip(current_metrics, final_weights_array_this_iter)}
            if verbose: print(f"  *** New best overall NDCG found: {best_overall_ndcg:.6f} with {len(best_overall_metrics)} metrics ***")


        # --- Ablation / Pruning (Reverted Logic) ---
        if len(current_metrics) <= min_metrics:
            if verbose: print(f"\nReached minimum number of metrics ({min_metrics}). Stopping pruning.")
            break

        metric_to_prune = None
        # Use the score achieved in *this* iteration (potentially with optimized weights) as the baseline for comparison
        baseline_ndcg_for_pruning = optimized_ndcg_this_iter if optimization_successful else evaluate_average_ndcg(active_weights_array, current_metrics, current_eval_dataset, k, n_jobs)
        if not np.isfinite(baseline_ndcg_for_pruning):
            print("  Error: Baseline NDCG for pruning is non-finite. Cannot prune. Stopping.")
            break
        baseline_ndcg_for_pruning = -baseline_ndcg_for_pruning # Convert back to positive NDCG


        min_ndcg_drop = float('inf') # We want the smallest drop (or largest increase) when removing

        if verbose: print(f"  Evaluating metric importance via ablation (baseline NDCG {baseline_ndcg_for_pruning:.6f})...")

        # Prepare arguments for parallel ablation evaluation
        ablation_tasks = []
        # Use the weights determined for this iteration (optimized or previous)
        weights_for_ablation_eval = {name: w for name, w in zip(current_metrics, final_weights_array_this_iter)}

        for i, name_to_remove in enumerate(current_metrics):
            temp_metrics = current_metrics[:i] + current_metrics[i+1:]
            if not temp_metrics: continue
            # Use weights corresponding to the temp_metrics set
            temp_weights_array = np.array([weights_for_ablation_eval.get(name, 0.0) for name in temp_metrics])
            # Optional: Re-normalize temp_weights_array? Might be better not to, to purely see impact of removal.
            ablation_tasks.append(joblib.delayed(evaluate_average_ndcg)(temp_weights_array, temp_metrics, current_eval_dataset, k, n_jobs)) # Use n_jobs here too

        if ablation_tasks:
            # Parallel execution of ablation checks
            ablation_scores = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(ablation_tasks)

            for i, name_to_remove in enumerate(current_metrics):
                if i < len(ablation_scores):
                    score_without = ablation_scores[i]
                    if isinstance(score_without, (float, int, np.number)) and np.isfinite(score_without):
                        ndcg_without_metric = -score_without # Convert back to positive NDCG
                        ndcg_drop = baseline_ndcg_for_pruning - ndcg_without_metric # How much score dropped (positive) or gained (negative)
                        if verbose: print(f"    Removing '{name_to_remove}': NDCG={ndcg_without_metric:.6f}, Drop={ndcg_drop:.6f}")

                        if ndcg_drop < min_ndcg_drop:
                            min_ndcg_drop = ndcg_drop
                            metric_to_prune = name_to_remove
                    else:
                        print(f"    Warning: Non-finite score ({score_without}) when evaluating removal of '{name_to_remove}'. Skipping.")
                else: # Should not happen if lengths match
                    print(f"    Warning: Mismatch between metrics and ablation scores for '{name_to_remove}'.")

        # Prune the least important metric
        if metric_to_prune:
            if verbose: print(f"  Pruning metric '{metric_to_prune}' (Removing caused smallest NDCG drop: {min_ndcg_drop:.6f})")
            current_metrics.remove(metric_to_prune)
            current_weights_dict.pop(metric_to_prune, None) # Remove from weights dict for next iter
            # Optional: Re-normalize remaining weights in current_weights_dict? Helps next initial guess.
            active_sum = sum(current_weights_dict.values())
            if active_sum > 1e-10:
                for name in current_metrics: current_weights_dict[name] = current_weights_dict.get(name, 0.0) / active_sum
            else: num_rem = len(current_metrics); [current_weights_dict.update({name: 1.0 / num_rem if num_rem>0 else 0.0}) for name in current_metrics]

        else:
            if verbose: print("  Could not determine metric to prune this iteration (e.g., all removals failed or hurt score significantly). Stopping.")
            break


    # --- Final Reporting ---
    print(f"\n--- Iterative Optimization Finished ---")
    # Use the best results recorded during iterations
    if best_overall_metrics and best_overall_weights:
        print(f"Best config found with {len(best_overall_metrics)} metrics:")
        print(f"  Metrics: {sorted(best_overall_metrics)}")
        print(f"  Avg NDCG@{k} (score from optimization): {best_overall_ndcg:.6f}")

        # Optional: Re-run final evaluation on full dataset if sampling was used
        if eval_dataset_sample_size and eval_dataset_sample_size < len(OPTIMIZER_EVAL_DATA_CACHE):
            print(f"  Recalculating final score on full dataset...")
            final_full_score_val = evaluate_average_ndcg(list(best_overall_weights.values()), best_overall_metrics, OPTIMIZER_EVAL_DATA_CACHE, k, n_jobs)
            if np.isfinite(final_full_score_val): print(f"  Avg NDCG@{k} (on full dataset): {-final_full_score_val:.6f}")
            else: print("  Could not calculate score on full dataset.")


        final_sum = sum(best_overall_weights.values())
        final_best_weights = {k: v / final_sum if final_sum > 1e-10 else v for k, v in best_overall_weights.items()}
    else: print("No suitable metric combination found or optimization failed early."); final_best_weights = None

    return best_overall_metrics, final_best_weights, best_overall_ndcg

# === Magnetic Truncation Evaluation ===
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