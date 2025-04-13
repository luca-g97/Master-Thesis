# Standard Library Imports
import joblib
import concurrent.futures
import copy
import itertools
import math
import time
import os
import shutil
import sys
import threading
from collections import Counter, defaultdict # Grouped imports from the same module

# Third-Party Library Imports
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn # Common alias for nn module
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.stats import (entropy, kendalltau, kurtosis, pearsonr, skew, # Sorted alphabetically
                         spearmanr, wasserstein_distance) # Line break for readability

# Local/Application-Specific Imports
import LLM_GPT2 as GPT2
import LLM_Small1x1 as Small1x1

# --- High-Level Configuration Flags ---
llm: bool = False                  # Use LLM specific logic
metricsEvaluation: bool = False    # Perform metrics evaluation
mtEvaluation: bool = True          # Perform magnitude truncation evaluation
useBitNet: bool = False            # Use BitNet specific logic/model

# --- Environment, Data & File Paths ---
device: str = ""                   # Computation device (e.g., "cuda", "cpu")
baseDirectory: str = "./LookUp"    # Base directory for lookup files or results
chosenDataSet: str = ""            # Identifier for the dataset being used
fileName: str = ""                 # Specific file name being processed
sourceArray: str = ""              # Placeholder or path for source data array

# --- Model Architecture & Parameters ---
contextLength: int = 1             # Sequence context length
layerSizes: list = []              # List of layer sizes (e.g., [768, 768, ...])
hidden_sizes: list = []            # Hidden layer sizes (potentially redundant with layerSizes?)
layers: list = []                  # Maybe holds layer objects or names?
totalLayers: int = 0               # Total number of layers in the model
relevantLayerIndices: list = []    # Indices of specific layers to focus on/analyze

# --- External Library/Tool Placeholders ---
io: str = ""                       # Placeholder for io module
pd: str = ""                       # Placeholder for pandas
pa: str = ""                       # Placeholder for pyarrow
pq: str = ""                       # Placeholder for parquet
zstd: str = ""                     # Placeholder for zstandard
levenshtein: str = ""              # Placeholder for levenshtein library
cma: str = ""                      # Placeholder for CMA-ES optimization library

# --- Core Runtime State & Processing Indices ---
layer: int = 0                     # Index of the current layer being processed
source: int = 0                    # Index of the current source/input being processed
currentLayer: int = 0              # Used for initialization of Customizable_RENN

# --- Primary Activation/Data Storage ---
dictionaryForSourceLayerNeuron: list = [] # List to store activations temporarily by source
dictionaryForLayerNeuronSource: list = [] # List to store activations temporarily by layer
activationsBySources: list = []           # List to store activations grouped by source
activationsByLayers: list = []            # List to store activations grouped by layer

# --- Metrics-Specific Data Storage ---
NumberOfComponents: int = 45 # Optimally between 44 and 47 (tested over various tries)
layersToCheck: list = []
metricsDictionaryForSourceLayerNeuron: list = [] 
metricsDictionaryForLayerNeuronSource: list = []
metricsActivationsBySources: list = []
metricsActivationsByLayers: list = []

# --- Magnitude Truncation ---
mmDictionaryForSourceLayerNeuron: list = []
mtDictionaryForLayerNeuronSource: list = []
mtActivationsBySources: list = []
mtActivationsByLayers: list = []

def _normalize_safe(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=float)
    elif arr.dtype != float:
        arr = arr.astype(float)

    if arr.size == 0: # Handle empty array
        return np.array([])

    min_val = np.min(arr)
    # Use np.maximum to ensure non-negativity robustly, adding epsilon after potential shift
    shifted_arr = np.maximum(0.0, arr - min_val) + 1e-10
    arr_sum = np.sum(shifted_arr)

    if arr_sum < 1e-10: # Check if sum is effectively zero
        # Handle case where all elements are the same (or effectively zero)
        return np.full(arr.shape, 1.0 / arr.size) if arr.size > 0 else np.array([])
    else:
        return shifted_arr / arr_sum

METRICS = {
    # === 3. Statistical distances (implicitly vs baseline comparison vector) ===
    # Note: Mahalanobis/Std Euclidean use variance computed relative to the baseline vector 'c'
    'Mahalanobis': lambda d, c, v: np.sqrt(np.sum((d - c)**2 / v)),
    'Standardized Euclidean': lambda d, c, v: np.sqrt(np.sum((d - c)**2 / v)),
    'Chi-square': lambda d, c: np.sum(np.where((d + c) > 0, (d - c)**2 / (d + c + 1e-10), 0)),
    'Jensen-Shannon': lambda d, c: distance.jensenshannon(_normalize_safe(d), _normalize_safe(c)),
    'KL Divergence': lambda d, c: entropy(_normalize_safe(d), _normalize_safe(c)), # P=data, Q=baseline
    'KL Divergence Reversed': lambda d, c: entropy(_normalize_safe(c), _normalize_safe(d)), # P=baseline, Q=data
    'Wasserstein': lambda d, c: wasserstein_distance(d, c),

    # === 4. Discrete metrics (comparing processed data vs processed baseline) ===
    'Levenshtein (Rounded Strings)': lambda s1, s2: levenshtein(s1, s2),
    'Hamming (Rounded Values)': lambda d_r, c_r: np.count_nonzero(d_r != c_r),
    'Jaccard (Rounded Sets)': lambda s1, s2: len(s1 & s2) / max(len(s1 | s2), 1),
    'Sørensen–Dice (Rounded Sets)': lambda s1, s2: 2 * len(s1 & s2) / max((len(s1) + len(s2)), 1),

    # === 5. Intrinsic Statistical Properties (of data vector `d`, ignore `c`) ===
    'Mean': lambda d, c: np.mean(d),
    'Median': lambda d, c: np.median(d),
    'Variance': lambda d, c: np.var(d),
    'Standard Deviation': lambda d, c: np.std(d),
    'Skewness': lambda d, c: skew(d),
    'Kurtosis': lambda d, c: kurtosis(d),
    'Min': lambda d, c: np.min(d),
    'Max': lambda d, c: np.max(d),
    'Peak-to-Peak Range': lambda d, c: np.ptp(d),
    'Shannon Entropy': lambda d, c: entropy(_normalize_safe(d)),

    # === 6. Intrinsic Norms & Sparsity (of data vector `d`, ignore `c`) ===
    'L2 Norm': lambda d, c: np.linalg.norm(d, ord=2), # Magnitude of data vector
    'L1 Norm': lambda d, c: np.linalg.norm(d, ord=1), # Magnitude of data vector
    'L_inf Norm': lambda d, c: np.linalg.norm(d, ord=np.inf), # Max abs value in data vector
    'L0 Norm (eps=1e-6)': lambda d, c: np.count_nonzero(np.abs(d) > 1e-6),
    'L1/L2 Ratio': lambda d, c: np.linalg.norm(d, ord=1) / (np.linalg.norm(d, ord=2) + 1e-10), # Sparsity measure

    # === 1. L-family distances (implicitly vs baseline comparison vector) ===
    'L2 norm (Euclidean)': lambda d, c: np.sqrt(np.sum((d - c)**2)),
    'Squared Euclidean': lambda d, c: np.sum((d - c)**2),
    'L1 norm (Manhattan)': lambda d, c: np.sum(np.abs(d - c)),
    'Canberra': lambda d, c: np.sum(np.abs(d - c) / (np.abs(d) + np.abs(c) + 1e-10)),
    'L∞ norm (Chebyshev)': lambda d, c: np.max(np.abs(d - c)),
    'Lp norm (Minkowski p=3)': lambda d, c: np.sum(np.abs(d - c)**3)**(1/3),

    # === 2. Correlation measures (implicitly vs baseline reference vectors) ===
    'Cosine Similarity': lambda d, c: (1 - distance.cosine(d, c) if (np.linalg.norm(d) > 1e-9 and np.linalg.norm(c) > 1e-9) else 0.0),
    'Pearson Correlation': lambda d, ref: np.corrcoef(d, ref)[0, 1] if np.std(d) > 1e-9 else 0.0,
    'Spearman Correlation': lambda d, ref: spearmanr(d, ref).correlation if np.std(d) > 1e-9 else 0.0,
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
            metricsActivationsBySources = np.zeros((train_samples, totalLayers, len(METRICS_TO_USE)), dtype=np.float128)
            metricsActivationsByLayers = np.zeros((totalLayers, train_samples, len(METRICS_TO_USE)), dtype=np.float128)

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
        metricsDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, len(METRICS_TO_USE)), dtype=np.float128)
        metricsDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, len(METRICS_TO_USE)), dtype=np.float128)
        mtDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, NumberOfComponents), dtype=np.float128)
        mtDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, NumberOfComponents), dtype=np.float128)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader, model, llmType, filename, sourceOffset, lstm)

    return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

METRIC_WEIGHTS = {name: 0.0 for name in METRICS.keys()}
METRICS_TO_USE = {name: 1.0 for name in METRICS.keys()}
for metric in METRICS_TO_USE.keys():
    METRIC_WEIGHTS[metric] = METRICS_TO_USE[metric]
# Add to global initialization
mt_component_optimizer = None
optimal_components_overall = 46

OPTIMIZER_EVAL_DATA_CACHE = []

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
            metric_scores = {}
            metric_calculation_successful_overall = True # Flag to track if all metrics were processed

            num_metrics = len(METRICS_TO_USE)
            num_samples = currentMetricsLayer.shape[0]
            if currentMetricsLayer.shape[1] != num_metrics:
                print(f"Warning: Shape mismatch - currentMetricsLayer columns ({currentMetricsLayer.shape[1]}) != num_metrics ({num_metrics}).")
                metric_calculation_successful_overall = False
            if metricsOutputsToCheck[currentLayer].shape[0] != num_metrics:
                print(f"Warning: Shape mismatch - metricsOutputsToCheck length ({metricsOutputsToCheck[currentLayer].shape[0]}) != num_metrics ({num_metrics}).")
                metric_calculation_successful_overall = False

            # --- Proceed only if initial checks pass ---
            if metric_calculation_successful_overall:
                # Calculate raw differences (assuming shapes are compatible)
                raw_diffs = np.abs(currentMetricsLayer - metricsOutputsToCheck[currentLayer][np.newaxis, :])

                # Normalize per metric (column-wise)
                min_vals = np.min(currentMetricsLayer, axis=0)
                max_vals = np.max(currentMetricsLayer, axis=0)
                range_vals = max_vals - min_vals
                # Add epsilon only where range is close to zero to avoid inflating differences elsewhere
                epsilon = 1e-10
                safe_range = np.where(range_vals < epsilon, epsilon, range_vals)

                norm_samples = (currentMetricsLayer - min_vals) / safe_range
                # Apply same normalization to the reference point
                norm_ref = (metricsOutputsToCheck[currentLayer] - min_vals) / safe_range

                # Handle similarity metrics - flip scores so lower is better universally
                # Use names matching the keys in your METRICS dictionary
                similarity_metric_names = {'Cosine Similarity', 'Pearson Correlation', 'Spearman Correlation',
                                           'Jaccard (Rounded Sets)', 'Sørensen–Dice (Rounded Sets)'}
                metric_name_list = list(METRICS_TO_USE.keys()) # Get a fixed order

                for i, name in enumerate(metric_name_list):
                    if name in similarity_metric_names:
                        if i < norm_samples.shape[1]: # Check index validity
                            norm_samples[:, i] = 1.0 - norm_samples[:, i]
                        if i < norm_ref.shape[0]: # Check index validity
                            norm_ref[i] = 1.0 - norm_ref[i]

                # Store normalized absolute difference scores per metric
                for i, name in enumerate(metric_name_list):
                    if i < norm_samples.shape[1] and i < norm_ref.shape[0]:
                        score_diff = np.abs(norm_samples[:, i] - norm_ref[i])

                        # --- FIX 2: Handle potential NaNs from normalization/calculation ---
                        if np.any(np.isnan(score_diff)):
                            #print(f"Warning: NaN detected in score for metric '{name}'. Replacing with mean.")
                            if np.all(np.isnan(score_diff)):
                                score_diff.fill(1.0) # Assign default high difference if all are NaN
                            else:
                                mean_val = np.nanmean(score_diff)
                                score_diff = np.nan_to_num(score_diff, nan=mean_val)
                        metric_scores[name] = score_diff
                    else:
                        print(f"Warning: Index {i} for metric '{name}' out of bounds during score storage. Metric skipped.")
                        metric_calculation_successful_overall = False # Mark as potentially incomplete

            # --- Proceed only if metric calculation was successful ---
            if metric_calculation_successful_overall and metric_scores:
                # Use only metrics that were successfully calculated
                metrics_to_use = list(metric_scores.keys())

                if mode == "Sum":
                    # Ensure data being cached is valid
                    if metric_scores and target_indices is not None:
                        OPTIMIZER_EVAL_DATA_CACHE.append((copy.deepcopy(metric_scores), copy.deepcopy(target_indices)))
                    else:
                        print("Warning: Skipping caching due to potentially invalid metric_scores or target_indices.")

                # --- FIX 3: Robust combined score calculation ---
                weighted_scores_list = []
                metrics_used_in_mean = 0
                for name in metrics_to_use:
                    score_array = metric_scores.get(name) # Should exist if in metrics_to_use
                    weight = METRIC_WEIGHTS.get(name)     # Get weight (Corrected dict)

                    # Ensure score/weight exist and score is valid array
                    if score_array is not None and weight is not None and isinstance(score_array, np.ndarray) and score_array.size > 0:
                        weighted_scores_list.append(score_array * weight)
                        metrics_used_in_mean += 1
                    else:
                        print(f"Warning: Skipping metric '{name}' in combined score (missing score/weight/invalid score array).")


                if not weighted_scores_list or metrics_used_in_mean == 0:
                    print("Error: No valid metric scores available to combine for this sample. Cannot sort.")
                    tuples = ()
                else:
                    combined_scores = np.sum(weighted_scores_list, axis=0) / metrics_used_in_mean

                    # Ensure combined_scores is 1D before argsort
                    if combined_scores.ndim != 1:
                        print(f"Error: combined_scores has unexpected shape {combined_scores.shape}. Cannot sort.")
                        tuples = ()
                    else:
                        # Sort indices (lower combined score is better)
                        sorted_metric_indices = np.argsort(combined_scores)
                        # Ensure closestSources doesn't exceed available indices
                        num_indices_to_take = min(closestSources, len(sorted_metric_indices))
                        closest_metric_indices = sorted_metric_indices[:num_indices_to_take]

                        # --- FIX 4: Safe Indexing for Output Tuples ---
                        safe_tuples = []
                        max_layer_idx = currentMetricsLayer.shape[0] - 1
                        max_raw_diffs_idx = raw_diffs.shape[0] - 1 if 'raw_diffs' in locals() else -1 # Check if raw_diffs exists

                        for i in range(num_indices_to_take):
                            idx = closest_metric_indices[i]
                            # Check if index is valid for both arrays needed
                            if idx <= max_layer_idx and idx <= max_raw_diffs_idx:
                                safe_tuples.append((
                                    idx,
                                    currentMetricsLayer[idx],
                                    raw_diffs[idx]
                                ))
                            else:
                                print(f"Warning: Index {idx} out of bounds when creating output tuple (Layer Max: {max_layer_idx}, Diff Max: {max_raw_diffs_idx}). Skipping.")
                        tuples = tuple(safe_tuples)


                # Assign results
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

class MetricProcessor:
    def __init__(self, comparison_value=0.5, round1_prec=1, round2_prec=2):
        self.comparison_value = comparison_value
        self.round1_prec = round1_prec
        self.round2_prec = round2_prec

        self.comparison = None # Fixed comparison vector (e.g., all 0.5)
        self.reference = None  # Fixed reference vector (e.g., linspace)
        self.variances = None  # For Mahalanobis/Std Euclidean
        self.round_cache = {}  # For discrete metrics

    def preprocess(self, data):
        """One-time preprocessing for all metrics for a given data vector"""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float) # Ensure data is a numpy float array
        elif data.dtype != float:
            data = data.astype(float) # Ensure float for calculations

        # Base reference arrays
        self.comparison = np.full_like(data, self.comparison_value, dtype=float)
        self.reference = np.linspace(0, 1, len(data))

        # Variance cache (for Mahalanobis/Std Euclidean)
        # Variance is calculated for each dimension across the two values (data[i] and comparison[i])
        self.variances = np.var(np.vstack([data, self.comparison]), axis=0) + 1e-10

        # Discrete metric caches (using specified precision)
        fmt1 = f"{{:.{self.round1_prec}f}}"
        self.round_cache = {
            # Level 1 cache (strings for Levenshtein)
            'lev1_d_str': "".join([fmt1.format(x) for x in data]),
            'lev1_c_str': "".join([fmt1.format(x) for x in self.comparison]),

            # Level 2 cache (sets for Jaccard/Dice)
            'round2_d_set': set(np.round(data, self.round2_prec)),
            'round2_c_set': set(np.round(self.comparison, self.round2_prec)),

            # Level 2 cache (arrays for Hamming)
            'round2_d_arr': np.round(data, self.round2_prec),
            'round2_c_arr': np.round(self.comparison, self.round2_prec),
        }

    def calculate(self, data):
        """Calculate all metrics with preprocessed data"""
        if self.comparison is None:
            # Ensure preprocess has been called if calculate is called directly
            # This also ensures data is converted to numpy float array
            self.preprocess(data)
            # Need to re-assign data in case preprocess converted it
            if not isinstance(data, np.ndarray) or data.dtype != float:
                data = np.array(data, dtype=float) if not isinstance(data, np.ndarray) else data.astype(float)

        elif not isinstance(data, np.ndarray) or data.dtype != float:
            # Ensure data is float array even if preprocess was called separately
            data = np.array(data, dtype=float) if not isinstance(data, np.ndarray) else data.astype(float)


        results = {
            # === 3. Statistical distances ===
            'Mahalanobis': METRICS['Mahalanobis'](data, self.comparison, self.variances),
            'Standardized Euclidean': METRICS['Standardized Euclidean'](data, self.comparison, self.variances),
            'Chi-square': METRICS['Chi-square'](data, self.comparison),
            'Jensen-Shannon': METRICS['Jensen-Shannon'](data, self.comparison),
            'KL Divergence': METRICS['KL Divergence'](data, self.comparison),
            'KL Divergence Reversed': METRICS['KL Divergence Reversed'](data, self.comparison),
            'Wasserstein': METRICS['Wasserstein'](data, self.comparison),

            # === 4. Discrete metrics ===
            'Levenshtein (Rounded Strings)': METRICS['Levenshtein (Rounded Strings)'](self.round_cache['lev1_d_str'], self.round_cache['lev1_c_str']),
            'Hamming (Rounded Values)': METRICS['Hamming (Rounded Values)'](self.round_cache['round2_d_arr'], self.round_cache['round2_c_arr']),
            'Jaccard (Rounded Sets)': METRICS['Jaccard (Rounded Sets)'](self.round_cache['round2_d_set'], self.round_cache['round2_c_set']),
            'Sørensen–Dice (Rounded Sets)': METRICS['Sørensen–Dice (Rounded Sets)'](self.round_cache['round2_d_set'], self.round_cache['round2_c_set']),

            # === 5. Intrinsic Statistical Properties ===
            'Mean': METRICS['Mean'](data, self.comparison),
            'Median': METRICS['Median'](data, self.comparison),
            'Variance': METRICS['Variance'](data, self.comparison),
            'Standard Deviation': METRICS['Standard Deviation'](data, self.comparison),
            'Skewness': METRICS['Skewness'](data, self.comparison),
            'Kurtosis': METRICS['Kurtosis'](data, self.comparison),
            'Min': METRICS['Min'](data, self.comparison),
            'Max': METRICS['Max'](data, self.comparison),
            'Peak-to-Peak Range': METRICS['Peak-to-Peak Range'](data, self.comparison),
            'Shannon Entropy': METRICS['Shannon Entropy'](data, self.comparison),

            # === 6. Intrinsic Norms & Sparsity ===
            'L2 Norm': METRICS['L2 Norm'](data, self.comparison),
            'L1 Norm': METRICS['L1 Norm'](data, self.comparison),
            'L_inf Norm': METRICS['L_inf Norm'](data, self.comparison),
            'L0 Norm (eps=1e-6)': METRICS['L0 Norm (eps=1e-6)'](data, self.comparison),
            'L1/L2 Ratio': METRICS['L1/L2 Ratio'](data, self.comparison),

            # === 1. L-family distances ===
            'L2 norm (Euclidean)': METRICS['L2 norm (Euclidean)'](data, self.comparison),
            'Squared Euclidean': METRICS['Squared Euclidean'](data, self.comparison),
            'L1 norm (Manhattan)': METRICS['L1 norm (Manhattan)'](data, self.comparison),
            'Canberra': METRICS['Canberra'](data, self.comparison),
            'L∞ norm (Chebyshev)': METRICS['L∞ norm (Chebyshev)'](data, self.comparison),
            'Lp norm (Minkowski p=3)': METRICS['Lp norm (Minkowski p=3)'](data, self.comparison),

            # === 2. Correlation measures ===
            'Cosine Similarity': METRICS['Cosine Similarity'](data, self.comparison),
            'Pearson Correlation': METRICS['Pearson Correlation'](data, self.reference),
            'Spearman Correlation': METRICS['Spearman Correlation'](data, self.reference),
        }
        return results

processor = MetricProcessor()
def createMetricsArray(data):
    processor.preprocess(data)
    results_dict = processor.calculate(data)

    # Return results in a fixed order (based on the METRICS dictionary order)
    return [results_dict[key] for key in METRICS_TO_USE.keys()]

# Use float32 for potentially large score arrays to save memory if precision allows
SCORE_DTYPE = np.float32
# Default value for missing metric scores in preprocessed data
MISSING_SCORE_VALUE = np.nan # Using NaN requires handling in combined score calculation
# Suggest a reasonable number of threads, adjust as needed based on your system
# Using cpu_count can be aggressive, maybe start lower.
NUM_ABLATION_THREADS = max(1, os.cpu_count() // 2 if os.cpu_count() else 4)

def calculate_ndcg_dynamic(retrieved_item_indices, target_set, k):
    """
    Calculates NDCG@k with dynamic log calculation (no precomputation).
    Args:
        retrieved_item_indices: np.array or list of item indices, sorted by relevance (best first).
        target_set: set containing the relevant item indices.
        k: The cutoff for NDCG calculation.
    """
    # Ensure retrieved_item_indices is not empty before slicing
    if len(retrieved_item_indices) == 0:
        return 0.0

    k_capped = min(k, len(retrieved_item_indices))
    if k_capped <= 0:
        return 0.0

    dcg = 0.0
    num_targets = len(target_set)

    # Calculate DCG
    for i in range(k_capped):
        idx = retrieved_item_indices[i]
        rank = i + 1
        if idx in target_set:
            # Calculate log dynamically
            dcg += 1.0 / math.log2(rank + 1)

    # Calculate IDCG
    idcg = 0.0
    ideal_ranks = min(k, num_targets)
    for i in range(ideal_ranks):
        rank = i + 1
        # Calculate log dynamically
        idcg += 1.0 / math.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


# calculate_combined_score -> calculate_combined_score_optimized
def calculate_combined_score_optimized(scores_array_sample, # Shape: (num_all_metrics, num_items)
                                       weights_array, # Shape: (num_active_metrics,)
                                       active_indices_mask): # Shape: (num_all_metrics,) boolean
    """
    Calculates the combined score using preprocessed score arrays and active indices mask.
    Assumes lower score is better for ranking (used for argsort later).
    Handles potential NaN values in scores using np.nanmean.
    """
    try:
        # 1. Select scores for active metrics using the boolean mask
        # Resulting shape: (num_active_metrics, num_items)
        active_scores = scores_array_sample[active_indices_mask, :]

        # Check if active_scores is empty
        if active_scores.shape[0] == 0:
            return None # No active metrics to combine

        # 3. Apply weights using broadcasting
        reshaped_weights = weights_array.reshape(-1, 1) # Shape (num_active_metrics, 1)

        if reshaped_weights.shape[0] != active_scores.shape[0]:
            print(f"CRITICAL ERROR: Shape mismatch in combined score. Weights: {reshaped_weights.shape[0]}, Scores: {active_scores.shape[0]}")
            return None # Prevent broadcasting error

        # NaN handling strategy: multiply weights first. If score was NaN, result is NaN.
        # If weight is 0, result is 0 (unless score is Inf/NaN).
        # np.nanmean will ignore NaNs during averaging.
        weighted_scores = active_scores * reshaped_weights # Shape: (num_active_metrics, num_items)

        # 4. Calculate the mean score per item, ignoring NaNs
        combo_score = np.nanmean(weighted_scores, axis=0) # Shape: (num_items,)

        # Handle case where a column (item) was all NaN (or only had NaN scores contributing)
        # Replace resulting NaNs with a high value (bad score), assuming lower is better.
        # Also handle potential +/- infinity from calculations involving np.inf in scores
        combo_score = np.nan_to_num(combo_score, nan=np.finfo(combo_score.dtype).max,
                                    posinf=np.finfo(combo_score.dtype).max,
                                    neginf=np.finfo(combo_score.dtype).min)


        # 5. Final checks
        if combo_score.ndim != 1:
            # This case should be unlikely with nanmean on axis 0 unless input was strange
            print(f"Warning: Combined score is not 1D (ndim={combo_score.ndim}). Check input.")
            return None

        return combo_score # Shape: (num_items,)

    except Exception as e:
        # Reduce verbosity by default
        # print(f"Error in calculate_combined_score_optimized: {type(e).__name__} - {e}")
        # traceback.print_exc() # Uncomment for detailed debugging
        return None

# ================== Objective Function (Optimized) ==================

# evaluate_average_ndcg -> evaluate_average_ndcg_optimized
def evaluate_average_ndcg_optimized(weights_array, # Corresponds to active_metric_indices
                                    active_metric_indices, # List of integer indices
                                    metric_index_map, # Dict: {name: index} - Needed? No, just for context maybe
                                    all_metric_names, # List: all names in order - Needed for num_all_metrics
                                    preprocessed_eval_dataset, # List: [(scores_np_array, target_set), ...]
                                    k):
    """
    Calculates -1 * Avg NDCG@k using preprocessed data and optimized helpers.
    Designed to be robust and ALWAYS return a float/int for optimization.
    """
    # --- Top-level Try-Except for utmost safety ---
    try:
        # --- Input Validation ---
        if not isinstance(weights_array, np.ndarray):
            weights_array = np.array(weights_array, dtype=float)
        if not isinstance(active_metric_indices, list) or \
                len(weights_array) != len(active_metric_indices):
            print("ERROR evaluate_average_ndcg_optimized: Invalid input types/lengths.")
            return 1.0 # Worst score (representing +inf objective value)

        if not preprocessed_eval_dataset:
            # print("Warning evaluate_average_ndcg_optimized: preprocessed_eval_dataset is empty.")
            return 1.0

        # --- Create mask and ensure non-negative weights ---
        num_all_metrics = len(all_metric_names)
        active_indices_mask = np.zeros(num_all_metrics, dtype=bool)
        # Ensure indices are within bounds
        valid_indices = [idx for idx in active_metric_indices if 0 <= idx < num_all_metrics]
        if len(valid_indices) != len(active_metric_indices):
            print("ERROR evaluate_average_ndcg_optimized: Contains invalid metric indices.")
            return 1.0
        active_indices_mask[valid_indices] = True

        # Ensure weights correspond to the valid indices mask count
        if np.sum(active_indices_mask) != len(weights_array):
            print("ERROR evaluate_average_ndcg_optimized: Mismatch between active index count and weights array length.")
            return 1.0

        current_weights = np.maximum(0.0, weights_array) # Ensure non-negative

        total_ndcg = 0.0
        evaluated_count = 0

        # --- Loop through preprocessed evaluation data ---
        for sample_idx, sample_data in enumerate(preprocessed_eval_dataset):
            # --- Validate sample structure (already validated during preprocessing, but double check) ---
            if not isinstance(sample_data, tuple) or len(sample_data) != 2: continue
            scores_array_sample, target_set = sample_data

            if not isinstance(scores_array_sample, np.ndarray) or not isinstance(target_set, set): continue
            # Check if dimensions match expected (num_all_metrics, num_items)
            if scores_array_sample.ndim != 2 or scores_array_sample.shape[0] != num_all_metrics: continue
            if not target_set: continue # Skip samples with no targets

            num_items = scores_array_sample.shape[1]
            if num_items == 0: continue # Skip samples with no items

            # --- Calculate combined score for this sample ---
            combined_score = None
            try:
                combined_score = calculate_combined_score_optimized(
                    scores_array_sample, current_weights, active_indices_mask
                )

                if combined_score is None:
                    # print(f"Debug: Combined score calculation failed for sample {sample_idx}")
                    continue # Skip if helper failed

                # Check type, dimension AFTER calculation
                if not isinstance(combined_score, np.ndarray) or combined_score.ndim != 1:
                    # print(f"Debug: Combined score invalid type/dim for sample {sample_idx}")
                    continue

                # Ensure score length matches number of items
                if len(combined_score) != num_items:
                    print(f"ERROR evaluate_average_ndcg_optimized: Combined score length mismatch sample {sample_idx}. Expected {num_items}, got {len(combined_score)}")
                    continue

                # --- Get ranking (lower score is better -> argsort ascending) ---
                # argsort returns indices that would sort the array.
                # We need the item indices (0 to num_items-1) sorted by their score.
                sorted_item_indices = np.argsort(combined_score)

            except Exception as score_e:
                # print(f"Error evaluate_average_ndcg_optimized: Score/Sort failed Sample {sample_idx}. Error: {type(score_e).__name__}. Skipping.")
                # traceback.print_exc() # Uncomment for debugging
                continue # Skip this sample

            # --- Calculate NDCG ---
            try:
                # Pass the *indices* of the items in their ranked order
                ndcg = calculate_ndcg_dynamic(sorted_item_indices, target_set, k)
                if not isinstance(ndcg, (float, int, np.number)) or np.isnan(ndcg) or np.isinf(ndcg):
                    print(f"Warning: Invalid NDCG value ({ndcg}) calculated for sample {sample_idx}. Skipping.")
                    continue

                total_ndcg += ndcg
                evaluated_count += 1
            except Exception as ndcg_e:
                # print(f"Error evaluate_average_ndcg_optimized: calculate_ndcg failed Sample {sample_idx}. Error: {type(ndcg_e).__name__}. Skipping.")
                # traceback.print_exc() # Uncomment for debugging
                continue # Skip this sample

        # --- Calculate final return value ---
        if evaluated_count == 0:
            # print("Warning evaluate_average_ndcg_optimized: No samples evaluated successfully.")
            return_value = 1.0 # Max penalty (worst score)
        else:
            average_ndcg = total_ndcg / evaluated_count
            # Ensure average is not NaN/Inf (shouldn't happen if individual ndcgs are checked)
            if np.isnan(average_ndcg) or np.isinf(average_ndcg):
                print(f"CRITICAL ERROR evaluate_average_ndcg_optimized: Average NDCG is NaN/Inf ({average_ndcg}). Returning 1.0")
                return_value = 1.0
            else:
                return_value = -average_ndcg # Return negative for minimization

        # --- Final type check ---
        if isinstance(return_value, (float, int, np.number)):
            return float(return_value) # Ensure standard float
        else:
            # This should be almost impossible now, but keep as safety net
            print(f"CRITICAL ERROR evaluate_average_ndcg_optimized: Final return value non-numeric! Type: {type(return_value)}. Returning 1.0")
            return 1.0

    # --- Catch ANY unexpected exception within the function ---
    except Exception as e:
        print(f"CRITICAL ERROR evaluate_average_ndcg_optimized: Uncaught exception! Error: {type(e).__name__} - {e}.")
        return 1.0 # Return worst score

# ================== Preprocessing Function ==================
def preprocess_dataset(eval_dataset_original, all_metric_names, metric_index_map):
    """
    Converts the dataset list of (dict, list/set) to list of (np.array, set).
    Args:
        eval_dataset_original: The input dataset in the original format.
        all_metric_names: Sorted list of all possible metric names.
        metric_index_map: Dictionary mapping metric name to its index (0 to N-1).
    Returns:
        List of tuples: [(scores_array, target_set), ...], where scores_array
        is a numpy array of shape (num_all_metrics, num_items) with SCORE_DTYPE,
        and target_set is a set of target indices.
    """
    preprocessed_data = []
    num_all_metrics = len(all_metric_names)

    print(f"Starting dataset preprocessing for {len(eval_dataset_original)} samples...")
    start_time = time.time()

    for i, sample_data in enumerate(eval_dataset_original):
        if not isinstance(sample_data, tuple) or len(sample_data) != 2:
            print(f"Warning: Skipping invalid sample structure at index {i} during preprocessing.")
            continue
        metric_scores_sample, target_indices_sample = sample_data

        if not isinstance(metric_scores_sample, dict) or not hasattr(target_indices_sample, '__iter__'):
            print(f"Warning: Skipping invalid sample types at index {i} during preprocessing (scores not dict or targets not iterable).")
            continue

        # Determine the number of items in this sample
        num_items = 0
        first_key = next(iter(metric_scores_sample), None)
        if first_key:
            try:
                # Handle potential non-sequence score values early
                if isinstance(metric_scores_sample[first_key], (np.ndarray, list, tuple)):
                    num_items = len(metric_scores_sample[first_key])
                else:
                    print(f"Warning: Metric '{first_key}' score in sample {i} is not array-like. Skipping sample.")
                    continue
            except TypeError: # Handle cases where len() is not defined
                print(f"Warning: Could not determine length of scores for metric '{first_key}' in sample {i}. Skipping sample.")
                continue

        if num_items == 0:
            # print(f"Debug: Skipping sample {i} with 0 items during preprocessing.")
            continue # Skip samples with no items to rank

        # Create the score array for this sample: (num_all_metrics, num_items)
        # Initialize with the chosen missing value indicator
        scores_array = np.full((num_all_metrics, num_items), MISSING_SCORE_VALUE, dtype=SCORE_DTYPE)

        # Populate the array
        for metric_name, scores in metric_scores_sample.items():
            if metric_name in metric_index_map:
                metric_idx = metric_index_map[metric_name]
                try:
                    # Ensure scores are convertible to a NumPy array and have the correct length
                    scores_np = np.asarray(scores, dtype=SCORE_DTYPE)
                    if scores_np.shape == (num_items,):
                        scores_array[metric_idx, :] = scores_np
                    else:
                        print(f"Warning: Shape mismatch for metric '{metric_name}' in sample {i}. Expected ({num_items},), got {scores_np.shape}. Filling with NaN.")
                        # Keep the default MISSING_SCORE_VALUE (NaN)
                except Exception as e:
                    print(f"Warning: Could not process metric '{metric_name}' in sample {i}. Error: {e}. Filling with NaN.")
                    # Keep the default MISSING_SCORE_VALUE (NaN)

        # Convert targets to set for efficient lookup
        try:
            target_set = set(target_indices_sample)
            # Optional: Check if target indices are within the range of item indices (0 to num_items-1)
            # if any(t < 0 or t >= num_items for t in target_set):
            #     print(f"Warning: Sample {i} has target indices outside the valid range [0, {num_items-1}).")
        except Exception as e:
            print(f"Warning: Could not convert targets to set for sample {i}. Error: {e}. Skipping sample.")
            continue

        preprocessed_data.append((scores_array, target_set))

    end_time = time.time()
    print(f"Dataset preprocessing finished in {end_time - start_time:.2f} seconds. {len(preprocessed_data)} samples processed.")
    return preprocessed_data


# ================== Ablation Helper (for parallel execution) ==================
def _evaluate_ablation_task(metric_index_to_remove,
                            current_metric_indices,
                            weights_for_ablation_array, # Weights corresponding to current_metric_indices
                            metric_index_map, # Pass through needed? No.
                            all_metric_names, # Needed for objective function
                            preprocessed_eval_dataset, # The efficient data
                            k): # NDCG@k parameter
    """
    Wrapper function to evaluate NDCG when one metric is removed.
    Runs evaluate_average_ndcg_optimized for the subset of metrics.
    Returns tuple: (metric_index_to_remove, resulting_objective_score)
                   Objective score is -NDCG, or 1.0 if evaluation fails.
    """
    task_start_time = time.time()
    try:
        # 1. Determine the indices and weights for this ablation run
        temp_indices = [idx for idx in current_metric_indices if idx != metric_index_to_remove]

        if not temp_indices:
            # print(f"Debug Ablation: No metrics left after removing index {metric_index_to_remove}")
            return metric_index_to_remove, 1.0 # Worst score if no metrics left

        # Select weights corresponding to the remaining temp_indices
        # Create a mapping from original index to its position in the 'current_metric_indices' list
        original_pos_map = {idx: i for i, idx in enumerate(current_metric_indices)}
        weight_indices_to_keep = [original_pos_map[idx] for idx in temp_indices]

        # Check if weights_for_ablation_array has enough elements
        if max(weight_indices_to_keep) >= len(weights_for_ablation_array):
            print(f"CRITICAL ERROR Ablation Task: Index out of bounds for weights. Index: {max(weight_indices_to_keep)}, Weights len: {len(weights_for_ablation_array)}")
            return metric_index_to_remove, 1.0

        temp_weights_array = weights_for_ablation_array[weight_indices_to_keep]

        # Optional: Re-normalize weights? The original didn't, likely not necessary just for comparison.
        # temp_sum = np.sum(temp_weights_array)
        # if temp_sum > 0: temp_weights_array /= temp_sum

        # 2. Call the objective function with the reduced set
        objective_score = evaluate_average_ndcg_optimized(
            weights_array=temp_weights_array,
            active_metric_indices=temp_indices,
            metric_index_map=None, # Not strictly needed by objective function
            all_metric_names=all_metric_names,
            preprocessed_eval_dataset=preprocessed_eval_dataset,
            k=k
        )
        # task_duration = time.time() - task_start_time
        # print(f"Debug Ablation Task Index {metric_index_to_remove}: Score={objective_score:.6f}, Time={task_duration:.2f}s")

        return metric_index_to_remove, objective_score

    except Exception as e:
        print(f"CRITICAL ERROR during ablation task for index {metric_index_to_remove}: {type(e).__name__} - {e}")
        return metric_index_to_remove, 1.0 # Return worst score on error


# ================== Main Optimizer Function (Optimized & Multithreaded Ablation) ==================
def optimize_metrics_and_weights_scipy(
        k, # Int: k for NDCG@k
        min_metrics=5, # Int: Minimum number of metrics to keep
        optimizer_method='Powell',# String: Method for scipy.optimize.minimize
        optimizer_options=None, # Dict: Options for the chosen method
        verbose=False, # Bool: Print progress details
        num_threads=NUM_ABLATION_THREADS # Int: Number of threads for ablation
):
    main_start_time = time.time()

    # --- Initial Setup & Validation ---
    if not METRICS_TO_USE:
        print("Error: METRICS_TO_USE dictionary is empty.")
        return None, None, -1.0
    if not OPTIMIZER_EVAL_DATA_CACHE:
        print("Error: eval_dataset_original argument is empty.")
        return None, None, -1.0

    all_metric_names = sorted(list(METRICS_TO_USE.keys()))
    metric_index_map = {name: i for i, name in enumerate(all_metric_names)}
    num_all_metrics = len(all_metric_names)

    # Basic check of dataset structure before potentially long preprocessing
    try:
        _scores, _targets = OPTIMIZER_EVAL_DATA_CACHE[0]
        if not isinstance(_scores, dict) or not hasattr(_targets, '__iter__'):
            raise ValueError("Invalid eval_dataset structure.")
    except (IndexError, TypeError, ValueError) as e:
        print(f"Error: Invalid eval_dataset_original structure: {e}")
        return None, None, -1.0

    # --- Preprocess Dataset ---
    preprocessed_eval_dataset = preprocess_dataset(OPTIMIZER_EVAL_DATA_CACHE, all_metric_names, metric_index_map)
    if not preprocessed_eval_dataset:
        print("Error: Dataset preprocessing resulted in an empty dataset.")
        return None, None, -1.0

    # --- Initialize State (using indices) ---
    current_metric_indices = list(range(num_all_metrics)) # Start with all metrics

    # Initialize weights (equal, non-negative, normalized)
    initial_weight_val = 1.0 / num_all_metrics if num_all_metrics > 0 else 1.0
    # Keep track of weights corresponding to 'current_metric_indices'
    current_weights_array = np.full(len(current_metric_indices), initial_weight_val, dtype=float)

    best_overall_ndcg = -1.0 # Use NDCG score directly now (-1.0 is invalid)
    best_overall_indices = None
    best_overall_weights_array = None

    # --- SciPy Optimizer Default Options ---
    default_optimizer_options = {'maxiter': 100, 'disp': False}
    if optimizer_method in ['Nelder-Mead']: default_optimizer_options['adaptive'] = True
    if optimizer_options: default_optimizer_options.update(optimizer_options)

    # --- Iterative Optimization and Pruning ---
    iteration = 0
    max_iterations = num_all_metrics - min_metrics # Max pruning steps

    while len(current_metric_indices) > min_metrics:
        iteration += 1
        iter_start_time = time.time()
        if iteration > max_iterations + 5: # Safety break
            print("Warning: Exceeded maximum expected iterations + safety margin. Breaking loop.")
            break

        num_current_metrics = len(current_metric_indices)
        if verbose: print(f"\n--- Iteration {iteration}/{max_iterations + 1}: Optimizing {num_current_metrics} metrics ---")

        # 1. Optimize weights for the current set of metric indices
        initial_guess_array = current_weights_array # Use weights from previous iter/initial state

        # Normalize initial guess (helps some optimizers)
        initial_guess_sum = np.sum(initial_guess_array)
        if initial_guess_sum > 0: initial_guess_array /= initial_guess_sum
        else: initial_guess_array = np.full(num_current_metrics, 1.0 / num_current_metrics if num_current_metrics > 0 else 1.0)


        optimized_objective_score_this_iter = 1.0 # Default to worst objective score
        optimization_successful_this_iter = False
        optimized_weights_array_this_iter = None

        # Define bounds (non-negative) for methods that support them
        bounds = [(0, None) for _ in current_metric_indices] if optimizer_method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None

        try:
            if verbose: print(f"DEBUG Iter {iteration}: Calling scipy.optimize.minimize (method={optimizer_method})...")
            opt_start_time = time.time()

            # Pass the necessary arguments to the optimized objective function
            opt_result = minimize(
                evaluate_average_ndcg_optimized,
                x0=initial_guess_array,
                args=(current_metric_indices, metric_index_map, all_metric_names, preprocessed_eval_dataset, k),
                method=optimizer_method,
                bounds=bounds,
                options=default_optimizer_options
            )
            opt_duration = time.time() - opt_start_time
            if verbose: print(f"DEBUG Iter {iteration}: SciPy minimize call took {opt_duration:.2f} seconds.")

            if opt_result.success:
                raw_optimized_weights = opt_result.x
                # Ensure weights are non-negative
                optimized_weights_array_this_iter = np.maximum(0.0, raw_optimized_weights)
                # Normalize weights to sum to 1 for consistency
                opt_sum = np.sum(optimized_weights_array_this_iter)
                if opt_sum > 0: optimized_weights_array_this_iter /= opt_sum
                else: # Handle all-zero case if it somehow occurs
                    optimized_weights_array_this_iter = np.full(num_current_metrics, 1.0 / num_current_metrics if num_current_metrics > 0 else 1.0)


                # Re-evaluate with the final, normalized, non-negative weights to get the definitive score
                final_score_check = evaluate_average_ndcg_optimized(
                    optimized_weights_array_this_iter, current_metric_indices,
                    metric_index_map, all_metric_names,
                    preprocessed_eval_dataset, k)

                if isinstance(final_score_check, (float, int, np.number)) and -1.0 <= final_score_check <= 1.0: # Check score validity
                    optimized_objective_score_this_iter = final_score_check
                    current_weights_array = optimized_weights_array_this_iter # Update weights for next iter/ablation
                    optimization_successful_this_iter = True
                    #if verbose: print(f"SciPy Optimize Result: Success=True, Objective Score = {optimized_objective_score_this_iter:.6f} (Avg NDCG = {-optimized_objective_score_this_iter:.6f})")
                else:
                    print(f"CRITICAL WARNING Iter {iteration}: SciPy optimize succeeded but re-evaluation yielded invalid score ({final_score_check}).")
                    # Keep optimization_successful_this_iter = False
                    # Retain previous weights in current_weights_array
            else:
                print(f"Warning: SciPy optimize failed in iteration {iteration}. Message: {opt_result.message}")
                # Keep optimization_successful_this_iter = False
                # Retain previous weights in current_weights_array

        except ValueError as ve:
            print(f"ValueError during SciPy optimize call (may be from objective function): {ve}")
            # traceback.print_exc()
        except Exception as e:
            print(f"CRITICAL ERROR Iter {iteration}: Exception during scipy.optimize.minimize call. Error: {type(e).__name__} - {e}")
            # Keep optimization_successful_this_iter = False

        # --- Update best overall result ---
        current_ndcg_this_iter = -optimized_objective_score_this_iter if optimization_successful_this_iter else -1.0
        if optimization_successful_this_iter and current_ndcg_this_iter > best_overall_ndcg:
            best_overall_ndcg = current_ndcg_this_iter
            best_overall_indices = current_metric_indices[:] # Store copy
            best_overall_weights_array = optimized_weights_array_this_iter.copy()
            print(f"*** New best overall NDCG found: {best_overall_ndcg:.6f} with {len(best_overall_indices)} metrics ***")
        elif not optimization_successful_this_iter:
            if verbose: print("Optimization failed this iteration, best overall score not updated.")


        # --- Ablation / Pruning (Multithreaded) ---
        if not optimization_successful_this_iter:
            if verbose: print("Skipping pruning step due to optimization failure in this iteration.")
            iter_duration = time.time() - iter_start_time
            if verbose: print(f"--- Iteration {iteration} finished (skipped prune). Duration: {iter_duration:.2f}s ---")
            continue # Skip to next iteration

        if len(current_metric_indices) <= min_metrics:
            if verbose: print(f"\nReached minimum number of metrics ({min_metrics}). Stopping pruning.")
            break # Stop the main loop

        if verbose: print(f"Evaluating metric importance for pruning ({num_threads} threads)...")
        ablation_start_time = time.time()
        metric_to_prune_index = None
        min_ndcg_drop = float('inf')
        ablation_results = {} # Store {removed_index: score_without}

        # Use the successfully optimized weights from *this* iteration for ablation
        weights_for_ablation = current_weights_array

        # Use ThreadPoolExecutor for parallel evaluation
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create futures for each ablation task
            futures = {
                executor.submit(
                    _evaluate_ablation_task,
                    metric_idx_to_remove,
                    current_metric_indices,
                    weights_for_ablation,
                    None, # metric_index_map not needed
                    all_metric_names,
                    preprocessed_eval_dataset,
                    k
                ): metric_idx_to_remove
                for metric_idx_to_remove in current_metric_indices
            }

            for future in concurrent.futures.as_completed(futures):
                removed_index = futures[future]
                try:
                    _, objective_score_without = future.result() # Result is (removed_idx, objective_score)
                    ablation_results[removed_index] = objective_score_without
                except Exception as exc:
                    print(f'CRITICAL ERROR: Ablation task for index {removed_index} generated an exception: {exc}')
                    ablation_results[removed_index] = 1.0 # Assign worst score on task error

        # Now analyze the results
        if verbose: print("Ablation evaluations complete. Analyzing results...")
        for removed_index, objective_score_without in ablation_results.items():
            # Check if the evaluation was successful (score is not 1.0)
            if objective_score_without <= 0: # Valid objective score (-NDCG)
                ndcg_without_metric = -objective_score_without
                # optimized_ndcg_this_iter = -optimized_objective_score_this_iter
                ndcg_drop = current_ndcg_this_iter - ndcg_without_metric

                if ndcg_drop < min_ndcg_drop:
                    min_ndcg_drop = ndcg_drop
                    metric_to_prune_index = removed_index
                # Optional: Print score for each removal
                # if verbose: print(f" - Removing index {removed_index} -> NDCG = {ndcg_without_metric:.6f} (Drop = {ndcg_drop:.6f})")

            # else: # Ablation evaluation failed for this metric
            #     if verbose: print(f" - Evaluation failed for removing index {removed_index}")


        ablation_duration = time.time() - ablation_start_time
        if verbose: print(f"Ablation analysis took {ablation_duration:.2f} seconds.")

        # 3. Prune the least important metric index
        if metric_to_prune_index is not None:
            metric_name_to_prune = all_metric_names[metric_to_prune_index] # Get name for logging
            print(f"Pruning metric '{metric_name_to_prune}' (Index: {metric_to_prune_index}, smallest NDCG drop/largest gain: {min_ndcg_drop:.6f})")

            # Remove the index and adjust the weights array for the next iteration
            prune_pos = current_metric_indices.index(metric_to_prune_index)
            current_metric_indices.pop(prune_pos)
            current_weights_array = np.delete(current_weights_array, prune_pos)

            # Optional: Re-normalize weights after pruning
            current_sum = np.sum(current_weights_array)
            if current_sum > 0: current_weights_array /= current_sum
            else: # Handle case if all remaining weights became zero
                num_remaining = len(current_weights_array)
                current_weights_array = np.full(num_remaining, 1.0 / num_remaining if num_remaining > 0 else 1.0)

        else:
            if verbose: print("Could not determine metric to prune this iteration (e.g., all ablation evaluations failed or caused large drops). Stopping.")
            break # Stop the main loop

        iter_duration = time.time() - iter_start_time
        if verbose: print(f"--- Iteration {iteration} finished. Duration: {iter_duration:.2f}s ---")


    # --- Final Steps ---
    main_duration = time.time() - main_start_time
    print(f"\n--- Iterative Optimization Finished ({main_duration:.2f} seconds) ---")

    final_best_metrics = None
    final_best_weights = None

    if best_overall_indices is not None and best_overall_weights_array is not None:
        # Convert best indices back to names
        final_best_metrics = [all_metric_names[i] for i in best_overall_indices]
        # Create the final weights dictionary using names
        final_best_weights = {name: weight
                              for name, weight in zip(final_best_metrics, best_overall_weights_array)}

        # Optional: Ensure final weights dict sums to 1 (should already be due to normalization)
        final_sum = sum(final_best_weights.values())
        if final_sum > 0 and not math.isclose(final_sum, 1.0):
            print("Normalizing final best weights...")
            final_best_weights = {name: w / final_sum for name, w in final_best_weights.items()}

        print(f"Best configuration found with {len(final_best_metrics)} metrics:")
        print(f"  Metrics: {sorted(final_best_metrics)}")
        print(f"  Avg NDCG@{k}: {best_overall_ndcg:.6f}")
        # Optional: Print weights
        # print(f"  Weights: { {n: f'{w:.4f}' for n, w in sorted(final_best_weights.items())} }")

    else:
        print("No suitable combination found during optimization.")
        best_overall_ndcg = -1.0 # Ensure return score is -1.0 if nothing found

    # Return names, dict, and the NDCG score
    return final_best_metrics, final_best_weights, best_overall_ndcg

def calculate_average_rank_of_relevant(retrieved_item_indices, target_set):
    """
    Calculates the average rank of relevant items found in the list.
    Lower is better. Returns a penalty if no relevant items are found.
    """
    if not target_set: return float('inf') # No targets to find

    total_rank = 0
    relevant_found_count = 0
    max_possible_rank = len(retrieved_item_indices)

    for i, idx in enumerate(retrieved_item_indices):
        rank = i + 1
        if idx in target_set:
            total_rank += rank
            relevant_found_count += 1

    if relevant_found_count == 0:
        # Assign a penalty rank worse than the worst possible average rank
        return float(max_possible_rank + 1)
    else:
        return total_rank / relevant_found_count

def optimize_feature_subset_local_search(
        k_for_ndcg, # k is now required
        seed_metric_names=['Cosine Similarity', 'L1 norm (Manhattan)', 'L2 norm (Euclidean)', 'Lp norm (Minkowski p=3)', 'L∞ norm (Chebyshev)', 'Pearson Correlation', 'Spearman Correlation'],
        all_metric_names_list=METRICS,
        eval_dataset_original=OPTIMIZER_EVAL_DATA_CACHE,
        final_optimizer_method='Nelder-Mead',
        final_optimizer_options=None,
        max_add=5,
        max_remove=5,
        verbose=False # Default to False
):
    """
    Selects best feature subset near a seed set based on MIN AVG RANK of relevant items
    (using equal weights), then optimizes weights and calculates final NDCG@k score
    for that single best subset.
    """
    main_start_time = time.time()
    # Scores/results related to Avg Rank stage (Lower is better)
    rank_eval_goal = 'Average Rank'
    # Scores/results related to final NDCG stage
    final_metric_goal = 'NDCG'
    default_bad_ndcg_score = -1.0 # Represents failed NDCG optimization

    # --- Initial Setup & Validation ---
    if not all_metric_names_list or not seed_metric_names or not eval_dataset_original or k_for_ndcg is None:
        print("Error: Invalid arguments (all_metric_names_list, seed_metrics, dataset, k_for_ndcg required).")
        return None, None, default_bad_ndcg_score
    all_metric_names_set = set(all_metric_names_list)
    all_metric_names_sorted = sorted(list(all_metric_names_set))
    metric_index_map = {name: i for i, name in enumerate(all_metric_names_sorted)}
    validated_seed_names = [name for name in seed_metric_names if name in all_metric_names_set]
    if len(validated_seed_names) != len(seed_metric_names): print("Warning: Some seed metrics ignored (not in all_metric_names_list).")
    if not validated_seed_names: print("Error: No valid seed metrics."); return None, None, default_bad_ndcg_score
    seed_metric_names = validated_seed_names
    seed_indices = frozenset(metric_index_map[name] for name in seed_metric_names)
    other_metric_names = [name for name in all_metric_names_sorted if name not in seed_metric_names]
    other_indices = frozenset(metric_index_map[name] for name in other_metric_names)

    print(f"Starting two-stage search: Seed size={len(seed_indices)}, Explore=+/-{max_add}/{max_remove}, Subset Eval=EqualWeightAvgRank, Final Eval=OptimizedNDCG@{k_for_ndcg}")

    # --- Preprocess Dataset ---
    preprocessed_eval_dataset = preprocess_dataset(eval_dataset_original, all_metric_names_sorted, metric_index_map)
    if not preprocessed_eval_dataset: print("Error: Preprocessing failed."); return None, None, default_bad_ndcg_score

    # --- Generate Candidate Subsets ---
    candidate_subsets_indices = set()
    candidate_subsets_indices.add(seed_indices)
    # (Same generation logic using itertools.combinations)
    for r in range(1, max_remove + 1):
        if r >= len(seed_indices):
            if r == len(seed_indices) and r > 0: candidate_subsets_indices.add(frozenset())
            break
        for combo_indices in itertools.combinations(seed_indices, len(seed_indices) - r):
            candidate_subsets_indices.add(frozenset(combo_indices))
    for a in range(1, max_add + 1):
        if a > len(other_indices): break
        for combo_indices in itertools.combinations(other_indices, a):
            new_set = seed_indices.union(frozenset(combo_indices))
            candidate_subsets_indices.add(new_set)
    candidate_subsets_indices.discard(frozenset()) # Discard empty set
    num_subsets_to_eval = len(candidate_subsets_indices)
    print(f"Generated {num_subsets_to_eval} unique candidate subsets to evaluate.")
    if num_subsets_to_eval == 0: print("Error: No candidate subsets generated."); return None, None, default_bad_ndcg_score
    if num_subsets_to_eval > 5000: print(f"*** WARNING: Evaluating {num_subsets_to_eval} subsets (equal weights, AvgRank) may take time! ***")

    # --- Stage 1: Evaluate Subsets using EQUAL WEIGHTS and Average Rank ---
    rank_results = {} # Store { frozenset(indices) -> avg_rank_score }
    evaluated_count = 0
    eval_start_time = time.time()
    print(f"Stage 1: Evaluating {num_subsets_to_eval} subsets using Equal Weight Average Rank...")

    for indices_set in candidate_subsets_indices:
        evaluated_count += 1
        current_subset_indices = sorted(list(indices_set))
        num_current_metrics = len(current_subset_indices)
        if num_current_metrics == 0: continue

        # Minimal progress print
        if evaluated_count % 10000 == 0: print(f"  Evaluated {evaluated_count}/{num_subsets_to_eval} subsets (Equal Weight AvgRank)...")

        equal_weights_array = np.full(num_current_metrics, 1.0 / num_current_metrics, dtype=float)
        active_indices_mask = np.zeros(len(all_metric_names_sorted), dtype=bool)
        # Handle potential index out of bounds if map is wrong
        try:
            active_indices_mask[current_subset_indices] = True
        except IndexError:
            print(f"Error: Invalid indices found in subset {evaluated_count}. Skipping.")
            rank_results[indices_set] = float('inf') # Assign worst possible rank
            continue


        # Calculate Average Rank for this subset with equal weights
        total_avg_rank_sum = 0.0
        sample_evaluated_count = 0
        overall_avg_rank_for_subset = float('inf') # Default to worst rank

        for sample_data in preprocessed_eval_dataset:
            scores_array_sample, target_set = sample_data
            num_items = scores_array_sample.shape[1]
            if num_items == 0: continue

            combined_score = calculate_combined_score_optimized(scores_array_sample, equal_weights_array, active_indices_mask)
            if combined_score is None: continue
            if not isinstance(combined_score, np.ndarray) or combined_score.ndim != 1 or len(combined_score) != num_items: continue
            try: sorted_item_indices = np.argsort(combined_score)
            except Exception: continue

            try:
                # *** Use calculate_average_rank_of_relevant helper ***
                avg_rank_sample = calculate_average_rank_of_relevant(sorted_item_indices, target_set)
                # Check for Inf explicitly - skip samples where subset found no relevant items
                if math.isinf(avg_rank_sample): continue

                total_avg_rank_sum += avg_rank_sample
                sample_evaluated_count += 1
            except Exception: continue

        if sample_evaluated_count > 0:
            overall_avg_rank_for_subset = total_avg_rank_sum / sample_evaluated_count

        rank_results[indices_set] = overall_avg_rank_for_subset # Store positive AvgRank score (lower is better)

    eval_duration = time.time() - eval_start_time
    print(f"Stage 1 finished in {eval_duration:.2f} seconds.")

    # --- Find Best Subset Based on Equal Weight Average Rank Score ---
    if not rank_results: print("Error: No subsets evaluated in Stage 1."); return None, None, default_bad_ndcg_score
    # Find the key (frozenset of indices) corresponding to the MINIMUM Average Rank score
    best_indices_set_via_rank = min(rank_results, key=rank_results.get)
    best_avg_rank_equal_weights = rank_results[best_indices_set_via_rank]

    # Check if even the best rank is infinite (means no subset found relevant items)
    if math.isinf(best_avg_rank_equal_weights):
        print("Error: All evaluated subsets failed to find any relevant items in the dataset.")
        return None, None, default_bad_ndcg_score

    best_subset_indices_final = sorted(list(best_indices_set_via_rank))
    best_metric_names_final = [all_metric_names_sorted[i] for i in best_subset_indices_final]

    print(f"\nBest subset identified via Equal Weight Avg Rank ({len(best_metric_names_final)} metrics):")
    print(f"  Equal Weight Avg Rank Score: {best_avg_rank_equal_weights:.6f} (Lower is better)")
    print(f"--- Stage 2: Optimizing weights & getting final NDCG@{k_for_ndcg} for this best subset ---")

    # --- Stage 2: Get Final Weights and Score (NDCG) for the BEST Subset Found ---
    num_best_metrics = len(best_subset_indices_final)
    final_weights_dict = None
    final_optimized_ndcg_score = default_bad_ndcg_score

    if best_subset_indices_final: # If the best subset is not empty
        stage2_start_time = time.time()
        initial_guess_array = np.full(num_best_metrics, 1.0 / num_best_metrics, dtype=float)
        bounds = [(0, None) for _ in best_subset_indices_final] if final_optimizer_method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None
        opt_options = {'maxiter': 150, 'disp': False}
        if final_optimizer_method in ['Nelder-Mead']: opt_options['adaptive'] = True
        if final_optimizer_options: opt_options.update(opt_options)
        # Args for the FINAL objective function (NDCG)
        args_for_objective = (
            best_subset_indices_final, metric_index_map, all_metric_names_sorted,
            preprocessed_eval_dataset, k_for_ndcg
        )

        try:
            # *** Run full optimization ONCE using NDCG objective ***
            final_opt_result = minimize(
                evaluate_average_ndcg_optimized, # Use NDCG objective here
                x0=initial_guess_array, args=args_for_objective,
                method=final_optimizer_method, bounds=bounds, options=opt_options
            )
            stage2_duration = time.time() - stage2_start_time
            print(f"Stage 2 optimization finished in {stage2_duration:.2f} seconds.")

            if final_opt_result.success:
                # Check if objective function returned error code
                if final_opt_result.fun >= 1.0:
                    print("Warning: Final optimization run succeeded but objective returned error code. Weights not available.")
                else:
                    final_raw_weights = final_opt_result.x
                    final_weights = np.maximum(0.0, final_raw_weights)
                    final_sum = np.sum(final_weights)
                    if final_sum > 0: final_weights /= final_sum
                    else: final_weights = np.full(num_best_metrics, 1.0 / num_best_metrics, dtype=float)
                    final_weights_dict = {name: weight for name, weight in zip(best_metric_names_final, final_weights)}
                    # Final score is the optimized NDCG
                    final_optimized_ndcg_score = -float(final_opt_result.fun)
                    final_optimized_ndcg_score = max(0.0, min(1.0, final_optimized_ndcg_score)) # Clamp actual NDCG to [0, 1]
            else:
                print("Warning: Final optimization run (NDCG) failed. Weights not available. No final NDCG score.")

        except Exception as e:
            print(f"Error during final optimization run (NDCG): {type(e).__name__}")

    # --- Final Report ---
    main_duration = time.time() - main_start_time
    print(f"\n--- Two-Stage Optimization Finished ({main_duration:.2f} seconds) ---")
    if best_metric_names_final:
        print(f"Best subset found (selected via Equal Weight AvgRank, final score is Optimized NDCG):")
        print(f"  Metrics ({len(best_metric_names_final)}): {sorted(best_metric_names_final)}")
        print(f"  Final Optimized NDCG@{k_for_ndcg} Score: {final_optimized_ndcg_score:.6f}")
        if final_weights_dict:
            print(f"  Optimized Weights (for NDCG): {{ {', '.join([f'{repr(n)}: {w:.4f}' for n, w in sorted(final_weights_dict.items())])} }}")
        else: print("  Optimized Weights: Not available (final optimization failed).")
        # Add context about the score used for selection
        if not math.isinf(best_avg_rank_equal_weights):
            print(f"  (Equal Weight Avg Rank score for this subset was: {best_avg_rank_equal_weights:.6f})")
    else:
        # This case should only be hit if generation failed or all subsets had infinite avg rank
        print("No suitable subset found during local search.")
        final_optimized_ndcg_score = default_bad_ndcg_score

    return best_metric_names_final, final_weights_dict, final_optimized_ndcg_score


# MRR-Test
def calculate_reciprocal_rank(retrieved_item_indices, target_set):
    """
    Calculates the Reciprocal Rank (RR) for a single ranked list.
    RR = 1 / rank of the first relevant item. Returns 0 if no relevant item is found.

    Args:
        retrieved_item_indices: np.array or list of item indices, sorted by relevance (best first).
        target_set: set containing the relevant item indices.

    Returns:
        float: The Reciprocal Rank (1/rank) or 0.0.
    """
    if not target_set: # Handle case where target set is empty
        return 0.0

    for i, idx in enumerate(retrieved_item_indices):
        rank = i + 1
        if idx in target_set:
            return 1.0 / rank # Return 1/rank of the first relevant item found

    # If no relevant item was found in the retrieved list
    return 0.0

# evaluate_average_ndcg_optimized -> evaluate_average_mrr
def evaluate_average_mrr(weights_array, # Corresponds to active_metric_indices
                         active_metric_indices, # List of integer indices
                         metric_index_map, # Not strictly needed here, but passed for consistency
                         all_metric_names, # Needed for num_all_metrics
                         preprocessed_eval_dataset, # List: [(scores_np_array, target_set), ...]
                         # k is not typically used in MRR, but we might keep it if needed elsewhere
                         # For now, we calculate RR based on the full sorted list
                         k=None # k parameter ignored for MRR calculation itself
                         ):
    """
    Calculates -1 * Avg MRR for a given weight vector using preprocessed data.
    Designed to be robust and ALWAYS return a float/int for optimization.
    """
    # --- Top-level Try-Except for utmost safety ---
    try:
        # --- Input Validation --- (Same as NDCG version)
        if not isinstance(weights_array, np.ndarray):
            weights_array = np.array(weights_array, dtype=float)
        if not isinstance(active_metric_indices, list) or \
                len(weights_array) != len(active_metric_indices):
            print("ERROR evaluate_average_mrr: Invalid input types/lengths.")
            return 1.0 # Worst score (representing +inf objective value)

        if not preprocessed_eval_dataset:
            return 1.0

        # --- Create mask and ensure non-negative weights --- (Same as NDCG version)
        num_all_metrics = len(all_metric_names)
        active_indices_mask = np.zeros(num_all_metrics, dtype=bool)
        valid_indices = [idx for idx in active_metric_indices if 0 <= idx < num_all_metrics]
        if len(valid_indices) != len(active_metric_indices):
            print("ERROR evaluate_average_mrr: Contains invalid metric indices.")
            return 1.0
        active_indices_mask[valid_indices] = True
        if np.sum(active_indices_mask) != len(weights_array):
            print("ERROR evaluate_average_mrr: Mismatch between active index count and weights array length.")
            return 1.0
        current_weights = np.maximum(0.0, weights_array)

        total_rr = 0.0
        evaluated_count = 0

        # --- Loop through preprocessed evaluation data ---
        for sample_idx, sample_data in enumerate(preprocessed_eval_dataset):
            # --- Validate sample structure --- (Same as NDCG version)
            if not isinstance(sample_data, tuple) or len(sample_data) != 2: continue
            scores_array_sample, target_set = sample_data
            if not isinstance(scores_array_sample, np.ndarray) or not isinstance(target_set, set): continue
            if scores_array_sample.ndim != 2 or scores_array_sample.shape[0] != num_all_metrics: continue
            # No need to check target_set here, calculate_reciprocal_rank handles empty set

            num_items = scores_array_sample.shape[1]
            if num_items == 0: continue

            # --- Calculate combined score for this sample --- (Same as NDCG version)
            combined_score = None
            try:
                combined_score = calculate_combined_score_optimized(
                    scores_array_sample, current_weights, active_indices_mask
                )
                if combined_score is None: continue
                if not isinstance(combined_score, np.ndarray) or combined_score.ndim != 1: continue
                if len(combined_score) != num_items:
                    print(f"ERROR evaluate_average_mrr: Combined score length mismatch sample {sample_idx}.")
                    continue

                # --- Get ranking --- (Same as NDCG version)
                sorted_item_indices = np.argsort(combined_score)

            except Exception as score_e:
                # print(f"Error evaluate_average_mrr: Score/Sort failed Sample {sample_idx}. Error: {type(score_e).__name__}. Skipping.")
                continue

            # --- Calculate Reciprocal Rank (RR) --- (DIFFERENT PART)
            try:
                rr = calculate_reciprocal_rank(sorted_item_indices, target_set)
                if not isinstance(rr, (float, int, np.number)) or np.isnan(rr) or np.isinf(rr):
                    print(f"Warning: Invalid RR value ({rr}) calculated for sample {sample_idx}. Skipping.")
                    continue

                total_rr += rr
                evaluated_count += 1
            except Exception as rr_e:
                # print(f"Error evaluate_average_mrr: calculate_rr failed Sample {sample_idx}. Error: {type(rr_e).__name__}. Skipping.")
                continue

        # --- Calculate final return value ---
        if evaluated_count == 0:
            # print("Warning evaluate_average_mrr: No samples evaluated successfully.")
            return_value = 1.0 # Max penalty (worst objective score)
        else:
            average_mrr = total_rr / evaluated_count
            if np.isnan(average_mrr) or np.isinf(average_mrr):
                print(f"CRITICAL ERROR evaluate_average_mrr: Average MRR is NaN/Inf ({average_mrr}). Returning 1.0")
                return_value = 1.0
            else:
                # We want to MAXIMIZE MRR, so return NEGATIVE MRR for minimization
                return_value = -average_mrr

        # --- Final type check --- (Same as NDCG version)
        if isinstance(return_value, (float, int, np.number)):
            return float(return_value)
        else:
            print(f"CRITICAL ERROR evaluate_average_mrr: Final return value non-numeric! Type: {type(return_value)}. Returning 1.0")
            return 1.0

    # --- Catch ANY unexpected exception --- (Same as NDCG version)
    except Exception as e:
        print(f"CRITICAL ERROR evaluate_average_mrr: Uncaught exception! Error: {type(e).__name__} - {e}.")
        return 1.0

# _evaluate_ablation_task -> _evaluate_ablation_task_mrr
def _evaluate_ablation_task_mrr(metric_index_to_remove,
                                current_metric_indices,
                                weights_for_ablation_array,
                                all_metric_names,
                                preprocessed_eval_dataset):
    """
    Wrapper function to evaluate MRR when one metric is removed.
    Runs evaluate_average_mrr for the subset of metrics.
    Returns tuple: (metric_index_to_remove, resulting_objective_score)
                   Objective score is -MRR, or 1.0 if evaluation fails.
    """
    try:
        temp_indices = [idx for idx in current_metric_indices if idx != metric_index_to_remove]
        if not temp_indices:
            return metric_index_to_remove, 1.0

        original_pos_map = {idx: i for i, idx in enumerate(current_metric_indices)}
        weight_indices_to_keep = [original_pos_map[idx] for idx in temp_indices]

        if max(weight_indices_to_keep) >= len(weights_for_ablation_array):
            print(f"CRITICAL ERROR Ablation Task MRR: Index out of bounds for weights.")
            return metric_index_to_remove, 1.0

        temp_weights_array = weights_for_ablation_array[weight_indices_to_keep]

        # Call the MRR objective function
        objective_score = evaluate_average_mrr(
            weights_array=temp_weights_array,
            active_metric_indices=temp_indices,
            metric_index_map=None,
            all_metric_names=all_metric_names,
            preprocessed_eval_dataset=preprocessed_eval_dataset,
            k=None # k ignored by MRR objective
        )
        return metric_index_to_remove, objective_score

    except Exception as e:
        print(f"CRITICAL ERROR during MRR ablation task for index {metric_index_to_remove}: {type(e).__name__} - {e}")
        return metric_index_to_remove, 1.0 # Return worst score on error

# optimize_metrics_and_weights_scipy -> optimize_metrics_and_weights_scipy_mrr
def optimize_metrics_and_weights_scipy_mrr(
        min_metrics=5,
        optimizer_method='Powell',
        optimizer_options=None,
        verbose=True,
        num_threads=NUM_ABLATION_THREADS
):
    main_start_time = time.time()

    # --- Initial Setup & Validation --- (Same as NDCG version)
    if not METRICS: print("Error: METRICS dictionary is empty."); return None, None, 0.0
    if not OPTIMIZER_EVAL_DATA_CACHE: print("Error: eval_dataset_original argument is empty."); return None, None, 0.0
    all_metric_names = sorted(list(METRICS.keys()))
    metric_index_map = {name: i for i, name in enumerate(all_metric_names)}
    num_all_metrics = len(all_metric_names)
    try:
        _scores, _targets = OPTIMIZER_EVAL_DATA_CACHE[0]
        if not isinstance(_scores, dict) or not hasattr(_targets, '__iter__'): raise ValueError("Invalid structure.")
    except (IndexError, TypeError, ValueError) as e: print(f"Error: Invalid structure: {e}"); return None, None, 0.0

    # --- Preprocess Dataset --- (Same as NDCG version)
    preprocessed_eval_dataset = preprocess_dataset(OPTIMIZER_EVAL_DATA_CACHE, all_metric_names, metric_index_map)
    if not preprocessed_eval_dataset: print("Error: Preprocessing failed."); return None, None, 0.0

    # --- Initialize State --- (Same as NDCG version)
    current_metric_indices = list(range(num_all_metrics))
    initial_weight_val = 1.0 / num_all_metrics if num_all_metrics > 0 else 1.0
    current_weights_array = np.full(len(current_metric_indices), initial_weight_val, dtype=float)

    # --- Track Best MRR Score --- (DIFFERENT)
    best_overall_mrr = 0.0 # Initialize MRR score (0 is minimum possible)
    best_overall_indices = None
    best_overall_weights_array = None

    # --- SciPy Optimizer Defaults --- (Same as NDCG version)
    default_optimizer_options = {'maxiter': 100, 'disp': False}
    if optimizer_method in ['Nelder-Mead']: default_optimizer_options['adaptive'] = True
    if optimizer_options: default_optimizer_options.update(optimizer_options)

    # --- Iterative Optimization and Pruning ---
    iteration = 0
    max_iterations = num_all_metrics - min_metrics

    while len(current_metric_indices) > min_metrics:
        iteration += 1
        iter_start_time = time.time()
        if iteration > max_iterations + 5: print("Warning: Exceeded max iterations."); break

        num_current_metrics = len(current_metric_indices)
        if verbose: print(f"\n--- Iteration {iteration}/{max_iterations + 1}: Optimizing {num_current_metrics} metrics for MRR ---")

        # 1. Optimize weights for MRR (using negative MRR objective)
        initial_guess_array = current_weights_array
        initial_guess_sum = np.sum(initial_guess_array)
        if initial_guess_sum > 0: initial_guess_array /= initial_guess_sum
        else: initial_guess_array = np.full(num_current_metrics, 1.0 / num_current_metrics if num_current_metrics > 0 else 1.0)

        optimized_objective_score_this_iter = 1.0 # Default worst objective (-MRR=1 -> MRR=-1)
        optimization_successful_this_iter = False
        optimized_weights_array_this_iter = None
        bounds = [(0, None) for _ in current_metric_indices] if optimizer_method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None

        try:
            if verbose: print(f"DEBUG Iter {iteration}: Calling scipy.optimize.minimize (objective: -MRR, method={optimizer_method})...")
            opt_start_time = time.time()

            # *** CALL MRR OBJECTIVE FUNCTION ***
            opt_result = minimize(
                evaluate_average_mrr, # Use the MRR objective
                x0=initial_guess_array,
                args=(current_metric_indices, metric_index_map, all_metric_names, preprocessed_eval_dataset), # k is ignored
                method=optimizer_method,
                bounds=bounds,
                options=default_optimizer_options
            )
            opt_duration = time.time() - opt_start_time
            if verbose: print(f"DEBUG Iter {iteration}: SciPy minimize call took {opt_duration:.2f} seconds.")

            if opt_result.success:
                raw_optimized_weights = opt_result.x
                optimized_weights_array_this_iter = np.maximum(0.0, raw_optimized_weights)
                opt_sum = np.sum(optimized_weights_array_this_iter)
                if opt_sum > 0: optimized_weights_array_this_iter /= opt_sum
                else: optimized_weights_array_this_iter = np.full(num_current_metrics, 1.0 / num_current_metrics if num_current_metrics > 0 else 1.0)

                # Re-evaluate with final weights to get definitive MRR score
                final_score_check = evaluate_average_mrr( # Call MRR objective again
                    optimized_weights_array_this_iter, current_metric_indices,
                    metric_index_map, all_metric_names,
                    preprocessed_eval_dataset)

                # Objective returns -MRR. Score check is valid if between -1 and 0 (MRR is 0 to 1)
                if isinstance(final_score_check, (float, int, np.number)) and -1.0 <= final_score_check <= 0.0:
                    optimized_objective_score_this_iter = final_score_check
                    current_weights_array = optimized_weights_array_this_iter
                    optimization_successful_this_iter = True
                    if verbose: print(f"SciPy Optimize Result: Success=True, Objective Score = {optimized_objective_score_this_iter:.6f} (Avg MRR = {-optimized_objective_score_this_iter:.6f})")
                else:
                    print(f"CRITICAL WARNING Iter {iteration}: SciPy optimize succeeded but re-evaluation yielded invalid score ({final_score_check}).")

            else:
                print(f"Warning: SciPy optimize failed in iteration {iteration}. Message: {opt_result.message}")

        except ValueError as ve: print(f"ValueError during SciPy optimize call: {ve}")
        except Exception as e: print(f"CRITICAL ERROR Iter {iteration}: Exception during minimize call: {type(e).__name__} - {e}");

        # --- Update best overall result --- (Using MRR)
        current_mrr_this_iter = -optimized_objective_score_this_iter if optimization_successful_this_iter else 0.0
        if optimization_successful_this_iter and current_mrr_this_iter > best_overall_mrr:
            best_overall_mrr = current_mrr_this_iter
            best_overall_indices = current_metric_indices[:]
            best_overall_weights_array = optimized_weights_array_this_iter.copy()
            if verbose: print(f"*** New best overall MRR found: {best_overall_mrr:.6f} with {len(best_overall_indices)} metrics ***")
        elif not optimization_successful_this_iter:
            if verbose: print("Optimization failed this iteration, best overall score not updated.")

        # --- Ablation / Pruning (Multithreaded for MRR) ---
        if not optimization_successful_this_iter:
            if verbose: print("Skipping pruning step due to optimization failure.")
            # ... (rest of iteration timing and continue)
            iter_duration = time.time() - iter_start_time
            if verbose: print(f"--- Iteration {iteration} finished (skipped prune). Duration: {iter_duration:.2f}s ---")
            continue

        if len(current_metric_indices) <= min_metrics:
            if verbose: print(f"\nReached minimum metrics ({min_metrics}). Stopping."); break

        if verbose: print(f"Evaluating metric importance for pruning ({num_threads} threads, objective: MRR)...")
        ablation_start_time = time.time()
        metric_to_prune_index = None
        # For MRR, the "drop" concept is slightly different. Removing a metric might *increase* MRR.
        # We want to remove the metric whose removal causes the *smallest decrease* or *largest increase* in MRR.
        # Since objective_score = -MRR, we want to remove the metric that results in the *highest* objective_score_without (closest to 0 or positive).
        # Alternatively, calculate MRR_drop = current_mrr_this_iter - mrr_without_metric. Remove metric with largest MRR_drop (least negative or most positive).
        # Let's stick to finding the largest MRR drop.
        max_mrr_drop = -float('inf') # Initialize with negative infinity
        ablation_results = {}

        weights_for_ablation = current_weights_array

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(
                    _evaluate_ablation_task_mrr, # Use MRR ablation task
                    metric_idx_to_remove,
                    current_metric_indices,
                    weights_for_ablation,
                    all_metric_names,
                    preprocessed_eval_dataset
                ): metric_idx_to_remove
                for metric_idx_to_remove in current_metric_indices
            }

            for future in concurrent.futures.as_completed(futures):
                removed_index = futures[future]
                try:
                    _, objective_score_without = future.result() # Score is -MRR
                    ablation_results[removed_index] = objective_score_without
                except Exception as exc:
                    print(f'CRITICAL ERROR: MRR Ablation task {removed_index} generated an exception: {exc}')
                    ablation_results[removed_index] = 1.0 # Assign worst score

        # Analyze MRR ablation results
        if verbose: print("Ablation evaluations complete. Analyzing results...")
        for removed_index, objective_score_without in ablation_results.items():
            # Check if evaluation was successful (objective score <= 0)
            if objective_score_without <= 0:
                mrr_without_metric = -objective_score_without
                mrr_drop = current_mrr_this_iter - mrr_without_metric # Higher drop means metric was important

                # We want to remove the metric whose removal causes the LEAST damage (smallest drop)
                # Actually, following original logic: remove metric causing smallest NDCG drop (largest gain if drop is negative)
                # Translated to MRR: Remove metric causing smallest MRR drop (largest gain if drop is negative).
                # We want to find the *minimum* drop. Let's rename min_ndcg_drop to min_perf_drop
                # Initialize min_perf_drop = float('inf')
                # If mrr_drop < min_perf_drop: update min_perf_drop, metric_to_prune = removed_index

                # Let's re-read the original code's goal for pruning:
                # "Pruning metric '{metric_to_prune}' (Removing caused smallest NDCG drop/largest gain: {min_ndcg_drop:.6f})"
                # So, find minimum ndcg_drop.

                # Apply same logic to MRR: Find the minimum mrr_drop
                if 'min_perf_drop' not in locals(): min_perf_drop = float('inf') # Initialize only once

                if mrr_drop < min_perf_drop:
                    min_perf_drop = mrr_drop
                    metric_to_prune_index = removed_index

                # Optional logging
                # if verbose: print(f" - Removing index {removed_index} -> MRR = {mrr_without_metric:.6f} (Drop = {mrr_drop:.6f})")
            # else: # Ablation evaluation failed
            # if verbose: print(f" - Evaluation failed for removing index {removed_index} (-MRR score: {objective_score_without})")


        ablation_duration = time.time() - ablation_start_time
        if verbose: print(f"Ablation analysis took {ablation_duration:.2f} seconds.")

        # 3. Prune the least important metric index (based on min MRR drop)
        if metric_to_prune_index is not None:
            metric_name_to_prune = all_metric_names[metric_to_prune_index]
            if verbose: print(f"Pruning metric '{metric_name_to_prune}' (Index: {metric_to_prune_index}, smallest MRR drop/largest gain: {min_perf_drop:.6f})") # Use min_perf_drop here

            prune_pos = current_metric_indices.index(metric_to_prune_index)
            current_metric_indices.pop(prune_pos)
            current_weights_array = np.delete(current_weights_array, prune_pos)
            current_sum = np.sum(current_weights_array)
            if current_sum > 0: current_weights_array /= current_sum
            else:
                num_remaining = len(current_weights_array)
                current_weights_array = np.full(num_remaining, 1.0 / num_remaining if num_remaining > 0 else 1.0)
        else:
            if verbose: print("Could not determine metric to prune this iteration. Stopping.")
            break

        iter_duration = time.time() - iter_start_time
        if verbose: print(f"--- Iteration {iteration} finished. Duration: {iter_duration:.2f}s ---")


    # --- Final Steps --- (Report MRR)
    main_duration = time.time() - main_start_time
    print(f"\n--- Iterative Optimization Finished ({main_duration:.2f} seconds) ---")

    final_best_metrics = None
    final_best_weights = None

    if best_overall_indices is not None and best_overall_weights_array is not None:
        final_best_metrics = [all_metric_names[i] for i in best_overall_indices]
        final_best_weights = {name: weight
                              for name, weight in zip(final_best_metrics, best_overall_weights_array)}
        final_sum = sum(final_best_weights.values())
        if final_sum > 0 and not math.isclose(final_sum, 1.0):
            # print("Normalizing final best weights...") # Should be normalized already
            final_best_weights = {name: w / final_sum for name, w in final_best_weights.items()}

        print(f"Best configuration found with {len(final_best_metrics)} metrics:")
        print(f"  Metrics: {sorted(final_best_metrics)}")
        print(f"  Avg MRR: {best_overall_mrr:.6f}") # Report MRR score
    else:
        print("No suitable combination found during optimization.")
        best_overall_mrr = 0.0 # Ensure return score is 0.0 if nothing found

    # Return names, dict, and the MRR score
    return final_best_metrics, final_best_weights, best_overall_mrr

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

def create_global_metric_combinations(max_metrics_to_add, max_metrics_to_remove, getIndices=False):

    # 1. Get all available metric names from the METRICS dictionary
    all_available_metrics_set = set(METRICS.keys())
    print(f"Total available metrics: {len(all_available_metrics_set)}")

    # 2. Define the current best combination as the baseline
    current_best_metrics_tuple = ('Cosine Similarity', 'L1 norm (Manhattan)', 'L2 norm (Euclidean)', 'Lp norm (Minkowski p=3)', 'L∞ norm (Chebyshev)', 'Pearson Correlation', 'Spearman Correlation')
    baseline_metrics_set = set(current_best_metrics_tuple)

    # 3. Verify baseline metrics are available (optional but recommended)
    if not baseline_metrics_set.issubset(all_available_metrics_set):
        missing = baseline_metrics_set - all_available_metrics_set
        print(f"\n*** Warning: The following baseline metrics are not found in the METRICS dictionary keys: {missing} ***")
        print("*** Please check the 'current_best_metrics_tuple' definition. ***\n")
        # Decide how to proceed: stop, use only valid ones, etc.
        # For now, we'll proceed using only the valid subset as the baseline.
        baseline_metrics_set = baseline_metrics_set.intersection(all_available_metrics_set)
        print(f"Using the valid subset as baseline: {sorted(list(baseline_metrics_set))}\n")
        if not baseline_metrics_set:
            print("Error: No valid baseline metrics remain. Stopping.")
            return


    # 4. Define the set of metrics available to be added
    other_metrics_set = all_available_metrics_set - baseline_metrics_set

    print(f"Defined baseline set with {len(baseline_metrics_set)} metrics.")
    print(f"Defined 'other' set with {len(other_metrics_set)} metrics available to add.")

    # --- Generate Combinations ---
    # Use a set to store combinations to automatically handle duplicates
    # Store combinations as sorted tuples for consistency
    combinations_to_evaluate = set()

    # 1. Add the baseline combination itself
    baseline_tuple = tuple(sorted(list(baseline_metrics_set)))
    if baseline_tuple: # Ensure baseline is not empty after validation
        combinations_to_evaluate.add(baseline_tuple)
        print(f"\nAdded baseline combination: {baseline_tuple}")
    else:
        print("\nBaseline combination is empty after validation, cannot add.")


    # 2. Generate combinations by ADDING metrics to the baseline
    print(f"\nGenerating combinations by adding up to {max_metrics_to_add} metrics from the 'other' set ({len(other_metrics_set)} available)...")
    count_before_add = len(combinations_to_evaluate)
    for k in range(1, max_metrics_to_add + 1):
        if k > len(other_metrics_set):
            print(f"  Cannot add {k} metrics, only {len(other_metrics_set)} available in 'other' set.")
            break # Stop adding if k exceeds available metrics

        added_in_step = 0
        for metrics_to_add in itertools.combinations(other_metrics_set, k):
            new_combo_set = baseline_metrics_set.union(set(metrics_to_add))
            new_tuple = tuple(sorted(list(new_combo_set)))
            if new_tuple not in combinations_to_evaluate:
                combinations_to_evaluate.add(new_tuple)
                added_in_step += 1

        if added_in_step > 0:
            print(f"  Added {added_in_step} new unique combinations by adding {k} metric(s). Total unique combos: {len(combinations_to_evaluate)}")
        else:
            print(f"  No new unique combinations generated by adding {k} metric(s).")


    # 3. Generate combinations by REMOVING metrics from the baseline
    print(f"\nGenerating combinations by removing up to {max_metrics_to_remove} metrics from the baseline set ({len(baseline_metrics_set)} available)...")
    count_before_remove = len(combinations_to_evaluate)
    for k in range(1, max_metrics_to_remove + 1):
        if k >= len(baseline_metrics_set): # Use >= because removing all baseline metrics is usually not desired
            print(f"  Skipping removal of {k} metrics (would leave {len(baseline_metrics_set) - k} metrics).")
            continue # Continue to check smaller k values if possible, but break might be safer depending on goal

        removed_in_step = 0
        for metrics_to_remove in itertools.combinations(baseline_metrics_set, k):
            new_combo_set = baseline_metrics_set.difference(set(metrics_to_remove))
            if new_combo_set: # Ensure the resulting set is not empty
                new_tuple = tuple(sorted(list(new_combo_set)))
                if new_tuple not in combinations_to_evaluate:
                    combinations_to_evaluate.add(new_tuple)
                    removed_in_step += 1

        if removed_in_step > 0:
            print(f"  Added {removed_in_step} new unique combinations by removing {k} metric(s). Total unique combos: {len(combinations_to_evaluate)}")
        else:
            print(f"  No new unique combinations generated by removing {k} metric(s).")


    # 4. Convert the set of tuples to a list and assign to global variable
    final_combinations_list = list(combinations_to_evaluate)
    
    if getIndices:
        final_combinations_list_indices = []
        for currentTuple in final_combinations_list:
            found_indices = []
            for metric in currentTuple:
                index = list(METRICS.keys()).index(metric)
                found_indices.append(index)
            final_combinations_list_indices.append(tuple(found_indices))
        final_combinations_list = list(final_combinations_list_indices)

    print(f"\n--- Generated a total of {len(final_combinations_list)} unique metric combinations to evaluate ---")
    
    return final_combinations_list









def identifyClosestSourcesByMetricCombination(closestSources, metricsOutputs, metrics_indices, mode=""):
    global layers 

    metricsDictionary = metricsActivationsByLayers

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

    metricsLayersToCheck = metricsDictionary[layerNumbersToCheck]
    metricsOutputsToCheck = metricsOutputs[layerNumbersToCheck]
    identifiedClosestMetricSources = np.empty((len(metricsLayersToCheck), closestSources), dtype=tuple)

    for currentLayer, currentMetricsLayer in enumerate(metricsLayersToCheck):
        currentMetricsLayer = currentMetricsLayer[:, metrics_indices]
        
        metric_scores = {}
        metric_calculation_successful_overall = True # Flag to track if all metrics were processed

        num_metrics = len(metrics_indices)
        num_samples = currentMetricsLayer.shape[0]

        currentMetricsLayerToCheck = np.array([metricsOutputsToCheck[currentLayer][idx] for idx in metrics_indices])
        
        if currentMetricsLayer.shape[1] != num_metrics:
            print(f"Warning: Shape mismatch - currentMetricsLayer columns ({currentMetricsLayer.shape[1]}) != num_metrics ({num_metrics}).")
            metric_calculation_successful_overall = False
        if currentMetricsLayerToCheck.shape[0] != num_metrics:
            print(f"Warning: Shape mismatch - metricsOutputsToCheck length ({currentMetricsLayerToCheck.shape[0]}) != num_metrics ({num_metrics}).")
            metric_calculation_successful_overall = False

        # --- Proceed only if initial checks pass ---
        if metric_calculation_successful_overall:
            # Calculate raw differences (assuming shapes are compatible)
            raw_diffs = np.abs(currentMetricsLayer - currentMetricsLayerToCheck[np.newaxis, :])

            # Normalize per metric (column-wise)
            min_vals = np.min(currentMetricsLayer, axis=0)
            max_vals = np.max(currentMetricsLayer, axis=0)
            range_vals = max_vals - min_vals
            # Add epsilon only where range is close to zero to avoid inflating differences elsewhere
            epsilon = 1e-10
            safe_range = np.where(range_vals < epsilon, epsilon, range_vals)

            norm_samples = (currentMetricsLayer - min_vals) / safe_range
            # Apply same normalization to the reference point
            norm_ref = (currentMetricsLayerToCheck - min_vals) / safe_range

            metric_keys_list = list(METRICS.keys())
            metrics_to_use = {metric_keys_list[index]: 1.0 for index in metrics_indices}
            # Handle similarity metrics - flip scores so lower is better universally
            # Use names matching the keys in your METRICS dictionary
            similarity_metric_names = {'Cosine Similarity', 'Pearson Correlation', 'Spearman Correlation',
                                       'Jaccard (Rounded Sets)', 'Sørensen–Dice (Rounded Sets)'}
            
            metric_name_list = list(metrics_to_use)

            for i, name in enumerate(metric_name_list):
                if name in similarity_metric_names:
                    if i < norm_samples.shape[1]: # Check index validity
                        norm_samples[:, i] = 1.0 - norm_samples[:, i]
                    if i < norm_ref.shape[0]: # Check index validity
                        norm_ref[i] = 1.0 - norm_ref[i]

            # Store normalized absolute difference scores per metric
            for i, name in enumerate(metric_name_list):
                if i < norm_samples.shape[1] and i < norm_ref.shape[0]:
                    score_diff = np.abs(norm_samples[:, i] - norm_ref[i])

                    # --- FIX 2: Handle potential NaNs from normalization/calculation ---
                    if np.any(np.isnan(score_diff)):
                        #print(f"Warning: NaN detected in score for metric '{name}'. Replacing with mean.")
                        if np.all(np.isnan(score_diff)):
                            score_diff.fill(1.0) # Assign default high difference if all are NaN
                        else:
                            mean_val = np.nanmean(score_diff)
                            score_diff = np.nan_to_num(score_diff, nan=mean_val)
                    metric_scores[name] = score_diff
                else:
                    print(f"Warning: Index {i} for metric '{name}' out of bounds during score storage. Metric skipped.")
                    metric_calculation_successful_overall = False # Mark as potentially incomplete

        # --- Proceed only if metric calculation was successful ---
        if metric_calculation_successful_overall and metric_scores:
            # Use only metrics that were successfully calculated
            metrics_to_use = list(metric_scores.keys())

            # --- FIX 3: Robust combined score calculation ---
            weighted_scores_list = []
            metrics_used_in_mean = 0
            for name in metrics_to_use:
                score_array = metric_scores.get(name) # Should exist if in metrics_to_use
                weight = METRIC_WEIGHTS.get(name)     # Get weight (Corrected dict)

                # Ensure score/weight exist and score is valid array
                if score_array is not None and weight is not None and isinstance(score_array, np.ndarray) and score_array.size > 0:
                    weighted_scores_list.append(score_array * weight)
                    metrics_used_in_mean += 1
                else:
                    print(f"Warning: Skipping metric '{name}' in combined score (missing score/weight/invalid score array).")


            if not weighted_scores_list or metrics_used_in_mean == 0:
                print("Error: No valid metric scores available to combine for this sample. Cannot sort.")
                tuples = ()
            else:
                combined_scores = np.sum(weighted_scores_list, axis=0) / metrics_used_in_mean

                # Ensure combined_scores is 1D before argsort
                if combined_scores.ndim != 1:
                    print(f"Error: combined_scores has unexpected shape {combined_scores.shape}. Cannot sort.")
                    tuples = ()
                else:
                    # Sort indices (lower combined score is better)
                    sorted_metric_indices = np.argsort(combined_scores)
                    # Ensure closestSources doesn't exceed available indices
                    num_indices_to_take = min(closestSources, len(sorted_metric_indices))
                    closest_metric_indices = sorted_metric_indices[:num_indices_to_take]

                    # --- FIX 4: Safe Indexing for Output Tuples ---
                    safe_tuples = []
                    max_layer_idx = currentMetricsLayer.shape[0] - 1
                    max_raw_diffs_idx = raw_diffs.shape[0] - 1 if 'raw_diffs' in locals() else -1 # Check if raw_diffs exists

                    for i in range(num_indices_to_take):
                        idx = closest_metric_indices[i]
                        # Check if index is valid for both arrays needed
                        if idx <= max_layer_idx and idx <= max_raw_diffs_idx:
                            safe_tuples.append((
                                idx
                            ))
                        else:
                            print(f"Warning: Index {idx} out of bounds when creating output tuple (Layer Max: {max_layer_idx}, Diff Max: {max_raw_diffs_idx}). Skipping.")
                    tuples = tuple(safe_tuples)

            identifiedClosestMetricSources[currentLayer] = tuples

    return identifiedClosestMetricSources, layerNumbersToCheck

def getMostUsedSourcesByMetrics(metricsSources, closestSources, evalSample=0, weightedMode="", info=True):
    metricsSourceCounter, metricsMostUsed = getMostUsedByMetrics(metricsSources, weightedMode, evaluation="Metrics")
    metricsCounter = Counter(metricsMostUsed)

    #if(info):
    #print("Total closest Sources (Per Neuron):", sourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", counter.most_common()[:closestSources])
    #if metricsEvaluation:
    #print("Total closest Sources (Metrics):", metricsSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", metricsCounter.most_common()[:closestSources])
    #if mtEvaluation:
    #print("Total closest Sources (MT):", mtSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", mtCounter.most_common()[:closestSources])
    return metricsCounter.most_common()[:closestSources]

def getMostUsedByMetrics(sources, mode="", evaluation=""):
    mostUsed = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
            for sourceNumber in layer:
                if(sourceNumber != 'None'):
                    mostUsed.append(sourceNumber)
                    sourceCounter += 1
    return sourceCounter, mostUsed