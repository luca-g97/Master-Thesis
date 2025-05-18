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
metricsEvaluation: bool = True    # Perform metrics evaluation
mtEvaluation: bool = True          # Perform magnitude truncation evaluation
useBitNet: bool = False            # Use BitNet specific logic/model
useOnlyBestMetrics: bool = False
ignore_near_zero_eval_activations = True

# --- Environment, Data & File Paths ---
device: str = ""                   # Computation device (e.g., "cuda", "cpu")
baseDirectory: str = "./LookUp"    # Base directory for lookup files or results
chosenDataSet: str = ""            # Identifier for the dataset being used
fileName: str = ""                 # Specific file name being processed
sourceArray: str = ""              # Placeholder or path for source data array
lastActualToken: str = ""          # The last token that is not used for padding

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

POTENTIAL_METRICS = {
    # === Comparison Metrics: L-family distances (d vs baseline `c`) ===
    # Proven useful. Lower is better (closer).
    'L2 norm (Euclidean)': lambda d, c: np.linalg.norm(d - c, ord=2),
    'L1 norm (Manhattan)': lambda d, c: np.linalg.norm(d - c, ord=1),
    'Canberra': lambda d, c: np.sum(np.abs(d - c) / (np.abs(d) + np.abs(c) + 1e-10)),
    'L∞ norm (Chebyshev)': lambda d, c: np.linalg.norm(d - c, ord=np.inf),
    'Lp norm (Minkowski p=3)': lambda d, c: np.power(np.sum(np.abs(d - c)**3), 1/3), # Appeared in 'best hits'

    # === Comparison Metrics: Correlation/Similarity measures (Converted to Distances) ===
    # Proven useful. All converted so Lower is better.
    'Cosine Distance': lambda d, c: distance.cosine(d, c) if (np.linalg.norm(d) > 1e-9 and np.linalg.norm(c) > 1e-9) else 1.0, # Range [0, 2] approx. 0 = identical direction. 1 = orthogonal. 2 = opposite. Lower is better.
    'Angular Distance': lambda d, c: np.arccos(np.clip(1.0 - distance.cosine(d, c), -1.0, 1.0)) / np.pi if (np.linalg.norm(d) > 1e-9 and np.linalg.norm(c) > 1e-9) else 0.5, # Range [0, 1]. Lower = smaller angle = more similar direction.
    'Pearson Correlation Distance': lambda d, ref: (1.0 - np.corrcoef(d, ref)[0, 1]) if (np.std(d) > 1e-9 and np.std(ref) > 1e-9) else 1.0, # Range [0, 2]. 0 = perfect positive correlation. 1 = no correlation. 2 = perfect negative correlation. Lower means higher positive correlation.
    'Spearman Correlation Distance': lambda d, ref: (1.0 - spearmanr(d, ref).correlation) if (np.std(d) > 1e-9 and np.std(ref) > 1e-9) else 1.0, # Range [0, 2]. 0 = perfect positive monotonic correlation. Lower means higher positive monotonic correlation.

    # === Comparison Metrics: Statistical distances (d vs baseline `c`) ===
    # Robust general-purpose comparison metrics. Lower is better.
    'Jensen-Shannon': lambda d, c: distance.jensenshannon(_normalize_safe(d), _normalize_safe(c)), # Range [0, 1]. Lower is better.
    'KL Divergence': lambda d, c: entropy(_normalize_safe(d), _normalize_safe(c)), # Non-negative. Lower is better. Hinted by results.
    'Wasserstein': lambda d, c: wasserstein_distance(d, c), # Non-negative. Lower is better.

    # === Intrinsic Statistical Properties (of data vector `d`) ===
    # Describes the data vector `d` itself. Some proven useful. Interpretation varies.
    'Mean': lambda d, c: np.mean(d), # Average activation
    'Median': lambda d, c: np.median(d), # Robust center
    'Standard Deviation': lambda d, c: np.std(d), # Spread
    'Median Absolute Deviation (MAD)': lambda d, c: np.median(np.abs(d - np.median(d))), # Robust spread
    'Interquartile Range (IQR)': lambda d, c: np.percentile(d, 75) - np.percentile(d, 25), # Robust spread
    'Skewness': lambda d, c: skew(d), # Asymmetry
    'Kurtosis': lambda d, c: kurtosis(d), # Tailedness/peakedness
    'Min': lambda d, c: np.min(d), # Minimum activation
    'Max': lambda d, c: np.max(d), # Maximum activation
    'Shannon Entropy': lambda d, c: entropy(_normalize_safe(d)), # Information content/uniformity [0, log(N)]

    # === Intrinsic Norms & Sparsity (of data vector `d`) ===
    # Measures magnitude and sparsity characteristics of `d`. Proven useful / fundamental.
    'L2 Norm': lambda d, c: np.linalg.norm(d, ord=2), # Euclidean magnitude (Energy)
    'L1 Norm': lambda d, c: np.linalg.norm(d, ord=1), # Manhattan magnitude
    'L_inf Norm': lambda d, c: np.linalg.norm(d, ord=np.inf), # Max absolute value (Peak)
    'L0 Norm (eps=1e-6)': lambda d, c: np.count_nonzero(np.abs(d) > 1e-6), # Sparsity count (lower is sparser)
    'L1/L2 Ratio': lambda d, c: np.linalg.norm(d, ord=1) / (np.linalg.norm(d, ord=2) + 1e-10), # Sparsity measure (higher -> denser)
}

# --- Metrics Excluded in This Final Filter ---
# Redundant: 'Squared Euclidean', 'Variance', 'Peak-to-Peak Range', 'Mahalanobis'
# Less Suitable/Relevant for Raw Activations: 'Chi-square', 'DTW', 'Levenshtein', 'Hamming', 'Jaccard', 'Sørensen–Dice'
# Complexity/Sequence: 'Approximate Entropy', 'Sample Entropy', 'Permutation Entropy', 'Dominant Frequency'
# Not doable since variance not calculable in hook: Standardized Euclidean 
# Other: 'KL Divergence Reversed', 'Coefficient of Variation' (can be derived from Mean/StdDev)

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
        sourceArray, hidden_sizes, llm, fileName, layersToCheck, lastActualToken

    if (llm):
        actualLayer = layer
        layerNeurons = layers[actualLayer][1]
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
        if(correctTypes):
            dictionaryForSourceLayerNeuron[source][layer,:layerNeurons] = relevantOutput

        if(llm):
            if relevantOutput.ndim == 3 and relevantOutput.shape[0] == 1:
                relevantOutput = relevantOutput.squeeze(0) # Now [seq_len, hidden_dim]
            
            if(actualLayer in layersToCheck or layersToCheck == []):
                # Option 1: Last token pooling - good for tasks where the final state of a sequence is most important (e.g., summarization, classification after sequential processing).
                #output = relevantOutput if relevantOutput.ndim == 1 else relevantOutput[lastActualToken]
    
                # Option 2: Mean token pooling - can be good for creating a "fingerprint" or average representation of the entire input sequence.
                output = relevantOutput if relevantOutput.ndim == 1 else np.mean(relevantOutput[:lastActualToken], axis=0)

                sourceNumber, sentenceNumber = chosenDataSet.getSourceAndSentenceIndex(source, fileName)
                if sourceNumber is not None and sentenceNumber is not None:
                    #print(f"Create File: LookUp/{fileName}/Layer{layer}/Source={result[0]}/Sentence{result[1]}-0")
                    append_structured_sparse(output[:layerNeurons], actualLayer, sourceNumber, sentenceNumber)
                    if metricsEvaluation:
                        metricsArray = createMetricsArray(output)
                        append_structured_sparse(metricsArray, actualLayer+"-Metrics", sourceNumber, sentenceNumber)
                    if mtEvaluation:
                        reduced = np.argsort(-np.abs(output))[:min(NumberOfComponents, output.shape[0])]
                        append_structured_sparse(reduced, actualLayer+"-MT", sourceNumber, sentenceNumber)
        else:
            output = relevantOutput if relevantOutput.ndim == 1 else relevantOutput[0]
            if metricsEvaluation:
                metricsArray = createMetricsArray(output)
                metricsDictionaryForSourceLayerNeuron[source][layer] = metricsArray
                metricsDictionaryForLayerNeuronSource[layer][source] = metricsArray
            if mtEvaluation:
                reduced = np.argsort(-np.abs(output))[:min(NumberOfComponents, output.shape[0])]
                mtDictionaryForSourceLayerNeuron[source][layer,:len(reduced)] = reduced
                mtDictionaryForLayerNeuronSource[layer][source,:len(reduced)] = reduced
            
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
    global source, layer, sourceArray, fileName, lastActualToken

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
                non_zero_indices = np.where(inputs[0].numpy() != 0)[0]
                lastActualToken = non_zero_indices[-1]
                
                actualSource, actualSentenceNumber = chosenDataSet.getSourceAndSentenceIndex(source, fileName)
                print(f"Saving all Activations for {fileName}-Source {tempSource} (Actual {fileName}-Source: {actualSource}:{actualSentenceNumber})")
            inputs = inputs.to(device)
            currentInput = inputs
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
    
    if not llm:
        return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

METRIC_WEIGHTS = {name: 0.0 for name in POTENTIAL_METRICS.keys()}
#Best metrics by Optimization for only one sample
#EVALUATION_METRICS = {'L1 norm (Manhattan)': 0.4550, 'Cosine Similarity': 1.3947, 'Pearson Correlation': 1.7145, 'Peak-to-Peak Range': 1.6832,
#                     'Variance': 0.7049, 'Spearman Correlation': 1.3736, 'L∞ norm (Chebyshev)': 0.8470, 'L2 norm (Euclidean)': 1.7258, 'Median': 0.7801}
#Best Metrics for one sample with 128 layer size by Average NDCG ~65%
#EVALUATION_METRICS = {'Canberra': 0.034262836221897686, 'Chi-square': 0.8155299806719645, 'Jaccard (Rounded Sets)': 0.001437592332473082, 'Jensen-Shannon': 0.0025480160518755336, 'Kurtosis': 0.0031398715863971147, 'L1 norm (Manhattan)': 0.12144434270728152, 'Max': 0.005106721107465592, 'Pearson Correlation': 0.010801096055247771, 'Skewness': 0.004429958518467075, 'Sørensen–Dice (Rounded Sets)': 0.0012995847469301114}
#Best Metrics for 10 samples with 1024 layer size by Average NDCG ~83%
#EVALUATION_METRICS = {'Canberra': 0.729337656812952, 'Kurtosis': 0.0012939579042749686, 'Max': 0.23866959715105898, 'Pearson Correlation': 0.009881806419878765, 'Spearman Correlation': 0.004248273259857096, 'Squared Euclidean': 0.01656870845197806}
#Bests Metrics for 10 samples with 2048 layer size by Average NDCG ~84%
EVALUATION_METRICS = {'Canberra': 0.6096463431205467, 'Chi-square': 0.03554298989041808, 'Jaccard (Rounded Sets)': 0.3382192228124472, 'Kurtosis': 0.0002865279259160476, 'Levenshtein (Rounded Strings)': 0.0002998769279219361, 'Max': 0.0096465702916513, 'Median': 0.0016993437007648648, 'Peak-to-Peak Range': 0.0010868364471923585, 'Pearson Correlation': 0.001783285431289672, 'Shannon Entropy': 0.000564316921047458, 'Skewness': 0.0012246865308042248}
#EVALUATION_METRICS = {'Cosine Similarity': 1.0, 'L1 norm (Manhattan)': 1.0, 'L∞ norm (Chebyshev)': 1.0, 'Pearson Correlation': 1.0, 'Spearman Correlation': 1.0}
METRICS_TO_USE = EVALUATION_METRICS if (len(EVALUATION_METRICS) > 0 and useOnlyBestMetrics) else {name: 1.0 for name in POTENTIAL_METRICS.keys()}

for metric in METRICS_TO_USE.keys():
    METRIC_WEIGHTS[metric] = METRICS_TO_USE[metric]

# Add to global initialization
mt_component_optimizer = None
optimal_components_overall = 46 # Ranging always between 44 and 47 overall

OPTIMIZER_EVAL_DATA_CACHE = []
EPSILON = 1e-7

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
    
    activationToCheckFor = [
        (idx * 2) + 1 for idx, (name, layerNumber, activation) in enumerate(layers)
        if getActivation(hidden_sizes, idx) != False
    ]
    activationLayersToCheck = dictionary[activationToCheckFor]
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
        for currentNeuron, neuron in enumerate(layer):
            if ignore_near_zero_eval_activations and np.abs(outputsToCheck[currentLayer][currentNeuron]) < EPSILON:
                identifiedClosestSources[currentLayer][currentNeuron] = tuple(('None', None, None) for i in range(closestSources))
            else:
                maxNeurons = layers[currentLayer][1]
                if not isinstance(maxNeurons, int):
                    maxNeurons = maxNeurons.out_features
                if currentNeuron < maxNeurons:
                    differences = np.abs(neuron - outputsToCheck[currentLayer][currentNeuron])
                    sorted_indices = np.argsort(differences)
                    closest_indices = sorted_indices[:closestSources]
    
                    # Store neuron results
                    tuples = tuple(
                        (closest_indices[i],
                         neuron[closest_indices[i]],
                         differences[closest_indices[i]])
                        for i in range(closestSources)
                    )
                    identifiedClosestSources[currentLayer][currentNeuron] = tuples

        if metricsEvaluation:
            metrics_indices = (5, 9, 10, 26, 22, 0, 3, 20, 13, 7, 8, 14) #Best indices so far
            currentMetricsLayer = currentMetricsLayer[:, metrics_indices]
            currentMetricsLayerToCheck = np.array([metricsOutputsToCheck[currentLayer][idx] for idx in metrics_indices])

            reference_data = currentMetricsLayerToCheck
            current_data = currentMetricsLayer
            epsilon = 1e-10 # Small value to prevent division by zero

            # --- 1. Min-Max Scaling ---
            # Calculate min and max along axis 0 (feature-wise/column-wise)
            min_vals = np.min(reference_data, axis=0)
            max_vals = np.max(reference_data, axis=0)
            range_vals = max_vals - min_vals

            # Apply Min-Max scaling: (X - min) / (range + epsilon)
            # Using broadcasting: reference_data (M, D), min/max/range (D,)
            norm_ref_minmax = (reference_data - min_vals) / (range_vals + epsilon)
            norm_curr_minmax = (current_data - min_vals) / (range_vals + epsilon)
            # Note: Features where min == max will result in 0 after scaling due to epsilon.

            # --- 2. Z-Score Normalization (Standardization) ---
            # Calculate mean and standard deviation along axis 0 (feature-wise)
            mean_vals = np.mean(reference_data, axis=0)
            std_vals = np.std(reference_data, axis=0)

            # Apply Z-score scaling: (X - mean) / (std_dev + epsilon)
            norm_ref_zscore = (reference_data - mean_vals) / (std_vals + epsilon)
            norm_curr_zscore = (current_data - mean_vals) / (std_vals + epsilon)
            # Note: Features with zero standard deviation will result in 0 after scaling.

            # Option A: Use Z-score normalized data (Often a good default)

            #normalized_currentMetricsLayer = norm_curr_zscore
            #normalized_metricsOutputsToCheck = norm_ref_zscore

            # Option B: Use Min-Max normalized data (Uncomment to use)
            #normalized_currentMetricsLayer = norm_curr_minmax
            #normalized_metricsOutputsToCheck = norm_ref_minmax

            # Option C: Use Original data (Uncomment to use)
            normalized_currentMetricsLayer = current_data
            normalized_metricsOutputsToCheck = reference_data

            # --- 4. Proceed with Distance Calculation (using the selected normalized data) ---
            # Calculate L1 distance (Manhattan)
            metrics_differences = np.sum(np.abs(normalized_currentMetricsLayer - normalized_metricsOutputsToCheck), axis=1)

            # Or calculate L2 distance (Euclidean) - often suitable for normalized data
            #metrics_differences = np.linalg.norm(normalized_currentMetricsLayer - normalized_metricsOutputsToCheck, axis=1)

            #metrics_differences = np.sum(np.abs(currentMetricsLayer - metricsOutputsToCheck[currentLayer][np.newaxis, :]), axis=1)
            metrics_sorted_indices = np.argsort(metrics_differences)
            metrics_closest_indices = metrics_sorted_indices[:closestSources]

            # Store results
            tuples = tuple(
                (metrics_closest_indices[i],
                 currentMetricsLayer[metrics_closest_indices[i]],
                 metrics_differences[metrics_closest_indices[i]])
                for i in range(closestSources)
            )

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
    mostUsedSourcesPerLayer = []
    mostUsed = []
    differences = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        mostUsedPerLayer = []
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
                                if sourceNumber not in mostUsedPerLayer:
                                    mostUsedPerLayer.append(sourceNumber)
        for sourceInLayer in mostUsedPerLayer:
            mostUsedSourcesPerLayer.append(sourceInLayer)

    return sourceCounter, mostUsed, differences, mostUsedSourcesPerLayer

def getMostUsedFromDataFrame(df, evalSample, closestSources, weightedMode=""):
    # Filter entries for the specific evalSample
    relevant_entries = df[df.index.get_level_values('evalSample') == evalSample]

    # Use value_counts to count occurrences of each source directly
    sources = relevant_entries['source']

    # Filter out invalid sources ('None')
    valid_entries = relevant_entries[sources != 'None']
    #ascending_order = True  # Sort by ascending for lowest total weights

    #if weightedMode == "Sum":
        # Group by 'source' and sum the 'difference' column as weights
        #weighted_counts = valid_entries.groupby('source')['difference'].sum()
    #elif weightedMode == "Mean":
        # Group by 'source' and calculate the average of 'difference'
        #weighted_counts = valid_entries.groupby('source')['difference'].mean()
    #else:
        # Default behavior: Count occurrences
        #weighted_counts = valid_entries['source'].value_counts()
        #ascending_order = False  # Sort by descending for highest counts

    mostUsed = valid_entries['source'].tolist()    
    differences = valid_entries['difference'].tolist()

    # Sort weighted sources by the determined order
    #sorted_sources = weighted_counts.sort_values(ascending=ascending_order).head(closestSources)
    # Total weight (sum or mean) or total count for closest sources
    #total_weight = sorted_sources.sum()

    # Print the total weight (sum, mean, or total count depending on the mode)
    #print(f"Total Weight for Weighted Mode={weightedMode}: {total_weight}")

    # Convert to a Counter-like output (sorted already by the determined order)
    #counter = [(source, weight) for source, weight in sorted_sources.items()]

    #print(f"Total closest Sources (Weighted Mode={weightedMode}):", total_weight,
    #      "|", closestSources, "closest Sources in format [SourceNumber, Weight]:", counter)

    # Fix: Convert the 'source' column of valid_entries to a list
    #sourceCounter = valid_entries['source'].value_counts().sum()  # Total count of valid sources
    #mostUsed = valid_entries['source'].tolist()  # Extract 'source' as a list

    return mostUsed, differences

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
        mostUsed, sourceDifferences = getMostUsedFromDataFrame(sources, evalSample, closestSources, weightedMode)
    else:
        sourceCounter, mostUsed, sourceDifferences, mostUsedSourcesPerLayer = getMostUsed(sources, weightedMode)
        if metricsEvaluation:
            metricsSourceCounter, metricsMostUsed, metricsDifferences, _ = getMostUsed(metricsSources, weightedMode, evaluation="Metrics")
        if mtEvaluation:
            mtSourceCounter, mtMostUsed, mtDifferences, _ = getMostUsed(mtSources, weightedMode, evaluation="Magnitude Truncation")

    counter = weighted_counter(mostUsed, sourceDifferences)
    if llm:
        return counter.most_common()[:closestSources]
    else:
        metricsCounter = weighted_counter(metricsMostUsed, metricsDifferences)
        mtCounter = weighted_counter(mtMostUsed, mtDifferences)
        mostUsedSourcesPerLayerCounter = Counter(mostUsedSourcesPerLayer)
        return counter.most_common()[:closestSources], metricsCounter.most_common()[:closestSources], mtCounter.most_common()[:closestSources], mostUsedSourcesPerLayerCounter.most_common()[:closestSources]

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
            **{f"Neuron {idx}": val for idx, val in zip(normalized_sparse_array.data, normalized_sparse_array.data)},
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

def identifyClosestLLMSources(evalSamples, evalOffset, closestSources, onlyOneEvaluation=False, layersToCheck=[], trainPathToUse="Training", info=True):
    global layers, layerSizes, fileName

    trainPath = os.path.join(baseDirectory, trainPathToUse)
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
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

class MetricProcessor:
    def __init__(self, comparison_value=0.5, metrics_to_use=METRICS_TO_USE):
        # Allow passing the list/set of metrics to use
        self.metrics_to_calculate = set(metrics_to_use)
        self.comparison_value = comparison_value

        self.comparison = None # Fixed comparison vector (e.g., all 0.5)
        self.reference = None  # Fixed reference vector (e.g., linspace) - Calculated if needed
        self.variances = None  # For Std Euclidean - Calculated if needed

    def _ensure_float_array(self, data):
        """Helper to ensure data is a numpy float array, handles lists/non-float arrays."""
        if not isinstance(data, np.ndarray):
            arr = np.array(data, dtype=float)
            return arr if arr.ndim > 0 else arr.reshape(1) # Ensure at least 1D
        elif data.dtype != float:
            return data.astype(float)
        # Ensure at least 1D even if input was 0D array
        return data if data.ndim > 0 else data.reshape(1)


    def preprocess(self, data):
        """One-time preprocessing for metrics for a given data vector."""
        data = self._ensure_float_array(data)
        # Handle empty input data gracefully
        if data.size == 0:
            print("Warning: Input data for preprocess is empty.")
            self.comparison = np.array([], dtype=float)
            self.reference = None
            self.variances = None
            return # Cannot proceed with empty data

        # Base comparison array (needed by many comparison metrics)
        self.comparison = np.full_like(data, self.comparison_value, dtype=float)

        # --- Conditional Preprocessing ---

        # Calculate reference only if Pearson/Spearman Correlation Distance are requested
        needs_reference = 'Pearson Correlation Distance' in self.metrics_to_calculate or \
                          'Spearman Correlation Distance' in self.metrics_to_calculate
        if needs_reference:
            # Reference needs same number of elements as flattened data vector
            num_elements = data.size
            if num_elements > 0:
                self.reference = np.linspace(0, 1, num_elements)
            else:
                self.reference = np.array([], dtype=float)
        else:
            self.reference = None

        # Calculate variances only if Standardized Euclidean is requested
        # Warning: Variance calculation method here is unusual. Provide 'v' externally if possible.
        if 'Standardized Euclidean' in self.metrics_to_calculate:
            num_elements = data.size
            if num_elements > 0:
                # Calculate variance for each dimension across the two vectors
                # Ensure consistent shape for vstack, even for multi-D arrays
                stacked_data = np.vstack([data.flatten(), self.comparison.flatten()])
                # Calculate variance along axis 0, add epsilon
                self.variances = np.var(stacked_data, axis=0, ddof=0) + 1e-10 # Use ddof=0 for population var of 2 points

                # Ensure variance shape matches flattened data shape
                if self.variances.shape != data.flatten().shape:
                    print(f"Warning: Variance shape {self.variances.shape} mismatch with data shape {data.flatten().shape}")
                    self.variances = None # Disable if shapes mismatch unexpectedly
            else:
                self.variances = np.array([], dtype=float)
        else:
            self.variances = None


    def calculate(self, data):
        """Calculate requested metrics using FINAL_POTENTIAL_METRICS"""
        original_data = data # Keep reference
        data = self._ensure_float_array(data)
        if data.size == 0:
            print("Warning: Input data for calculate is empty. Returning empty dict.")
            return {}

        # Ensure preprocess has been called or call it if comparison isn't set or shape mismatches
        if self.comparison is None or self.comparison.shape != data.shape:
            self.preprocess(original_data) # Use original data
            # Reload data as float array after preprocess
            data = self._ensure_float_array(original_data)
            if data.size == 0: return {} # Check again if preprocess handled empty input

        results = {}
        metrics_to_run = self.metrics_to_calculate

        # Helper function to add metric if requested
        def add_metric(name, *args):
            # Check if metric is actually requested before proceeding
            if name not in metrics_to_run:
                return

            metric_func = POTENTIAL_METRICS.get(name)
            if metric_func is None:
                print(f"Warning: Metric function for '{name}' not found in dictionary.")
                results[name] = np.nan
                return

            # Check if required conditional args are available
            required_arg = None
            is_arg_required = False
            if name == 'Standardized Euclidean':
                required_arg = self.variances
                is_arg_required = True
            elif name == 'Pearson Correlation Distance' or name == 'Spearman Correlation Distance':
                required_arg = self.reference
                is_arg_required = True

            if is_arg_required and required_arg is None:
                #print(f"Debug: Skipping metric '{name}' because required argument (variance/reference) is None.")
                results[name] = np.nan # Record as NaN if requested but unavailable
                return

            try:
                # Pass only the arguments the specific metric needs
                num_expected_args = metric_func.__code__.co_argcount
                actual_args = args[:num_expected_args]
                results[name] = metric_func(*actual_args)
            except Exception as e:
                print(f"Warning: Could not calculate metric '{name}'. Error: {e}")
                results[name] = np.nan

        # --- Calculate Selected Metrics ---

        # Comparison Metrics: L-family distances
        add_metric('L2 norm (Euclidean)', data, self.comparison)
        add_metric('L1 norm (Manhattan)', data, self.comparison)
        add_metric('Canberra', data, self.comparison)
        add_metric('L∞ norm (Chebyshev)', data, self.comparison)
        add_metric('Lp norm (Minkowski p=3)', data, self.comparison)

        # Comparison Metrics: Correlation/Similarity Distances
        add_metric('Cosine Distance', data, self.comparison)
        add_metric('Angular Distance', data, self.comparison)
        add_metric('Pearson Correlation Distance', data, self.reference) # Uses reference
        add_metric('Spearman Correlation Distance', data, self.reference) # Uses reference

        # Comparison Metrics: Statistical Distances
        add_metric('Standardized Euclidean', data, self.comparison, self.variances) # Uses variances
        add_metric('Jensen-Shannon', data, self.comparison)
        add_metric('KL Divergence', data, self.comparison)
        add_metric('Wasserstein', data, self.comparison)

        # Intrinsic Statistical Properties
        add_metric('Mean', data, self.comparison)
        add_metric('Median', data, self.comparison)
        add_metric('Standard Deviation', data, self.comparison)
        add_metric('Median Absolute Deviation (MAD)', data, self.comparison)
        add_metric('Interquartile Range (IQR)', data, self.comparison)
        add_metric('Skewness', data, self.comparison)
        add_metric('Kurtosis', data, self.comparison)
        add_metric('Min', data, self.comparison)
        add_metric('Max', data, self.comparison)
        add_metric('Shannon Entropy', data, self.comparison)

        # Intrinsic Norms & Sparsity
        add_metric('L2 Norm', data, self.comparison)
        add_metric('L1 Norm', data, self.comparison)
        add_metric('L_inf Norm', data, self.comparison)
        add_metric('L0 Norm (eps=1e-6)', data, self.comparison)
        add_metric('L1/L2 Ratio', data, self.comparison)

        return results

processor = MetricProcessor()
def createMetricsArray(data):
    processor.preprocess(data)
    results_dict = processor.calculate(data)

    # Return results in a fixed order (based on the METRICS dictionary order)
    return [results_dict[key] for key in METRICS_TO_USE.keys()]

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
    all_available_metrics_set = set(POTENTIAL_METRICS.keys())
    #print(f"Total available metrics: {len(all_available_metrics_set)}")

    # 2. Define the current best combination as the baseline
    current_best_metrics_tuple = ('Cosine Distance', 'Jensen-Shannon', 'KL Divergence', 'Kurtosis', 'L1 norm (Manhattan)', 'L1/L2 Ratio', 'L2 Norm', 'L2 norm (Euclidean)', 'L∞ norm (Chebyshev)', 'Max', 'Median', 'Pearson Correlation Distance', 'Shannon Entropy', 'Spearman Correlation Distance', 'Standard Deviation')
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

    #print(f"Defined baseline set with {len(baseline_metrics_set)} metrics.")
    #print(f"Defined 'other' set with {len(other_metrics_set)} metrics available to add.")

    # --- Generate Combinations ---
    # Use a set to store combinations to automatically handle duplicates
    # Store combinations as sorted tuples for consistency
    combinations_to_evaluate = set()

    # 1. Add the baseline combination itself
    baseline_tuple = tuple(sorted(list(baseline_metrics_set)))
    if baseline_tuple: # Ensure baseline is not empty after validation
        combinations_to_evaluate.add(baseline_tuple)
        #print(f"\nAdded baseline combination: {baseline_tuple}")
    else:
        print("\nBaseline combination is empty after validation, cannot add.")


    # 2. Generate combinations by ADDING metrics to the baseline
    #print(f"\nGenerating combinations by adding up to {max_metrics_to_add} metrics from the 'other' set ({len(other_metrics_set)} available)...")
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

        #if added_in_step > 0:
        #    print(f"  Added {added_in_step} new unique combinations by adding {k} metric(s). Total unique combos: {len(combinations_to_evaluate)}")
        #else:
        #    print(f"  No new unique combinations generated by adding {k} metric(s).")


    # 3. Generate combinations by REMOVING metrics from the baseline
    #print(f"\nGenerating combinations by removing up to {max_metrics_to_remove} metrics from the baseline set ({len(baseline_metrics_set)} available)...")
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

        #if removed_in_step > 0:
        #    print(f"  Added {removed_in_step} new unique combinations by removing {k} metric(s). Total unique combos: {len(combinations_to_evaluate)}")
        #else:
        #    print(f"  No new unique combinations generated by removing {k} metric(s).")


    # 4. Convert the set of tuples to a list and assign to global variable
    final_combinations_list = list(combinations_to_evaluate)

    if getIndices:
        final_combinations_list_indices = []
        for currentTuple in final_combinations_list:
            found_indices = []
            for metric in currentTuple:
                index = list(POTENTIAL_METRICS.keys()).index(metric)
                found_indices.append(index)
            final_combinations_list_indices.append(tuple(found_indices))
        final_combinations_list = list(final_combinations_list_indices)

    print(f"\n--- Generated a total of {len(final_combinations_list)} unique metric combinations to evaluate ---")

    return final_combinations_list

def identifyClosestSourcesByMetricCombination(closestSources, metricsOutputs, metrics_indices, metric_weights=METRIC_WEIGHTS, mode=""):
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
        currentMetricsLayerToCheck = np.array([metricsOutputsToCheck[currentLayer][idx] for idx in metrics_indices])

        reference_data = currentMetricsLayerToCheck
        current_data = currentMetricsLayer
        epsilon = 1e-10 # Small value to prevent division by zero

        # --- 1. Min-Max Scaling ---
        # Calculate min and max along axis 0 (feature-wise/column-wise)
        min_vals = np.min(reference_data, axis=0)
        max_vals = np.max(reference_data, axis=0)
        range_vals = max_vals - min_vals

        # Apply Min-Max scaling: (X - min) / (range + epsilon)
        # Using broadcasting: reference_data (M, D), min/max/range (D,)
        norm_ref_minmax = (reference_data - min_vals) / (range_vals + epsilon)
        norm_curr_minmax = (current_data - min_vals) / (range_vals + epsilon)
        # Note: Features where min == max will result in 0 after scaling due to epsilon.

        # --- 2. Z-Score Normalization (Standardization) ---
        # Calculate mean and standard deviation along axis 0 (feature-wise)
        mean_vals = np.mean(reference_data, axis=0)
        std_vals = np.std(reference_data, axis=0)

        # Apply Z-score scaling: (X - mean) / (std_dev + epsilon)
        norm_ref_zscore = (reference_data - mean_vals) / (std_vals + epsilon)
        norm_curr_zscore = (current_data - mean_vals) / (std_vals + epsilon)
        # Note: Features with zero standard deviation will result in 0 after scaling.

        # Option A: Use Z-score normalized data (Often a good default)
        #normalized_currentMetricsLayer = norm_curr_zscore
        #normalized_metricsOutputsToCheck = norm_ref_zscore

        # Option B: Use Min-Max normalized data (Uncomment to use)
        #normalized_currentMetricsLayer = norm_curr_minmax
        #normalized_metricsOutputsToCheck = norm_ref_minmax

        # Option C: Use Original data (Uncomment to use)
        normalized_currentMetricsLayer = current_data
        normalized_metricsOutputsToCheck = reference_data


        # --- 4. Proceed with Distance Calculation (using the selected normalized data) ---
        # Calculate L1 distance (Manhattan)
        metrics_differences = np.sum(np.abs(normalized_currentMetricsLayer - normalized_metricsOutputsToCheck), axis=1)

        # Or calculate L2 distance (Euclidean) - often suitable for normalized data
        #metrics_differences = np.linalg.norm(normalized_currentMetricsLayer - normalized_metricsOutputsToCheck, axis=1)

        #metrics_differences = np.sum(np.abs(currentMetricsLayer - metricsOutputsToCheck[currentLayer][np.newaxis, :]), axis=1)
        metrics_sorted_indices = np.argsort(metrics_differences)
        metrics_closest_indices = metrics_sorted_indices[:closestSources]

        # Store results
        tuples = tuple(
            (metrics_closest_indices[i],
             metrics_differences[metrics_closest_indices[i]])
            for i in range(closestSources)
        )

        # Assign results
        identifiedClosestMetricSources[currentLayer] = tuples

    return identifiedClosestMetricSources, layerNumbersToCheck

def getMostUsedSourcesByMetrics(metricsSources, closestSources, evalSample=0, weightedMode="", info=True):
    sourceCounter, mostUsed, differences = getMostUsedByMetrics(metricsSources)
    metricsCounter = weighted_counter(mostUsed, differences)

    #if(info):
    #print("Total closest Sources (Metrics):", metricsSourceCounter, " | ", closestSources, " closest Sources (", weightedMode, ") in format: [SourceNumber, Occurances]: ", metricsCounter.most_common()[:closestSources])
    return metricsCounter.most_common()[:closestSources]

def getMostUsedByMetrics(sources):
    mostUsed = []
    differences = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        for sourceNumber, difference in layer:
            if(sourceNumber != 'None'):
                mostUsed.append(sourceNumber)
                differences.append(difference)
                sourceCounter += 1
    return sourceCounter, mostUsed, differences