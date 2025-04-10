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

layer, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, totalLayers = 0, 0, [], [], [], [], 0
llm, metricsEvaluation, useBitNet, layerSizes, device, hidden_sizes, layers, currentLayer, relevantLayerIndices = False, False, False, [], "", [], 0, [], []
sourceArray, fileName, contextLength, io, pd, pa, pq, zstd, levenshtein, cma, chosenDataSet, baseDirectory = "", "", 1, "", "", "", "", "", "", "", "", "./LookUp"
metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, metricsActivationsBySources, metricsActivationsByLayers, layersToCheck = [], [], [], [], []
mmDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource, mtActivationsBySources, mtActivationsByLayers = [], [], [], []

# ================== Helper Functions (Standalone) ==================

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
            metricsActivationsBySources = np.zeros((train_samples, totalLayers, len(METRICS)), dtype=np.float128)
            metricsActivationsByLayers = np.zeros((totalLayers, train_samples, len(METRICS)), dtype=np.float128)

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
        metricsDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, len(METRICS)), dtype=np.float128)
        metricsDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, len(METRICS)), dtype=np.float128)
        mtDictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, NumberOfComponents), dtype=np.float128)
        mtDictionaryForLayerNeuronSource = np.zeros((totalLayers, eval_samples, NumberOfComponents), dtype=np.float128)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader, model, llmType, filename, sourceOffset, lstm)

        create_global_metric_combinations(4, 4)
    return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, metricsDictionaryForSourceLayerNeuron, metricsDictionaryForLayerNeuronSource, mtDictionaryForSourceLayerNeuron, mtDictionaryForLayerNeuronSource

# Global configuration - tested with 10000 evaluation samples and 60000 trainSamples on Seed0
#METRIC_WEIGHTS = {'L2 norm (Euclidean)': 0.84595279427617855154, 'Squared Euclidean': 0.83224573934538148085, 'L1 norm (Manhattan)': 0.7687561660244726608, 'Canberra': 0.96355450386131677833, 'L∞ norm (Chebyshev)': 0.85905570563456966126, 'Lp norm (Minkowski p=3)': 0.9605228998431233623, 'Cosine Similarity': 1.4085102017289147866, 'Pearson Correlation': 1.2791490776542367033, 'Spearman Correlation': 0.88216770387512854216, 'Mahalanobis': 0.88823148424610403, 'Standardized Euclidean': 0.88823148424610403, 'Chi-square': 0.7844054846319496534, 'Jensen-Shannon': 1.0729674415824971881, 'Levenshtein': 0.39255559509664496065, 'Hamming': 0.88822753562351450306, 'Jaccard/Tanimoto': 0.88822753562351450306, 'Sørensen–Dice': 0.88822753562351450306}
# Global configuration - tested with 10000 evaluation samples and 60000 trainSamples on Seed0
# Global configuration - tested with 100 evaluation samples and 10000 trainSamples on Seed0
#METRIC_WEIGHTS = {'L2 norm (Euclidean)': 1.3113559725740969809, 'Squared Euclidean': 1.2101180209756891844, 'L1 norm (Manhattan)': 1.1251426462974175736, 'Canberra': 1.7421718881126872844, 'L∞ norm (Chebyshev)': 0.50953942577044365075, 'Lp norm (Minkowski p=3)': 1.3860242085827914397, 'Cosine Similarity': 0.8859925301938055847, 'Pearson Correlation': 0.9120567724775818127, 'Spearman Correlation': 1.4976966778957922496, 'Mahalanobis': 0.85707488088405925353, 'Standardized Euclidean': 0.85707488088405925353, 'Chi-square': 0.8520277648058949888, 'Jensen-Shannon': 0.9095651680450349729, 'Levenshtein': 1.1135734660492960539, 'Hamming': 1.1325316811990171291, 'Jaccard/Tanimoto': 0.7118675443837889661, 'Sørensen–Dice': 0.71295894304878325054}
METRIC_WEIGHTS = {name: 1.0 for name in METRICS.keys()}
#METRIC_WEIGHTS = {'L2 norm (Euclidean)': 0.4746196803972474, 'Squared Euclidean': 0.2101787020369187, 'L1 norm (Manhattan)': 0.1738521386220618, 'Canberra': 0.6619277451044928, 'L∞ norm (Chebyshev)': 13.075694243733738, 'Lp norm (Minkowski p=3)': 0.10445952929476779, 'Cosine Similarity': 0.2994576185342686, 'Pearson Correlation': 0.14767663238708353, 'Spearman Correlation': 0.22614705931869764, 'Mahalanobis': 0.015219902755839967, 'Standardized Euclidean': 0.05142937168438708, 'Chi-square': 0.5674737495422129, 'Jensen-Shannon': 0.357395925225805, 'Levenshtein': 0.31252626193825506, 'Hamming': 0.16867044457406632, 'Jaccard/Tanimoto': 0.006171341104304164, 'Sørensen–Dice': 0.147099653745872}
metrics_optimizer = None  # Will be initialized on first call
# Add to global initialization
mt_component_optimizer = None
optimal_components_overall = 0

GLOBAL_COMBINATION_PERFORMANCE = {}
# This list will be populated by the function and store all combinations to test.
ALL_METRIC_COMBINATIONS_TO_TEST = []
OPTIMIZER_EVAL_DATA_CACHE = []

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

            num_metrics = len(METRICS)
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
                metric_name_list = list(METRICS.keys()) # Get a fixed order

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
                            print(f"Warning: NaN detected in score for metric '{name}'. Replacing with mean.")
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

def createMetricsArray(data):
    processor = MetricProcessor()
    processor.preprocess(data)
    results = processor.calculate(data)

    return list(results.values())

class NeuronMetricProcessor:
    def __init__(self, query):
        """Initialize with proper handling of both scalar and vector inputs"""
        # Convert input to 1D numpy array
        self.query = np.asarray(query, dtype=np.float64).flatten()

        # Handle scalar case (0-dim array)
        if self.query.ndim == 0:
            self.query = np.array([self.query.item()])

        # Precompute string representations for discrete metrics
        self.query_round_lev1_str = ''.join(f"{x:.1f}" for x in np.round(self.query, 1))
        self.query_round2 = set(np.round(self.query, 2))

        # Precompute norms and stats for efficiency
        self.query_norm = self.query / (np.linalg.norm(self.query) + 1e-10)
        self.query_sum = self.query.sum()
        self.query_min = self.query.min()
        self.query_max = self.query.max()

    def calculate(self, data):
        """Calculate all metrics for data against query, handling both scalars and vectors"""
        # Convert input to 1D numpy array
        data_array = np.asarray(data, dtype=np.float64).flatten()
        if data_array.ndim == 0:
            data_array = np.array([data_array.item()])

        # Ensure compatible shapes
        if data_array.shape != self.query.shape:
            raise ValueError(f"Shape mismatch: query {self.query.shape} vs data {data_array.shape}")

        # Precompute common values
        diff = data_array - self.query
        abs_diff = np.abs(diff)
        data_round_lev1_str = ''.join(f"{x:.1f}" for x in np.round(data_array, 1))
        data_round2 = set(np.round(data_array, 2))

        # Variance between data and query
        variances = np.var([data_array, self.query], axis=0, ddof=0) + 1e-10

        # Calculate all metrics
        metrics = {
            # L-family distances
            'L2': np.sqrt(np.sum(diff**2)),
            'Squared L2': np.sum(diff**2),
            'L1': np.sum(abs_diff),
            'Canberra': np.sum(abs_diff / (np.abs(data_array) + np.abs(self.query) + 1e-10)),
            'Chebyshev': np.max(abs_diff),
            'Minkowski3': np.sum(abs_diff**3)**(1/3),

            # Correlation measures
            'Cosine': 1 - distance.cosine(data_array, self.query),
            'Pearson': np.corrcoef(data_array, self.query)[0, 1] if np.std(data_array) > 1e-9 else 0,
            'Spearman': spearmanr(data_array, self.query).correlation if np.std(data_array) > 1e-9 else 0,

            # Statistical distances
            'Mahalanobis': np.sqrt(np.sum(diff**2 / variances)),
            'Std Euclidean': np.sqrt(np.sum(diff**2 / variances)),
            'Chi-square': np.sum(np.where((data_array + self.query) > 0, diff**2 / (data_array + self.query + 1e-10), 0)),
            'Jensen-Shannon': distance.jensenshannon(
                (data_array - self.query_min) / (self.query_max - self.query_min + 1e-10),
                (self.query - self.query_min) / (self.query_max - self.query_min + 1e-10)
            ),

            # Discrete metrics
            'Levenshtein': levenshtein(data_round_lev1_str, self.query_round_lev1_str),
            'Hamming': np.count_nonzero(np.round(data_array, 2) != np.round(self.query, 2)),
            'Jaccard': len(data_round2 & self.query_round2) / max(len(data_round2 | self.query_round2), 1),
            'Sørensen': 2 * len(data_round2 & self.query_round2) / max(len(data_round2) + len(self.query_round2), 1)
        }

        return metrics

def sort_vectors(query, vectors, weights=None):
    """
    Sort vectors by similarity to query using composite metric score
    Args:
        query: scalar or vector to compare against
        vectors: list/array of scalars or vectors
        weights: optional dict of metric weights
    Returns:
        Tuple of (sorted_indices, composite_scores)
    """
    # Convert all vectors to arrays and check shapes
    processor = NeuronMetricProcessor(query)
    processed_vectors = []

    for v in vectors:
        vec = np.asarray(v, dtype=np.float64).flatten()
        if vec.ndim == 0:
            vec = np.array([vec.item()])
        if vec.shape != processor.query.shape:
            raise ValueError(f"Vector shape {vec.shape} doesn't match query shape {processor.query.shape}")
        processed_vectors.append(vec)

    # Calculate all metrics
    metrics = [processor.calculate(vec) for vec in processed_vectors]

    # Normalization parameters
    metric_names = list(metrics[0].keys())
    is_similarity = {
        'Cosine': True, 'Pearson': True, 'Spearman': True,
        'Jaccard': True, 'Sørensen': True
    }

    # Convert to distance space and normalize
    normalized = {}
    for name in metric_names:
        values = [m[name] for m in metrics]

        # Convert similarities to distances
        if is_similarity.get(name, False):
            if name in ['Cosine', 'Pearson', 'Spearman']:
                values = [(1 - v)/2 for v in values]  # Convert [-1,1] to [0,1]
            else:
                values = [1 - v for v in values]  # Convert [0,1] to [0,1]

        # Robust normalization using IQR
        q25, q75 = np.percentile(values, [25, 75])
        iqr = max(q75 - q25, 1e-9)
        normalized[name] = (values - np.median(values)) / iqr

    # Create weighted composite score
    default_weights = {
        'L2': 1.0, 'Squared L2': 0.8, 'L1': 1.0, 'Canberra': 0.9,
        'Chebyshev': 0.7, 'Minkowski3': 0.8, 'Cosine': 1.0,
        'Pearson': 1.0, 'Spearman': 1.0, 'Mahalanobis': 0.9,
        'Std Euclidean': 0.8, 'Chi-square': 0.8, 'Jensen-Shannon': 0.9,
        'Levenshtein': 0.7, 'Hamming': 0.6, 'Jaccard': 0.8, 'Sørensen': 0.8
    }

    weights = weights or default_weights
    composite = np.zeros(len(processed_vectors))
    for name in metric_names:
        composite += normalized[name] * weights.get(name, 1.0)

    # Return sorted indices and scores
    sorted_indices = np.argsort(composite)
    return sorted_indices, composite[sorted_indices]

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
        }
        return results

class MetricWeightOptimizer:
    def __init__(self, metric_names, top_k=10, learning_rate=0.01, # Reduced default LR for Adam
                 reg_strength=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.metric_names = list(metric_names)
        self.weights = {name: 1.0 / len(self.metric_names) for name in self.metric_names} # Start normalized
        self.top_k = top_k
        # Adam parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {name: 0.0 for name in self.metric_names} # Adam 1st moment vector
        self.v = {name: 0.0 for name in self.metric_names} # Adam 2nd moment vector

        self.reg_strength = reg_strength
        self.iteration = 0
        self.margin = 0.05  # Minimum score difference margin
        self.best_weights = self.weights.copy() # Initialize best with starting weights
        self.best_score = -np.inf
        # self.gradient_history = {name: [] for name in self.metric_names} # Can keep if needed for debugging

    def _calculate_gradient(self, metric_scores, target_indices):
        # Ensure metric_scores keys match self.metric_names order if calculation relies on implicit order
        # Using explicit lookup by name is safer
        gradients = {name: 0.0 for name in self.metric_names}
        combined = np.sum([self.weights[name] * metric_scores[name]
                           for name in self.metric_names if name in metric_scores], axis=0) # Check key exists

        # Ensure combined is not empty if metric_scores was missing keys
        if combined.size == 0:
            print("Warning: combined score is empty in gradient calculation.")
            return gradients # Return zero gradients

        predicted_ranking = np.argsort(combined)
        # Ensure target_indices is not longer than available scores
        k_target = min(self.top_k, len(target_indices))
        k_pred = min(self.top_k, len(predicted_ranking))

        predicted_top_k = predicted_ranking[:k_pred]
        target_set = set(target_indices[:k_target])

        # --- Calculate pairwise gradients (User's original logic) ---
        # Note: This logic assumes lower 'combined' score is better
        for target_idx in target_set:
            # Ensure target_idx is a valid index for metric_scores arrays
            if target_idx >= combined.shape[0]: continue

            for pred_idx in predicted_top_k:
                # Ensure pred_idx is a valid index
                if pred_idx >= combined.shape[0]: continue

                if pred_idx == target_idx:
                    continue

                # Margin-based gradient calculation
                score_diff = combined[pred_idx] - combined[target_idx]

                # If prediction is worse than target (higher score) or not better by margin
                if score_diff > -self.margin:
                    # This pair violates the desired ranking or margin
                    direction = (1 if pred_idx not in target_set else -1)
                    for name in self.metric_names:
                        if name in metric_scores: # Check key exists
                            # Ensure indices are valid for this metric's score array
                            if pred_idx < len(metric_scores[name]) and target_idx < len(metric_scores[name]):
                                feature_diff = metric_scores[name][pred_idx] - metric_scores[name][target_idx]
                                gradients[name] += feature_diff * direction
                            else:
                                print(f"Warning: Index out of bounds for metric '{name}' in gradient calc.")
                        # else: Metric score missing, gradient remains 0

        # Apply L2 regularization (gradient ascent maximizes score, reg minimizes weight magnitude)
        for name in self.metric_names:
            gradients[name] -= 2 * self.reg_strength * self.weights[name]

        return gradients
        # --- End User's original gradient logic ---


    def update_weights(self, metric_scores, target_indices):
        # Input validation basic check
        if not metric_scores or target_indices is None or len(target_indices) == 0:
            print("Warning: Skipping weight update due to empty scores or targets.")
            return # Skip update if inputs are invalid

        gradients = self._calculate_gradient(metric_scores, target_indices)
        self.iteration += 1

        updated_weights = {}
        # --- Adam Update ---
        for name in self.metric_names:
            # Clip gradient to prevent extreme values before Adam update
            grad = np.clip(gradients.get(name, 0.0), -1.0, 1.0) # Use .get for safety

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad**2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1**self.iteration)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[name] / (1 - self.beta2**self.iteration)
            # Update weights
            updated_weights[name] = self.weights[name] + self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # --- End Adam Update ---


        # --- Apply Constraints: Non-negativity and Sum-to-One Normalization ---
        total_weight = 0.0
        for name in self.metric_names:
            # Ensure weights don't die completely, set a small floor
            updated_weights[name] = max(updated_weights.get(name, 0.0), 1e-5)
            total_weight += updated_weights[name]

        # Normalize to sum to 1
        if total_weight > 0:
            for name in self.metric_names:
                self.weights[name] = updated_weights[name] / total_weight
        else:
            # Fallback: Reinitialize to equal weights if total becomes zero (shouldn't happen with floor)
            print("Warning: Total weight became zero, reinitializing.")
            num_metrics = len(self.metric_names)
            self.weights = {name: 1.0 / num_metrics for name in self.metric_names}
        # --- End Constraints ---


        # Track best weights based on evaluation on the *current* data
        # Note: Evaluating on the training sample might lead to overfitting tracking
        current_score = self._evaluate(metric_scores, target_indices)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_weights = self.weights.copy()
            # print(f"Iter {self.iteration}: New best score: {self.best_score:.4f}") # Optional debug

    def _evaluate(self, metric_scores, target_indices):
        # Check if weights exist before combining
        if not self.weights: return 0.0

        combined = np.sum([self.weights.get(name, 0) * metric_scores.get(name, np.array([np.inf])) # Handle missing scores/weights
                           for name in self.metric_names], axis=0)

        if combined.size == 0: return 0.0 # Handle empty combined array

        # Ensure target_indices is not longer than available scores
        num_available = combined.shape[0]
        k_target = min(self.top_k, len(target_indices), num_available)
        k_pred = min(self.top_k, num_available)

        if k_target == 0 or k_pred == 0: return 0.0 # Cannot evaluate if no targets/predictions

        predicted_indices = np.argsort(combined)[:k_pred]
        predicted_set = set(predicted_indices)
        target_set = set(target_indices[:k_target])

        # Calculate Precision@k style score (Jaccard Index of top K sets)
        intersection = len(predicted_set & target_set)
        union = len(predicted_set | target_set) # Or use self.top_k for denominator? Let's use union for Jaccard.
        # return intersection / self.top_k # Classic Precision@K
        return intersection / union if union > 0 else 0.0 # Jaccard Index


    def get_weights(self):
        # Return the best weights found so far based on evaluation
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

def create_global_metric_combinations(max_metrics_to_add, max_metrics_to_remove):
    global ALL_METRIC_COMBINATIONS_TO_TEST

    # 1. Get all available metric names from the METRICS dictionary
    all_available_metrics_set = set(METRICS.keys())
    print(f"Total available metrics: {len(all_available_metrics_set)}")

    # 2. Define the current best combination as the baseline
    current_best_metrics_tuple = ('Canberra', 'Cosine Similarity', 'L1 norm (Manhattan)', 'L2 norm (Euclidean)', 'Lp norm (Minkowski p=3)', 'L∞ norm (Chebyshev)', 'Pearson Correlation', 'Spearman Correlation', 'L0 Norm (eps=1e-6)', 'L1/L2 Ratio', 'Variance')
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
    ALL_METRIC_COMBINATIONS_TO_TEST = final_combinations_list

    print(f"\n--- Generated a total of {len(final_combinations_list)} unique metric combinations to evaluate ---")

# --- NDCG Calculation (Unchanged from your version) ---
def calculate_ndcg(retrieved_indices, target_set, k):
    k = min(k, len(retrieved_indices))
    if k <= 0 or not target_set: return 0.0
    dcg, idcg = 0.0, 0.0
    num_targets = len(target_set)
    for i, idx in enumerate(retrieved_indices[:k]):
        rank = i + 1
        if idx in target_set: dcg += 1.0 / math.log2(rank + 1) # Correct log base 2
    for i in range(min(k, num_targets)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

# --- Helper to Calculate Combined Score (Unchanged) ---
def calculate_combined_score(metric_scores_sample, weights_dict, active_metrics):
    try:
        weighted_scores = [
            metric_scores_sample[name] * weights_dict[name]
            for name in active_metrics if name in metric_scores_sample
        ]
        if not weighted_scores: return None
        combo_score = np.mean(weighted_scores, axis=0)
        return combo_score if combo_score.ndim == 1 else None
    except KeyError as e:
        # print(f"KeyError in calculate_combined_score: Metric '{e}' not found.") # Reduce verbosity
        return None
    except Exception as e:
        # print(f"Error in calculate_combined_score: {type(e).__name__} - {e}") # Reduce verbosity
        return None

# --- Helper function for parallel NDCG evaluation ---
# --- OPTIMIZED with Vectorized Combined Score ---
def _process_single_sample_ndcg(sample_data, current_weights, active_metric_names, k):
    """Processes one sample from eval_dataset for parallel evaluation. Vectorized."""
    try:
        if not isinstance(sample_data, tuple) or len(sample_data) != 2: return None
        metric_scores_sample, target_indices_sample = sample_data
        if not isinstance(metric_scores_sample, dict) or not hasattr(target_indices_sample, '__iter__'): return None

        target_set = set(target_indices_sample)
        if not target_set: return None

        num_source_items = -1
        source_indices = []
        try:
            # Determine num_source_items safely
            first_key = next(iter(metric_scores_sample))
            first_val = metric_scores_sample[first_key]
            if isinstance(first_val, np.ndarray):
                num_source_items = len(first_val)
                source_indices = list(range(num_source_items))
            else: return None
        except (StopIteration, KeyError, TypeError): return None
        if num_source_items <= 0: return None

        # --- Vectorized Combined Score Calculation ---
        active_metrics_in_sample = [name for name in active_metric_names if name in metric_scores_sample]
        if not active_metrics_in_sample: return None

        weights_active = np.array([current_weights.get(name, 0.0) for name in active_metrics_in_sample]) # Use .get for safety if active_metrics might differ

        try: # Ensure all score arrays have the correct length
            data_matrix_list = []
            for name in active_metrics_in_sample:
                scores = metric_scores_sample[name]
                if not isinstance(scores, np.ndarray) or len(scores) != num_source_items:
                    # print(f"Warning: Inconsistent data for metric {name} in sample.")
                    return None # Skip sample if data inconsistent
                data_matrix_list.append(scores)
            if not data_matrix_list: return None # Should not happen if active_metrics_in_sample is not empty
            data_matrix = np.array(data_matrix_list) # Shape (M_active, N_src)
        except Exception as e:
            # print(f"Warning: Error creating data matrix: {e}")
            return None # Skip sample if error during matrix creation

        # Handle potential NaNs/Infs in data matrix *before* weighting
        if not np.all(np.isfinite(data_matrix)):
            data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0) # Replace non-finite with 0

        # Filter for positive weights BEFORE summing
        pos_weights_mask = weights_active > 1e-10
        weights_filtered = weights_active[pos_weights_mask]
        data_matrix_filtered = data_matrix[pos_weights_mask, :]

        if weights_filtered.size == 0: return None # No active metrics with positive weight
        total_w = np.sum(weights_filtered)
        if total_w < 1e-10: return None

        # Calculate weighted sum per source item using broadcasting
        weighted_sum = np.sum(data_matrix_filtered * weights_filtered[:, np.newaxis], axis=0) # Shape (N_src,)
        combined_scores_vec = weighted_sum / total_w # Shape (N_src,)

        # Check for non-finite results and filter
        valid_scores_mask = np.isfinite(combined_scores_vec)
        if not np.any(valid_scores_mask): return None

        final_scores = combined_scores_vec[valid_scores_mask]
        valid_source_indices = np.array(source_indices)[valid_scores_mask]
        # --- End Vectorized Calculation ---

        if final_scores.size == 0: return None

        # --- Get ranking ---
        sorted_indices_within_valid = np.argsort(final_scores)
        retrieved_original_indices = valid_source_indices[sorted_indices_within_valid]

        # --- Calculate NDCG ---
        ndcg = calculate_ndcg(retrieved_original_indices, target_set, k)
        return ndcg if np.isfinite(ndcg) else None

    except Exception: # Catch any broad error during sample processing
        # import traceback; traceback.print_exc(); # For deep debugging
        return None


# --- Objective Function (OPTIMIZED with Parallelization) ---
def evaluate_average_ndcg(weights_array, active_metric_names, eval_dataset, k, n_jobs=-1):
    """ Objective function using parallel processing via joblib. """
    final_ndcg_scores = []
    try:
        if not isinstance(weights_array, np.ndarray): weights_array = np.array(weights_array, dtype=float)
        if weights_array.ndim != 1 or len(weights_array) != len(active_metric_names): return 1.0
        if not eval_dataset: return 1.0
        current_weights = {name: max(0.0, weight) for name, weight in zip(active_metric_names, weights_array)}

        # Use joblib for parallel execution
        results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib.delayed(_process_single_sample_ndcg)(sample_data, current_weights, active_metric_names, k)
            for sample_data in eval_dataset
        )
        final_ndcg_scores = [score for score in results if score is not None]
        if not final_ndcg_scores: return 1.0
        average_ndcg = np.mean(final_ndcg_scores)
        return -average_ndcg # Return negative for minimization

    except Exception as e: # Catch errors during parallel setup/aggregation
        print(f"CRITICAL ERROR evaluate_average_ndcg: Uncaught exception! Error: {type(e).__name__} - {e}. Returning 1.0")
        return 1.0


# --- Main Optimizer Function (Using SciPy + Optimized Ablation Pruning) ---
def optimize_metrics_and_weights_scipy(
        k,                                  # Int: k for NDCG@k
        min_metrics=5,                      # Int: Minimum number of metrics to keep
        optimizer_method='L-BFGS-B',        # String: Method ('L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead')
        optimizer_options=None,             # Dict: Options, e.g., {'maxiter': 75, 'ftol': 1e-5}
        eval_dataset_sample_size=None,      # Int or None: Number of samples for optimization steps
        n_jobs=-1,                          # Int: Number of parallel jobs (-1 for all cores)
        initial_weights=None,               # Dict: Optional initial weights
        verbose=True                        # Bool: Print progress details
):
    """
    Optimizes metric selection/weights using iterative ABLATION pruning.
    Includes parallel objective evaluation, dataset sampling, and tunable optimizer.
    """
    if not OPTIMIZER_EVAL_DATA_CACHE: print("Error: eval_dataset is empty."); return None, None, -np.inf
    if not METRICS: print("Error: METRICS dictionary is empty."); return None, None, -np.inf

    all_metric_names = sorted(list(METRICS.keys()))
    current_metrics = all_metric_names[:]

    # Initialize weights
    if initial_weights: current_weights_dict = {name: max(0.0, initial_weights.get(name, 0.0)) for name in all_metric_names}
    else: current_weights_dict = {name: 1.0 for name in all_metric_names}
    num_metrics_init = len(all_metric_names)
    initial_sum = sum(current_weights_dict.values())
    if initial_sum > 1e-10: current_weights_dict = {name: w / initial_sum for name, w in current_weights_dict.items()}
    else: current_weights_dict = {name: 1.0 / num_metrics_init if num_metrics_init > 0 else 0.0 for name in all_metric_names}

    best_overall_ndcg = -np.inf # Initialize low
    best_overall_metrics = current_metrics[:]
    best_overall_weights = current_weights_dict.copy()

    # --- Prepare Optimizer Options ---
    default_options = {'maxiter': 75, 'disp': False, 'ftol': 1e-5, 'gtol': 1e-4} # More aggressive defaults
    if optimizer_method in ['Nelder-Mead']: default_options.pop('ftol', None); default_options.pop('gtol', None); default_options['adaptive'] = True
    user_options = optimizer_options or {} # Ensure user_options is a dict
    combined_options = {**default_options, **user_options} # User options override defaults


    iteration = 0
    max_iterations = len(all_metric_names) - min_metrics + 1

    # --- Prepare eval dataset (full or sampled) ---
    if eval_dataset_sample_size and eval_dataset_sample_size < len(OPTIMIZER_EVAL_DATA_CACHE):
        if verbose: print(f"Sampling {eval_dataset_sample_size} entries from eval_dataset (size {len(OPTIMIZER_EVAL_DATA_CACHE)}) for optimization.")
        current_eval_dataset = random.sample(OPTIMIZER_EVAL_DATA_CACHE, eval_dataset_sample_size)
    else:
        current_eval_dataset = OPTIMIZER_EVAL_DATA_CACHE
        if verbose and eval_dataset_sample_size: print(f"Sample size >= dataset size or None. Using full dataset ({len(OPTIMIZER_EVAL_DATA_CACHE)}).")
    if not current_eval_dataset: print("Error: Eval dataset empty."); return None, None, -np.inf

    # --- Main Optimization Loop ---
    while len(current_metrics) > min_metrics:
        iteration += 1
        if iteration > max_iterations + 10: print("Warning: Exceeded max iterations + buffer."); break # Increased buffer
        num_current_metrics = len(current_metrics)
        if num_current_metrics == 0: print("Error: No metrics left to optimize."); break
        if verbose: print(f"\n--- Iteration {iteration}/{max_iterations}: Optimizing {num_current_metrics} metrics ---")

        # 1. Optimize weights for the current set
        # Use last known weights as initial guess, or uniform if dict empty
        initial_guess_array = np.array([current_weights_dict.get(name, 1.0 / num_current_metrics if num_current_metrics>0 else 0.0) for name in current_metrics])
        bounds = [(0, None) for _ in current_metrics] if optimizer_method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None

        optimized_ndcg_this_iter = -np.inf
        final_weights_array_this_iter = initial_guess_array # Default to guess if optim fails
        optimization_successful = False

        try:
            if verbose: print(f"  Running scipy.optimize.minimize (method={optimizer_method}, options={combined_options})...")
            opt_result = minimize(
                evaluate_average_ndcg, x0=initial_guess_array,
                args=(current_metrics, current_eval_dataset, k, n_jobs), # Pass dataset & n_jobs
                method=optimizer_method, bounds=bounds, options=combined_options
            )

            # Check success (more lenient check)
            if opt_result.success or 'tolerance' in opt_result.message.lower() or 'iteration' in opt_result.message.lower():
                temp_final_weights = np.maximum(0.0, opt_result.x); temp_sum = np.sum(temp_final_weights)
                if temp_sum > 1e-10: temp_final_weights /= temp_sum
                else: temp_final_weights = np.full(num_current_metrics, 1.0/num_current_metrics if num_current_metrics > 0 else 0.0)

                final_weights_array_this_iter = temp_final_weights # Use the result from optimizer

                # Re-evaluate score with these weights (on same dataset)
                final_score_check = evaluate_average_ndcg(final_weights_array_this_iter, current_metrics, current_eval_dataset, k, n_jobs)

                if np.isfinite(final_score_check):
                    optimized_ndcg_this_iter = -final_score_check
                    # Update current_weights_dict immediately for next iter's guess / ablation
                    current_weights_dict = {name: w for name, w in zip(current_metrics, final_weights_array_this_iter)}
                    if verbose: print(f"  Optimization Result: Success={opt_result.success}, Status='{opt_result.message}', Optimized Avg NDCG = {optimized_ndcg_this_iter:.6f}")
                    optimization_successful = True
                else: print(f"  Warning Iter {iteration}: Optimizer finished but re-eval non-finite ({final_score_check}). Using previous weights.")
            else: 
                if verbose: print(f"  Optimization Failed Iter {iteration}. Message: {opt_result.message}. Using previous weights.")

        except Exception as e: print(f"  CRITICAL ERROR Iter {iteration}: Exception during minimize: {type(e).__name__} - {e}. Using previous weights.")

        # --- Update Best Overall Result ---
        current_score_to_compare = optimized_ndcg_this_iter # Use score from this iter if finite
        if current_score_to_compare > best_overall_ndcg: # Comparison works even if -np.inf
            best_overall_ndcg = current_score_to_compare
            best_overall_metrics = current_metrics[:]
            best_overall_weights = {name: w for name, w in zip(current_metrics, final_weights_array_this_iter)} # Store weights that yielded best score
            if verbose: print(f"  *** New best overall NDCG found: {best_overall_ndcg:.6f} with {len(best_overall_metrics)} metrics ***")

        # --- Ablation Pruning (Parallelized) ---
        if len(current_metrics) <= min_metrics: 
            if verbose: print(f"\nReached min metrics ({min_metrics})."); break

        metric_to_prune = None
        # Use the score from this iteration as baseline (even if optimization failed, use score with previous weights)
        baseline_score_val = evaluate_average_ndcg(final_weights_array_this_iter, current_metrics, current_eval_dataset, k, n_jobs)
        if not np.isfinite(baseline_score_val): print("  Error: Baseline NDCG for pruning is non-finite. Cannot prune. Stopping."); break
        baseline_ndcg_for_pruning = -baseline_score_val # Convert to positive NDCG

        min_ndcg_drop = float('inf') # Smallest drop is best removal candidate

        if verbose: print(f"  Evaluating metric importance via parallel ablation (baseline NDCG {baseline_ndcg_for_pruning:.6f})...")
        ablation_tasks = []
        weights_for_ablation_eval = {name: w for name, w in zip(current_metrics, final_weights_array_this_iter)} # Use current iter weights

        for i, name_to_remove in enumerate(current_metrics):
            temp_metrics = current_metrics[:i] + current_metrics[i+1:]
            if not temp_metrics: continue
            temp_weights_array = np.array([weights_for_ablation_eval.get(name, 0.0) for name in temp_metrics])
            ablation_tasks.append(joblib.delayed(evaluate_average_ndcg)(temp_weights_array, temp_metrics, current_eval_dataset, k, n_jobs))

        if ablation_tasks:
            ablation_scores = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(ablation_tasks)
            valid_ablation_metrics = [m for i, m in enumerate(current_metrics) if len(current_metrics[:i] + current_metrics[i+1:]) > 0] # Metrics actually tested

            for i, name_to_remove in enumerate(valid_ablation_metrics): # Iterate metrics tested
                if i < len(ablation_scores):
                    score_without = ablation_scores[i]
                    if np.isfinite(score_without):
                        ndcg_without_metric = -score_without
                        ndcg_drop = baseline_ndcg_for_pruning - ndcg_without_metric
                        if verbose >= 2: print(f"    Removing '{name_to_remove}': NDCG={ndcg_without_metric:.6f}, Drop={ndcg_drop:.6f}")
                        if ndcg_drop < min_ndcg_drop: min_ndcg_drop = ndcg_drop; metric_to_prune = name_to_remove
                    # else: print(f"    Warning: Non-finite score removing '{name_to_remove}'.")
                # else: print error? Should not happen

        # Prune
        if metric_to_prune:
            if verbose: print(f"  Pruning metric '{metric_to_prune}' (Smallest NDCG drop: {min_ndcg_drop:.6f})")
            current_metrics.remove(metric_to_prune)
            current_weights_dict.pop(metric_to_prune, None) # Remove from weights dict
            # Re-normalize remaining weights in current_weights_dict for next initial guess
            active_sum = sum(current_weights_dict.values())
            num_rem = len(current_metrics)
            if active_sum > 1e-10:
                for name in current_metrics: current_weights_dict[name] = current_weights_dict.get(name, 0.0) / active_sum
            elif num_rem > 0: # Handle sum being zero or dict empty
                for name in current_metrics: current_weights_dict[name] = 1.0 / num_rem
        else: 
            if verbose: print("  Could not determine metric to prune (or all removals hurt significantly). Stopping."); break

    # --- Final Reporting ---
    print(f"\n--- Iterative Optimization Finished ({iteration-1} iterations) ---")
    if best_overall_metrics and best_overall_weights is not None: # Check weights explicitly
        print(f"Best config found with {len(best_overall_metrics)} metrics:")
        print(f"  Metrics: {sorted(best_overall_metrics)}")
        print(f"  Avg NDCG@{k} (score from optimization on {'sampled' if eval_dataset_sample_size else 'full'} dataset): {best_overall_ndcg:.6f}")

        # Optional: Recalculate on full dataset if sampling was used
        if eval_dataset_sample_size and eval_dataset_sample_size < len(OPTIMIZER_EVAL_DATA_CACHE):
            print(f"  Recalculating final score on full dataset ({len(OPTIMIZER_EVAL_DATA_CACHE)} entries)...")
            final_weights_values = [best_overall_weights.get(name, 0.0) for name in best_overall_metrics] # Get weights in correct order
            final_full_score_val = evaluate_average_ndcg(final_weights_values, best_overall_metrics, OPTIMIZER_EVAL_DATA_CACHE, k, n_jobs)
            if np.isfinite(final_full_score_val): print(f"  Avg NDCG@{k} (on full dataset): {-final_full_score_val:.6f}")
            else: print("  Could not calculate score on full dataset.")

        # Prepare final weights (normalized)
        final_sum = sum(best_overall_weights.values()); final_best_weights = {k: v / final_sum if final_sum > 1e-10 else v for k, v in best_overall_weights.items()}
    else: print("No suitable metric combination found or optimization failed early."); final_best_weights = None

    return best_overall_metrics, final_best_weights, best_overall_ndcg