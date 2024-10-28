import torch
from torch import nn
import numpy as np
from collections import Counter
import LLM_Small1x1 as Small1x1
import LLM_Verdict as Verdict
import scipy.sparse
import math

layer, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, totalLayers = "", "", "", "", "", "", ""
llm, layerSizes, device, hidden_sizes, layers, layers, currentLayer, relevantLayerIndices, useBitNet = "", "", "", "", "", [], 0, [], False

def initializePackages(devicePackage, seed="", useBitLinear=False):
    global device, np, torch, useBitNet 
    device = devicePackage
    useBitNet = useBitLinear
    if(seed != ""):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
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
    global layer
    global source
    global dictionaryForSourceLayerNeuron
    global dictionaryForLayerNeuronSource
    global hidden_sizes
    global llm

    if not (isinstance(module, nn.Sequential) or isinstance(module, Small1x1.FeedForward) or isinstance(module, Small1x1.TransformerBlock) or isinstance(module, nn.Dropout) or isinstance(module, Verdict.FeedForward) or isinstance(module, Verdict.TransformerBlock)):
        if (llm == True):
            actualLayer = layer
            layerNeurons = layers[actualLayer][1]
            if(source >= dictionaryForSourceLayerNeuron.shape[0]):
                return
        else:
            actualLayer = int(layer/2)
            layerNeurons = layers[actualLayer][1].out_features

        correctTypes = False
        if(llm == False):
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
            dictionaryForSourceLayerNeuron[source][layer,:layerNeurons] = relevantOutput
            # if(source == 0):
            #   print(relevantOutput, dictionaryForSourceLayerNeuron[source][layer,:layerNeurons])

            #Use for array structure like: [layer, neuron, source]
            output = relevantOutput if len(relevantOutput.shape) == 1 else relevantOutput[0]
            for neuronNumber, neuron in enumerate(output):
                if neuronNumber < layerNeurons:
                    dictionaryForLayerNeuronSource[layer][neuronNumber][source] = neuron
                else:
                    break

        if(layer % 2 == 0 and llm != True):
            if(checkIfActivationLayerExists(hidden_sizes, actualLayer)):
                layer += 1
            elif(layer == (len(layers)*2)-2):
                layer = 0
            else:
                layer += 2
        else:
            if((layer == (len(layers)*2)-1 and llm != True) or (layer == (len(layers))-1 and llm == True)):
                layer = 0
            else:
                layer += 1    

def attachHooks(hookLoader, model, llmType = False):
    global source, layer

    hooks = []  # Store the handles for each hook
    outputs = np.array([])

    for name, module in model.named_modules():
        if not isinstance(module, CustomizableRENN):
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    with torch.no_grad():
        # Forward Pass
        for source, (inputs, labels) in enumerate(hookLoader):
            layer = 0
            # Uncomment for array structure like: [source, layer, neuron]
            if not llmType:
                inputs = inputs.float()
            inputs = inputs.to(device)
            _ = model(inputs)

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

def createDictionaries(hidden_sizes, totalLayersParameter, train_samples):
    global activationsBySources, activationsByLayers, totalLayers, layerSizes
    totalLayers = totalLayersParameter
    layerSizes = [size[1] for size in hidden_sizes[:]]
    if useBitNet:
        activationsBySources = np.zeros((train_samples, totalLayers, np.max(layerSizes)), dtype=int)
        activationsByLayers = np.zeros((totalLayers, np.max(layerSizes), train_samples), dtype=int)
    else:
        activationsBySources = np.zeros((train_samples, totalLayers, np.max(layerSizes)), dtype=np.float128)
        activationsByLayers = np.zeros((totalLayers, np.max(layerSizes), train_samples), dtype=np.float128)
    print("Hook-Dictionaries created")# - ", "Activations by Sources (Shape): ", activationsBySources.shape, " | ", "Activations by Layers (Shape): ", activationsByLayers.shape)

def runHooks(train_dataloader, model, layersParameter=layers, llmType = False):
    global layers, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, llm
    
    #Variables for usage within the hook
    llm = llmType
    layers = layersParameter
    dictionaryForSourceLayerNeuron = activationsBySources
    dictionaryForLayerNeuronSource = activationsByLayers

    attachHooks(train_dataloader, model, llmType)
    activationsBySources = dictionaryForSourceLayerNeuron
    activationsByLayers = dictionaryForLayerNeuronSource
    print("Hooks finished successfully")

def initializeHook(train_dataloader, model, hidden_sizesParameter, train_samples):
    global totalLayers, layer, hidden_sizes, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers
    
    print("Initializing Hooks")
    hidden_sizes = hidden_sizesParameter
    totalLayers = len(layers)*2
    createDictionaries(hidden_sizes, totalLayers, train_samples)
    runHooks(train_dataloader, model, layers)

def initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource
    
    dictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, np.max(layerSizes)))
    dictionaryForLayerNeuronSource = np.zeros((totalLayers, np.max(layerSizes), eval_samples))

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader, model)

    return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource

def identifyClosestSources(closestSources, outputs, mode = ""):
    global layers
    
    dictionary = activationsByLayers
    if(mode == "Sum"):
        layerNumbersToCheck = [idx*2 for idx, (name, layerNumber, activation) in enumerate(layers)]
    elif(mode == "Activation"):
        layerNumbersToCheck = [(idx*2)+1 for idx, (name, layerNumber, activation) in enumerate(layers) if getActivation(hidden_sizes, idx) != False]
    else:
        layerNumbersToCheck = [idx for idx, _ in enumerate(layers)]

    layersToCheck = dictionary[layerNumbersToCheck]
    outputsToCheck = outputs[layerNumbersToCheck]
    identifiedClosestSources = np.empty((len(layersToCheck), np.max(layerSizes), closestSources), dtype=tuple)

    #print(len(layersToCheck), len(outputsToCheck))
    for currentLayer, layer in enumerate(layersToCheck):
        for currentNeuron, neuron in enumerate(layer):
            maxNeurons = layers[currentLayer][1] if mode == "" else layers[currentLayer][1].out_features
            if(currentNeuron < maxNeurons):
                #print(currentLayer, currentNeuron, len(neuron), outputsToCheck[currentLayer][currentNeuron], outputsToCheck.shape)
                differencesBetweenSources = np.abs(neuron - np.full(len(neuron), outputsToCheck[currentLayer][currentNeuron]))
                sortedSourceIndices = np.argsort(differencesBetweenSources) # for highest difference uncomment this: [::-1]
                closestSourceIndices = sortedSourceIndices[:closestSources]
                tuples = tuple((closestSourceIndices[i], neuron[closestSourceIndices[i]], abs(neuron[closestSourceIndices[i]]-outputsToCheck[currentLayer][currentNeuron])) for i in range(closestSources))
                identifiedClosestSources[currentLayer][currentNeuron] = tuples
    return identifiedClosestSources, outputsToCheck, layerNumbersToCheck

def getMostUsed(sources, mode=""):
    mostUsed = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        for currentNeuron, neuron in enumerate(layer):
            maxNeurons = layers[currentLayer][1] if mode == "" else layers[currentLayer][1].out_features
            if(currentNeuron < maxNeurons):
                for sourceNumber, value, difference in neuron:
                    mostUsed.append(sourceNumber)
                    sourceCounter += 1
    return sourceCounter, mostUsed

def getMostUsedSources(sources, closestSources, weightedMode=""):
    weightedSources = []

    sourceCounter, mostUsed = getMostUsed(sources, weightedMode)
    counter = Counter(mostUsed)

    print("Total closest Sources :" , sourceCounter, " | ", closestSources, " closest Sources (",weightedMode,") in format: [SourceNumber, Occurances]: ", counter.most_common()[:closestSources])
    return counter.most_common()[:closestSources]
def save_sparse_3d_array(array, filename):
    # Flatten the 3D array to a 2D array where each row represents one element
    shape = array.shape
    flat_array = array.reshape(-1)

    # Convert flattened array to COO format (ignoring zeros)
    sparse_array = scipy.sparse.coo_matrix(flat_array)

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

def analyzeData(analyzeActivationsBySources = True):
    dictionary = activationsBySources
    if(analyzeActivationsBySources):
        dictionary = activationsByLayers

    save_sparse_3d_array(dictionary, 'SparseArray.txt')
    unique_values = getValuesCount(dictionary)
    getValueClusters(dictionary)
    getMinimumPrecision(unique_values)
    print("\n")

    
                
                
