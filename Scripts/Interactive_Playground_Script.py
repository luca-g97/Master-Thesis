import os
import logging
import timeit

# Set up dependencies
logging.info("Installing dependencies...")
os.system("pip install -q lorem==0.1.1 tiktoken==0.8.0 stanza ipywidgets==7.7.1 plotly pyarrow zstandard")
import stanza
stanza.download('en', verbose=False)

#Libraries related to Torch
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
#Libraries needed for MNIST dataset
from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import to_categorical
#Libraries needed for HSV-RGB
import colorsys
#!pip install -q numpy==1.26.4 scipy spacy
#!python -m spacy download en_core_web_sm -q
#import spacy
import lorem
import tiktoken
import random
from transformers import GPT2Tokenizer
#!pip install -q datasets
#from datasets import load_dataset
import stanza
# Suppress logging from stanza
nlp = stanza.Pipeline('en', verbose=False)
stanza.download('en', verbose=False)

#Libraries needed for Visualization
from IPython.display import display, clear_output
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'colab'

#Libraries needed for Zipping
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

#Set the correct device. Prefer a graphics card (cuda) if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device type: ", device)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def check_if_all_files_are_present(expected_files):
    for item in expected_files:
        if not os.path.exists(item):
            logging.warning(f"Missing: {item}")
            return False
    return True

def main():
    logging.info("Starting Interactive AI Playground...")

    # Check for required files
    expected_files = ['Customizable_RENN.py', 'Images_HSVRGB.py', 'Images_MNIST.py',
                      'LLM_Small1x1.py', 'LLM_GPT2.py', 'LLM_LSTM.py', 'Widgets.py', 'Datasets']

    if not check_if_all_files_are_present(expected_files):
        logging.info("Cloning repository...")
        os.system("rm -rf ./Interactive-AI-Playground")
        os.system("git clone https://github.com/luca-g97/Master-Thesis.git ./Interactive-AI-Playground")
        os.system("mv ./Interactive-AI-Playground/Google_Colab-Interactive_AI_Playground/* ./")
        os.system("rm -rf ./Interactive-AI-Playground/")

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Using device: {device}")

if __name__ == "__main__":
    main()

# Initialization: MNIST Dataset
import Images_MNIST as MNIST
MNIST.initializePackages(mnist, to_categorical, nn, DataLoader, device)
trainSetMNIST, testSetMNIST = MNIST.createTrainAndTestSet()

# HSV -> RGB Conversion
import Images_HSVRGB as HSVRGB
HSVRGB.initializePackages(colorsys, go, pio, DataLoader, device)
# Set visualize to True if you want to see the 3D-cube containing both the train and test samples
trainSetHSVRGB, testSetHSVRGB = HSVRGB.createTrainAndTestSet(50000, 10000, visualize=False)

# Small 1x1: LLM-Sourcecheck
import LLM_Small1x1 as Small1x1
Small1x1.initializePackages(random, lorem, device, tiktoken, DataLoader)
small1x1 = Small1x1.createTrainAndTestSet(100)

# WikiText2: GPT2-Sourcecheck
import LLM_GPT2 as GPT2
GPT2.initializePackages(random, lorem, device, tiktoken, DataLoader, nlp, GPT2Tokenizer)
# Choose one of the following as needed:
# gpt2train, gpt2test = GPT2.createTrainSet()
# gpt2train, gpt2test = GPT2.createWikiTrainSet("sports")
gpt2train, gpt2test = GPT2.createWikiText2TrainSet()
# gpt2train, gpt2test = GPT2.createEnglishWikiTrainSet("./english_wikipedia/data/train-00021-of-00022-8014350d27e6cde7.parquet")

# WikiText2: LSTM-Sourcecheck
import LLM_LSTM as LSTM
LSTM.initializePackages(device, DataLoader)
lstmTrain, lstmTest = LSTM.createTrainAndTestSet()

# Organize datasets for easy access
datasets = {
    "MNIST": (trainSetMNIST, testSetMNIST),
    "HSV-RGB": (trainSetHSVRGB, testSetHSVRGB),
    "Small 1x1": (small1x1[:int(len(small1x1) * 0.8)], small1x1[int(len(small1x1) * 0.8):int(len(small1x1) * 0.9)]),
    "WikiText2 (GPT2)": (gpt2train, gpt2test),
    "WikiText2 (LSTM)": (lstmTrain, lstmTest)
}

layerAmount = 2
learning_rate = 0.0004
epochs = 10
seed = 0
useBitLinear = False
normalLayers = [['Linear', 128, 'ReLU'], ['Linear', 128, 'ReLU']]
neuronChoices = [((0, 128), True), ((0, 128), True)]

datasetChoice = "WikiText2 (GPT2)"
chosenDataSet = ""

# Selection Initialization
if datasetChoice == "MNIST":
    chosenDataSet = MNIST
elif datasetChoice == "HSV-RGB":
    chosenDataSet = HSVRGB
elif datasetChoice == "Small 1x1":
    chosenDataSet = Small1x1
elif datasetChoice == "WikiText2 (GPT2)":
    chosenDataSet = GPT2
elif datasetChoice == "WikiText2 (LSTM)":
    chosenDataSet = LSTM

train_samples = 10#len(datasets[datasetChoice][0])
test_samples = len(datasets[datasetChoice][1])
eval_samples = 10#len(datasets[datasetChoice][1])

if datasetChoice in ["Small 1x1", "WikiText2 (GPT2)"]:
    LLM_Layers, TransformerBlockLayer = chosenDataSet.setGPTSettings(layerAmount, learning_rate, epochs)
    chosenDataSet.setGPTSettings(layerAmount, learning_rate, epochs)
    hidden_sizes = [LLM_Layers[0]]
    for _ in range(layerAmount):
        hidden_sizes.append(TransformerBlockLayer)
    hidden_sizes.append(LLM_Layers[1])
    hidden_sizes = [item for sublist in hidden_sizes for item in sublist]
elif(datasetChoice == "WikiText2 (LSTM)"):
    hidden_sizes = LSTM.get_hidden_sizes(layerAmount, train_samples)
else:
    hidden_sizes = [
        (normalLayer.value, normalLayerSize.value, activationLayer.value)
        for normalLayer, normalLayerSize, activationLayer in normalLayers
    ]
    hidden_sizes.append(('Linear', 10, 'ReLU'))

visualizeCustom = [
    ((normalLayer[0], normalLayer[1]), activationLayer)
    for normalLayer, activationLayer in neuronChoices]
visualizeCustom.append(((0, 10), True))

# Extract settings from Widgets
batch_size_training = 64
batch_size_test = 64
loss_function = "MSE"
optimizer = "Adam"

closestSources = 10
showClosestMostUsedSources = 5
visualizationChoice = "Weighted"

# @title Click `Show code` in the code cell. { display-mode: "form" }

import numpy as np
from collections import defaultdict

# 3 integers (evalSample, layer, neuron) and a string (source of 20 bytes) and 3 floats (value, difference)
element_memory = (4 * 3) + 20 + (8 * 3)  # 3 integers + 1 string + 3 floats
# element_memory = 12 bytes for integers + 20 bytes for string + 24 bytes for floats = 56 bytes

# Initialize total elements counters
total_non_zero_elements = 0
total_non_zero_elements_in_list = 0

# Loop through layers and calculate total non-zero elements
for layer in hidden_sizes:
    neurons_in_layer = layer[1]
    # Calculate total non-zero elements for each layer (eval_samples * neurons_in_layer * closestSources)
    total_non_zero_elements += eval_samples * neurons_in_layer * closestSources
    # Calculate total elements for the list (eval_samples * neurons_in_layer * train_samples)
    total_non_zero_elements_in_list += eval_samples * neurons_in_layer * train_samples

# Total memory usage (in bytes) for both sparse tensor and list
total_storage_bytes = total_non_zero_elements * element_memory
total_storage_for_list_bytes = total_non_zero_elements_in_list * element_memory

# Convert bytes to megabytes (MB)
total_storage_mb = total_storage_bytes / (1024 ** 2)
total_storage_for_list_mb = total_storage_for_list_bytes / (1024 ** 2)

# Print results
print(f"Total memory usage for the sparse tensor: {total_storage_mb:.2f} MB")
print(f"Total memory usage for the list: {total_storage_for_list_mb:.2f} MB")

import Customizable_RENN as RENN
RENN.initializePackages(device, io, pd, pa, pq, zstd, chosenDataSet, seed, useBitLinear)

"""# Training"""

chosenDataSet.initializeDatasets(train_samples, test_samples, eval_samples, batch_size_training, batch_size_test, seed)
chosenDataSet.trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs)

"""# Sourcecheck Customisable RENN"""

elapsed_time = timeit.timeit(
    lambda: chosenDataSet.initializeHook(hidden_sizes, train_samples),
    number=1  # Run it once
)
print(f"Time taken: {elapsed_time:.2f} seconds")

"""# Visualization Customisable RENN"""

elapsed_time = timeit.timeit(
    lambda: chosenDataSet.visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, True),
    number=1  # Run it once
)
print(f"Time taken: {elapsed_time:.2f} seconds")