import os
import logging
import timeit

# Set up dependencies
logging.info("Installing dependencies...")
os.system("pip install -q lorem==0.1.1 tiktoken==0.8.0 stanza ipywidgets==7.7.1 plotly pyarrow zstandard optuna python-Levenshtein cma")
import stanza
import optuna
from Levenshtein import distance as levenshtein
import cma
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

import sys
from subprocess import run
sys.path.append('/tf/.local/lib/python3.11/site-packages')
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=True)
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize,sent_tokenize

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

# Initialization: MNIST Dataset
import MNISTFinalEvaluation as MNIST
MNIST.initializePackages(mnist, to_categorical, nn, DataLoader, pd, optuna, device)
trainSetMNIST, testSetMNIST = MNIST.createTrainAndTestSet()

#Not adjusting the following values, since they would be part of the model beforehand
test_samples = len(testSetMNIST)
learning_rate = 0.0004
epochs = 100
useBitLinear = False

batch_size_training = 64
batch_size_test = 64
loss_function = "MSE"
optimizer = "Adam"

closestSources = 42
showClosestMostUsedSources = 3
visualizationChoice = "Weighted"

# Initialization of fixed values
eval_samples = 10#len(testSetMNIST)
INTEGER_LIMIT = 4294967295
seeds = [random.randint(0, INTEGER_LIMIT) for _ in range(10)]
layerSizes = [128, 512, 2048, 8192]
train_samples = [100, 1000, 10000, len(trainSetMNIST)]
activation_types = ['ReLU', 'Sigmoid', 'Tanh']

for seed in seeds:
    for layerSize in layerSizes:
        for train_sample_count in train_samples:
            for activationType in activation_types:
                normalLayers = [['Linear', layerSize, activationType], ['Linear', layerSize, activationType]]
                neuronChoices = [((0, layerSize), True), ((0, layerSize), True)]
                
                hidden_sizes = [(normalLayer, normalLayerSize, activationLayer)
                    for normalLayer, normalLayerSize, activationLayer in normalLayers]
                hidden_sizes.append(('Linear', 10, activationType))
                
                visualizeCustom = [
                    ((normalLayer[0], normalLayer[1]), activationLayer)
                    for normalLayer, activationLayer in neuronChoices]
                visualizeCustom.append(((0, 10), True))
                
                import RENNFinalEvaluation as RENN
                RENN.initializePackages(device, io, pd, pa, pq, zstd, levenshtein, cma, MNIST, seed, useBitLinear)
                
                """# Training"""
                
                MNIST.initializeDatasets(train_sample_count, test_samples, eval_samples, batch_size_training, batch_size_test, seed)
                MNIST.trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs)
                
                """# Sourcecheck Customisable RENN"""
                
                elapsed_time = timeit.timeit(
                    lambda: MNIST.initializeHook(hidden_sizes, train_sample_count),
                    number=1  # Run it once
                )
                print(f"Time taken: {elapsed_time:.2f} seconds")
                
                """# Visualization Customisable RENN"""
                
                elapsed_time = timeit.timeit(
                    lambda: MNIST.visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, True),
                    number=1  # Run it once
                )
                print(f"Time taken: {elapsed_time:.2f} seconds")