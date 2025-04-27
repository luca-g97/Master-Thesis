import os
import logging
import timeit

# Set up dependencies
logging.info("Installing dependencies...")
import sys
from subprocess import run
sys.path.append('/tf/.local/lib/python3.11/site-packages')
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
os.system("pip install -q lorem==0.1.1 tiktoken==0.8.0 stanza ipywidgets==7.7.1 plotly pyarrow zstandard optuna python-Levenshtein cma")
import stanza
import optuna
from Levenshtein import distance as levenshtein
import cma

#Additional libraries for evaluation loop
import json
import itertools

#Libraries related to Torch
import torch
from torch.utils.data import DataLoader
from torch import nn
#Libraries needed for MNIST dataset
from tqdm import tqdm
from keras.datasets import mnist
from keras.utils import to_categorical
import random

#Libraries needed for Visualization
import plotly.io as pio
pio.renderers.default = 'colab'

#Libraries needed for Zipping
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- Local Module Imports ---
# Ensure these modules are in the same directory or Python path
try:
    import MNISTFinalEvaluation as MNIST
    import RENNFinalEvaluation as RENN
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure MNISTFinalEvaluation.py and RENNFinalEvaluation.py are accessible.")
    sys.exit(1)


# --- Initial Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
pio.renderers.default = 'colab' # Adjust if not using Colab

# Set the correct device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device type: {device}")

# --- Configuration ---
# Fixed values
eval_samples = 2  # Reduced for testing, adjust as needed
INTEGER_LIMIT = 4294967295
seeds = [random.randint(0, INTEGER_LIMIT) for _ in range(10)]
layerSizes = [128, 512, 2048]
train_samples_options = [100, 10000, 60000] # Assuming 60k is full MNIST train set size
#activation_types = ['ReLU', 'Sigmoid', 'Tanh']

# MNIST specific settings (assuming these are fixed for the loop)
test_samples = 10000 # Assuming full MNIST test set size
learning_rate = 0.0004
epochs = 100 # Consider reducing for faster testing loops
useBitLinear = False
batch_size_training = 64
batch_size_test = 64
loss_function = "MSE"
optimizer = "Adam"

# RENN specific settings
closestSources = 25 # Adjusted because best results seemed likely to be on lower bounds
showClosestMostUsedSources = 3
visualizationChoice = "Weighted"

# Checkpoint file
CHECKPOINT_FILE = "training_checkpoint.json"

# --- Helper Functions ---
def load_checkpoint():
    """Loads the last saved checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                # Validate checkpoint structure if necessary
                if all(k in checkpoint for k in ["seed", "layerSize", "train_sample_count"]):
                    logging.info(f"Resuming from checkpoint: {checkpoint}")
                    return checkpoint
                else:
                    logging.warning("Invalid checkpoint file structure. Starting from scratch.")
                    return None
        except json.JSONDecodeError:
            logging.warning(f"Could not decode {CHECKPOINT_FILE}. Starting from scratch.")
            return None
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            return None
    return None

def save_checkpoint(params):
    """Saves the current progress (parameters of the next iteration)."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(params, f)
        # logging.info(f"Checkpoint saved: {params}") # Optional: Log every save
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

def clear_checkpoint():
    """Removes the checkpoint file upon successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            logging.info("All iterations completed. Checkpoint file removed.")
        except Exception as e:
            logging.error(f"Error removing checkpoint file: {e}")


# --- Main Training Loop ---
def run_training_loop():
    """Executes the main training and evaluation loop."""

    # Load MNIST data once
    logging.info("Loading MNIST dataset...")
    # Assuming MNIST.initializePackages also loads data if not already done
    # If createTrainAndTestSet loads data every time, it should be refactored in MNIST module
    MNIST.initializePackages(mnist, to_categorical, nn, DataLoader, pd, optuna, device)
    trainSetMNIST_full, testSetMNIST = MNIST.createTrainAndTestSet() # Load full sets once
    logging.info("MNIST dataset loaded.")

    # Initialize RENN packages once
    # Note: RENN.initializePackages might need specific seed handling if it sets global seeds
    # Passing a dummy seed here, actual seed is used per iteration later.
    RENN.initializePackages(device, io, pd, pa, pq, zstd, levenshtein, cma, MNIST, seeds[0], useBitLinear)


    # Define the parameter space using itertools.product
    param_space = list(itertools.product(seeds, layerSizes, train_samples_options))

    # Load checkpoint and determine starting point
    checkpoint = load_checkpoint()
    start_index = 0
    if checkpoint:
        try:
            # Find the index corresponding to the checkpoint
            start_index = param_space.index(
                (checkpoint['seed'], checkpoint['layerSize'], checkpoint['train_sample_count'])
            )
            logging.info(f"Found checkpoint. Resuming from iteration {start_index + 1}/{len(param_space)}.")
        except ValueError:
            logging.warning("Checkpoint parameters not found in current parameter space. Starting from scratch.")
            start_index = 0 # Start from beginning if checkpoint is invalid for current space


    # Iterate through the parameter space starting from the checkpoint
    for i, (seed, layerSize, train_sample_count) in enumerate(tqdm(param_space, desc="Overall Progress", initial=start_index)):

        if i < start_index:
            continue # Skip iterations before the checkpoint

        logging.info(f"\n--- Starting Iteration {i+1}/{len(param_space)} ---")
        logging.info(f"Params: seed={seed}, layerSize={layerSize}, train_samples={train_sample_count}")

        # --- Configure iteration-specific settings ---
        # Set seed for reproducibility for this specific iteration
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Adjust train_sample_count if it exceeds the dataset size
        actual_train_samples = min(train_sample_count, len(trainSetMNIST_full))
        if actual_train_samples != train_sample_count:
            logging.warning(f"Requested {train_sample_count} train samples, but dataset only has {len(trainSetMNIST_full)}. Using {actual_train_samples}.")


        # Define model layers for this iteration
        # Using fixed structure as per original code, adjust if needed
        normalLayers = [['Linear', layerSize, 'ReLU'], ['Linear', layerSize, 'ReLU']]
        neuronChoices = [((0, layerSize), True), ((0, layerSize), True)] # Assuming these ranges are correct

        hidden_sizes = [(normalLayer, normalLayerSize, activationLayer)
                        for normalLayer, normalLayerSize, activationLayer in normalLayers]
        hidden_sizes.append(('Linear', 10, 'ReLU')) # Output layer

        visualizeCustom = [
            ((normalLayer[0], normalLayer[1]), activationLayer) # Assuming layer indices match normalLayers
            for normalLayer, activationLayer in neuronChoices]
        visualizeCustom.append(((0, 10), True)) # Output layer visualization choice

        # --- Run MNIST Operations ---
        try:
            logging.info("Initializing MNIST datasets for current iteration...")
            # Pass the correctly sized subset of training data if needed by initializeDatasets
            # This depends on how MNIST.initializeDatasets uses the train_sample_count
            MNIST.initializeDatasets(actual_train_samples, test_samples, eval_samples, batch_size_training, batch_size_test, seed)

            logging.info("Training MNIST model...")
            MNIST.trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs)

            logging.info("Initializing RENN hook...")
            start_time = timeit.default_timer()
            # Pass the actual number of samples used for training
            MNIST.initializeHook(hidden_sizes, actual_train_samples)
            elapsed_time = timeit.default_timer() - start_time
            logging.info(f"RENN Hook Initialization Time: {elapsed_time:.2f} seconds")

            logging.info("Running RENN Visualization...")
            start_time = timeit.default_timer()
            evaluation_name = f"S={seed}-L={layerSize}-T={train_sample_count}"
            MNIST.visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, evaluation_name, True)
            elapsed_time = timeit.default_timer() - start_time
            logging.info(f"RENN Visualization Time: {elapsed_time:.2f} seconds")

            # --- Save Checkpoint ---
            # Checkpoint saved *after* successful completion of the iteration
            # Get the parameters for the *next* iteration
            if i + 1 < len(param_space):
                next_params = {
                    "seed": param_space[i+1][0],
                    "layerSize": param_space[i+1][1],
                    "train_sample_count": param_space[i+1][2],
                    "activationType": param_space[i+1][3]
                }
                save_checkpoint(next_params)
            else:
                # Last iteration completed successfully
                clear_checkpoint()


        except Exception as e:
            logging.error(f"Error during iteration {i+1} with params: {seed, layerSize, train_sample_count}")
            logging.error(f"Error details: {e}", exc_info=True) # Log traceback
            # Decide if you want to stop or continue to the next iteration
            # Option 1: Stop the script
            print(f"\n--- CRITICAL ERROR ENCOUNTERED. Stopping script. ---")
            print(f"Failed on iteration {i+1}. Checkpoint points to this failed iteration for restart.")
            # Save checkpoint pointing to the *current* failed iteration to retry it next time
            current_params = {
                "seed": seed,
                "layerSize": layerSize,
                "train_sample_count": train_sample_count,
            }
            save_checkpoint(current_params)
            sys.exit(1) # Exit the script
            # Option 2: Log error and continue (uncomment below and comment sys.exit(1))
            # print(f"\n--- ERROR ENCOUNTERED. Skipping to next iteration. ---")
            # continue # Skip to the next iteration in param_space

    logging.info("--- All iterations completed successfully! ---")

# --- Script Execution ---
if __name__ == "__main__":
    run_training_loop()