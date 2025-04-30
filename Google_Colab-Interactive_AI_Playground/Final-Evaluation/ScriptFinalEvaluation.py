import os
import logging
import timeit
import sys
import atexit # To manage shutdown order if needed
import json
import itertools
import random
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from subprocess import run

# --- Tee Class for duplicating streams ---
# (No changes needed to Tee class itself - unchanged from previous version)
class Tee:
    """Helper class to duplicate stream output to multiple destinations (e.g., console and file)."""
    def __init__(self, *files):
        self.files = files
        self.closed = False # Flag to indicate if closed

    def write(self, obj):
        if self.closed:
            # print("Warning: Write attempt on closed Tee object.", file=sys.__stderr__) # Optional warning
            return
        for f in self.files:
            try:
                # Check if the underlying file object itself is closed
                if hasattr(f, 'closed') and f.closed:
                    # print(f"Warning: Underlying file {f} is closed. Skipping write.", file=sys.__stderr__)
                    continue
                f.write(obj)
                # No flush here by default, rely on explicit flush or program exit flush
            except ValueError as e:
                # Catch "I/O operation on closed file" errors specifically
                if 'closed file' in str(e).lower():
                    # print(f"Warning: Write attempt on closed underlying file {f}.", file=sys.__stderr__)
                    pass # Ignore writes to closed files gracefully
                else:
                    print(f"Error writing to stream {f}: {e}", file=sys.__stderr__) # Log other errors
            except Exception as e:
                # Handle potential errors during write (e.g., file closed unexpectedly)
                print(f"Error writing to stream {f}: {e}", file=sys.__stderr__)


    def flush(self):
        if self.closed:
            # print("Warning: Flush attempt on closed Tee object.", file=sys.__stderr__) # Optional warning
            return
        for f in self.files:
            try:
                # Check if the underlying file object itself is closed before flushing
                if hasattr(f, 'closed') and f.closed:
                    # print(f"Warning: Underlying file {f} is closed. Skipping flush.", file=sys.__stderr__)
                    continue
                f.flush()
            except ValueError as e:
                # Catch "I/O operation on closed file" errors specifically
                if 'closed file' in str(e).lower():
                    # print(f"Warning: Flush attempt on closed underlying file {f}.", file=sys.__stderr__)
                    pass # Ignore flushes to closed files gracefully
                else:
                    print(f"Error flushing stream {f}: {e}", file=sys.__stderr__) # Log other errors
            except Exception as e:
                # Avoid printing errors during shutdown if file is already closed by finally block
                if 'closed file' not in str(e).lower():
                    print(f"Error flushing stream {f}: {e}", file=sys.__stderr__)

    def close(self):
        """Closes the streams this Tee object *owns* (specifically added file handles)."""
        if self.closed:
            return
        # Only close handles that Tee might uniquely own (like log_file_handle).
        # Don't close original_stdout/stderr here.
        for f in self.files:
            # Check if it's the file handle we track globally
            # (Better: Pass ownership info during init or track owned files internally)
            global current_log_file_handle # Check against the currently active handle
            if f is current_log_file_handle:
                try:
                    if not f.closed:
                        # print(f"Tee closing its file handle: {f.name}", file=sys.__stderr__) # Debug print
                        f.close()
                except Exception as e:
                    print(f"Error closing Tee's file handle: {e}", file=sys.__stderr__)
        self.closed = True # Mark Tee as closed regardless of individual file close success

    def isatty(self):
        # Help libraries like tqdm determine if output is interactive
        # Assume TTY if original stdout is TTY
        try:
            # Check the first file object, assuming it's the original stream (stdout/stderr)
            return self.files[0].isatty()
        except:
            return False # Default to false if error or no files


# --- Global variables for stream management ---
original_stdout = sys.stdout
original_stderr = sys.stderr
# These will now hold the *current* iteration's resources
current_log_file_handle = None
current_file_handler = None # Keep track of the FileHandler object
tee_stdout = None
tee_stderr = None
root_logger = logging.getLogger() # Get root logger once
log_formatter = logging.Formatter("%(asctime)s - %(message)s") # Define formatter once
# --- Global variables End ---

# --- Initial Logging Setup (Console Only) ---
# Configure root logger basic settings
root_logger.setLevel(logging.INFO)
# Remove any pre-existing handlers (e.g., from previous runs in interactive env)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
# Add Console Handler (using original stdout) - this stays constant
console_handler = logging.StreamHandler(original_stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
logging.info("--- Script Initializing ---")
logging.info("Console logging initialized.")
# --- Initial Logging Setup End ---


# --- Cleanup Function ---
def cleanup_streams_and_logging():
    """Restores streams, closes the *last* Tee handle/logger, and shuts down logging."""
    global original_stdout, original_stderr, current_log_file_handle, current_file_handler, tee_stdout, tee_stderr, root_logger

    logging.info("--- Initiating Cleanup ---") # Log cleanup start

    # 1. Restore original stdout and stderr FIRST
    streams_restored = False
    if sys.stdout is tee_stdout and tee_stdout is not None:
        sys.stdout = original_stdout
        streams_restored = True
    if sys.stderr is tee_stderr and tee_stderr is not None:
        sys.stderr = original_stderr
        streams_restored = True # Even if only one was restored

    if streams_restored:
        print("--- Standard streams restored ---", file=original_stdout) # Use original streams now

    # 2. Close the Tee objects (which attempts to close the file handle they use)
    # These might be None if setup failed early or loop didn't run
    if tee_stdout and not tee_stdout.closed:
        try:
            # print(f"Cleanup: Closing tee_stdout...", file=original_stderr) # Debug print
            tee_stdout.close()
        except Exception as e:
            print(f"Error closing tee_stdout during cleanup: {e}", file=original_stderr)
    if tee_stderr and not tee_stderr.closed:
        try:
            # print(f"Cleanup: Closing tee_stderr...", file=original_stderr) # Debug print
            tee_stderr.close()
        except Exception as e:
            print(f"Error closing tee_stderr during cleanup: {e}", file=original_stderr)

    # 3. Remove the specific FileHandler for the last iteration
    if current_file_handler:
        try:
            # print(f"Cleanup: Removing file handler for {current_file_handler.baseFilename}...", file=original_stderr) # Debug print
            root_logger.removeHandler(current_file_handler)
            # Closing the handler itself also closes its underlying stream if opened by the handler
            current_file_handler.close()
        except Exception as e:
            print(f"Error removing/closing file handler during cleanup: {e}", file=original_stderr)

    # 4. Explicitly close the log file handle opened for Tee (as a safeguard)
    # Tee.close() should have handled this, but double-check.
    if current_log_file_handle and not current_log_file_handle.closed:
        try:
            # print(f"Cleanup: Final check closing log file handle {current_log_file_handle.name}...", file=original_stderr) # Debug print
            current_log_file_handle.close()
        except Exception as e:
            print(f"Error closing log_file_handle during final cleanup safeguard: {e}", file=original_stderr)

    # 5. Shutdown the logging system
    # print("Cleanup: Shutting down logging system...", file=original_stderr) # Debug print
    logging.shutdown()

    # Final message to original console
    print(f"--- Script finished. Cleanup complete. Individual logs saved per iteration. ---", file=original_stdout)

# Register cleanup function
atexit.register(cleanup_streams_and_logging)
# --- Cleanup Function End ---


# Set up dependencies (assuming they might be needed early)
if '/tf/.local/lib/python3.11/site-packages' not in sys.path:
    if os.path.exists('/tf/.local/lib/python3.11/site-packages'):
        sys.path.append('/tf/.local/lib/python3.11/site-packages')
        logging.info("Added /tf/.local/lib/python3.11/site-packages to sys.path")
    else:
        logging.warning("/tf/.local/lib/python3.11/site-packages not found, imports might fail if not installed globally.")

# --- Library Import and Installation ---
# User requested these lines remain for local setup
try:
    logging.info("Installing/Checking dependencies...")
    # >>> Lines for library installation <<<
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    os.system("pip install -q lorem==0.1.1 tiktoken==0.8.0 stanza ipywidgets==7.7.1 plotly pyarrow zstandard optuna python-Levenshtein cma")
    # >>> End of lines for library installation <<<

    logging.info("Importing libraries...")
    import torch # Import torch early for device check
    from torch.utils.data import DataLoader
    from torch import nn
    import stanza # stanza seems unused later, but keeping import as requested
    import optuna
    from Levenshtein import distance as levenshtein
    import cma
    from tqdm import tqdm
    from keras.datasets import mnist
    from keras.utils import to_categorical
    import plotly.io as pio
    # Imports moved up: json, itertools, random, io, pd, pa, pq, zstd
    logging.info("Libraries imported successfully.")
except ImportError as e:
    logging.error(f"Failed to import required libraries after installation attempt: {e}", exc_info=True)
    sys.exit(f"Exiting due to import failure: {e}") # atexit cleanup will run
except Exception as e:
    logging.error(f"Error during dependency installation or import: {e}", exc_info=True)
    sys.exit(f"Exiting due to setup error: {e}")
# --- Library Import and Installation End ---

# Setup plotting and device
pio.renderers.default = 'colab' # Adjust if not using Colab
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Using device type: {device}")
# print(f"Using device type: {device}") # This will be printed later after redirection

# --- Local Module Imports ---
try:
    import MNISTFinalEvaluation as MNIST
    import RENNFinalEvaluation as RENN
except ImportError as e:
    logging.error(f"Error importing local modules: {e}", exc_info=True)
    print(f"Error importing local modules: {e}", file=sys.__stderr__) # Print directly if redirection failed
    print("Please ensure MNISTFinalEvaluation.py and RENNFinalEvaluation.py are accessible.")
    sys.exit(1) # atexit cleanup will run
# --- Local Module Imports End ---


# --- Configuration ---
#eval_samples = 100 # Used in MNIST.initializeDatasets
INTEGER_LIMIT = 4294967295
seeds = [random.randint(0, INTEGER_LIMIT) for _ in range(10)]
# Defined by subjective measures: Needed to still be readable and within the first 10 entries per class
good_test_samples = [13, 101, 14, 37, 35, 72, 32, 68, 27, 56, 53, 120, 21, 50, 0, 34, 110, 181, 12, 16]
bad_test_samples = [25, 46, 43, 18, 33, 59, 66, 41, 61, 62]
# Structure: List of tuples, where each tuple is (TypeNameString, ListOfIndices)
test_samples_options = [("Good", good_test_samples), ("Bad", bad_test_samples)]
layerSizes = [128, 512, 2048]
train_samples_options = [100, 10000, 60000]
test_samples = 2000 # General test set size used in MNIST.initializeDatasets
learning_rate = 0.0004
epochs = 100
useBitLinear = False # Passed to RENN.initializePackages
batch_size_training = 64 # Used in MNIST.initializeDatasets
batch_size_test = 64 # Used in MNIST.initializeDatasets
loss_function = "MSE" # Passed to MNIST.trainModel
optimizer = "Adam" # Passed to MNIST.trainModel
closestSources = 35 # Passed to MNIST.visualize - Testing +/-25 so between 10 and 60
showClosestMostUsedSources = 3 # Passed to MNIST.visualize
visualizationChoice = "Weighted" # Passed to MNIST.visualize
CHECKPOINT_FILE = "evaluation_checkpoint.json"
# --- Configuration End ---

# --- Helper Functions ---
def load_checkpoint():
    """Loads the last saved checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                if all(k in checkpoint for k in ["seed", "layerSize", "train_sample_count", "sample_type_name"]):
                    logging.info(f"Resuming from checkpoint: {checkpoint}")
                    return checkpoint
                else:
                    logging.warning("Invalid checkpoint file structure (missing keys). Starting from scratch.")
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
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}") # This goes to current log + console

def clear_checkpoint():
    """Removes the checkpoint file upon successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            logging.info("All iterations completed. Checkpoint file removed.")
        except Exception as e:
            logging.error(f"Error removing checkpoint file: {e}")
# --- Helper Functions End ---


# --- Main Training Loop ---
def run_training_loop():
    """Executes the main training and evaluation loop."""
    global current_log_file_handle, current_file_handler, tee_stdout, tee_stderr, root_logger, log_formatter

    # Load MNIST data once
    logging.info("Loading MNIST dataset...")
    MNIST.initializePackages(mnist, to_categorical, nn, DataLoader, pd, optuna, device)
    trainSetMNIST_full, testSetMNIST = MNIST.createTrainAndTestSet() # Load full sets once
    logging.info("MNIST dataset loaded.")

    # Initialize RENN packages once
    # Using first seed for RENN init? Okay if intended. Pass useBitLinear here.
    RENN.initializePackages(device, io, pd, pa, pq, zstd, levenshtein, cma, MNIST, seeds[0], useBitLinear)

    param_space = list(itertools.product(seeds, layerSizes, train_samples_options, test_samples_options))

    # Load checkpoint and determine starting point
    checkpoint = load_checkpoint()
    start_index = 0
    if checkpoint:
        try:
            # Reconstruct the tuple to find in param_space, including the sample_type tuple
            checkpoint_seed = checkpoint['seed']
            checkpoint_layerSize = checkpoint['layerSize']
            checkpoint_train_sample_count = checkpoint['train_sample_count']
            checkpoint_sample_type_name = checkpoint['sample_type_name']

            # Find the corresponding sample_type tuple from the options
            checkpoint_sample_type_tuple = next((st for st in test_samples_options if st[0] == checkpoint_sample_type_name), None)

            if checkpoint_sample_type_tuple:
                chk_tuple = (checkpoint_seed, checkpoint_layerSize, checkpoint_train_sample_count, checkpoint_sample_type_tuple)
                start_index = param_space.index(chk_tuple)
                logging.info(f"Found checkpoint. Resuming from iteration {start_index + 1}/{len(param_space)}.")
            else:
                logging.warning(f"Sample type name '{checkpoint_sample_type_name}' from checkpoint not found in current 'test_samples_options'. Starting from scratch.")
                start_index = 0

        except ValueError:
            logging.warning("Checkpoint parameters combination not found in current parameter space. Starting from scratch.")
            start_index = 0
        except KeyError:
            # This case is handled by the check in load_checkpoint, but added robustness here.
            logging.warning("Checkpoint file is missing expected keys. Starting from scratch.")
            start_index = 0
        except Exception as e:
            logging.error(f"Error processing checkpoint data: {e}. Starting from scratch.")
            start_index = 0


    # Iterate through the parameter space starting from the checkpoint
    iterator = tqdm(param_space, desc="Overall Progress", initial=start_index)
    for i, (seed, layerSize, train_sample_count, sample_type) in enumerate(iterator):

        # --- PRE-ITERATION CLEANUP (for resources from previous iteration) ---
        # (Unchanged from previous version)
        if sys.stdout is tee_stdout and tee_stdout is not None: sys.stdout = original_stdout
        if sys.stderr is tee_stderr and tee_stderr is not None: sys.stderr = original_stderr
        if tee_stdout: tee_stdout.close()
        if tee_stderr: tee_stderr.close()
        if current_file_handler:
            root_logger.removeHandler(current_file_handler)
            current_file_handler.close()
            current_file_handler = None
        if current_log_file_handle and not current_log_file_handle.closed:
            current_log_file_handle.close()
            current_log_file_handle = None
        # --- PRE-ITERATION CLEANUP End ---

        # --- START OF CURRENT ITERATION ---
        if i < start_index:
            continue # Skip iterations before the checkpoint

        sample_type_name = sample_type[0] # e.g., "Good" or "Bad"
        sample_indices = sample_type[1] # List of indices
        evaluation_name = f"S={seed}-L={layerSize}-T={train_sample_count}-{sample_type_name}"
        current_log_filename = f"{evaluation_name}.log"

        try:
            # --- DYNAMIC LOGGING & REDIRECTION SETUP for this iteration ---
            # (Largely unchanged, uses updated current_log_filename)
            # 1. Open the new log file
            current_log_file_handle = open(current_log_filename, 'a', encoding='utf-8')
            # 2. Create and add the File Handler
            current_file_handler = logging.FileHandler(current_log_filename, mode='a', encoding='utf-8')
            current_file_handler.setFormatter(log_formatter)
            root_logger.addHandler(current_file_handler)
            # 3. Create Tee objects
            tee_stdout = Tee(original_stdout, current_log_file_handle)
            tee_stderr = Tee(original_stderr, current_log_file_handle)
            # 4. Redirect stdout and stderr
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            # --- DYNAMIC LOGGING & REDIRECTION SETUP End ---

            # --- Log Start of Iteration ---
            logging.info(f"\n--- Starting Iteration {i+1}/{len(param_space)} ---")
            logging.info(f"Logging to console and '{current_log_filename}'")
            logging.info(f"Params: seed={seed}, layerSize={layerSize}, train_samples={train_sample_count}, sample_type='{sample_type_name}'")
            print(f"--- Print output test for iteration {i+1} (goes to console and {current_log_filename}) ---")
            print(f"Using device type: {device}") # Print device info now

            # --- Configure iteration-specific settings ---
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logging.debug(f"Seeds set for iteration {i+1}: Python random={seed}, Torch={seed}")

            actual_train_samples = min(train_sample_count, len(trainSetMNIST_full))
            if actual_train_samples != train_sample_count:
                logging.warning(f"Requested {train_sample_count} train samples, but dataset only has {len(trainSetMNIST_full)}. Using {actual_train_samples}.")

            # Define model layers (Unchanged)
            normalLayers = [['Linear', layerSize, 'ReLU'], ['Linear', layerSize, 'ReLU']]
            neuronChoices = [((0, layerSize), True), ((0, layerSize), True)]
            hidden_sizes = [(nl[0], nl[1], nl[2]) for nl in normalLayers]
            hidden_sizes.append(('Linear', 10, 'ReLU')) # Output layer
            visualizeCustom = [((nc[0][0], nc[0][1]), nc[1]) for nc in neuronChoices]
            visualizeCustom.append(((0, 10), True)) # Visualize output layer neurons
            logging.debug(f"Model hidden_sizes defined for iteration {i+1}")

            # --- Run MNIST Operations ---
            logging.info("Initializing MNIST datasets for current iteration...")
            # Uses eval_samples, batch_size_training, batch_size_test, seed
            MNIST.initializeDatasets(actual_train_samples, test_samples, sample_indices, batch_size_training, batch_size_test, seed)

            logging.info("Training MNIST model...")
            # Uses loss_function, optimizer, learning_rate, epochs
            MNIST.trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs)

            logging.info("Initializing RENN hook...")
            start_time = timeit.default_timer()
            # Uses hidden_sizes, actual_train_samples
            MNIST.initializeHook(hidden_sizes, actual_train_samples)
            elapsed_time = timeit.default_timer() - start_time
            logging.info(f"RENN Hook Initialization Time: {elapsed_time:.2f} seconds")

            logging.info("Running RENN Visualization...")
            start_time = timeit.default_timer()
            # Uses closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, evaluation_name
            MNIST.visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, evaluation_name, True)
            elapsed_time = timeit.default_timer() - start_time
            logging.info(f"RENN Visualization Time: {elapsed_time:.2f} seconds")
            # --- Run MNIST Operations End ---

            logging.info(f"--- Finished Iteration {i+1}/{len(param_space)} ---")

            # --- Save Checkpoint (pointing to the *next* iteration) ---
            if i + 1 < len(param_space):
                next_iter_params = param_space[i+1]
                next_params = {
                    "seed": next_iter_params[0],
                    "layerSize": next_iter_params[1],
                    "train_sample_count": next_iter_params[2],
                    "sample_type_name": next_iter_params[3][0],
                }
                save_checkpoint(next_params)
                logging.debug(f"Checkpoint saved for next iteration: {next_params}")
            else:
                clear_checkpoint() # Last iteration completed successfully

        except Exception as e:
            # Log error thoroughly
            logging.error(f"Error during iteration {i+1} with params: seed={seed}, layerSize={layerSize}, train_samples={train_sample_count}, sample_type='{sample_type_name}'")
            logging.error(f"Error details: {e}", exc_info=True)
            # Print error details to stderr (which is Tee'd)
            print(f"\n--- CRITICAL ERROR ENCOUNTERED IN ITERATION {i+1}. Stopping script. ---", file=sys.stderr)
            print(f"Failed on iteration {i+1}. Params: seed={seed}, layerSize={layerSize}, train_samples={train_sample_count}, sample_type='{sample_type_name}'", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            print(f"Log file for this failed iteration: '{current_log_filename}'", file=sys.stderr)
            print(f"Checkpoint (if saved) points to the start of this failed iteration for restart.", file=sys.stderr)

            # Save checkpoint pointing to the *failed* iteration for easy restart
            current_params = {
                "seed": seed,
                "layerSize": layerSize,
                "train_sample_count": train_sample_count,
                "sample_type_name": sample_type_name
            }
            save_checkpoint(current_params)

            # Exit - atexit cleanup function will run
            sys.exit(1)

        # Ensure output buffers are flushed before the next iteration potentially closes files
        sys.stdout.flush()
        sys.stderr.flush()
        # --- END OF ITERATION LOOP ---

    logging.info("--- All iterations completed successfully! ---")

# --- Script Execution ---
if __name__ == "__main__":
    main_success = False
    try:
        run_training_loop()
        main_success = True # Mark success if loop completes without sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("--- KeyboardInterrupt detected. Exiting script gracefully. ---")
        print("\n--- KeyboardInterrupt detected. Exiting script gracefully. ---", file=original_stderr)
        # Cleanup is handled by atexit
    except SystemExit as e:
        if e.code != 0:
            logging.warning(f"Script exited early with code: {e.code}. See logs for details.")
            print(f"--- Script exited early with code: {e.code}. See logs for details. ---", file=original_stderr)
        else:
            logging.info("Script exited normally.") # e.g. sys.exit(0) called somewhere
        # Cleanup is handled by atexit
    except Exception as e:
        try:
            logging.exception("--- SCRIPT TERMINATED DUE TO UNHANDLED EXCEPTION OUTSIDE MAIN LOOP ---")
        except Exception as log_err:
            print(f"--- Logging failed during exception handling: {log_err} ---", file=original_stderr)
        print(f"--- SCRIPT TERMINATED DUE TO UNHANDLED EXCEPTION OUTSIDE MAIN LOOP: {e} ---", file=original_stderr)
        # Cleanup is handled by atexit
    finally:
        if main_success:
            logging.info("--- Main script block finished (Success). Preparing for cleanup. ---")
        else:
            print("--- Main script block finished (Terminated Early or Error). Preparing for cleanup. ---", file=original_stdout) # Use original stream
        # atexit cleanup runs automatically
        pass
# --- Script Execution End ---