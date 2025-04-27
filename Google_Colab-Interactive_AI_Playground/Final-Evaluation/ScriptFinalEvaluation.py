import os
import logging
import timeit
import sys
# <<< ADDED/MODIFIED START >>>
import atexit # To manage shutdown order if needed
# <<< ADDED/MODIFIED END >>>


# --- Tee Class for duplicating streams ---
# <<< ADDED/MODIFIED START >>> Added close() method and closed attribute
class Tee:
    """Helper class to duplicate stream output to multiple destinations (e.g., console and file)."""
    def __init__(self, *files):
        self.files = files
        self.closed = False # Flag to indicate if closed

    def write(self, obj):
        if self.closed:
            # Optionally raise an error or just ignore write attempts on closed Tee
            # print("Warning: Write attempt on closed Tee object.", file=sys.__stderr__)
            return
        for f in self.files:
            try:
                f.write(obj)
                # No flush here by default, rely on explicit flush or program exit flush
            except Exception as e:
                # Handle potential errors during write (e.g., file closed unexpectedly)
                print(f"Error writing to stream {f}: {e}", file=sys.__stderr__)


    def flush(self):
        if self.closed:
            # print("Warning: Flush attempt on closed Tee object.", file=sys.__stderr__)
            return
        for f in self.files:
            try:
                f.flush()
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
            # Check if it's the file handle we opened for Tee
            # This is a bit fragile; assumes log_file_handle is globally accessible for check
            # A better way might be to pass ownership info during init
            global log_file_handle
            if f is log_file_handle:
                try:
                    if not f.closed:
                        f.close()
                except Exception as e:
                    print(f"Error closing Tee's file handle: {e}", file=sys.__stderr__)
        self.closed = True

    def isatty(self):
        # Help libraries like tqdm determine if output is interactive
        # Assume TTY if original stdout is TTY
        try:
            # Check the first file object, assuming it's the original stream
            return self.files[0].isatty()
        except:
            return False # Default to false if error or no files

# <<< ADDED/MODIFIED END >>>

# --- Global variables for stream management ---
# <<< ADDED/MODIFIED START >>> Define these globally for access in finally/atexit
original_stdout = sys.stdout
original_stderr = sys.stderr
log_file_handle = None
tee_stdout = None
tee_stderr = None
# <<< ADDED/MODIFIED END >>>

# --- Setup Logging and Redirection ---
LOG_FILENAME = "evaluation_log.txt"

try:
    # Configure logging FIRST, potentially before redirection helps avoid issues
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicates or conflicts
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler (Append Mode) - added before redirection
    # Use 'utf-8' encoding for broader compatibility
    log_file_handle = open(LOG_FILENAME, 'a', encoding='utf-8')
    file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler (using original stdout) - added before redirection
    # Important: Pass original_stdout explicitly
    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Now, redirect stdout and stderr using Tee
    tee_stdout = Tee(original_stdout, log_file_handle)
    tee_stderr = Tee(original_stderr, log_file_handle)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    # --- Initial Log Messages ---
    logging.info("--- Script Started ---")
    logging.info(f"Logging INFO and above to console and '{LOG_FILENAME}'")
    logging.info(f"Redirecting stdout and stderr to console and '{LOG_FILENAME}'")
    print("--- Print output test after redirection ---") # Test print redirection

except Exception as e:
    # If setup fails, log to original stderr and exit
    print(f"FATAL ERROR during logging/redirection setup: {e}", file=sys.__stderr__)
    if log_file_handle and not log_file_handle.closed:
        log_file_handle.close()
    # Restore original streams just in case partially redirected
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    sys.exit("Exiting due to logging setup failure.")

# --- Cleanup Function ---
# <<< ADDED/MODIFIED START >>> Define cleanup explicitly
def cleanup_streams_and_logging():
    """Restores streams, closes Tee handle, and shuts down logging."""
    global original_stdout, original_stderr, log_file_handle, tee_stdout, tee_stderr

    # Check if streams were redirected before trying to restore
    streams_restored = False
    if sys.stdout is tee_stdout:
        sys.stdout = original_stdout
        streams_restored = True
    if sys.stderr is tee_stderr:
        sys.stderr = original_stderr
        streams_restored = True

    # Close the Tee object (which closes the log_file_handle it uses)
    # This prevents "I/O operation on closed file" if logging tries flushing later
    if tee_stdout:
        tee_stdout.close() # This sets the internal 'closed' flag
    if tee_stderr:
        tee_stderr.close()

    # Close the file handle opened explicitly for the FileHandler if different
    # Note: In this setup, FileHandler and Tee use the same filename,
    # but potentially different handles if not careful. Here, we ensured
    # log_file_handle is used by Tee, and FileHandler opens its own.
    # Let logging shutdown handle closing the FileHandler's stream.

    # Shutdown logging system AFTER restoring streams
    logging.shutdown()

    # Explicitly close the handle used by Tee if it wasn't closed by Tee.close()
    # (Tee.close() should handle this, but as a safeguard)
    if log_file_handle and not log_file_handle.closed:
        try:
            log_file_handle.close()
        except Exception as e:
            print(f"Error closing log_file_handle during final cleanup: {e}", file=sys.__stderr__)


    # Final message to original console
    if streams_restored:
        print(f"--- Script finished. Streams restored. Log file '{LOG_FILENAME}' contains full output. ---", file=original_stdout)

# Register cleanup function to be called on normal exit and specific signals
atexit.register(cleanup_streams_and_logging)
# <<< ADDED/MODIFIED END >>>


# Set up dependencies
# import sys # Already imported
from subprocess import run
# Check if running in a non-standard environment and adjust path if needed
if '/tf/.local/lib/python3.11/site-packages' not in sys.path:
    if os.path.exists('/tf/.local/lib/python3.11/site-packages'):
        sys.path.append('/tf/.local/lib/python3.11/site-packages')
        logging.info("Added /tf/.local/lib/python3.11/site-packages to sys.path")
    else:
        logging.warning("/tf/.local/lib/python3.11/site-packages not found, imports might fail if not installed globally.")
# run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
# os.system("pip install -q lorem==0.1.1 tiktoken==0.8.0 stanza ipywidgets==7.7.1 plotly pyarrow zstandard optuna python-Levenshtein cma")

# Import libraries
try:
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
    import json
    import itertools
    import torch
    from torch.utils.data import DataLoader
    from torch import nn
    from tqdm import tqdm
    from keras.datasets import mnist
    from keras.utils import to_categorical
    import random
    import plotly.io as pio
    import io
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard as zstd
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}", exc_info=True)
    print(f"Failed to import required libraries: {e}") # Also print to redirected streams
    sys.exit(1) # atexit cleanup will run


# Setup logging # Handled above
pio.renderers.default = 'colab' # Adjust if not using Colab

# --- Local Module Imports ---
try:
    import MNISTFinalEvaluation as MNIST
    import RENNFinalEvaluation as RENN
except ImportError as e:
    logging.error(f"Error importing local modules: {e}", exc_info=True)
    print(f"Error importing local modules: {e}")
    print("Please ensure MNISTFinalEvaluation.py and RENNFinalEvaluation.py are accessible.")
    sys.exit(1) # atexit cleanup will run


# --- Initial Setup ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Using device type: {device}")
print(f"Using device type: {device}")

# --- Configuration ---
# (Configuration remains the same)
eval_samples = 100
INTEGER_LIMIT = 4294967295
seeds = [random.randint(0, INTEGER_LIMIT) for _ in range(10)]
layerSizes = [128, 512, 2048]
train_samples_options = [100, 10000, 60000]
test_samples = 2000
learning_rate = 0.0004
epochs = 100
useBitLinear = False
batch_size_training = 64
batch_size_test = 64
loss_function = "MSE"
optimizer = "Adam"
closestSources = 25
showClosestMostUsedSources = 3
visualizationChoice = "Weighted"
CHECKPOINT_FILE = "evaluation_checkpoint.json"

# --- Helper Functions ---
# (Helper functions remain the same: load_checkpoint, save_checkpoint, clear_checkpoint)
def load_checkpoint():
    """Loads the last saved checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
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
    MNIST.initializePackages(mnist, to_categorical, nn, DataLoader, pd, optuna, device)
    trainSetMNIST_full, testSetMNIST = MNIST.createTrainAndTestSet() # Load full sets once
    logging.info("MNIST dataset loaded.")

    # Initialize RENN packages once
    RENN.initializePackages(device, io, pd, pa, pq, zstd, levenshtein, cma, MNIST, seeds[0], useBitLinear)

    # Define the parameter space using itertools.product
    param_space = list(itertools.product(seeds, layerSizes, train_samples_options))

    # Load checkpoint and determine starting point
    checkpoint = load_checkpoint()
    start_index = 0
    if checkpoint:
        try:
            start_index = param_space.index(
                (checkpoint['seed'], checkpoint['layerSize'], checkpoint['train_sample_count'])
            )
            logging.info(f"Found checkpoint. Resuming from iteration {start_index + 1}/{len(param_space)}.")
        except ValueError:
            logging.warning("Checkpoint parameters not found in current parameter space. Starting from scratch.")
            start_index = 0

    # Iterate through the parameter space starting from the checkpoint
    # <<< MODIFIED: Wrap tqdm with flush calls to ensure output with Tee >>>
    iterator = tqdm(param_space, desc="Overall Progress", initial=start_index)
    for i, (seed, layerSize, train_sample_count) in enumerate(iterator):
        # Ensure tqdm output gets flushed through Tee
        sys.stdout.flush()
        sys.stderr.flush()

        if i < start_index:
            continue

        logging.info(f"\n--- Starting Iteration {i+1}/{len(param_space)} ---")
        logging.info(f"Params: seed={seed}, layerSize={layerSize}, train_samples={train_sample_count}")
        print(f"\n--- Starting Iteration {i+1}/{len(param_space)} ---")
        print(f"Params: seed={seed}, layerSize={layerSize}, train_samples={train_sample_count}")

        # --- Configure iteration-specific settings ---
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.debug(f"Seeds set for iteration {i+1}: Python random={seed}, Torch={seed}")

        actual_train_samples = min(train_sample_count, len(trainSetMNIST_full))
        if actual_train_samples != train_sample_count:
            logging.warning(f"Requested {train_sample_count} train samples, but dataset only has {len(trainSetMNIST_full)}. Using {actual_train_samples}.")

        # Define model layers
        normalLayers = [['Linear', layerSize, 'ReLU'], ['Linear', layerSize, 'ReLU']]
        neuronChoices = [((0, layerSize), True), ((0, layerSize), True)]
        hidden_sizes = [(nl[0], nl[1], nl[2]) for nl in normalLayers]
        hidden_sizes.append(('Linear', 10, 'ReLU'))
        visualizeCustom = [((nc[0][0], nc[0][1]), nc[1]) for nc in neuronChoices]
        visualizeCustom.append(((0, 10), True))
        logging.debug(f"Model hidden_sizes defined for iteration {i+1}")

        # --- Run MNIST Operations ---
        try:
            logging.info("Initializing MNIST datasets for current iteration...")
            MNIST.initializeDatasets(actual_train_samples, test_samples, eval_samples, batch_size_training, batch_size_test, seed)

            logging.info("Training MNIST model...")
            MNIST.trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs)

            logging.info("Initializing RENN hook...")
            start_time = timeit.default_timer()
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
            if i + 1 < len(param_space):
                next_params = {
                    "seed": param_space[i+1][0],
                    "layerSize": param_space[i+1][1],
                    "train_sample_count": param_space[i+1][2],
                }
                save_checkpoint(next_params)
                logging.debug(f"Checkpoint saved for next iteration: {next_params}")
            else:
                clear_checkpoint() # Last iteration completed successfully

        except Exception as e:
            # Log error thoroughly
            logging.error(f"Error during iteration {i+1} with params: {seed, layerSize, train_sample_count}")
            logging.error(f"Error details: {e}", exc_info=True)
            # Print error details to stderr (which is Tee'd)
            print(f"\n--- CRITICAL ERROR ENCOUNTERED IN ITERATION {i+1}. Stopping script. ---", file=sys.stderr)
            print(f"Failed on iteration {i+1}. Params: {seed, layerSize, train_sample_count}", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            print(f"Checkpoint (if saved) points to the start of this failed iteration for restart.", file=sys.stderr)

            # Save checkpoint pointing to the failed iteration
            current_params = {"seed": seed, "layerSize": layerSize, "train_sample_count": train_sample_count}
            save_checkpoint(current_params)

            # Exit - atexit cleanup function will run
            sys.exit(1)

        # <<< ADDED: Explicit flush after iteration if tqdm doesn't handle it well with Tee >>>
        sys.stdout.flush()
        sys.stderr.flush()


    logging.info("--- All iterations completed successfully! ---")
    print("--- All iterations completed successfully! ---")

# --- Script Execution ---
if __name__ == "__main__":
    main_success = False
    try:
        run_training_loop()
        main_success = True # Mark success if loop completes
    except KeyboardInterrupt:
        logging.warning("--- KeyboardInterrupt detected. Exiting script gracefully. ---")
        print("\n--- KeyboardInterrupt detected. Exiting script gracefully. ---", file=sys.stderr)
        # Cleanup is handled by atexit
    except SystemExit as e:
        # Raised by sys.exit() within the loop on error
        logging.warning(f"Script exited early with code: {e.code}. See logs for details.")
        # Cleanup is handled by atexit
    except Exception as e:
        # Catch unexpected errors outside the main loop's try/except
        logging.exception("--- SCRIPT TERMINATED DUE TO UNHANDLED EXCEPTION OUTSIDE MAIN LOOP ---")
        print(f"--- SCRIPT TERMINATED DUE TO UNHANDLED EXCEPTION OUTSIDE MAIN LOOP: {e} ---", file=sys.stderr)
        # Cleanup is handled by atexit
    finally:
        # <<< MODIFIED: The actual cleanup logic is now in the atexit handler >>>
        # Optional: Log that the main block is finished before atexit runs
        if main_success:
            logging.info("--- Main script block finished (Success). Preparing for cleanup. ---")
        else:
            logging.warning("--- Main script block finished (Terminated Early or Error). Preparing for cleanup. ---")
        # The 'atexit' registered function 'cleanup_streams_and_logging' will execute now or on exit.