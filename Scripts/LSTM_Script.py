#IMPORTING LIBRARIES AND MODULES - Torch 2.5.1+cu121 needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import os
import sys
from subprocess import run
sys.path.append('/tf/.local/lib/python3.11/site-packages')
run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=True)
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize,sent_tokenize
from collections import defaultdict
import gc
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

#General
seq_len = 6
batch_size = 256
train_path, test_path, model_path = "./train.txt", "./test.txt", "./Wiki_Model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug = True

#Model
embedding_dim = 100
padding_idx = 0
hidden_size = 128
dropout_p = 0.1
num_layers = 2

#Training
num_train_sentences = 25000
lr = 0.001
num_epochs = 10

#Test
num_test_sentences = 6000

#Generation
init = "The"
sent_len = 100
topk = 1

#Preinitialize
word_to_idx, idx_to_word = {}, {}
train_sources, train_titles, train_source_structure = [], [], []
test_sources, test_titles, test_source_structure = [], [], []

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return ""
    except IOError as e:
        print(f"Error reading the file {filepath}: {e}")
        return ""

# Patterns for data cleaning
patterns = [
    (re.compile(r'\('), ''),  # Remove open brackets
    (re.compile(r'\)'), ''),  # Remove close brackets
    (re.compile(r"=.*="), ''),  # Remove headings
    (re.compile(r"<unk>"), ''),  # Remove unknown tokens
    (re.compile(r"-"), ' '),  # Exchange hyphens for spaces
    (re.compile(r"[^\w.' ]"), ''),  # Remove non-alphanumeric, except specific symbols
]

def clean_data(data, name):
    titles, sources = create_sources(data)

    for pattern, replacement in patterns:
        data = pattern.sub(replacement, data)

    titles, sources, source_structure = create_source_structure(data, name, titles, sources)

    return data, titles, sources, source_structure

def create_sources(data):
    # Define regex for valid headings (not subheadings)
    heading_pattern = r'(?<!\= )= [^=]+ =(?!= )'  # Matches = Title = but not = = Subheading = =

    # Find all valid headings
    titles = [source.replace('= ', '').replace(' =', '') for source in re.findall(heading_pattern, data)]

    # Split the text, ensuring it splits before the heading
    split_text = re.split(f'(?={heading_pattern})', data)

    # Print the adjusted titles
    print("Titles: ", titles[:10])

    sources = []
    for source in split_text:
        if re.match(heading_pattern, source):
            for pattern, replacement in patterns:
                source = pattern.sub(replacement, source)
            sources.append(source)

    return titles, sources

def create_source_structure(data, name, titles, sources):
    source_structure = []

    # Write sources to the output file
    with open(f"{name}_sources.txt", 'w', encoding='utf-8') as f:
        for source_number, (title, source) in enumerate(zip(titles, sources)):
            f.write(f"{title}:\n")  # Write the title
            source_structure.append([])

            # Split data into sentences and sequences
            sentences, _ = split_data(source)
            _, sequences = create_sequences(sentences)

            for sentence_number, (sentence, sequence) in enumerate(zip(sentences, sequences)):
                # Append the sequence to the source structure
                source_structure[source_number].append(sequence)
                # Write to file
                f.write(f"{source_number}-{sentence_number}{sequence}: {sentence}\n")

            f.write('\n')  # Add a newline after each title for better readability

    return titles, sources, source_structure

def create_sequences(sentences):
    sent_sequences = []

    current_offset = 0
    sequences = []
    for sentence in sentences:
        sequence_offset = 0
        words_in_sent = sentence.split(' ')
        for j in range(1, len(words_in_sent)):
            if(j <= seq_len) or j > seq_len and j < len(words_in_sent) or j > len(words_in_sent) - seq_len:
                sequence_offset += 1
            if j <= seq_len:
                sent_sequences.append(words_in_sent[:j])
            elif j > seq_len and j < len(words_in_sent):
                sent_sequences.append(words_in_sent[j - seq_len:j])
            elif j > len(words_in_sent) - seq_len:
                sent_sequences.append(words_in_sent[j - seq_len:])

        sequences.append([current_offset, current_offset+sequence_offset])
        current_offset += sequence_offset

    return sent_sequences, sequences

def split_data(data, num_sentences=-1):
    sentences = sent_tokenize(data) if num_sentences == -1 else sent_tokenize(data)[:num_sentences]
    words = sorted({word for sent in sentences for word in sent.split()})
    words.insert(0, "")  # Add an empty string for padding
    return sentences, words

def convert_data(sentences, words):
    # Step 1: Generate sent_sequences
    sent_sequences, _ = create_sequences(sentences)

    # Step 2: Split into predictors and class_labels
    predictors = [seq[:-1] for seq in sent_sequences]
    class_labels = [seq[-1] for seq in sent_sequences]

    # Step 3: Manually pad predictors with empty strings
    pad_predictors = []
    for pred in predictors:
        emptypad = [''] * (seq_len - len(pred) - 1)
        emptypad.extend(pred)
        pad_predictors.append(emptypad)

    # Step 4: Create word-to-index and index-to-word mappings
    word_ind = {word: idx for idx, word in enumerate(words)}
    ind_word = {idx: word for idx, word in enumerate(words)}

    # Step 5: Convert predictors and class_labels to indices
    for i in range(len(pad_predictors)):
        pad_predictors[i] = [word_ind[pad_predictors[i][j]] for j in range(len(pad_predictors[i]))]
        class_labels[i] = word_ind[class_labels[i]]

    # Step 6: Convert to tensors
    pad_predictors = torch.tensor(pad_predictors, dtype=torch.long)
    class_labels = torch.tensor(class_labels, dtype=torch.long)

    return pad_predictors, class_labels

class LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, hidden_size, dropout_p, num_layers, device):
        super(LSTM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.device = device

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

        # LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_p)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_embeddings)

    def init_hidden(self, batch_size):
        """Initializes hidden and cell states to zeros."""
        state_h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        state_c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return state_h, state_c

    def forward(self, input_sequence, state_h, state_c):
        """Forward pass through the model."""
        embed_input = self.embedding(input_sequence)
        output, (state_h, state_c) = self.lstm(embed_input, (state_h, state_c))
        logits = self.fc(output[:, -1, :])  # Take only the last time step
        return logits, (state_h, state_c)

    def get_layer_info(self):
        """Retrieve detailed information about each layer."""
        layer_info = {
            'embedding': {
                'weights_shape': self.embedding.weight.shape,
                'weights': self.embedding.weight.cpu().detach().numpy(),
                'padding_idx': self.padding_idx
            },
            'lstm': {
                'input_weights': self.lstm.weight_ih_l0.cpu().detach().numpy(),
                'hidden_weights': self.lstm.weight_hh_l0.cpu().detach().numpy(),
                'biases': self.lstm.bias_ih_l0.cpu().detach().numpy(),
                'hidden_biases': self.lstm.bias_hh_l0.cpu().detach().numpy(),
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout_p
            },
            'fully_connected': {
                'weights_shape': self.fc.weight.shape,
                'weights': self.fc.weight.cpu().detach().numpy(),
                'bias_shape': self.fc.bias.shape,
                'bias': self.fc.bias.cpu().detach().numpy()
            }
        }
        return layer_info

    def topk_sampling(self, logits, topk):
        """Applies softmax and samples an index using top-k sampling."""
        logits_softmax = F.softmax(logits, dim=1)
        values, indices = torch.topk(logits_softmax[0], k=topk)
        sampling = torch.multinomial(values, 1)
        return indices[sampling].item()

def get_batch(pad_predictors, class_labels, batch_size):
    for i in range(0, len(pad_predictors), batch_size):
        if i+batch_size<len(pad_predictors):
            yield pad_predictors[i:i+batch_size], class_labels[i:i+batch_size]

def train_model(pad_predictors, class_labels, n_vocab):
    model = LSTM(n_vocab, embedding_dim, padding_idx, hidden_size, dropout_p, num_layers, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()  # Set model to training mode

        for x, y in get_batch(pad_predictors, class_labels, batch_size):
            # Reinitialize hidden states for current batch size
            current_batch_size = x.size(0)
            state_h, state_c = model.init_hidden(current_batch_size)

            # Move data and states to the appropriate device
            x, y = x.to(device), y.to(device)
            state_h, state_c = state_h.to(device), state_c.to(device)

            # Detach hidden states to prevent backpropagation through past batches
            state_h = state_h.detach()
            state_c = state_c.detach()

            # Forward pass and loss computation
            logits, (state_h, state_c) = model(x, state_h, state_c)
            loss = criterion(logits, y)

            # Backpropagation
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Clip gradients
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item() * current_batch_size

        # Normalize loss by total number of predictors
        total_loss /= len(pad_predictors)
        perplexity = np.exp(total_loss)

        # Logging progress
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss:.4f}, Perplexity: {perplexity:.4f}")

        # Generate text after each epoch
        gen_text = generate(model)
        print(f"Text generated after epoch {epoch + 1}:\n{gen_text}\n")

    return model, total_loss

def evaluate(model_path, test_file):
    global test_sources, test_titles, test_source_structure

    # Load and clean data
    test_data = load_data(test_file)
    data, test_sources, test_titles, test_source_structure = clean_data(test_data[:], "test")
    sentences, words = split_data(data, num_test_sentences)

    pad_predictors, class_labels = convert_data(sentences, words)
    print("Number of input sequences:", len(pad_predictors))

    # Load the model
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()

    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    # Processing in batches
    for i in range(0, len(pad_predictors), batch_size):
        x, y = pad_predictors[i:i + batch_size], class_labels[i:i + batch_size]
        x, y = x.to(device), y.to(device)

        current_batch_size = x.size(0)
        state_h, state_c = model.init_hidden(batch_size=current_batch_size)
        state_h, state_c = state_h.to(device), state_c.to(device)

        with torch.no_grad():
            logits, (state_h, state_c) = model(x, state_h, state_c)
            loss = criterion(logits, y)
            total_loss += loss.item() * current_batch_size

    total_loss /= len(pad_predictors)
    return total_loss, np.exp(total_loss)

def generate(model):
    model.eval()
    sentence = init
    input_indices = [word_to_idx[word] for word in init.split() if word in word_to_idx]

    with torch.no_grad():
        for _ in range(sent_len):
            # Pad or trim the input sequence to `seq_len`
            if len(input_indices) < seq_len:
                input_tensor = [0] * (seq_len - len(input_indices)) + input_indices
            else:
                input_tensor = input_indices[-seq_len:]

            input_tensor = torch.tensor([input_tensor]).to(device)
            state_h, state_c = model.init_hidden(1)

            logits, (state_h, state_c) = model(input_tensor, state_h, state_c)

            # Sample a word using top-k sampling
            word_idx = model.topk_sampling(logits, topk)
            word = idx_to_word.get(word_idx, "<UNK>")

            if word and word != sentence.split()[-1]:
                sentence += f" {word}"

            input_indices.append(word_idx)

    return sentence

def main():
    global word_to_idx, idx_to_word

    # Load and preprocess training data
    train_data = load_data(train_path)
    train_data, train_sources, train_titles, train_source_structure = clean_data(train_data, "train")
    sentences, words = split_data(train_data, num_train_sentences)

    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    pad_predictors, class_labels = convert_data(sentences, words)

    print("Number of input sequences: ", len(pad_predictors))

    # Train the model
    model, loss = train_model(pad_predictors, class_labels, n_vocab=len(words))

    # Generate a sentence
    generated_sentence = generate(model)

    print(f"Generated Sentence: {generated_sentence}")

    # Save the trained model
    torch.save(model, model_path)

    return loss, model

import timeit

if __name__ == "__main__":
    # Train the model
    start_time = timeit.default_timer()
    loss, model = main()
    train_time = timeit.default_timer() - start_time
    print(f"Training Time: {train_time:.2f} seconds")

    print(f"Train Loss: {loss:.4f}, Perplexity: {np.exp(loss):.4f}")


    # Evaluate the model on test data
    start_time = timeit.default_timer()
    test_loss, test_perplexity = evaluate(model_path, test_path)
    eval_time = timeit.default_timer() - start_time
    print(f"Evaluation Time: {eval_time:.2f} seconds")

    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}")

    # Write sources to the output file
    with open(f"results.txt", 'w', encoding='utf-8') as f:
        f.write(f"Generated Sentence: {generate(model)}\n")
        f.write(f"Train Loss: {loss:.4f}, Train Perplexity: {np.exp(loss):.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}\n")

def extract_layer_info(model):
    layer_info = []

    # Function to gather layer details
    def gather_layer_details(module):
        details = {}

        # Get layer type
        details['name'] = f"{module.__class__.__name__}"

        # Get layer parameters and shapes
        if hasattr(module, 'weight') and module.weight is not None:
            details['weights_shape'] = module.weight.shape

        if hasattr(module, 'bias') and module.bias is not None and isinstance(module.bias, torch.Tensor):
            details['bias_shape'] = module.bias.shape

        if isinstance(module, nn.LSTM):
            # Extract LSTM specific details
            #details['input_weights_shape'] = module.weight_ih_l0.shape
            details['weights_shape'] = module.weight_hh_l0.shape
            #details['input_bias_shape'] = module.bias_ih_l0.shape
            details['bias_shape'] = module.bias_hh_l0.shape

            # Get output size after forward pass for LSTM
            def hook(module, input, output):
                if output is not None:
                    output_size = output.size()
                    details['output_shape'] = output_size
                    details['output'] = output.cpu().detach().numpy()  # Move to CPU before conversion

            module.register_forward_hook(hook)

        if isinstance(module, nn.Embedding):
            # Extract Embedding specific details
            details['padding_idx'] = module.padding_idx

        return details

    # Traverse all layers and collect details, excluding the first layer
    first_layer_excluded = False
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Module) and not isinstance(layer, nn.Sequential):
            if not first_layer_excluded:
                first_layer_excluded = True
                continue  # Skip the first layer

            layer_info.append(gather_layer_details(layer))

    return layer_info

model = torch.load(model_path, map_location=device, weights_only=False).to(device)
layer_info = extract_layer_info(model)

# Display detailed information for each layer
for idx, layer_details in enumerate(layer_info):
    print(f"Layer {idx + 1}:")
    for key, value in layer_details.items():
        print(f"  {key}: {value}")
    print()