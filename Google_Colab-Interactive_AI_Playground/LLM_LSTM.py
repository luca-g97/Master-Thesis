import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import os
import Customizable_RENN as RENN
from torch.utils.data import Dataset
from collections import defaultdict
import gc
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

#General
seq_len = 6
batch_size = 256
train_path, test_path, model_path = "./Datasets/WikiText2Train.txt", "./Datasets/WikiText2Test.txt", "./Wiki_Model.pt"
debug = True

#Model
embedding_dim = 100
padding_idx = 0
hidden_size = 128
dropout_p = 0.1

#Generation
sent_len = 100
topk = 5

#Preinitialize
device, DataLoader, nltk = "", "", ""
word_to_idx, idx_to_word = {}, {}
train_sources, cleaned_train_data, train_titles, train_sentences, train_source_structure = [], "", [], [], []
test_sources, cleaned_test_data, test_titles, test_sentences, test_source_structure = [], "", [], [], []
train_samples, test_samples, eval_samples, train_loader, test_loader, eval_loader = "", "", "", "", "", ""
model, criterion_class, chosen_optimizer, layers = "", "", "", ""
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, eval_source_structure = [], [], []

def initializePackages(devicePackage, DataLoaderPackage, nltkPackage):
    global device, DataLoader, nltk

    device, DataLoader, nltk = devicePackage, DataLoaderPackage, nltkPackage

def get_hidden_sizes(num_layers, trainSamples):
    global model

    sentences, words = split_data(cleaned_train_data, trainSamples)

    model = LSTM(len(words), embedding_dim, padding_idx, hidden_size, dropout_p, num_layers, device).to(device)
    model.to(device)

    hidden_sizes = []
    hidden_sizes.append(['Embedding', len(words), embedding_dim])
    hidden_sizes.append(['LSTM', batch_size*num_layers, hidden_size])
    hidden_sizes.append(['Linear', len(words), hidden_size])

    return hidden_sizes

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
    #print("Titles: ", titles[:10])

    sources = []
    for source in split_text:
        if re.match(heading_pattern, source):
            for pattern, replacement in patterns:
                source = pattern.sub(replacement, source)
            sources.append(source)

    return titles, sources

def create_source_structure(data, name, titles, sources):
    source_structure = []
    current_offset = 0

    # Write sources to the output file
    with open(f"{name}_sources.txt", 'w', encoding='utf-8') as f:
        for source_number, (title, source) in enumerate(zip(titles, sources)):
            f.write(f"{title}:\n")  # Write the title
            source_structure.append([])

            # Split data into sentences and sequences
            sentences, _ = split_data(source)
            _, sequences, current_offset = create_sequences(sentences, current_offset)

            for sentence_number, (sentence, sequence) in enumerate(zip(sentences, sequences)):
                # Append the sequence to the source structure
                source_structure[source_number].append(sequence)
                # Write to file
                f.write(f"{source_number}-{sentence_number}{sequence}: {sentence}\n")

            f.write('\n')  # Add a newline after each title for better readability

    return titles, sources, source_structure

def create_sequences(sentences, current_offset=0):
    sent_sequences = []

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

    return sent_sequences, sequences, current_offset

def split_data(data, num_sentences=-1):
    sentences = nltk.sent_tokenize(data) if num_sentences == -1 else nltk.sent_tokenize(data)[:num_sentences]
    words = sorted({word for sent in sentences for word in sent.split()})
    words.insert(0, "")  # Add an empty string for padding
    return sentences, words

# Function to map a flattened index back to its source and sentence index
def getSourceAndSentenceIndex(flat_index, structure="Training"):
    if structure == "Training":
        structure = train_source_structure
    elif "Evaluation" in structure:
        structure = eval_source_structure
    else:
        raise ValueError("Invalid structure name. Must be 'Training' or contain 'Evaluation'!")

    counter = 0
    for sourceNumber, sentences in enumerate(structure):
        for sentence, (start, end) in enumerate(sentences):
            #if start <= flat_index < end: #For sequences
            if counter == flat_index: #For sentences
                sequence = flat_index - start
                return sourceNumber, sentence

            counter += 1

    raise ValueError("Flat index is out of bounds!")

def get_flat_index(source_index, sentence_index, structure="Training"):
    if structure == "Training":
        structure = train_source_structure
    elif "Evaluation" in structure:
        structure = eval_source_structure
    else:
        raise ValueError("Invalid structure name. Must be 'Training' or contain 'Evaluation'!")

    # Calculate the total number of sentences in all sources before the target source
    #flat_index = sum(len(sentences) for sentences in structure[:source_index]) #For sequences
    flat_index = sum(len(sentences) for sentences in structure[:source_index]) #For sentences

    # Add the sentence index within the target source
    flat_index += sentence_index

    return flat_index

def createTrainAndTestSet():
    global train_sources, cleaned_train_data, train_titles, train_sentences, train_source_structure
    global test_sources, cleaned_test_data, test_titles, test_sentences, test_source_structure

    # Load and preprocess training data
    train_data = load_data(train_path)
    cleaned_train_data, train_sources, train_titles, train_source_structure = clean_data(train_data, "train")
    train_sentences, words = split_data(cleaned_train_data)

    # Load and preprocess test data
    test_data = load_data(test_path)
    cleaned_test_data, test_sources, test_titles, test_source_structure = clean_data(test_data[:], "test")
    test_sentences, words = split_data(cleaned_test_data)

    print(f"Created a train set with {len(train_sentences)} sentences")

    return train_sentences, test_sentences

def prepare_data_loader(sentences, words, batch_size, shuffle=True):
    class TextDataset(Dataset):
        """Custom Dataset for text sequences."""
        def __init__(self, predictors, class_labels):
            self.predictors = predictors
            self.class_labels = class_labels

        def __len__(self):
            return len(self.predictors)

        def __getitem__(self, idx):
            return self.predictors[idx], self.class_labels[idx]

    if(batch_size > 1):
        # Step 1: Generate sent_sequences
        sent_sequences, _, _ = create_sequences(sentences)
    else:
        sent_sequences = [sentence.split(' ') for sentence in sentences]

    # Step 2: Split into predictors and class_labels
    predictors = [seq[:-1] for seq in sent_sequences]
    class_labels = [seq[-1] for seq in sent_sequences]

    # Step 3: Pad predictors with empty strings
    max_seq_len = max(len(seq) for seq in predictors)
    pad_predictors = []
    for pred in predictors:
        pad_predictors.append([''] * (max_seq_len - len(pred)) + pred)

    # Step 4: Create word-to-index mapping
    word_ind = {word: idx for idx, word in enumerate(words)}

    # Step 5: Convert predictors and class_labels to indices
    pad_predictors = [[word_ind[word] for word in pred] for pred in pad_predictors]
    class_labels = [word_ind[label] for label in class_labels]

    # Step 6: Convert to tensors
    pad_predictors = torch.tensor(pad_predictors, dtype=torch.long)
    class_labels = torch.tensor(class_labels, dtype=torch.long)

    # Step 7: Create a dataset and DataLoader
    dataset = TextDataset(pad_predictors, class_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    #print("Number of input sequences: ", len(pad_predictors))

    return dataloader

def initializeDatasets(train_samplesParameter, test_samplesParameter, eval_samplesParameter, batch_size_training="", batch_size_test="", seed=""):
    global torch, word_to_idx, idx_to_word, train_samples, test_samples, eval_samples, train_loader, test_loader, eval_loader

    train_samples, test_samples, eval_samples = train_samplesParameter, test_samplesParameter, eval_samplesParameter

    if(seed != ""):
        print("Setting seed number to ", seed)
        torch.manual_seed(seed)
    else:
        print("Setting random seed")

    sentences, words = split_data(cleaned_train_data, train_samples)
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    train_loader = prepare_data_loader(sentences, words, batch_size=batch_size, shuffle=False)

    sentences, words = split_data(cleaned_test_data, test_samples)
    test_loader = prepare_data_loader(sentences, words, batch_size=batch_size, shuffle=False)

    sentences, words = split_data(cleaned_test_data, eval_samples)
    eval_loader = prepare_data_loader(sentences, words, batch_size=1, shuffle=False)

    print("Created all dataloaders")

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

def initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate):
    global criterion_class, chosen_optimizer, layers

    if(loss_function == "MSE"):
        criterion_class = nn.MSELoss()  # For regression
    elif(loss_function == "Cross-Entropy"):
        criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

    if(optimizer == "Adam"):
        chosen_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer == "SGD"):
        chosen_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion_class = nn.CrossEntropyLoss(ignore_index=padding_idx)
    chosen_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs):
    global model

    initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate)
    print("Model initialized, Starting training")
    model = train_model(train_loader, epochs)

    # Save the trained model
    torch.save(model, model_path)

    print("Training finished")

def evaluate():
    model.eval()
    total_loss = 0

    for x, y in train_loader:
        with torch.no_grad():
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

            loss = criterion_class(logits, y)

            # Accumulate loss
            total_loss += loss.item() * current_batch_size

    # Normalize loss by total number of predictors
    total_loss /= len(test_loader.dataset)
    perplexity = np.exp(total_loss)

    # Generate text after each epoch
    text, _ = generate("The")

    return total_loss, perplexity, text

def train_model(train_loader, epochs):
    global model

    for epoch in range(epochs):
        total_loss = 0
        model.train()  # Set model to training mode

        for x, y in train_loader:
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

            loss = criterion_class(logits, y)

            # Backpropagation
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Clip gradients
            chosen_optimizer.step()

            # Accumulate loss
            total_loss += loss.item() * current_batch_size

        # Normalize loss by total number of predictors
        total_loss /= len(train_loader.dataset)
        perplexity = np.exp(total_loss)

        test_loss, test_perplexity, text = evaluate()

        # Logging progress
        print(f"Epoch [{epoch + 1}/{epochs}] -> Train-Loss: {total_loss:.4f}, Train-Perplexity: {perplexity:.4f}, Test-Loss: {test_loss:.4f}, Test-Perplexity: {test_perplexity:.4f}")
        print(f"Text generated after epoch {epoch + 1}: {text}\n")

    return model

def initializeHook(hidden_sizes, train_samples):
    RENN.createDictionaries(hidden_sizes, len(hidden_sizes), train_samples, llmType=True)

    sentences, words = split_data(cleaned_train_data, train_samples)
    train_loader = prepare_data_loader(sentences, words, batch_size=1, shuffle=False)

    RENN.runHooks(train_loader, model, hidden_sizes, True, lstm=True)

def generate(init):
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

    generated = sentence[len(init)+1:]

    return sentence, generated

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, analyze=False):
    global train_samples, test_samples, eval_samples, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, eval_source_structure

    #Generate sentences and get their activation values
    generatedEvals = [generate(test_sentences[evalSample])[1] for evalSample in range(eval_samples)]
    generatedEvalSentences = [split_data(generatedEvalSentence, 2)[0][0] for generatedEvalSentence in generatedEvals]
    generatedEvalSentences = [generatedEvalSentence+"." if "." not in generatedEvalSentence else generatedEvalSentence for generatedEvalSentence in generatedEvalSentences]

    # Split the combined sentences into sentences and words
    eval_source_structure = [create_sequences(generatedEvalSentences)[1]]
    sentences, words = split_data(" ".join(generatedEvalSentences))
    generatedEvalLoader = prepare_data_loader(sentences, words, batch_size=1, shuffle=False)
    RENN.initializeEvaluationHook(hidden_sizes, generatedEvalLoader, len(generatedEvalLoader), model, os.path.join("Evaluation", "Generated"), True, 0, True)

    #RENN.initializeEvaluationHook(hidden_sizes, eval_loader, eval_samples, model, os.path.join("Evaluation", "Sample"), True, train_samples)
    #closestSourcesEvaluation, closestSourcesGeneratedEvaluation = RENN.identifyClosestLLMSources(eval_samples, 0, closestSources)

    _, closestSourcesGeneratedEvaluation = RENN.identifyClosestLLMSources(len(generatedEvalLoader), 0, closestSources, True)

    for sampleNumber in range(eval_samples):
        #mostUsedEvalSources = RENN.getMostUsedSources(closestSourcesEvaluation, closestSources, sampleNumber, "Mean")
        mostUsedGeneratedEvalSources = RENN.getMostUsedSources(closestSourcesGeneratedEvaluation, closestSources, sampleNumber, "Mean")

        sample = test_sentences[sampleNumber]
        print("Evaluation Sample ", sampleNumber, ": ", sample)
        print("Follow up: ", generatedEvals[sampleNumber])
        print(f"Generated Source Sentence: {generatedEvalSentences[sampleNumber]}")
        print(f"Closest Sources for GeneratedEvaluation-Sample {sampleNumber} in format [SourceNumber, Occurrences, Source]:")
        for source, count in mostUsedGeneratedEvalSources[:closestSources]:
            tempSource = source.split(":")
            sourceNumber, sentenceNumber = int(tempSource[0]), int(tempSource[1])
            index = get_flat_index(sourceNumber, sentenceNumber)
            trainSentence = train_sentences[index]
            print(f"Source: {source}, Count: {count}, Sentence: {trainSentence}")
        #print("Whole List: ", [(source, count, train_sentences[get_flat_index(int(source.split(":")[0]), int(source.split(":")[1]))]) for source, count in mostUsedGeneratedEvalSources], "\n") #For sequences
        print("Whole List: ", [(source, count, train_sentences[get_flat_index(int(source.split(":")[0]), int(source.split(":")[1]))]) for source, count in mostUsedGeneratedEvalSources], "\n")

        if(analyze):
            _, closestSourcesGeneratedEvaluationForLSTMLayer = RENN.identifyClosestLLMSources(len(generatedEvalLoader), 0, closestSources, True, layersToCheck=[1])
            mostUsedGeneratedEvalSourcesForLSTMLayer = RENN.getMostUsedSources(closestSourcesGeneratedEvaluationForLSTMLayer, closestSources, sampleNumber, "Mean")
            print("Whole List: ", [(source, count) for source, count in mostUsedGeneratedEvalSources], "\n")
            print("Whole List: ", [(source, count) for source, count in mostUsedGeneratedEvalSourcesForLSTMLayer], "\n")