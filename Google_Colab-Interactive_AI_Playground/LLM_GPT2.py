import torch
from pip._internal.index import sources
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
import re
import numpy as np
import requests
import urllib.request
import Customizable_RENN as RENN
import pandas as pd

random, lorem, device, tiktoken, DataLoader, nlp, GPT2Tokenizer, nltk = "", "", "", "", "", "", "", ""
train_samples, test_samples, eval_samples = "", "", ""
GPT_CONFIG_124M, settings = "", ""
train_loader, val_loader, eval_loader, tokenizer, trainSentences, trainSentencesStructure, testSentences, testSentencesStructure = "", "", "", "", "", "", "", ""
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, layersToCheck = [], [], []

def initializePackages(randomPackage, loremPackage, devicePackage, tiktokenPackage, DataLoaderPackage, nlpPackage, GPT2TokenizerPackage, nltkPackage):
    global random, lorem, device, tiktoken, DataLoader, nlp, GPT2Tokenizer, nltk

    random, lorem, device, tiktoken, DataLoader, nlp, GPT2Tokenizer, nltk = randomPackage, loremPackage, devicePackage, tiktokenPackage, DataLoaderPackage, nlpPackage, GPT2TokenizerPackage, nltkPackage

def createTrainAndTestStructure(sentencesStructure, offset):
    train_sentences, test_sentences = [], []
    current_count = 0

    for sublist in sentencesStructure:
        if current_count + len(sublist) <= offset:
            train_sentences.append(sublist)
            current_count += len(sublist)
        else:
            split_idx = offset - current_count
            if split_idx > 0:
                train_sentences.append(sublist[:split_idx])
            test_sentences.append(sublist[split_idx:])
            test_sentences.extend(sentencesStructure[sentencesStructure.index(sublist) + 1:])
            break

    return train_sentences, test_sentences

def createTrainSet():
    global trainSentences, trainSentencesStructure, testSentences, testSentencesStructure

    file_path = "./Datasets/TheVerdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Remove any newline characters and extra spaces
    text_data = text_data.replace("\n", " ").strip()

    # Process the text with Stanza to extract sentences
    doc = nlp(text_data)
    sentencesStructure = [[sentence.text for sentence in doc.sentences]]

    # Flatten the list of sentences correctly
    sentences = [sentence for sublist in sentencesStructure for sentence in sublist]

    offset = int(len(sentences)*0.8)
    trainSentencesStructure, testSentencesStructure = createTrainAndTestStructure(sentencesStructure, offset)
    print(f"Created a train set with {len(sentences)} sentences")

    trainSentences, testSentences = sentences[:offset], sentences[offset:]

    return trainSentences, testSentences

# Wikipedia API endpoint for querying
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

def fetch_article_titles(category):
    """Fetch a list of article titles from a Wikipedia category."""
    titles = []
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'categorymembers',
        'cmtitle': f'Category:{category}',
        'cmlimit': 'max'  # Get the maximum number of articles
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params).json()
    for page in response.get('query', {}).get('categorymembers', []):
        titles.append(page['title'])
    return titles

def fetch_and_parse_content(title):
    """Fetch the content of a Wikipedia article and parse it."""
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params).json()
    pages = response.get('query', {}).get('pages', {})
    page = next(iter(pages.values()))
    content = page.get('extract', '')

    return content

def clean_wikipedia_content(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        # Remove references, citations, and template placeholders
        sentence = re.sub(r'\[\[File:.*?\]\]', '', sentence)  # Remove file/image links (e.g., [[File:Image.jpg]])
        sentence = re.sub(r'\[\[.*?\|.*?\]\]', '', sentence)  # Remove internal links with aliases (e.g., [[Link|Alias]])
        sentence = re.sub(r'\[\[.*?\]\]', '', sentence)      # Remove internal links (e.g., [[Link]])

        sentence = re.sub(r'\{.*?\}', '', sentence)      # Remove templates (e.g., {{Citation needed}} or any template)
        sentence = re.sub(r'==+.*?==+', '', sentence)    # Remove headings (e.g., == Heading ==)
        sentence = re.sub(r'===+.*?===+', '', sentence)  # Remove subheadings (e.g., === Subheading ===)
        sentence = re.sub(r'<!--.*?-->', '', sentence, flags=re.DOTALL)  # Remove comments
        sentence = re.sub(r'\<.*?\>', '', sentence)      # Remove any HTML tags (just in case)

        # Remove irrelevant sections like "See also", "External Links", "References", "Further reading"
        sentence = re.sub(r'\n(See also|External links|References|Further reading)\n.*?(\n|$)', '', sentence, flags=re.DOTALL)

        # Remove anything inside curly brackets (often for templates or references)
        sentence = re.sub(r'\{.*?\}', '', sentence)

        # Clean up extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        # Remove short sentences or fragments (less than 3 words)
        if len(sentence.split()) > 2:
            cleaned_sentences.append(sentence.strip())

    return cleaned_sentences

# Function to split cleaned content into sentences
def split_sentences(content, nlp):
    # Process the content with Stanza
    doc = nlp(content)
    # Remove non-informative sentences (those with just punctuation or short sentences)
    sentences = [sentence.text for sentence in doc.sentences if len(sentence.text.split()) > 2]
    return sentences

def createWikiTrainSet(category):
    global trainSentences, trainSentencesStructure, testSentences, testSentencesStructure

    # Fetch a list of article titles from the specified category
    titles = fetch_article_titles(category)
    print(f"Number of titles fetched from category '{category}': {len(titles)}")

    sentencesStructure = []  # Store all sentences from fetched articles

    # Fetch content from each title and split into sentences
    for title in titles:
        content = fetch_and_parse_content(title)

        # Tokenize the paragraph into sentences
        sentence_data = split_sentences(content, nlp)

        # Clean the Wikipedia content
        sentence_data = clean_wikipedia_content(sentence_data)

        # Add the list of sentences for this paragraph
        sentencesStructure.append(sentence_data)

    sentences = [sentence for sublist in sentencesStructure for sentence in sublist]

    offset = int(len(sentences)*0.8)
    trainSentencesStructure, testSentencesStructure = createTrainAndTestStructure(sentencesStructure, offset)
    print(f"Created a train set with {len(sentences)} sentences")

    trainSentences, testSentences = sentences[:offset], sentences[offset:]

    return trainSentences, testSentences

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

def clean_data(data):
    for pattern, replacement in patterns:
        data = pattern.sub(replacement, data)

    return data

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

def createWikiText2Set(path):
    data = load_data(path)

    titles, sources = create_sources(data)
    sources = [clean_data(source) for source in sources]

    # Convert paragraphs to individual sentences and group them by source
    sentencesStructure = []

    for source in sources:
        # Process the paragraph using NLTK
        sentences_data = nltk.sent_tokenize(source)

        # Add the list of sentences for this paragraph
        sentencesStructure.append(sentences_data)

    # Flatten the list of lists to have a total collection of sentences
    sentences = [sentence for sublist in sentencesStructure for sentence in sublist]

    return sentences, sentencesStructure

def createWikiText2TrainSet():
    global trainSentences, trainSentencesStructure, testSentences, testSentencesStructure

    testSentences, testSentencesStructure = createWikiText2Set("./Datasets/WikiText2Test.txt")
    trainSentences, trainSentencesStructure = createWikiText2Set("./Datasets/WikiText2Train.txt")

    print(f"Created a train set with {len(trainSentences)} sentences")

    return trainSentences, testSentences

def createEnglishWikiTrainSet(filePath):
    global trainSentences, trainSentencesStructure, testSentences, testSentencesStructure

    # Load the specific Parquet file using pandas
    df = pd.read_parquet(filePath)

    # Extract the relevant column ('maintext')
    paragraphs = df['maintext'].tolist()
    print(df['filename'][0])

    # Convert paragraphs to individual sentences and group them by source
    sentencesStructure = []

    for paragraph in paragraphs:
        # Process the paragraph through Stanza
        doc = nlp(paragraph)
        sentences_data = [sentence.text for sentence in doc.sentences]

        # Add the list of sentences for this paragraph
        sentencesStructure.append(sentences_data)

    # Flatten the list of lists to have a total collection of sentences
    sentences = [sentence for sublist in sentencesStructure for sentence in sublist]

    offset = int(len(sentences)*0.8)
    trainSentencesStructure, testSentencesStructure = createTrainAndTestStructure(sentencesStructure, offset)
    print(f"Created a train set with {len(sentences)} sentences")

    trainSentences, testSentences = sentences[:offset], sentences[offset:]

    return trainSentences, testSentences

# Function to map a flattened index back to its source and sentence index
def getSourceAndSentenceIndex(flat_index, structure="Training"):
    if structure == "Training":
        structure = trainSentencesStructure
    elif "Evaluation" in structure:
        structure = testSentencesStructure
    else:
        raise ValueError("Invalid structure name. Must be 'Training' or contain 'Evaluation'!")

    # Iterate over the sentences list of lists to find the corresponding sublist and sentence index
    current_index = 0
    for source_index, sublist in enumerate(structure):
        sublist_length = len(sublist)
        if flat_index < current_index + sublist_length:
            sentence_index = flat_index - current_index
            return source_index, sentence_index
        current_index += sublist_length
    return None  # Return None if the index is out of range

# Global definitions for special tokens
eos_token_id = 50256  # GPT-2 EOS token ID
pad_token_id = 0      # Arbitrary padding token ID (ensure it does not conflict with actual tokens)

def setGPTSettings(layerAmount, learningRate, epochs):
    global GPT_CONFIG_124M, settings, tokenizer, eos_token_id, pad_token_id

    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Context length (originally 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": layerAmount,  # Number of attention heads
        "n_layers": layerAmount, # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    # Initialize the tokenizer using tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    settings = {
        "learning_rate": learningRate,
        "weight_decay": 0.1,
        "batch_size": 8,
        "num_epochs": epochs
    }

    # (Optional) Define your LLM layers and transformer block layer as needed.
    LLM_Layers = [
        [
            ('Embedding', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
            ('Embedding', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
            ('Dropout', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"])
        ],
        [
            ('Sequential', GPT_CONFIG_124M["emb_dim"], 4 * GPT_CONFIG_124M["emb_dim"]),
            ('LayerNorm', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
            ('Linear', GPT_CONFIG_124M["vocab_size"], GPT_CONFIG_124M["emb_dim"])
        ]
    ]

    TransformerBlockLayer = [
        ('LayerNorm', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Dropout', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"]),
        ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('MultiHeadAttention', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Dropout', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"]),
        ('LayerNorm', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Linear', GPT_CONFIG_124M["emb_dim"], 4 * GPT_CONFIG_124M["emb_dim"]),
        ('GELU', 4 * GPT_CONFIG_124M["emb_dim"], 4 * GPT_CONFIG_124M["emb_dim"]),
        ('Linear', 4 * GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
        ('Sequential', GPT_CONFIG_124M["emb_dim"], 4 * GPT_CONFIG_124M["emb_dim"]),
        ('FeedForward', GPT_CONFIG_124M["emb_dim"], 4 * GPT_CONFIG_124M["emb_dim"]),
        ('Dropout', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"]),
        ('TransformerBlock', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"])
    ]

    return LLM_Layers, TransformerBlockLayer


# Updated Dataset that returns variable-length token lists
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, use_sliding_window=False, stride=None):
        self.input_ids = []
        self.target_ids = []
        stride = stride or max_length

        def process_with_sliding(token_ids):
            # Ensure EOS token is appended if not present.
            if not token_ids or token_ids[-1] != eos_token_id:
                token_ids.append(eos_token_id)
            # Use sliding window only if the token sequence is long enough.
            if len(token_ids) >= max_length:
                for i in range(0, len(token_ids) - max_length + 1, stride):
                    sample_ids = token_ids[i : i + max_length]
                    # Only add sample if it meets the required length.
                    if len(sample_ids) == max_length:
                        input_chunk = sample_ids[:-1]
                        target_chunk = sample_ids[1:]
                        self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                        self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
            else:
                # If the text is too short for sliding, fall back to single-sample mode.
                process_as_single_sample(token_ids)

        def process_as_single_sample(token_ids):
            # Ensure EOS token is appended.
            if not token_ids or token_ids[-1] != eos_token_id:
                token_ids.append(eos_token_id)
            # Truncate or pad the token_ids to exactly max_length.
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids += [pad_token_id] * (max_length - len(token_ids))
            input_chunk = token_ids[:-1]
            target_chunk = token_ids[1:]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

        # Process input based on type.
        if isinstance(txt, list):
            # Process each sentence.
            for sentence in txt:
                # Skip empty strings.
                sentence = sentence.strip() if isinstance(sentence, str) else ""
                if not sentence:
                    continue
                # Tokenize the sentence.
                token_ids = tokenizer.encode(sentence, allowed_special={"<|endoftext|>"})
                # Use sliding window if desired and if the sentence is long enough.
                if use_sliding_window and len(token_ids) >= max_length:
                    process_with_sliding(token_ids)
                else:
                    process_as_single_sample(token_ids)
        else:
            # txt is a single string.
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            if use_sliding_window and len(token_ids) >= max_length:
                process_with_sliding(token_ids)
            else:
                process_as_single_sample(token_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, use_sliding_window=False, stride=None, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, use_sliding_window, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=custom_collate_fn)
    return dataloader

def custom_collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
    return input_ids_padded, target_ids_padded

def createLLMLoader(sentences, batch_size, use_sliding_window=False, shuffle=False, drop_last=True):
    stride = 128 if use_sliding_window else None
    loader = create_dataloader_v1(
        sentences,
        batch_size=batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        use_sliding_window=use_sliding_window,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return loader

def createLLMLoaders(train_samplesParameter, test_samplesParameter, eval_samplesParameter):
    global train_loader, val_loader, eval_loader, train_samples, test_samples, eval_samples, tokenizer

    train_samples, test_samples, eval_samples = train_samplesParameter, test_samplesParameter, eval_samplesParameter
    # Assuming trainSentences and testSentences are defined globally
    train_sentences = " ".join(trainSentences[:train_samples])
    test_sentences = " ".join(testSentences[:test_samples])
    eval_sentences = testSentences[:eval_samples]

    train_loader = createLLMLoader(train_sentences, batch_size=1, use_sliding_window=True, shuffle=True)
    val_loader = createLLMLoader(test_sentences, batch_size=1, use_sliding_window=True, shuffle=True)
    eval_loader = createLLMLoader(eval_sentences, batch_size=1)
    return train_loader, val_loader, eval_loader

"""# Setup Customizable Network"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        in_idx = in_idx.long().to(device)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def initializeDatasets(train_samples, test_samples, eval_samples, batch_size_training="", batch_size_test="", seed=""):
    global torch, settings

    if(seed != ""):
        print("Setting seed number to ", seed)
        torch.manual_seed(seed)
    else:
        print("Setting random seed")

    createLLMLoaders(train_samples, test_samples, eval_samples)
    print("Created all dataloaders")

def initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate):
    global model, criterion_class, chosen_optimizer, layers, layersToCheck

    layersToCheck = [layerNumber for layerNumber, layer in enumerate(hidden_sizes) if layer[0] == "FeedForward"]

    input_size = len(torch.flatten(train_loader.dataset[0][0].clone().detach()))
    output_size = len(torch.flatten(train_loader.dataset[0][1].clone().detach()))

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    layers = hidden_sizes
    loss_function, optimizer = "Cross-Entropy", "Adam"

    if(loss_function == "MSE"):
        criterion_class = nn.MSELoss()  # For regression
    elif(loss_function == "Cross-Entropy"):
        criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

    if(optimizer == "Adam"):
        chosen_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer == "SGD"):
        chosen_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=pad_token_id)
    return loss


def calc_loss_loader(data_loader, num_batches=None):
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        #print("Dataloader: ", num_batches, ", ", len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch)
            total_loss += loss.item()
        else:
            break

    return total_loss / max(1, num_batches)

def train_model_simple(train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous epoch
            loss = calc_loss_batch(input_batch, target_batch)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            #if global_step % eval_freq == 0:
        model.eval()
        with torch.no_grad():
            train_loss, val_loss = evaluate_model(
                train_loader, val_loader, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
        #print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        print(f"Epoch {epoch+1}: "f"Train-Loss {train_loss:.4f}, Train-Perplexity {np.exp(train_loss):.4f}, Validation-Loss {val_loss:.4f}, Test-Perplexity {np.exp(val_loss):.4f},")


    return train_losses, val_losses, track_tokens_seen

def evaluate_model(train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()

def main():
    global tokenizer

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        train_loader, val_loader, optimizer,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=5,
        start_context="", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, train_loader, eval_loader

def trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs):
    global train_loader, eval_loader, tokenizer

    initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate)
    print("Model initialized, Starting training")
    _, _, _, train_dataloader, eval_dataloader = main()
    print("Training finished")

def initializeHook(hidden_sizes, train_samples):  
    RENN.createDictionaries(hidden_sizes, len(hidden_sizes), train_samples, llmType=True)

    samples = trainSentences[:train_samples]
    train_loader = createLLMLoader(samples, 1)

    RENN.runHooks(train_loader, model, hidden_sizes, True, layersToCheckParameter=layersToCheck)


def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond.to(device))
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx.to(device), idx_next.to(device)), dim=1)  # (batch_size, num_tokens+1)

    return idx

def getLLMPrediction(sample, singleSentence=False):
    global tokenizer

    # Convert text to token IDs
    input_ids = torch.tensor([tokenizer.encode(sample)], dtype=torch.long)

    # Generate new token IDs
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=150,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=1,  # Greedy decoding
            temperature=0.0  # Deterministic output
        )

    # Convert token IDs back to text
    prediction = tokenizer.decode(token_ids.squeeze().tolist()).strip()

    # Ensure the generated text does not include the prompt
    if prediction.startswith(sample):
        onlyPrediction = prediction[len(sample):].lstrip()
    else:
        onlyPrediction = prediction

    # If extracting only the first generated sentence
    if singleSentence:
        doc = nlp(onlyPrediction)
        first_sentence = doc.sentences[0].text if doc.sentences else onlyPrediction
        return first_sentence, onlyPrediction

    return sample, onlyPrediction

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, analyze=False):
    global train_samples, test_samples, eval_samples, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource

    #Generate sentences and get their activation values
    generatedEvalSentences, generatedPrediction = zip(*[getLLMPrediction(testSentences[evalSample], True) for evalSample in range(eval_samples)])
    print([generatedEvalSentence.replace('"', '\\"') for generatedEvalSentence in generatedEvalSentences])
    generatedEvalLoader = createLLMLoader(list(generatedEvalSentences), 1)
    RENN.initializeEvaluationHook(hidden_sizes, generatedEvalLoader, eval_samples, model, os.path.join("Evaluation", "Generated"), True, 0)

    #RENN.initializeEvaluationHook(hidden_sizes, eval_loader, eval_samples, model, os.path.join("Evaluation", "Sample"), True, train_samples)
    #closestSourcesEvaluation, closestSourcesGeneratedEvaluation = RENN.identifyClosestLLMSources(eval_samples, 0, closestSources)
    _, closestSourcesGeneratedEvaluation = RENN.identifyClosestLLMSources(eval_samples, 0, closestSources, True)

    for sampleNumber in range(eval_samples):
        #mostUsedEvalSources = RENN.getMostUsedSources(closestSourcesEvaluation, closestSources, sampleNumber, "Mean")
        #_ = RENN.getMostUsedSources(closestSourcesEvaluation, closestSources, sampleNumber, "Sum")
        mostUsedGeneratedEvalSources = RENN.getMostUsedSources(closestSourcesGeneratedEvaluation, closestSources, sampleNumber, "Mean")
        _ = RENN.getMostUsedSources(closestSourcesGeneratedEvaluation, closestSources, sampleNumber, "Sum")

        sample, prediction = getLLMPrediction(testSentences[sampleNumber])
        print("Evaluation Sample ", sampleNumber, ": ", sample.replace('\n', '').replace('<|endoftext|>', ''))
        print("Follow up: ", prediction.replace('\n', '').replace('<|endoftext|>', ''))
        #print(f"Closest Sources for Evaluation-Sample {sampleNumber} in format [SourceNumber, Occurrences, Source]:")
        #for source, count in mostUsedEvalSources[:closestSources]:
        #    tempSource = source.split(":")
        #    sourceNumber, sentenceNumber = int(tempSource[0]), int(tempSource[1])
        #    trainSentence = trainSentencesStructure[sourceNumber][sentenceNumber].replace('\n', '').replace('<|endoftext|>', '')
        #    print(f"Source: {source}, Count: {count}, Sentence: {trainSentence}")
        #print("Whole List: ", [(source, count, trainSentencesStructure[int(source.split(":")[0])][int(source.split(":")[1])].replace('\n', '').replace('<|endoftext|>', '')) for source, count in mostUsedEvalSources], "\n")
        print(f"Generated Source Sentence: {generatedEvalSentences[sampleNumber]}")
        print(f"Generated Source: {generatedPrediction[sampleNumber]}")
        print(f"Closest Sources for GeneratedEvaluation-Sample {sampleNumber} in format [SourceNumber, Occurrences, Source]:")
        for source, count in mostUsedGeneratedEvalSources[:closestSources]:
            tempSource = source.split(":")
            sourceNumber, sentenceNumber = int(tempSource[0]), int(tempSource[1])
            trainSentence = trainSentencesStructure[sourceNumber][sentenceNumber].replace('\n', '').replace('<|endoftext|>', '')
            print(f"Source: {source}, Count: {count}, Sentence: {trainSentence}")
        print("Whole List: ", [(source, count, trainSentencesStructure[int(source.split(":")[0])][int(source.split(":")[1])].replace('\n', '').replace('<|endoftext|>', '')) for source, count in mostUsedGeneratedEvalSources], "\n")
    #print(f"Time passed since start: {time_since_start(startTime)}"