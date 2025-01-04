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