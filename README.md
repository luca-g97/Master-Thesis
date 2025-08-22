# A White-Box Approach to LLM Explainability: Preserving k-NN Accuracy with Compressed Activation Analysis

**This repository contains the research and implementation for the master's thesis, "A White-Box Approach to LLM Explainability: Tracing Training Sources with RENN's Compressed Activation Analysis," completed at the University of Applied Sciences Upper Austria, Hagenberg Campus.** This research tackles a critical bottleneck in "white-box" AI explainability: the immense computational and storage cost of analyzing neuron activation data from Large Language Models (LLMs). While methods like k-Nearest Neighbors (k-NN) search on activation data are promising for understanding model behavior, they are often computationally infeasible due to the sheer volume and high dimensionality of the data.

This thesis develops and validates a framework to solve this problem by creating highly efficient compression and fingerprinting techniques for neuron activations. The primary goal is to make large-scale similarity searches practical while rigorously preserving the accuracy of the k-NN neighborhood structure.

---
## Key Contributions & Features

* **Efficient Activation Data Pipeline**: This work introduces a novel and highly efficient pipeline for storing and retrieving neuron activations. By synergistically combining sparse value normalization, Apache Parquet, and ZSTD compression, the pipeline achieves over **99% compression** with negligible reconstruction error, enabling high-fidelity k-NN analysis on what would otherwise be unmanageably large datasets.

* **Systematic Evaluation of Compression on k-NN Fidelity**: The thesis provides a comprehensive benchmark of how different compression strategies impact k-NN accuracy. The proposed `Sparse Integer Normalization` method is shown to be a superior, customizable solution that balances data fidelity and storage efficiency.

* **"Less is More" Principle in Fingerprinting**: Through systematic evaluation, this research demonstrates that for creating low-dimensional proxies ("fingerprints") of activation vectors, simple linear methods like **Principal Component Analysis (PCA)** consistently outperform more complex, non-linear, or hybrid approaches. This provides clear, practical guidance for creating efficient representations for rapid similarity searches.

* **Custom OLMo Dataset**: To facilitate this research, the `luca-g97/dolma-v1_7-50B-second-phase` dataset was manually created to approximate the specialized data mix used for the second-phase training of the OLMo model. This provides a rich, multi-source environment for evaluating the framework on a state-of-the-art model.

---

## Colab Notebooks

Explore the concepts and experiments of this framework using the Google Colab notebooks below.

### 📊 Final Evaluation Notebook

This notebook contains the scripts used for the final, systematic evaluation of different source-tracing workflow strategies on the MNIST dataset, which now forms the basis of the "Future Work" chapter. The code used to generate the performance heatmaps and the layer relevance table can be found here.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DClthGU46S0vywJZBIxelkA9F0LwAmMF?usp=sharing)

---
## Preliminary Research (Future Work)

The initial focus of this thesis was on direct source tracing using the **Retraceable Explainable Neural Network (RENN)** paradigm. While this line of inquiry could not be fully validated in time, the extensive preliminary research provides a strong foundation for future work. The core idea was to trace an LLM's output back to specific training samples by comparing neuron activation patterns. The notebooks below document this exploratory work.

### 🕹️ Interactive AI Playground

This notebook provides a hands-on "playground" environment to experiment with the foundational findings of the original RENN source-tracing concept. The modularly designed code can be found in the folder "Google_Colab-Interactive_AI_Playground".

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lApBBKHaF7xl0NQ5-gcwV5Mr5wHWHVlq?usp=sharing)

### 🔬 Main Experiments & Validation Notebook

This notebook contains the detailed experimental work and validation for the compression pipelines and fingerprinting techniques. It includes the comprehensive benchmarks on MNIST and the application of the framework to **GPT-2** and **OLMo** models, serving as the primary source for the thesis's empirical findings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/luca-g97/Master-Thesis/blob/main/Testing_Area.ipynb)

### 🤖 OLMo Notebook

This notebook documents the initial exploration and application of RENN source-tracing concepts to the OLMo architecture. It details how activation hooks were adapted for OLMo's specific features, such as SwiGLU activations and Rotary Positional Embeddings (RoPE).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/luca-g97/Master-Thesis/blob/main/OLMO_Playground.ipynb)

### Blending with kNN (Proof of Concept)

This visualization includes the scripts extensively tested on the MNIST and CIFAR-10 dataset. It clearly shows how the kNN blending technique can be used as a vector approximation being almost everytime superior against single closest source matching.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I6rC0XptSLn3dU3YholPDVQ2H9zQSb9s?usp=sharing)
