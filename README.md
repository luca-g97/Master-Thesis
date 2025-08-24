# A White-Box Approach to LLM Explainability: Preserving k-NN Accuracy with Compressed Activation Analysis

**This repository contains the research and implementation for the master's thesis, "A White-Box Approach to LLM Explainability: Tracing Training Sources with RENN's Compressed Activation Analysis," completed at the University of Applied Sciences Upper Austria, Hagenberg Campus.** 

This research tackles a critical bottleneck in "white-box" AI explainability: the immense computational and storage cost of analyzing neuron activation data from Large Language Models (LLMs). While methods like k-Nearest Neighbors (k-NN) search on activation data are promising for understanding model behavior, they are often computationally infeasible due to the sheer volume and high dimensionality of the data.

This thesis develops and validates a framework to solve this problem by creating highly efficient compression and fingerprinting techniques for neuron activations. The primary goal is to make large-scale similarity searches practical while rigorously preserving the accuracy of the k-NN neighborhood structure.

---

## Key Contributions & Features

* **Efficient Activation Data Pipeline**: This work introduces a novel and highly efficient pipeline for storing and retrieving neuron activations. By synergistically combining sparse value normalization, Apache Parquet, and ZSTD compression, the pipeline achieves over **99% compression** with negligible reconstruction error, enabling high-fidelity k-NN analysis on what would otherwise be unmanageably large datasets.

* **Systematic Evaluation of Compression on k-NN Fidelity**: The thesis provides a comprehensive benchmark of how different compression strategies impact k-NN accuracy. The proposed `Sparse Integer Normalization` method is shown to be a superior, customizable solution that balances data fidelity and storage efficiency.

* **"Less is More" Principle in Fingerprinting**: Through systematic evaluation, this research demonstrates that for creating low-dimensional proxies ("fingerprints") of activation vectors, simple linear methods like **Principal Component Analysis (PCA)** consistently outperform more complex, non-linear, or hybrid approaches. This provides clear, practical guidance for creating efficient representations for rapid similarity searches.

* **Custom OLMo Dataset**: To support this research, the `luca-g97/dolma-v1_7-50B-second-phase` [![Hugging Face Dataset](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg)](https://huggingface.co/datasets/luca-g97/dolma-v1_7-50B-second-phase) dataset was manually regenerated from the larger Dolma corpus. It was created to approximate the specialized, 50-billion-token data mixture used for the second phase of the OLMo model's training curriculum. This dataset provides a rich and realistic multi-source environment for evaluating the framework’s performance on a state-of-the-art model.

---

## Colab Notebooks for the Final Thesis

Explore the core concepts and validated experiments of the final thesis using the Google Colab notebooks below.

### Research: Compression & k-NN Fidelity Analysis (MNIST)
This notebook contains the detailed experimental work and validation for the compression pipelines and fingerprinting techniques on the MNIST dataset. It includes the comprehensive benchmarks for both lossless-style compression and the various fingerprinting methods, serving as the primary source for the thesis's empirical findings on k-NN preservation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uXqehALHx8PLbeQ1lTqFCnHq56oxNM98)

### Research: Metric Combination & Fingerprinting (MNIST)
This notebook details the initial, deep exploration into creating activation "fingerprints" from a wide array of statistical metrics. It includes the experiments with the "Expert Team" (Statistician, Geometer, Artist) concept and the various supervised and unsupervised models that led to the "less is more" conclusion.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11BHVYN_G9tJYvx23fKEbU19wuZ1YyzLN)

### GPT-2 (Proof of Concept)
This notebook adapts and applies the validated compression and fingerprinting concepts to a custom GPT-2 model. It serves as a proof-of-concept for scaling the analysis to transformer architectures and demonstrates the trade-offs between different methods in an LLM context.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12efrjrTAzVTUAVv6C8jootjBCZsjv0g9)

### OLMo (Proof of Concept)
This notebook documents the large-scale application of the compression framework to the OLMo-1B model. It details the creation of the custom data streaming pipeline for handling the massive Dolma dataset and validates the performance of the compression techniques on a state-of-the-art, open-source LLM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qEOz-SwjrGOrhkVxlNhXmxKTxcpoERh6)

---

## Preliminary Research (Future Work)

The initial focus of this thesis was on direct source tracing using the **Retraceable Explainable Neural Network (RENN)** paradigm. While this line of inquiry could not be fully validated in time, the extensive preliminary research provides a strong foundation for future work. The core idea was to trace an LLM's output back to specific training samples by comparing neuron activation patterns. The notebooks below document this exploratory work.

### Interactive AI Playground

This notebook provides a hands-on "playground" environment to experiment with the foundational findings of the original RENN source-tracing concept. The modularly designed code can be found in the folder "Google_Colab-Interactive_AI_Playground".

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lApBBKHaF7xl0NQ5-gcwV5Mr5wHWHVlq?usp=sharing)

### OLMo Playground

This notebook documents the initial exploration and application of RENN source-tracing concepts to the OLMo architecture. It details how activation hooks were adapted for OLMo's specific features, such as SwiGLU activations and Rotary Positional Embeddings (RoPE).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/luca-g97/Master-Thesis/blob/main/OLMO_Playground.ipynb)

### Future Work: Final Evaluation (RENN Workflow)
This notebook contains the scripts used for the systematic evaluation of different source-tracing workflow strategies on the MNIST dataset. The findings from this notebook, such as the superiority of global, weighted blending, now form the basis of the recommendations in the "Future Work" chapter.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DClthGU46S0vywJZBIxelkA9F0LwAmMF)

### Appendix: Blending with kNN (Input Space)

This visualization includes the scripts extensively tested on the MNIST and CIFAR-10 dataset. It clearly shows how the kNN blending technique can be used as a vector approximation being almost everytime superior against single closest source matching.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I6rC0XptSLn3dU3YholPDVQ2H9zQSb9s?usp=sharing)

### Experimental: Testing Area

This notebook contains the detailed experimental work and validation for the initial compression pipelines and fingerprinting techniques. It includes comprehensive tests on various topics, such as pre-evaluations for Compression, Sentence Similarity, best Sentence Tokenizer, etc.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/luca-g97/Master-Thesis/blob/main/Testing_Area.ipynb)
