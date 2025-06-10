# A White-Box Approach to LLM Explainability: The RENN Framework

]**This repository contains the research and implementation for the master's thesis, "A White-Box Approach to LLM Explainability: Tracing Training Sources with RENN's Compressed Activation Analysis," completed at the University of Applied Sciences Upper Austria, Media Department.** 

This project introduces the **Retraceable Explainable Neural Network (RENN)** framework, a novel "white-box" approach designed to enhance the transparency of Large Language Models (LLMs). RENN traces the origins of a model's output back to specific training data samples by analyzing and comparing neuron activation patterns.

The core challenge with modern LLMs is their "black box" nature, which makes it difficult to understand their decision-making processes. RENN addresses this by creating a verifiable link between a model's predictions and its training data, fostering greater trust and accountability.

---

## Key Contributions & Features

* **Source Tracing via Neuron Activations**: Implements a methodology to identify influential training sources by comparing the linear neuron activations of a model's output with a database of activations from its training data. This has been validated on architectures like **GPT-2** and **OLMo**.
* **Efficient Activation Data Pipeline**: To manage the massive volume of activation data, this work introduces a highly efficient storage and retrieval pipeline. This pipeline uses sparse value normalization, Apache Parquet, and ZSTD compression to achieve over **99% compression** with minimal loss of fidelity.
* **Metric-Based Activation "Fingerprints"**: A key innovation of this research is the concept of "metric fingerprints". This technique creates a low-dimensional proxy for high-dimensional activation vectors using a combination of 14 statistical metrics (e.g., $L_{1}$, $L_{2}$, $L_{\infty}$ norms, Pearson Correlation, Shannon Entropy). These fingerprints drastically reduce computational overhead for similarity searches while preserving accuracy, making large-scale source tracing feasible.
* **Custom OLMo Dataset**: For this research, the `luca-g97/dolma-v1_7-50B-second-phase` dataset was manually created to approximate the specialized data mix used for the second-phase training of the OLMo model. This provides a rich, multi-source environment for evaluating tracing precision.

---

## Colab Notebooks

Explore the concepts and experiments of the RENN framework using the Google Colab notebooks below.

### Interactive AI Playground

This notebook provides a hands-on "playground" environment to experiment with the foundational findings of the RENN framework. It allows users to explore source tracing and the impact of different configurations in an accessible way. The modular designed code can be found inside this repository in the folder "Google_Colab-Interactive_AI_Playground".

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luca-g97/Master-Thesis/blob/main/Interactive_AI_Playground.ipynb)

### Testing Area

This notebook contains the detailed experimental work and validation for the metric fingerprinting approach. It serves as an informational source detailing the empirical tests that confirmed the effectiveness of using low-dimensional metric combinations for efficient and accurate activation similarity assessment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luca-g97/Master-Thesis/blob/main/Testing_Area.ipynb)

### OLMo Notebook

This notebook documents the exploration and application of RENN concepts to the OLMo architecture. It details how activation hooks were adapted for OLMo's specific features, such as SwiGLU activations and Rotary Positional Embeddings (RoPE).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](PLACEHOLDER_LINK_FOR_OLMO_NOTEBOOK)

### Final Evaluation Scripts

This collection includes the scripts used for the final, comprehensive strategy evaluation on the MNIST dataset, as detailed in Chapter 5. The code used for the creation of the table can be found inside this repository in the folder "Google_Colab-Interactive_AI_Playground/Final-Evaluation".

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](PLACEHOLDER_LINK_FOR_FINAL_EVALUATION_SCRIPTS)
