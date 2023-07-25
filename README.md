# Probabilistic Concept Bottleneck Models

This is code for "Probabilistic Concept Bottleneck Models."<br>
[ArXiv](https://arxiv.org/abs/2306.01574) | [OpenReview](https://openreview.net/forum?id=yOxy3T0d6e)

Part of code is borrowed from [Evaluating Weakly Supervised Object Localization Methods Right](https://github.com/clovaai/wsolevaluation), [Probabilistic Cross-Modal Embedding](https://github.com/naver-ai/pcme), and [Polysemous Visual-Semantic Embedding (PVSE)](https://github.com/yalesong/pvse).


## Abstract

Interpretable models are designed to make decisions in a human-interpretable manner. Representatively, Concept Bottleneck Models (CBM) follow a two-step process of concept prediction and class prediction based on the predicted concepts. CBM provides explanations with high-level concepts derived from concept predictions; thus, reliable concept predictions are important for trustworthiness. In this study, we address the ambiguity issue that can harm reliability. While the existence of a concept can often be ambiguous in the data, CBM predicts concepts deterministically without considering this ambiguity. To provide a reliable interpretation against this ambiguity, we propose Probabilistic Concept Bottleneck Models (ProbCBM). By leveraging probabilistic concept embeddings, ProbCBM models uncertainty in concept prediction and provides explanations based on the concept and its corresponding uncertainty. This uncertainty enhances the reliability of the explanations. Furthermore, as class uncertainty is derived from concept uncertainty in ProbCBM, we can explain class uncertainty by means of concept uncertainty. Code is publicly available at https://github.com/ejkim47/prob-cbm.


## Usage

### Step 1. Prepare Dataset

For the dataset (CUB), please refer to the [page](https://github.com/yewsiang/ConceptBottleneck).
Please change the 'data_root' and 'metadataroot' in a config file.

### Step 2. Train a model

An example command line for the train+eval:
```bash
python main.py --config ./configs/config_exp.yaml --gpu {gpu_num}
```
