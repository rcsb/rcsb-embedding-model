# RCSB Embedding Model: A Deep Learning Approach for 3D Structure Embeddings

## Overview
RCSB Embedding Model is a PyTorch-based neural network that transforms macromolecular 3D structures into vector embeddings.

Preprint: [Multi-scale structural similarity embedding search across entire proteomes](https://www.biorxiv.org/content/10.1101/2025.02.28.640875v1).

A web-based implementation using this model for structure similarity search is available at [rcsb-embedding-search](http://embedding-search.rcsb.org).

If you are interested in training the model with a new dataset, visit the [rcsb-embedding-search repository](https://github.com/bioinsilico/rcsb-embedding-search), which provides scripts and documentation for training.

---

## Embedding Model
The embedding model is trained to predict structural similarity by approximating TM-scores using cosine distances between embeddings. It consists of two main components:

- **Protein Language Model (PLM)**: Computes residue-level embeddings from a given 3D structure.
- **Residue Embedding Aggregator**: A transformer-based neural network that aggregates these residue-level embeddings into a single vector.

![Embedding model architecture](assets/embedding-model-architecture.png)

### **Protein Language Model (PLM)**
Residue-wise embeddings of protein structures are computed using the [ESM3](https://www.evolutionaryscale.ai/) generative protein language model.

### **Residue Embedding Aggregator**
The aggregation component consists of six transformer encoder layers, each with a 3,072-neuron feedforward layer and ReLU activations. After processing through these layers, a summation pooling operation is applied, followed by 12 fully connected residual layers that refine the embeddings into a single 1,536-dimensional vector.

---

## How to Use the Model
This repository provides the tools to compute embeddings for 3D macromolecular structure data.

### **Installation**
`pip install rcsb-embedding-model`

### **Requirements**
Ensure you have the following dependencies installed:
- `python >= 3.10`
- `esm`
- `torch`

### **Generating Residue Embeddings**
ESM3 embeddings for the 3D structures can be calculated as:

```python
from rcsb_embedding_model import RcsbStructureEmbedding

mmcif_file = "<path_to_file>/<name>.cif"
model = RcsbStructureEmbedding()
res_embedding = model.residue_embedding(
    structure_src=mmcif_file,
    format="mmcif",
    chain_id='A'
)
```

### **Generating Protein Structure Embeddings**
Protein 3D structure embedding can be calculated as:

```python
from rcsb_embedding_model import RcsbStructureEmbedding

mmcif_file = "<path_to_file>/<name>.cif"
model = RcsbStructureEmbedding()
res_embedding = model.residue_embedding(
    structure_src=mmcif_file,
    format="mmcif",
    chain_id='A'
)
structure_embedding = model.aggregator_embedding(
    res_embedding
)
```

### **Pretrained Model**
You can download a pretrained Residue Embedding Aggregator model from [Hugging Face](https://huggingface.co/jseguramora/rcsb-embedding-model/resolve/main/rcsb-embedding-model.pt).

---

## Questions & Issues
For any questions or comments, please open an issue on this repository.

---

## License
This software is released under the BSD 3-Clause License. See the full license text below.

### BSD 3-Clause License

Copyright (c) 2024, RCSB Protein Data Bank, UC San Diego

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

