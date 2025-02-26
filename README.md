
RCSB Embedding Model
---

RCSB Embedding Model is a pytorch model that transforms macromolecule 3D structures into vectors.

A web application implementing this method is available at [rcsb-embedding-search](http://embedding-search.rcsb.org).

If you are interested in training the model with a new dataset the repo [rcsb-embedding-search](https://github.com/bioinsilico/rcsb-embedding-search)
provides different scripts and documentation for this task.

### Embedding Model
The embedding model was trained to predict structure similarity approximating TM-scores with embedding cosine distance.
The model consists of two components:
- A protein language model PLM that computes residue-level embeddings a given 3D structure
- A transformer-based neural network that aggregates these residue-level embeddings into a single vector

![Embedding model architecture](assets/embedding-model-architecture.png)

#### PLM
Protein residue-level embeddings are computed with the [ESM](https://www.evolutionaryscale.ai/) generative protein language model.

#### Residue Embedding Aggregator

The aggregator consists of six transformer encoder layers, with 3,072 neurons feedforward layer and ReLU activations.
Following the encoders, a summation pooling operation and 12 fully connected residual layers aggregate the resulting embeddings into a single 1,536-dimensional vector.

### How to use the model
This repository contains the Residue Embedding Aggregator model. First, you will need to calculate ESM3 embeddings for the 3D structures.
The script `examples/esm_embeddings.py` shows how to do this using [biotite](https://www.biotite-python.org/).

You can download a pretrained Residue Embedding Aggregator model from [huggingface](https://huggingface.co/jseguramora/rcsb-embedding-model/resolve/main/rcsb-embedding-model.pt) and loaded as a pytorch model as shown in the code above.

```python
    if torch.cuda.is_available():
        weights = torch.load(modelfile, weights_only=True)
    else:
        weights = torch.load(modelfile, weights_only=True, map_location='cpu')

    aggregator_model = ResidueEmbeddingAggregator()
    aggregator_model.load_state_dict(weights)

    structure_vect = aggregator_model(esm3_embeddigns)
```

Questions
---
Please, open an issue for questions or comments.

License
---
BSD 3-Clause License

Copyright (c) 2024, RCSB Protein Data Bank, UC San Diego

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
