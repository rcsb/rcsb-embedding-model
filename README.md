# FoldMatch

**Version** 0.2.0


## Overview

FoldMatch is a Python toolkit to encode macromolecular 3D structures into fixed-length vector embeddings for efficient large-scale structure similarity search and clustering.

Reference: [Multi-scale structural similarity embedding search across entire proteomes](https://doi.org/10.1093/bioinformatics/btag058).

A web-based implementation using this tool for structure similarity search is available at [rcsb-embedding-search](http://embedding-search.rcsb.org).

If you are interested in training a new model with a new structure dataset, visit the [rcsb-embedding-search repository](https://github.com/bioinsilico/rcsb-embedding-search), which provides scripts and documentation for training.


## Features

- **Residue-level embeddings** computed using the ESM3 protein language model  
- **Structure-level embeddings** aggregated via a transformer-based aggregator network 
- **Fast and efficient** FAISS-based similarity search
- **Structural clustering** using the Leiden algorithm for biological assembly identification
- **Command-line interface** implemented with Typer for high-throughput inference workflows  
- **Python API** for interactive embedding computation and integration into analysis pipelines  
- **High-performance inference** leveraging PyTorch Lightning, with multi-node and multi-GPU support  

---

## Installation

### From PyPI

```bash
pip install foldmatch
```

### From Source (Development)

```bash
git clone https://github.com/rcsb/foldmatch.git
cd foldmatch
pip install -e .
```

**Requirements:**

- Python ≥ 3.12
- ESM 3.2.3
- Lightning 2.6.1
- Typer 0.24.1
- Biotite 1.6.0
- FAISS 1.13.2
- igraph 1.0.0
- leidenalg 0.11.0
- PyTorch with CUDA support (recommended for GPU acceleration)

**Optional Dependencies:**

- `faiss-gpu` for GPU-accelerated similarity search (instead of `faiss-cpu`)

### Download Pre-trained Models

Before using the package, download the pre-trained ESM3 and aggregator models:

```bash
fm-inference download-models
```

---

## Usage

The package provides two main interfaces:
1. **Command-line Interface (CLI)** for batch processing and high-throughput workflows
2. **Python API** for interactive use and integration into custom pipelines

---

## Command-Line Interface (CLI)

The CLI provides two main command groups: `fm-inference` for computing embeddings and `fm-search` for similarity search operations.

### Inference Commands

#### `fm-inference residue-embedding`

Calculate residue-level embeddings using ESM3. Outputs are stored as PyTorch tensor files.

```bash
fm-inference residue-embedding \
  --src-file data/structures.csv \
  --output-path results/residue_embeddings \
  --structure-format mmcif \
  --batch-size 8 \
  --devices auto
```

**Key Options:**
- `--src-file`: CSV file with 4 columns: Structure Name | File Path/URL | Chain ID | Output Name
- `--output-path`: Directory to store tensor files
- `--output-format`: `separated` (individual files) or `grouped` (single JSON)
- `--output-name`: Filename when using `grouped` format (default: `inference`)
- `--structure-format`: `mmcif`, `binarycif`, or `pdb`
- `--min-res-n`: Minimum residue count for chain filtering (default: 0)
- `--batch-size`: Batch size for processing (default: 1)
- `--num-workers`: Data loader workers (default: 0)
- `--num-nodes`: Number of nodes for distributed inference (default: 1)
- `--accelerator`: Device type - `auto`, `cpu`, `cuda`, `gpu` (default: `auto`)
- `--devices`: Device indices (can specify multiple with `--devices 0 --devices 1`) or `auto`
- `--strategy`: Lightning distribution strategy (default: `auto`)

---

#### `fm-inference structure-embedding`

Calculate complete structure embeddings (residue + aggregator) from structural files. Outputs stored as a single DataFrame.

```bash
fm-inference structure-embedding \
  --src-file data/structures.csv \
  --output-path results/structure_embeddings \
  --output-name embeddings \
  --batch-size 4 \
  --devices 0 --devices 1
```

**Key Options:**
- Same as `residue-embedding`, plus:
- `--output-name`: Output DataFrame filename (default: `inference`)

---

#### `fm-inference chain-embedding`

Aggregate residue embeddings into chain-level embeddings. Requires pre-computed residue embeddings.

```bash
fm-inference chain-embedding \
  --src-file data/structures.csv \
  --res-embedding-location results/residue_embeddings \
  --output-path results/chain_embeddings \
  --batch-size 4
```

**Key Options:**
- `--res-embedding-location`: Directory containing residue embedding tensor files
- All other options similar to `residue-embedding`

---

#### `fm-inference assembly-embedding`

Aggregate residue embeddings into assembly-level embeddings.

```bash
fm-inference assembly-embedding \
  --src-file data/assemblies.csv \
  --res-embedding-location results/residue_embeddings \
  --output-path results/assembly_embeddings \
  --min-res-n 10 \
  --max-res-n 10000
```

**Key Options:**
- `--src-file`: CSV with columns: Structure Name | File Path/URL | Assembly ID | Output Name
- `--res-embedding-location`: Directory with pre-computed residue embeddings
- `--min-res-n`: Minimum residues per chain (default: 0)
- `--max-res-n`: Maximum total residues for assembly (default: unlimited)

---

#### `fm-inference complete-embedding`

End-to-end pipeline: compute residue, chain, and assembly embeddings in one command.

```bash
fm-inference complete-embedding \
  --src-chain-file data/chains.csv \
  --src-assembly-file data/assemblies.csv \
  --output-res-path results/residues \
  --output-chain-path results/chains \
  --output-assembly-path results/assemblies \
  --batch-size-res 8 \
  --batch-size-chain 4 \
  --batch-size-assembly 2
```

**Key Options:**
- `--src-chain-file`: Chain input CSV
- `--src-assembly-file`: Assembly input CSV
- `--output-res-path`, `--output-chain-path`, `--output-assembly-path`: Output directories
- `--batch-size-res`, `--num-workers-res`, `--num-nodes-res`: Residue embedding settings
- `--batch-size-chain`, `--num-workers-chain`: Chain embedding settings
- `--batch-size-assembly`, `--num-workers-assembly`, `--num-nodes-assembly`: Assembly settings

---

#### `fm-inference download-models`

Download ESM3 and aggregator models from Hugging Face.

```bash
fm-inference download-models
```

---

### Search Commands

#### `fm-search build-db`

Build a FAISS database from structure files for similarity search.

```bash
fm-search build-db \
  --structure-dir data/pdb_files \
  --output-db databases/my_structures \
  --tmp-dir tmp \
  --granularity chain \
  --min-res 10 \
  --use-gpu-index
```

**Key Options:**
- `--structure-dir`: Directory containing structure files
- `--output-db`: Database path (prefix for `.index` and `.metadata` files)
- `--tmp-dir`: Temporary directory for intermediate files
- `--structure-format`: `mmcif`, `binarycif`, or `pdb`
- `--granularity`: `chain` or `assembly` level embeddings
- `--file-extension`: Filter files by extension (e.g., `.cif`, `.bcif`, `.pdb`)
- `--min-res`: Minimum residue count (default: 10)
- `--use-gpu-index`: Use GPU for FAISS index construction
- `--accelerator`, `--devices`, `--strategy`: Inference device settings
- `--batch-size-res`, `--num-workers-res`, `--num-nodes-res`: Residue embedding settings
- `--batch-size-aggregator`, `--num-workers-aggregator`, `--num-nodes-aggregator`: Aggregator settings

---

#### `fm-search update-db`

Update an existing FAISS database with new or replacement structure files. Structures with IDs already present in the database are replaced; new IDs are added. The FAISS index is fully rebuilt after merging.

```bash
fm-search update-db \
  --structure-dir data/new_structures \
  --output-db databases/my_structures \
  --tmp-dir tmp \
  --structure-format mmcif \
  --granularity chain \
  --min-res 10 \
  --batch-size-res 8
```

**Key Options:**
- `--structure-dir`: Directory containing new or updated structure files
- `--output-db`: Path to the existing FAISS database to update
- `--tmp-dir`: Temporary directory for intermediate files
- `--structure-format`: `mmcif`, `binarycif`, or `pdb`
- `--granularity`: `chain` or `assembly` level embeddings
- `--file-extension`: Filter files by extension (e.g., `.cif`, `.bcif`, `.pdb`)
- `--min-res`: Minimum residue count (default: 10)
- `--use-gpu-index`: Use GPU for FAISS index construction
- `--accelerator`, `--devices`, `--strategy`: Inference device settings
- `--batch-size-res`, `--num-workers-res`, `--num-nodes-res`: Residue embedding settings
- `--batch-size-aggregator`, `--num-workers-aggregator`, `--num-nodes-aggregator`: Aggregator settings
- `--log-level`: Logging level - `info`, `warn`, or `debug` (default: `info`)

---

#### `fm-search query`

Search the database for structures similar to a query structure.

```bash
fm-search query \
  --db-path databases/my_structures \
  --query-structure query.cif \
  --structure-format mmcif \
  --granularity chain \
  --top-k 100 \
  --threshold 0.8 \
  --output-csv results.csv
```

**Key Options:**
- `--db-path`: Path to FAISS database
- `--query-structure`: Query structure file
- `--structure-format`: `mmcif` or `pdb`
- `--granularity`: `chain` or `assembly` search mode
- `--chain-id`: Specific chain to search (optional)
- `--assembly-id`: Specific assembly ID (optional)
- `--top-k`: Number of results per query (default: 100)
- `--threshold`: Minimum similarity score (default: 0.8)
- `--output-csv`: Export results to CSV (optional)
- `--min-res`: Minimum residue filter (default: 10)
- `--max-res`: Maximum residue filter (optional)
- `--device`: `cuda`, `cpu`, or `auto`
- `--use-gpu-index`: Use GPU for FAISS search

---

#### `fm-search query-db`

Compare all entries from a query database against a subject database.

```bash
fm-search query-db \
  --query-db-path databases/query_set \
  --subject-db-path databases/target_set \
  --top-k 100 \
  --threshold 0.8 \
  --output-csv comparisons.csv
```

**Key Options:**
- `--query-db-path`: Query database path
- `--subject-db-path`: Subject database to search
- `--top-k`: Results per query (default: 100)
- `--threshold`: Similarity threshold (default: 0.8)
- `--output-csv`: Export results to CSV
- `--use-gpu-index`: Use GPU acceleration

---

#### `fm-search stats`

Display database statistics.

```bash
fm-search stats --db-path databases/my_structures
```

---

#### `fm-search cluster`

Cluster database embeddings using the Leiden algorithm.

```bash
fm-search cluster \
  --db-path databases/my_structures \
  --threshold 0.8 \
  --resolution 1.0 \
  --output clusters.csv \
  --max-neighbors 1000 \
  --min-cluster-size 5
```

**Key Options:**
- `--db-path`: Database path
- `--threshold`: Similarity threshold for edge creation (default: 0.8)
- `--resolution`: Leiden resolution parameter - higher values create more clusters (default: 1.0)
- `--output`: Output file (`.csv` or `.json`)
- `--max-neighbors`: Maximum neighbors per chain (default: 1000)
- `--min-cluster-size`: Filter out smaller clusters (optional)
- `--use-gpu-index`: Use GPU for FAISS operations
- `--seed`: Random seed for reproducibility (optional)

---

#### `fm-search similarity-graph`

Build a similarity graph from database embeddings and export it in [GraphML](http://graphml.graphdrawing.org/) format. Each node represents a chain (identified by its chain ID) and each edge carries a `weight` attribute with the cosine similarity score between the two connected chains.

```bash
fm-search similarity-graph \
  --db-path databases/my_structures \
  --threshold 0.8 \
  --output similarity_graph.graphml \
  --max-neighbors 1000
```

**Key Options:**
- `--db-path`: Database path
- `--threshold`: Minimum similarity score to create an edge (default: 0.8)
- `--output`: Output GraphML file (default: `similarity_graph.graphml`)
- `--max-neighbors`: Maximum neighbors per chain considered during k-NN search (default: 1000)
- `--use-gpu-index`: Use GPU for FAISS operations
- `--log-level`: Logging verbosity level (default: `info`)

The resulting GraphML file can be loaded directly into tools such as [Gephi](https://gephi.org/), [Cytoscape](https://cytoscape.org/), or Python's [NetworkX](https://networkx.org/):

```python
import networkx as nx
G = nx.read_graphml("similarity_graph.graphml")
```

---

## Python API

The `RcsbStructureEmbedding` class provides methods for computing embeddings programmatically.

### Basic Usage

```python
from foldmatch import FoldMatch

# Initialize model
model = FoldMatch(min_res=10, max_res=5000)

# Load models (optional - loads automatically on first use)
model.load_models()  # Auto-detects CUDA
# or specify device:
# import torch
# model.load_models(device=torch.device("cuda:0"))
```

### Methods

#### `load_models(device=None)`

Load both residue and aggregator models.

```python
import torch
model.load_models(device=torch.device("cuda"))
```

---

#### `load_residue_embedding(device=None)`

Load only the ESM3 residue embedding model.

```python
model.load_residue_embedding()
```

---

#### `load_aggregator_embedding(device=None)`

Load only the aggregator model.

```python
model.load_aggregator_embedding()
```

---

#### `residue_embedding(src_structure, structure_format='mmcif', chain_id=None, assembly_id=None)`

Compute per-residue embeddings for a structure.

**Parameters:**
- `src_structure`: File path, URL, or file-like object
- `structure_format`: `'mmcif'`, `'binarycif'`, or `'pdb'`
- `chain_id`: Specific chain ID (optional, uses all chains if None)
- `assembly_id`: Assembly ID for biological assembly (optional)

**Returns:** `torch.Tensor` of shape `[num_residues, embedding_dim]`

```python
# Single chain
residue_emb = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)

# All chains concatenated
all_residues = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif"
)

# Biological assembly
assembly_residues = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    assembly_id="1"
)
```

---

#### `residue_embedding_by_chain(src_structure, structure_format='mmcif', chain_id=None)`

Compute per-residue embeddings separately for each chain.

**Returns:** `dict[str, torch.Tensor]` mapping chain IDs to embeddings

```python
chain_embeddings = model.residue_embedding_by_chain(
    src_structure="1abc.cif",
    structure_format="mmcif"
)
# Returns: {'A': tensor(...), 'B': tensor(...), ...}

# Get specific chain
chain_a = model.residue_embedding_by_chain(
    src_structure="1abc.cif",
    chain_id="A"
)
```

---

#### `residue_embedding_by_assembly(src_structure, structure_format='mmcif', assembly_id=None)`

Compute residue embeddings for an assembly.

**Returns:** `dict[str, torch.Tensor]` mapping assembly ID to concatenated embeddings

```python
assembly_emb = model.residue_embedding_by_assembly(
    src_structure="1abc.cif",
    structure_format="mmcif",
    assembly_id="1"
)
# Returns: {'1': tensor(...)}
```

---

#### `sequence_embedding(sequence)`

Compute residue embeddings from amino acid sequence (no structural information).

**Parameters:**
- `sequence`: Amino acid sequence string (plain or FASTA format)

**Returns:** `torch.Tensor` of shape `[sequence_length, embedding_dim]`

```python
# Plain sequence
seq_emb = model.sequence_embedding("ACDEFGHIKLMNPQRSTVWY")

# FASTA format
fasta = """>Protein1
ACDEFGHIKLMNPQRSTVWY
ACDEFGHIKLMNPQRSTVWY"""
seq_emb = model.sequence_embedding(fasta)
```

---

#### `aggregator_embedding(residue_embedding)`

Aggregate residue embeddings into a single structure-level vector.

**Parameters:**
- `residue_embedding`: `torch.Tensor` from residue embedding methods

**Returns:** `torch.Tensor` of shape `[1536]`

```python
residue_emb = model.residue_embedding("1abc.cif", chain_id="A")
structure_emb = model.aggregator_embedding(residue_emb)
```

---

#### `structure_embedding(src_structure, structure_format='mmcif', chain_id=None, assembly_id=None)`

End-to-end: compute residue embeddings and aggregate in one call.

```python
# Complete structure embedding
structure_emb = model.structure_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)
# Returns: tensor of shape [1536]
```

---

### Complete Example

```python
from foldmatch import FoldMatch
import torch

# Initialize
model = FoldMatch(min_res=10, max_res=5000)

# Option 1: Full structure embedding (one-shot)
embedding = model.structure_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)

# Option 2: Step-by-step with residue embeddings
residue_emb = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)
structure_emb = model.aggregator_embedding(residue_emb)

# Option 3: Process multiple chains
chain_embeddings = model.residue_embedding_by_chain(
    src_structure="1abc.cif"
)
for chain_id, res_emb in chain_embeddings.items():
    chain_emb = model.aggregator_embedding(res_emb)
    print(f"Chain {chain_id}: {chain_emb.shape}")

# Sequence-based embedding
seq_emb = model.sequence_embedding("ACDEFGHIKLMNPQRSTVWY")
structure_from_seq = model.aggregator_embedding(seq_emb)
```

See the `examples/` and `tests/` directories for more use cases.

---

## Model Architecture

The embedding model is trained to predict structural similarity by approximating TM-scores using cosine distances between embeddings. It consists of two main components:

- **Protein Language Model (PLM)**: Computes residue-level embeddings from a given 3D structure.
- **Residue Embedding Aggregator**: A transformer-based neural network that aggregates these residue-level embeddings into a single vector.

![Embedding model architecture](assets/embedding-model-architecture.png)

### **Protein Language Model (PLM)**
Residue-wise embeddings of protein structures are computed using the [ESM3](https://www.evolutionaryscale.ai/) generative protein language model.

### **Residue Embedding Aggregator**
The aggregation component consists of six transformer encoder layers, each with a 3,072-neuron feedforward layer and ReLU activations. After processing through these layers, a summation pooling operation is applied, followed by 12 fully connected residual layers that refine the embeddings into a single 1,536-dimensional vector.

---

## Testing

After installation, run the test suite:

```bash
pytest
```

---

## Citation

Segura, J., et al. (2026). *Multi-scale structural similarity embedding search across entire proteomes*. (https://doi.org/10.1093/bioinformatics/btag058)

---

## License

This project uses the EvolutionaryScale ESM-3 model and is distributed under the
[Cambrian Non-Commercial License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement).