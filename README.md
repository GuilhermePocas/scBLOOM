
# scBLOOM: Improving Cell Type Classification by Bridging Single-cell Expression and Biological Knowledge

## Overview

![scBLOOM Overview](docs/overview.png)
*Figure 1. Overview of the scBLOOM workflow.*



**scBLOOM** (single-cell Biologically-enhanced Learning from Ontologies and Omics-based Models) is a computational workflow geared towards cell type classification, with a particular focus on accuratly predicting rare cell types. It achieves this by bridging large-scale, expression-only Foundation Models, known for their accurate predictions, with domain-specific biological knowledge drawn from large graphs of relevant information.

The workflow employs two methods for this purpose, the biological knowledge integration method [scNET, 2025](https://github.com/madilabcode/scNET), and the Foundation Model [scGPT, 2024](https://github.com/bowang-lab/scGPT).


##  Installation

### **1. Setting up**
To clone the repository, use the following command:
```
git clone https://github.com/your-username/scBLOOM.git
```

The required packages can ben instaled trough the use of the ``` requirements.txt ``` file located at ```./Docs ```. Python 10.16** is recommended.

A dockerfile for the required enviroment is also provided, but GPU access ir required.

##  API

### Networks

scNET utilizes a gene similarity graph based in PPI information. In ``` ./networks ``` this graph is presented, as well as three others constructed from our own methods. 

These can be found in the corresponding directories, and in order to run them the corresponding knowledge sources must be downloaded from * and placed in ``` ./knowledge sources ```. Then, to run the network creation methods simply run the following command:

``` 
python create_METHOD_graph.py --top_p 100 --var_genes 2000
```

Whith the following args:

- **top_p(int)**: the number of pairs to consider for each gene, higher number might generate really large graphs.

- **var_genes(int)**: the number of highly variable genes to be selected.

### scBLOOM

To run the scBLOOM pipeline, first load a scRNAseq dataset using Scanpy, and then choose the appropriate scGPT model checkpoint and place it in ```/src/scgpt/checkpoints```. Then, run the following function:

```
scnet_embs, scgpt_embs, avg_embs, conq_embs, labels = run_scBLOOM(model_name, data, network, dir, scnet_epochs, scgpt_epochs)
```

with the following args:

- **model_name(str)**: the name of the current run.

- **data(AnnData)**: a scRNAseq AnnData object.

- **network(str)**: the file path of the gene similarity network.

- **dir(str)**: the output directory where the results and embeddings will be saved.

- **scnet_epochs(int)**: the number of steps when training scNET.

- **scgpt_epochs(int)**: the number of steps when training scGPT.

The resulting output objects are:

- **scnet_embs**: the embeddings from the scNET model.

- **scgpt_embs**: the embeddings from the scGPT model.

- **avg_embs**: the scBLOOM embeddings with the average aggregation strategy.

- **conq_embs**: the scBLOOM embeddings with the concatenation aggregation strategy.

- **labels**: the cell type labels, in the same order as the embeddings.

An example script is provided in ```/main.py```, using the pbmc3k dataset from Scanpy, and the human scGPT model checkpoint, available here *. To run the example simply run the following command:

```
python main.py --model_name test --network PPI.csv --scnet_epochs 800 --scgpt_epochs 60
```
