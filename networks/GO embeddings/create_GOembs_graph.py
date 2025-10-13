
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import scipy.sparse as sp
from anndata import AnnData
from tqdm import tqdm
from numpy.linalg import norm
from itertools import combinations

TOP_PAIRS=100

def read_go_embeddings(file):
    go_embs = {}
    with open(file, "r") as file:
        next(file)
        for line in file:
            parts = [item for item in line.strip().split(" ") if item]
            key = int(parts[0])  
            values = list(map(float, parts[1:]))
            go_embs[key] = values
    return go_embs

def decode_go_ids_embeddings(embs, ids_file):
    with open(ids_file, 'rb') as fp:
        go_id_dict = pickle.load(fp)
    id_go_dict = {value: key for key, value in go_id_dict.items()}
    decoded_embs = {id_go_dict[key]: values for key, values in embs.items()}
    return decoded_embs

#used_genes_file = "../datasets/genes/genes.csv"
expression_file = "../../data/pbmc3k/pbmc3k_raw.h5ad"
annotations_file = "../knowledge sources/goa_human.gaf"
go_embeddings_file = "../knowledge sources/go-terms-75.emd"
go_ids_file = "../knowledge sources/go_id_dict"

#annotations
df_gaf = pd.read_csv(annotations_file, sep="\t", comment="!", header=None, names=[
    "DB", "GeneID", "GeneSymbol", "Qualifier", "GO_Term", "Reference", 
    "Evidence", "WithFrom", "Aspect", "DB_Object_Name", "DB_Object_Synonym", 
    "DB_Object_Type", "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID"
],low_memory=False)

#only annotations from bp subontology
df_gaf = df_gaf[df_gaf["Aspect"] == "P"]

#rnaseq data 2000 most variable genes
adata = sc.read(expression_file)
adata.var_names = adata.var_names.str.upper()
sc.pp.log1p(adata)

# Filter GO annotations first
go_genes = set(df_gaf["GeneSymbol"])
adata = adata[:, adata.var_names.isin(go_genes)].copy()

# Compute top X highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var['highly_variable']].copy()

list_genes = adata.var.index.tolist()

#filter only used go terms
df_gaf_filtered = df_gaf[df_gaf["GeneSymbol"].isin(list_genes)]
filtered_terms = df_gaf_filtered.groupby("GeneSymbol")["GO_Term"].apply(list).to_dict()

gene_pairs = list(combinations(list(filtered_terms.keys()), 2))

#PRODUCE GENE EMBEDDINGS
go_embs = read_go_embeddings(go_embeddings_file)
decoded_embs = decode_go_ids_embeddings(go_embs, go_ids_file)

gene_embeddings = {}

for gene, go_list in filtered_terms.items():
    embedding_sum = 0
    sum_weights = 0

    for go_term in go_list:
        if go_term in decoded_embs:

            embedding_sum += np.array(decoded_embs[go_term]) #* ic_values[go_term]
            sum_weights += 1#ic_values[go_term]
        else:
            print(f"Term {go_term} not found in embeddings")

    if sum_weights > 0:
        gene_embeddings[gene] = embedding_sum/sum_weights
    else:
        print(f"Gene {gene} does not have an embedding")
        gene_embeddings[gene] = np.zeros((75,))


edges = {}
for g1, g2 in tqdm(gene_pairs, desc="Making the network"):
    emb1 = gene_embeddings[g1]
    emb2 = gene_embeddings[g2]
    norm1 = norm(emb1)
    norm2 = norm(emb2)

    if norm1 == 0 or norm2 == 0:
        continue  # Skip this pair due to zero vector
    cosine = np.dot(emb1, emb2) / (norm1 * norm2)

    if g1 not in edges:
        edges[g1] = []
    edges[g1].append((g1,g2,cosine))

final_edges=[]
for g1, pairs in tqdm(edges.items(), desc="Pruning the network"):
    top = sorted(pairs, key=lambda x: x[2], reverse=True)[:TOP_PAIRS]
    final_edges.extend(top)

# Convert to DataFrame
df_edges = pd.DataFrame(final_edges, columns=["g1_symbol", "g2_symbol", "conn"])

#normalize
min_conn = df_edges["conn"].min()
max_conn = df_edges["conn"].max()
if max_conn > min_conn:
    df_edges["conn"] = (df_edges["conn"] - min_conn) / (max_conn - min_conn)
else:
    df_edges["conn"] = 0.0
df_edges = df_edges[df_edges["conn"] > 0]


# Save to CSV
df_edges.to_csv("../pbmc-GOembs.csv", index_label="id")
