import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from itertools import combinations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top_p", type=int, required=False, default=100)
parser.add_argument("--var_genes", type=int, required=False, default=3000)
args = parser.parse_args()

expression_file = "./../data/pbmc3k/pbmc3k_raw.h5ad"
transcripts_file = "./knowledge sources/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv"

#annotations
df_trans = pd.read_csv(transcripts_file, sep="\t", comment="!")
#gene expression
adata = sc.read(expression_file)
adata.var_names = adata.var_names.str.upper()
sc.pp.log1p(adata)

# Filter GO annotations first
trans_genes = set(df_trans['Name.Target'])
adata = adata[:, adata.var_names.isin(trans_genes)].copy()

# Compute top X highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=args.var_genes)
adata = adata[:, adata.var['highly_variable']].copy()

list_genes = adata.var.index.tolist()

#filter only used go terms
filtered_df_trans = df_trans[df_trans["Name.Target"].isin(list_genes)]
filtered_terms = filtered_df_trans.groupby("Name.Target")["Name.TF"].apply(list).to_dict()

gene_pairs = list(combinations(list(filtered_terms.keys()), 2))

edges = {}

for g1, g2 in tqdm(gene_pairs, desc="Making the network"):
    terms1 = filtered_terms.get(g1, set())
    terms2 = filtered_terms.get(g2, set())

    common = len(set(terms1) & set(terms2))
    
    if g1 not in edges:
            edges[g1] = []
    edges[g1].append((g1,g2,common))

final_edges = []

for g1, pairs in tqdm(edges.items(), desc="Pruning the network"):
    # Sort by overlap (highest first), keep top 50
    top = sorted(pairs, key=lambda x: x[2], reverse=True)[:args.top_p]
    final_edges.extend(top)

# Convert to DataFrame
df_edges = pd.DataFrame(final_edges, columns=["g1_symbol", "g2_symbol", "conn"])

df_edges["conn"] = np.log1p(df_edges["conn"])

min_conn = df_edges["conn"].min()
max_conn = df_edges["conn"].max()

df_edges["conn"] = (df_edges["conn"] - min_conn) / (max_conn - min_conn)

df_edges = df_edges[df_edges["conn"] > 0]

print(df_edges["conn"].value_counts())

# Save to CSV
df_edges.to_csv("pbmc-TF.csv", index_label="id")