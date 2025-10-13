import os
import scNET
import argparse
import scanpy as sc
import pandas as pd
import faulthandler
import numpy as np
faulthandler.enable()
from datetime import datetime
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
from scgpt import run_scGPT
import mlflow
from test_embeddings import test_embeddings
import pickle
from sklearn.decomposition import PCA

def load_dataset(file, processed_file):
    obj = sc.read(file)
    obj.var_names = obj.var_names.str.capitalize()
    obj_proc = sc.read(processed_file)
    obj_proc.var_names = obj_proc.var_names.str.capitalize()

    obj.obs['celltype'] = obj.obs_names.map(lambda x: obj_proc.obs['louvain'].get(x, "unknown"))
    obj = obj[obj.obs['celltype'] != "unknown"]
    obj.var_names_make_unique()

    return obj

def train_scnet(obj, dir, cfg):
    obj_scNET = obj.copy()
    obj_scNET.raw = obj_scNET

    with mlflow.start_run(run_name="scNET"):
        mlflow.log_param("model", "scNET")
        mlflow.log_param("epochs", cfg["max_epoch"])
        mlflow.log_param("model_name", cfg["model_name"])
        
        scNET_dir = dir + "/scNET"
        os.makedirs(scNET_dir, exist_ok=True)
        #scNET.run_scNET(cfg["annotation_file"], obj_scNET, scNET_dir ,pre_processing_flag=cfg["pre_processing_flag"], human_flag=cfg["human_flag"],
        #                 number_of_batches=cfg["number_of_batches"], split_cells=cfg["split_cells"], max_epoch=cfg["max_epoch"],
        #                 model_name = cfg["model_name"], clf_loss=cfg["clf_loss"])

    embedded_genes, embedded_cells, node_features , out_features, ids =  scNET.load_embeddings(cfg["model_name"], scNET_dir)
    recon_obj = scNET.create_reconstructed_obj(node_features, out_features, obj_scNET)

    return pd.DataFrame(embedded_cells, index=recon_obj.obs_names)


def train_scgpt(obj, dir, cfg):

    obj = obj.copy()

    obj.var_names = obj.var_names.str.upper()
    sc.pp.normalize_total(obj, target_sum=1e4)
    sc.pp.log1p(obj)
    sc.pp.highly_variable_genes(obj, n_top_genes=2000)
    obj = obj[:, obj.var['highly_variable']].copy()
    
    train_idx, test_idx = train_test_split(
        obj.obs_names,
        test_size=0.4,
        stratify=obj.obs["celltype"],  # Ensure class balance
        random_state=42
    )

    obj_scGPT_test = obj[test_idx].copy()
    obj_scGPT_train = obj[train_idx].copy()

    obj_scGPT_train.obs["batch_id"] = obj_scGPT_train.obs["str_batch"] = "0"
    obj_scGPT_test.obs["batch_id"] = obj_scGPT_test.obs["str_batch"] = "1"
    obj_scGPT = obj_scGPT_train.concatenate(obj_scGPT_test, batch_key="str_batch")

    with mlflow.start_run(run_name="scGPT"):
        mlflow.log_param("model", "scGPT")
        for k, v in cfg.items():
            mlflow.log_param(k, v)

        scGPT_dir = dir + "/scGPT"
        os.makedirs(scGPT_dir, exist_ok=True)
        print(obj_scGPT.shape)
        run_scGPT(cfg["model_name"], cfg ,obj_scGPT, scGPT_dir)

    with open(scGPT_dir + "/test_cell_embeddings.pkl", "rb") as f:
        test_cell_embeddings = pickle.load(f)
    scgpt_cell_embeddings = pd.DataFrame.from_dict(test_cell_embeddings, orient="index")
    scgpt_cell_embeddings.index.name = "cell_id"
    scgpt_cell_embeddings.index = scgpt_cell_embeddings.index.str.replace(r"-[01]$", "", regex=True)

    return scgpt_cell_embeddings

def combine_embeddings(obj, scnet_emb, scgpt_emb):

    common_test_idx = scgpt_emb.index.intersection(scnet_emb.index)

    print(f"Test cells in common: {len(common_test_idx)} / {len(scgpt_emb.index)}")

    # Subset to only common cells
    common_scnet_embs = scnet_emb.loc[common_test_idx]
    common_scgpt_embs = scgpt_emb.loc[common_test_idx]

    #subset labels
    common_labels = obj.obs.loc[common_test_idx, "celltype"].values


    avg_combined_embs = pd.DataFrame((common_scgpt_embs.values + common_scnet_embs.values) / 2, index=common_test_idx)

    test_conq_combined_embs = pd.concat([common_scgpt_embs, common_scnet_embs], axis=1)
    pca = PCA(n_components=512)
    conq_combined_embs = pd.DataFrame(
        pca.fit_transform(test_conq_combined_embs),
        index=common_test_idx
    )

    return common_scnet_embs, common_scgpt_embs, avg_combined_embs, conq_combined_embs, common_labels

