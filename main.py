import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
os.makedirs('/tmp/numba_cache', exist_ok=True)
os.makedirs('/tmp/matplotlib_config', exist_ok=True)
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
from test_embeddings_cross import test_scNET, test_scGPT_cross, test_combined_cross
import pickle
from sklearn.decomposition import PCA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")

parser = argparse.ArgumentParser(description="Run scNET with custom parameters.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
parser.add_argument("--dataset", type=str, required=True, help="Dataset used")
parser.add_argument("--scnet_epochs", type=int, help="Number of training epochs", default=1)
parser.add_argument("--scgpt_epochs", type=int, help="Number of training epochs", default=1)
args = parser.parse_args()

SAVE_DIR = "output/" + args.dataset + "/" + args.model_name
os.makedirs(SAVE_DIR, exist_ok=True)

#file for gene similarity graph, in Data folder of scNET
ANN_FILE = "Data/pbmc-commonGOh_interaction_network.csv"

mlflow.set_experiment(args.model_name)

##pbmc3k
if args.dataset == "pbmc":
    obj = sc.read("./data/pbmc3k/pbmc3k_raw.h5ad")
    obj.var_names = obj.var_names.str.capitalize()
    obj_proc = sc.read("./data/pbmc3k/pbmc3k_processed.h5ad")
    obj_proc.var_names = obj_proc.var_names.str.capitalize()
    obj.obs['celltype'] = obj.obs_names.map(lambda x: obj_proc.obs['louvain'].get(x, "unknown"))
    obj = obj[obj.obs['celltype'] != "unknown"]
    obj.var_names_make_unique()

obj_scNET = obj.copy()
obj_scNET.raw = obj_scNET

if args.scnet_epochs > 0:
    ## TRAIN SCNET ###
    with mlflow.start_run(run_name="scNET"):
        mlflow.log_param("model", "scNET")
        mlflow.log_param("epochs", args.scnet_epochs)
        mlflow.log_param("model_name", args.model_name)
        
        scNET_dir = SAVE_DIR + "/scNET"
        os.makedirs(scNET_dir, exist_ok=True)
        scNET.run_scNET(ANN_FILE, obj_scNET, scNET_dir ,pre_processing_flag=True, human_flag=False, number_of_batches=1, split_cells=True,
                        max_epoch=args.scnet_epochs, model_name = args.model_name, clf_loss=False)

    embedded_genes, embedded_cells, node_features , out_features, ids =  scNET.load_embeddings(args.model_name, scNET_dir)
    recon_obj = scNET.create_reconstructed_obj(node_features, out_features, obj_scNET)


##### TRAIN SCGPT ######
if args.scgpt_epochs > 0:
    parameters = dict(
        seed=0,
        dataset_name="pbmc",
        do_train=True,
        load_model="./scgpt/checkpoints/scGPT_human",
        mask_ratio=0,
        epochs=args.scgpt_epochs,
        n_bins=51,
        MVC=False, # Masked value prediction for cell embedding
        ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=20,
        layer_size=75,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.3,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene = False,
        freeze = False, #freeze
        DSBN = False,  # Domain-spec batchnorm
    )


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
        for k, v in parameters.items():
            mlflow.log_param(k, v)

        scGPT_dir = SAVE_DIR + "/scGPT"
        os.makedirs(scGPT_dir, exist_ok=True)
        run_scGPT(args.model_name, parameters ,obj_scGPT, scGPT_dir)

    with open(scGPT_dir + "/train_cell_embeddings.pkl", "rb") as f:
        train_cell_embeddings = pickle.load(f)
    train_scgpt = pd.DataFrame.from_dict(train_cell_embeddings, orient="index")
    train_scgpt.index.name = "cell_id"

    with open(scGPT_dir + "/test_cell_embeddings.pkl", "rb") as f:
        test_cell_embeddings = pickle.load(f)
    test_scgpt = pd.DataFrame.from_dict(test_cell_embeddings, orient="index")
    test_scgpt.index.name = "cell_id"

    train_labels = obj_scGPT.obs.loc[train_scgpt.index, "celltype"].values
    test_labels = obj_scGPT.obs.loc[test_scgpt.index, "celltype"].values

results_dir = SAVE_DIR + "/results"
os.makedirs(results_dir, exist_ok=True)

if args.scnet_epochs > 0 and args.scgpt_epochs > 0:
    ##Prepare combined embeddings
    emb_scnet = pd.DataFrame(embedded_cells, index=recon_obj.obs_names)

    test_scgpt.index = test_scgpt.index.str.replace(r"-[01]$", "", regex=True)
    obj_scGPT.obs.index = obj_scGPT.obs.index.str.replace(r"-[01]$", "", regex=True)

    common_test_idx = test_scgpt.index.intersection(emb_scnet.index)

    print(f"Test cells in common: {len(common_test_idx)} / {len(test_scgpt.index)}")


    # Subset to only common cells
    test_scgpt_common = test_scgpt.loc[common_test_idx]
    test_scnet_common = emb_scnet.loc[common_test_idx]
    print(test_scnet_common.shape)

    #subset labels
    test_labels_common = obj_scGPT.obs.loc[common_test_idx, "celltype"].values
    print(test_labels_common.shape)

    test_avg_combined_embs = (test_scgpt_common.values + test_scnet_common.values) / 2
    test_conq_combined_embs = pd.concat([test_scgpt_common, test_scnet_common], axis=1).values

    #pca = PCA(n_components=512)
    #test_conq_pca = pca.fit_transform(test_conq_combined_embs)  # Or concatenate all embeddings

    #test_scnet_pca = pca.transform(test_scnet_common)
    #test_scgpt_pca = pca.transform(test_scgpt_common.values)
    #test_avg_pca   = pca.transform(test_avg_combined_embs)

    #pca = PCA(n_components=512)
    #pca.fit(test_conq_combined_embs)
    #test_conq_pca = pca.transform(test_conq_combined_embs)

with mlflow.start_run(run_name="Combined"):
    mlflow.log_param("model", "Combined")
    #test individual embeddings
    if args.scnet_epochs > 0:
      print("Testing scNET")
      test_scNET(test_scnet_common, test_labels_common, results_dir)
    if args.scgpt_epochs > 0:
      print("Testing scGPT")
      test_scGPT_cross(test_scgpt_common.values, test_labels_common, results_dir)
    if args.scnet_epochs > 0 and args.scgpt_epochs > 0:
        print("Testing Combined Avg")
        test_combined_cross(test_avg_combined_embs, test_labels_common, results_dir, "avg")
        print("Testing Combined Conq")
        test_combined_cross(test_conq_combined_embs, test_labels_common, results_dir, "conq")



print("Finished")