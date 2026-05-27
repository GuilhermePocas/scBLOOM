import os
from .scNET import run_scNET, load_embeddings, create_reconstructed_obj
import scanpy as sc
import pandas as pd
import faulthandler
faulthandler.enable()
from sklearn.model_selection import train_test_split
from .scgpt import run_scGPT
import pickle



def train_scnet(obj, dir, cfg):
    obj_scNET = obj.copy()
    obj_scNET.raw = obj_scNET

        
    scNET_dir = dir + "/scNET"
    os.makedirs(scNET_dir, exist_ok=True)
    run_scNET(cfg["annotation_file"], obj_scNET, scNET_dir ,pre_processing_flag=cfg["pre_processing_flag"], human_flag=cfg["human_flag"],
                        number_of_batches=cfg["number_of_batches"], split_cells=cfg["split_cells"], max_epoch=cfg["max_epoch"],
                        model_name = cfg["model_name"], clf_loss=cfg["clf_loss"])

    embedded_genes, embedded_cells, node_features , out_features, ids =  load_embeddings(cfg["model_name"], scNET_dir)
    recon_obj = create_reconstructed_obj(node_features, out_features, obj_scNET)

    scnet_cell_embeddings = pd.DataFrame(embedded_cells, index=recon_obj.obs_names)
    scnet_labels = recon_obj.obs['celltype']

    return scnet_cell_embeddings, scnet_labels


def train_scgpt(obj, dir, cfg, training_obj=None):

    obj_scGPT = obj.copy()
    obj_scGPT.var_names = obj_scGPT.var_names.str.upper()

    if training_obj is None:

        train_idx, test_idx = train_test_split(
            obj_scGPT.obs_names,
            test_size=cfg['test_size'],
            stratify=obj.obs["celltype"],  # Ensure class balance
            random_state=42
        )

        obj_scGPT_test = obj_scGPT[test_idx].copy()
        obj_scGPT_train = obj_scGPT[train_idx].copy()

    else:

        obj_scGPT_train = training_obj
        obj_scGPT_test = obj_scGPT.copy()

    #obj_scGPT_train.obs["batch_id"] = obj_scGPT_train.obs["str_batch"] = "0"
    #obj_scGPT_test.obs["batch_id"] = obj_scGPT_test.obs["str_batch"] = "1"
    obj_scGPT_batch = obj_scGPT_train.concatenate(obj_scGPT_test, batch_key="str_batch")

    scGPT_dir = dir + "/scGPT"
    os.makedirs(scGPT_dir, exist_ok=True)
    run_scGPT(cfg["model_name"], cfg ,obj_scGPT_batch, scGPT_dir)

    with open(scGPT_dir + "/test_cell_embeddings.pkl", "rb") as f:
        test_cell_embeddings = pickle.load(f)
    scgpt_cell_embeddings = pd.DataFrame.from_dict(test_cell_embeddings, orient="index")
    scgpt_cell_embeddings.index.name = "cell_id"

    scgpt_labels = obj_scGPT_batch.obs.loc[scgpt_cell_embeddings.index, "celltype"].values
    scgpt_cell_embeddings.index = scgpt_cell_embeddings.index.str.replace(r"-[01]$", "", regex=True)

    return scgpt_cell_embeddings, scgpt_labels

def combine_embeddings(obj, scnet_emb, scgpt_emb, dir):

    common_test_idx = scgpt_emb.index.intersection(scnet_emb.index)
    obj_common = obj[common_test_idx].copy()
    print(f"Test cells in common: {len(common_test_idx)} / {len(scgpt_emb.index)}")

    # Subset to only common cells
    common_scnet_embs = scnet_emb.loc[common_test_idx]
    common_scgpt_embs = scgpt_emb.loc[common_test_idx]

    #subset labels
    common_labels = obj.obs.loc[common_test_idx, "celltype"].values

    avg_combined_embs = pd.DataFrame((common_scgpt_embs.values + common_scnet_embs.values) / 2, index=common_test_idx)
    conq_combined_embs = pd.concat([common_scgpt_embs, common_scnet_embs], axis=1)

    obj_common.X = obj_common.X.toarray()
    obj_common.obsm["X_scnet"] = common_scnet_embs.values
    obj_common.obsm["X_scgpt"] = common_scgpt_embs.values
    obj_common.obsm["X_combined_avg"] = avg_combined_embs.values
    obj_common.obsm["X_combined_conc"] = conq_combined_embs.values

    scRepresenter_dir = dir + "/scRepresenter"
    os.makedirs(scRepresenter_dir, exist_ok=True)
    obj_common.write(scRepresenter_dir + "/Embs.h5ad")

    return common_scnet_embs, common_scgpt_embs, avg_combined_embs, conq_combined_embs, common_labels



