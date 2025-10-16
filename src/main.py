import os
import argparse
import faulthandler
import torch
from .utils import train_scnet, train_scgpt, combine_embeddings, write_embeddings
import mlflow


def run_scBLOOM(model_name, obj, network, results_dir, scnet_epochs=0, scgpt_epochs=0):

    # --- Set up directories ---
    ANN_FILE = "../networks/" + network

    scnet_cell_embeddings = None
    scnet_labels = None
    scgpt_cell_embeddings = None
    scgpt_labels = None

        ## TRAIN SCNET ###
    if scnet_epochs > 0:

        parameters_scnet = dict(
            annotation_file=ANN_FILE,
            pre_processing_flag=True,
            human_flag=False,
            number_of_batches=1,
            split_cells=True,
            max_epoch=scnet_epochs,
            model_name=model_name,
            clf_loss=False,
        )

        scnet_cell_embeddings, scnet_labels = train_scnet(obj, results_dir, parameters_scnet)
        


    ##### TRAIN SCGPT ######
    if scgpt_epochs > 0:
        parameters_scgpt = dict(
            model_name=model_name,
            seed=0,
            dataset_name="pbmc",
            do_train=True,
            load_model="./src/scgpt/checkpoints/scGPT_human",
            mask_ratio=0,
            epochs=scgpt_epochs,
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

        scgpt_cell_embeddings, scgpt_labels = train_scgpt(obj, results_dir, parameters_scgpt)

    #### COMBINE BOTH EMBEDDINGS #####
    if scnet_epochs > 0 and scgpt_epochs > 0:

        common_scnet_embs, common_scgpt_embs, avg_combined_embs, conq_combined_embs, common_labels = \
        combine_embeddings(obj, scnet_cell_embeddings, scgpt_cell_embeddings)
        write_embeddings(avg_combined_embs, conq_combined_embs, results_dir)

        return common_scnet_embs, common_scgpt_embs, avg_combined_embs, conq_combined_embs, common_labels
    

    elif scnet_epochs > 0:
        return scnet_cell_embeddings, None, None, None, scnet_labels

    elif scgpt_epochs > 0:
        return None, scgpt_cell_embeddings, None, None, scgpt_labels

    else:
        print("⚠️ No training performed — both scnet_epochs and scgpt_epochs are 0.")
        return None

    