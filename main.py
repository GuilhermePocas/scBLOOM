import os
import argparse
import faulthandler
import torch
from utils import load_dataset, train_scnet, train_scgpt, combine_embeddings
import mlflow
from test_embeddings import test_embeddings

faulthandler.enable()

def parse_args():

    parser = argparse.ArgumentParser(description="Run scNET with custom parameters.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--network", type=str, required=True, help="Dataset used")
    parser.add_argument("--scnet_epochs", type=int, help="Number of training epochs", default=0)
    parser.add_argument("--scgpt_epochs", type=int, help="Number of training epochs", default=0)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # --- Set up directories ---
    SAVE_DIR = "output/" + args.model_name
    os.makedirs(SAVE_DIR, exist_ok=True)
    ANN_FILE = "../networks/" + args.network

    # --- Set up MLflow ---
    mlflow.set_experiment(args.model_name)

    # --- load data ---
    file = "./data/pbmc3k/pbmc3k_raw.h5ad"
    processed_file = "./data/pbmc3k/pbmc3k_processed.h5ad"
    obj = load_dataset(file, processed_file)


    ## TRAIN SCNET ###
    if args.scnet_epochs > 0:

        parameters_scnet = dict(
            annotation_file=ANN_FILE,
            pre_processing_flag=True,
            human_flag=False,
            number_of_batches=1,
            split_cells=True,
            max_epoch=args.scnet_epochs,
            model_name=args.model_name,
            clf_loss=False,
        )

        scnet_cell_embeddings = train_scnet(obj, SAVE_DIR, parameters_scnet)
        


    ##### TRAIN SCGPT ######
    if args.scgpt_epochs > 0:
        parameters_scgpt = dict(
            model_name=args.model_name,
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

        scgpt_cell_embeddings = train_scgpt(obj, SAVE_DIR, parameters_scgpt)
        

    results_dir = SAVE_DIR + "/results"
    os.makedirs(results_dir, exist_ok=True)

    #### COMBINE BOTH EMBEDDINGS #####
    if args.scnet_epochs > 0 and args.scgpt_epochs > 0:

        common_scnet_embs, common_scgpt_embs, avg_combined_embs, conq_combined_embs, common_labels = \
        combine_embeddings(obj, scnet_cell_embeddings, scgpt_cell_embeddings)


    #### CLASSICATION TESTS #####
    with mlflow.start_run(run_name="Combined"):
        mlflow.log_param("model", "Combined")
        if args.scnet_epochs > 0:
            print("Testing scNET")
            test_embeddings(common_scnet_embs, common_labels, 'scNET',results_dir)
        
        if args.scgpt_epochs > 0:
            print("Testing scGPT")
            test_embeddings(common_scgpt_embs, common_labels, 'scGPT', results_dir)

        if args.scnet_epochs > 0 and args.scgpt_epochs > 0:
            print("Testing Combined Avg")
            test_embeddings(avg_combined_embs, common_labels, "avg", results_dir)
            print("Testing Combined Conq")
            test_embeddings(conq_combined_embs, common_labels, 'conq', results_dir)


    print("Finished all tasks successfully!!")

if __name__ == "__main__":
    main()