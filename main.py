import os
import argparse
import faulthandler
import scanpy as sc
from src.main import run_scBLOOM
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

def load_dataset(file, processed_file):
    obj = sc.read(file)
    obj.var_names = obj.var_names.str.capitalize()
    obj_proc = sc.read(processed_file)
    obj_proc.var_names = obj_proc.var_names.str.capitalize()

    obj.obs['celltype'] = obj.obs_names.map(lambda x: obj_proc.obs['louvain'].get(x, "unknown"))
    obj = obj[obj.obs['celltype'] != "unknown"]
    obj.var_names_make_unique()

    return obj

def main():
    args = parse_args()

    # --- Set up directories ---
    SAVE_DIR = "output/" + args.model_name
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Set up MLflow ---
    mlflow.set_experiment(args.model_name)

    # --- load data ---
    file = "./data/pbmc3k/pbmc3k_raw.h5ad"
    processed_file = "./data/pbmc3k/pbmc3k_processed.h5ad"
    obj = load_dataset(file, processed_file)

    common_scnet_embs, common_scgpt_embs, avg_combined_embs, conq_combined_embs, common_labels = \
        run_scBLOOM(args.model_name, obj, args.network, SAVE_DIR, args.scnet_epochs, args.scgpt_epochs)
    
    results_dir = SAVE_DIR + "/results"
    os.makedirs(results_dir, exist_ok=True)

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
            print("Testing scBLOOM Avg")
            test_embeddings(avg_combined_embs, common_labels, "avg", results_dir)
            print("Testing scBLOOM Conq")
            test_embeddings(conq_combined_embs, common_labels, 'conq', results_dir)


    print("Finished all tasks successfully!!")

if __name__ == "__main__":
    main()