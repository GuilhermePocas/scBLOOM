import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import mlflow


SCGPT_EPOCHS=5
SCNET_EPOCHS=15
AVG_EPOCHS=7
CONQ_EPOCHS=5
METHOD_COLOR = "Reds"

Methods_dic={
    'scNET' : (15, 'Blues'),
    'scGPT' : (5, 'Oranges'),
    'avg' : (7, 'Reds'),
    'conq' : (5, 'Reds')
}

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def test_embeddings( cell_embeddings, labels, method, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(f"{save_dir}/{method}_results.txt", "w+") as f:
        sys_stdout_backup = sys.stdout
        sys.stdout = f

        num_folds = 5
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        X = cell_embeddings
        y = labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_losses = []

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, drop_last=False)

            # Initialize model
            input_size = X.shape[1]
            num_classes = len(np.unique(y))
            model = Classifier(input_size, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model
            num_epochs = Methods_dic[method][0]
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_X, batch_y in train_loader:
                    
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    _, predicted_labels = torch.max(outputs, 1)
                    correct_predictions += (predicted_labels == batch_y).sum().item()
                    total_predictions += batch_y.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
                accuracy = correct_predictions / total_predictions

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")


            model.eval()
            with torch.inference_mode():
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                
                outputs = model(X_test_tensor)
                test_loss = criterion(outputs, y_test_tensor)

                _, predicted = torch.max(outputs, 1)
                y_true = y_test_tensor.cpu().numpy()
                y_pred = predicted.cpu().numpy()

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
                recall = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                
                print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

                fold_losses.append(test_loss.item())
                fold_accuracies.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

        print("\nFinal Cross-Validation Results:")
        print(f"Mean Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"Mean Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
        print(f"Mean Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
        print(f"Mean F1-Score: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        sys.stdout = sys_stdout_backup

        mlflow.log_metric(f"{method} - Classification Loss", np.mean(fold_losses))
        mlflow.log_metric(f"{method} - Classification Accuracy ", np.mean(fold_accuracies))
        mlflow.log_metric(f"{method} - Classification Precision", np.mean(fold_precisions))
        mlflow.log_metric(f"{method} - Classification Recall", np.mean(fold_recalls))
        mlflow.log_metric(f"{method} - Classification F1-Score", np.mean(fold_f1s))

        true_labels_decoded = label_encoder.inverse_transform(np.array(all_y_true))
        pred_labels_decoded = label_encoder.inverse_transform(np.array(all_y_pred))

        make_conf_matrix(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, Methods_dic[method][1])
        sys.stdout = sys_stdout_backup


def test_scNET( cell_embeddings, labels, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(save_dir + "/scNET_results2.txt", "w+") as f:
        sys_stdout_backup = sys.stdout
        sys.stdout = f
        num_folds = 5
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        X = cell_embeddings
        y = labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_losses = []

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, drop_last=False)

            # Initialize model
            input_size = X.shape[1]
            num_classes = len(np.unique(y))
            model = Classifier(input_size, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model
            num_epochs = SCNET_EPOCHS
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_X, batch_y in train_loader:
                    
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    _, predicted_labels = torch.max(outputs, 1)
                    correct_predictions += (predicted_labels == batch_y).sum().item()
                    total_predictions += batch_y.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
                accuracy = correct_predictions / total_predictions

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")


            model.eval()
            with torch.inference_mode():
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                
                outputs = model(X_test_tensor)
                test_loss = criterion(outputs, y_test_tensor)

                _, predicted = torch.max(outputs, 1)
                y_true = y_test_tensor.cpu().numpy()
                y_pred = predicted.cpu().numpy()

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
                recall = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                
                print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

                fold_losses.append(test_loss.item())
                fold_accuracies.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

        print("\nFinal Cross-Validation Results:")
        print(f"Mean Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"Mean Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
        print(f"Mean Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
        print(f"Mean F1-Score: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        sys.stdout = sys_stdout_backup

        mlflow.log_metric("scNet - Classification Loss", np.mean(fold_losses))
        mlflow.log_metric("scNet - Classification Accuracy ", np.mean(fold_accuracies))
        mlflow.log_metric("scNet - Classification Precision", np.mean(fold_precisions))
        mlflow.log_metric("scNet - Classification Recall", np.mean(fold_recalls))
        mlflow.log_metric("scNet - Classification F1-Score", np.mean(fold_f1s))

        true_labels_decoded = label_encoder.inverse_transform(np.array(all_y_true))
        pred_labels_decoded = label_encoder.inverse_transform(np.array(all_y_pred))

        make_conf_matrix(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, METHOD_COLOR)
        #plot_conf_matrix_pvalues(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, "scNET2", "Blues")

def test_scGPT_cross(cell_embeddings, labels, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(save_dir + "/scGPT_results2.txt", "w+") as f:
        sys_stdout_backup = sys.stdout
        sys.stdout = f

        num_folds = 5
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        X = cell_embeddings

        y = labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_losses = []

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size = 30, shuffle=True, drop_last=False)


            # Initialize model
            input_size = X.shape[1]
            num_classes = len(np.unique(y))
            model = Classifier(input_size, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model
            num_epochs = SCGPT_EPOCHS
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_X, batch_y in train_loader:
                    
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    _, predicted_labels = torch.max(outputs, 1)
                    correct_predictions += (predicted_labels == batch_y).sum().item()
                    total_predictions += batch_y.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
                accuracy = correct_predictions / total_predictions

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")


            model.eval()
            with torch.inference_mode():
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                
                outputs = model(X_test_tensor)
                test_loss = criterion(outputs, y_test_tensor)

                _, predicted = torch.max(outputs, 1)
                y_true = y_test_tensor.cpu().numpy()
                y_pred = predicted.cpu().numpy()

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
                recall = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                
                print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

                fold_losses.append(test_loss.item())
                fold_accuracies.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

        print("\nFinal Cross-Validation Results:")
        print(f"Mean Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"Mean Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
        print(f"Mean Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
        print(f"Mean F1-Score: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        sys.stdout = sys_stdout_backup

        mlflow.log_metric("scGPT - Classification Loss", np.mean(fold_losses))
        mlflow.log_metric("scGPT - Classification Accuracy ", np.mean(fold_accuracies))
        mlflow.log_metric("scGPT - Classification Precision", np.mean(fold_precisions))
        mlflow.log_metric("scGPT - Classification Recall", np.mean(fold_recalls))
        mlflow.log_metric("scGPT - Classification F1-Score", np.mean(fold_f1s))

        true_labels_decoded = label_encoder.inverse_transform(np.array(all_y_true))
        pred_labels_decoded = label_encoder.inverse_transform(np.array(all_y_pred))

        make_conf_matrix(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, "Oranges")
        #plot_conf_matrix_pvalues(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, "scGPT2", "Reds")


def test_combined_cross(cell_embeddings, labels, save_dir, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(save_dir + f"/combined_results2_{name}.txt", "w+") as f:
        sys_stdout_backup = sys.stdout
        sys.stdout = f

        num_folds = 5
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        X = cell_embeddings

        y = labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_losses = []

        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx] 
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if name == "conq":
                pca = PCA(n_components=512)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, drop_last=False)


            # Initialize model
            input_size = X_train.shape[1]
            num_classes = len(np.unique(y))
            model = Classifier(input_size, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model
            if name == "conq":
                num_epochs = CONQ_EPOCHS
            else:
                num_epochs = AVG_EPOCHS
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_X, batch_y in train_loader:
                    
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    _, predicted_labels = torch.max(outputs, 1)
                    correct_predictions += (predicted_labels == batch_y).sum().item()
                    total_predictions += batch_y.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                
                accuracy = correct_predictions / total_predictions

                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")


            model.eval()
            with torch.inference_mode():
                X_test_tensor = X_test_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                
                outputs = model(X_test_tensor)
                test_loss = criterion(outputs, y_test_tensor)

                _, predicted = torch.max(outputs, 1)
                y_true = y_test_tensor.cpu().numpy()
                y_pred = predicted.cpu().numpy()

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
                recall = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                
                print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

                fold_losses.append(test_loss.item())
                fold_accuracies.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)

                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

        print("\nFinal Cross-Validation Results:")
        print(f"Mean Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"Mean Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
        print(f"Mean Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
        print(f"Mean F1-Score: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        sys.stdout = sys_stdout_backup

        mlflow.log_metric("combined - Classification Loss", np.mean(fold_losses))
        mlflow.log_metric("combined - Classification Accuracy ", np.mean(fold_accuracies))
        mlflow.log_metric("combined - Classification Precision", np.mean(fold_precisions))
        mlflow.log_metric("combined - Classification Recall", np.mean(fold_recalls))
        mlflow.log_metric("combined - Classification F1-Score", np.mean(fold_f1s))

        true_labels_decoded = label_encoder.inverse_transform(np.array(all_y_true))
        pred_labels_decoded = label_encoder.inverse_transform(np.array(all_y_pred))
        
        if name == "conq":
            make_conf_matrix(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, METHOD_COLOR)
        else:
            make_conf_matrix(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, METHOD_COLOR)
        #plot_conf_matrix_pvalues(true_labels_decoded, pred_labels_decoded, label_encoder.classes_, save_dir, f"combined2_{name}", "Purples")



def make_conf_matrix(y_true, y_pred, labels, save_dir, color):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_df = pd.DataFrame(
        cm_normalized,
        index=labels,
        columns=labels
    )
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm_df,
        annot=True,
        fmt=".2f",
        cmap=color,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 18}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
    ax.set_xlabel("Predicted Label", fontsize=18)
    ax.set_ylabel("True Label", fontsize=18)
    #plt.title(f"{name} Confusion Matrix", fontsize=24)
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(save_dir, f"{name} confusion_matrix.png"), dpi=500)
    plt.close()
