import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from base_layer import DSSMLayer
from sklearn.metrics import roc_auc_score


class DSSMDataset(Dataset):
    def __init__(self, dataset):
        """
        data: np.ndarray
        """
        self.features = torch.tensor(np.delete(dataset, 2, axis=1), dtype=torch.long)
        self.labels = torch.tensor(dataset[:, 2], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DSSMTrainer:
    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        # Hyperparams
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.best_auc = 0.0
        self.best_epoch = 0
        self.model_save_path = "../model"

    def train_epoch(self,train_loader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for features, labels in tqdm(train_loader):
            # move data to device
            features = features.to(device)
            labels = labels.to(device)
            # forward
            self.optimizer.zero_grad()
            out,_,_ = self.model.forward(features)
            preds = out.squeeze()
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples

        return avg_loss

    def evaluate(self, dev_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(dev_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                preds,_,_ = self.model.forward(features)

                all_preds.append(preds)
                all_labels.append(labels)
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        return roc_auc_score(all_labels, all_preds)

    def save_model(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "dssm_best.pth"))

    def load_model(self):
        self.model.load_state_dict(
            torch.load("../model/dssm_best.pth"))
        self.model.to(self.device)
        print("Model loaded")

    def infer(self, features):
        self.model.eval()
        # Convert numpy to torch tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).long()
        features = features.to(self.device)

        with torch.no_grad():
            preds,_,_ = self.model.forward(features)
            preds = (preds >= 0.5).int()
            preds = preds.cpu().numpy().squeeze()

        return preds

    def run(self, train_loader=None, val_loader=None, epochs=1, mode="train_and_eval"):
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

            # TRAINING
            if mode in ["train_and_eval", "train_only"]:
                if train_loader is None:
                    raise ValueError("train_loader must be provided for training mode.")
                avg_loss = self.train_epoch(train_loader)
                print(f"Train Loss: {avg_loss:.4f}")

            else:
                avg_loss = None  # no training

            # EVALUATION
            if mode in ["train_and_eval", "eval_only"]:
                if val_loader is None:
                    raise ValueError("val_loader must be provided for evaluation mode.")
                auc = self.evaluate(val_loader)
                print(f"Val AUC: {auc:.4f}")

                # Save best model
                if mode in ["train_and_eval", "train_only"]:
                    if auc > self.best_auc:
                        self.best_auc = auc
                        self.best_epoch = epoch + 1
                        self.save_model()




if __name__ == "__main__":
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_csv = os.path.join("..", "data", "train", "train.csv")
    test_csv = os.path.join("..", "data", "test", "test.csv")

    # Load Data
    train_df = pd.read_csv(train_csv).to_numpy()
    test_df = pd.read_csv(test_csv).to_numpy()

    # set parameters
    epochs = 10
    batch_size = 128
    feature_num = 20597  # if will base on the len(index)
    embedding_size = 16
    lr = 0.01
    device = "mps"
    user_fields = [0, 2, 3]
    item_fields = [1, 4, 5, 6]

    # Dataloader
    train_dataset = DSSMDataset(train_df)
    test_dataset = DSSMDataset(test_df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=False
    )

    # Model
    model = DSSMLayer(feature_num, embedding_size, user_fields, item_fields).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # Trainer
    trainer = DSSMTrainer(model, loss_fn, optimizer, device=device)
    #
    # Run
    trainer.run(train_dataloader, test_dataloader, epochs=epochs)

    # Load model
    trainer.load_model()

    # Prediction
    features = np.array([
        [0, 5280, 2, 3, 39, 5281, 53],
        [0, 4479, 2, 3, 12, 4480, 182],
        [0, 7903, 2, 3, 79, 7904, 518]
    ])
    predictions = trainer.infer(features)
    print("Predictions:", predictions)