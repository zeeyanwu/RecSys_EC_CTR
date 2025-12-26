from Dssm_trainer import DSSMLayer, DSSMDataset, DSSMTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import redis

if __name__ == "__main__":
    # load index2id mapping
    index2id = {}
    with open("../data/index","r") as f:
        for line in f:
            key, idx = line.strip().split('\t')
            index2id[int(idx)] = key
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # set parameters
    epochs = 2
    batch_size = 100
    feature_num = 20597  # if will base on the len(index)
    embedding_size = 16
    lr = 0.01
    device = "mps"
    user_fields = [0, 2, 3]
    item_fields = [1, 4, 5, 6]

    # Model
    model = DSSMLayer(feature_num, embedding_size, user_fields, item_fields).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = DSSMTrainer(model, loss_fn, optimizer, device=device)

    trainer.load_model()

    # Prepare test data and dataloader
    test_df = pd.read_csv("../data/test/test.csv").to_numpy()
    test_dataset = DSSMDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Extract embeddings and build user/item dictionaries
    user_emb_dic = {}
    item_emb_dic = {}

    with torch.no_grad():
        for features, _ in tqdm(test_dataloader):
            features = features.to(device)

            user_index = features[:, user_fields[0]].cpu().numpy()
            item_index = features[:, item_fields[0]].cpu().numpy()

            # get embedding
            _,user_emb, item_emb = model.forward(features)

            user_emb = user_emb.cpu().numpy()
            item_emb = item_emb.cpu().numpy()

            # Fill dictionaries
            for k, v in zip(user_index, user_emb):
                user_emb_dic[index2id[int(k)]] = v
            for k, v in zip(item_index, item_emb):
                item_emb_dic[index2id[int(k)]] = v

    # Build item embedding for BallTree
    item_ids = list(item_emb_dic.keys())
    item_embs = np.stack([item_emb_dic[i] for i in item_ids])

    # Normalize embeddings if using cosine similarity
    item_embs_norm = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # Build BallTree for fast nearest neighbor search
    tree = BallTree(item_embs_norm, metric='euclidean', leaf_size=10)

    # Test query (optional)
    dist, ind = tree.query([item_embs[0]], k=20)
    print("Top-20 nearest items indices for first item:", ind)

    # ------------------------------
    # Write recall results to Redis and file
    # ------------------------------
    pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=2)
    r = redis.Redis(connection_pool=pool)

    recall_result_file = "../data/dssm_recall.result"
    with open(recall_result_file, "w") as fout:
        for user_id, user_emb in user_emb_dic.items():

            # Normalize if using cosine similarity
            user_emb_norm = user_emb / np.linalg.norm(user_emb)
            # Query top-20 items for this user
            dist, ind = tree.query(user_emb_norm.reshape(1, -1), k=20)

            top_items = [item_ids[i] for i in ind[0]]
            # Save to Redis
            r.set(str(user_id), ",".join(top_items), nx=True)
            # Save to file
            fout.write(f"{user_id}\t{','.join(top_items)}\n")

    print(f"Recall results saved to {recall_result_file}")







