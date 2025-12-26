"""
retriving candidate items for users before ranking
for offine evaluation: use the test set users and all items in your catalog (or test items).
For online serving: use all active users and all items in your catalog.

Embeddings	     |   Use data for recall?
User embeddings	 |   Users you want recommendations for (test or real users)
Item embeddings	 |   All items available for recommendation (catalog)

embedding extraction -> vector index -> recall results
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import redis
from tqdm import tqdm

from FM_trainer import FMTrainer,RecSysDataset
from FM_layer import FMLayer

if __name__ == "__main__":
    # load index2id mapping
    index2id = {}
    with open("../data/index","r") as f:
        for line in f:
            id, idx = line.strip().split('\t')
            index2id[int(idx)] = id

    # set parameters
    epochs = 2
    batch_size = 100
    feature_num = 20597  # if will base on the len(index)
    embedding_size = 16
    lr = 0.01
    device = "mps"
    user_fields = [0, 2, 3]
    item_fields = [1, 4, 5, 6]

    # Load trained pytorch model
    model = FMLayer(feature_num, embedding_size,user_fields,item_fields).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trainer
    trainer = FMTrainer(model, loss_fn, optimizer, device=device)

    # Load model
    trainer.load_model()

    model.eval()

    # Prepare test data and dataloader
    test_df = pd.read_csv("../data/test/test.csv").to_numpy()
    test_dataset = RecSysDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Extract embeddings and build user/item dictionaries
    user_emb_dic = {}
    item_emb_dic = {}

    with torch.no_grad():
        for features,_ in tqdm(test_dataloader):
            features = features.to(device)

            user_index = features[:, user_fields[0]].cpu().numpy()
            item_index = features[:, item_fields[0]].cpu().numpy()

            # get embedding
            _, user_emb, item_emb = model(features)

            user_emb = user_emb.cpu().numpy()
            item_emb = item_emb.cpu().numpy()

            # Fill dictionaries
            for k, v in zip(user_index, user_emb):
                user_emb_dic[index2id[int(k)]] = v
            for k, v in zip(item_index, item_emb):
                item_emb_dic[index2id[int(k)]] = v
    # Check
    # for k, v in user_emb_dic.items():
    #     print("User:", k, "Embedding:", v[:5], "...")  # show first 5 dimensions
    #     break

    # Build item embedding for BallTree
    item_ids = list(item_emb_dic.keys())
    item_embs = np.stack([item_emb_dic[i] for i in item_ids])

    # Normalize embeddings if using cosine similarity
    item_embs_norm = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # Build BallTree for fast nearest neighbor search
    tree = BallTree(item_embs_norm, metric='euclidean',leaf_size=10)

    # Test query (optional)
    dist, ind = tree.query([item_embs[0]], k=20)
    print("Top-20 nearest items indices for first item:", ind)

    # ------------------------------
    # Write recall results to Redis and file
    # ------------------------------
    pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=1)
    r = redis.Redis(connection_pool=pool)

    recall_result_file = "../data/fm_recall.result"
    with open(recall_result_file, "w") as fout:
        for user_id, user_emb in user_emb_dic.items():

            # Normalize if using cosine similarity
            user_emb_norm = user_emb / np.linalg.norm(user_emb)

            # Query top-20 items for this user
            dist, ind = tree.query(user_emb_norm.reshape(1,-1), k=20)

            top_items = [item_ids[i] for i in ind[0]]

            # Save to Redis
            r.set(str(user_id), ",".join(top_items), nx=True)
            # Save to file
            fout.write(f"{user_id}\t{','.join(top_items)}\n")

    print(f"Recall results saved to {recall_result_file}")





