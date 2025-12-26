import os
import json
import torch
import torch.nn as nn
from flask import Flask, request
from base_layer import DSSMLayer
from Dssm_trainer import DSSMDataset,DSSMTrainer
from sklearn.neighbors import BallTree

class OnlineRecall():
    def __init__(self, device='mps'):
        self.device = device

        # load id2index mapping
        self.id2index = {}
        with open("../data/index") as f:
            for line in f:
                key, idx = line.strip().split("\t")
                self.id2index[key] = int(idx)

        # set parameters
        epochs = 2
        batch_size = 100
        # feature_num = 20597  # if will base on the len(index)
        feature_num = len(self.id2index)
        embedding_size = 16
        lr = 0.01
        device = "mps"
        mlp_units = [64, 32]
        dropout = 0.2
        use_bn = True
        user_fields = [0, 2, 3]
        item_fields = [1, 4, 5, 6]

        # Init Model & Trainer
        self.model = DSSMLayer(feature_num, embedding_size, user_fields, item_fields).to(device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # # Trainer
        self.trainer = DSSMTrainer(self.model, self.loss_fn, self.optimizer, device=device)

        # load trained model
        self.trainer.load_model()
        self.model.eval()

        # ---------- Load test dataset (item features) ----------
        self.item_feature_dict = {}
        # with open("../data/raw_data/item_feature.dat", encoding="utf-8") as f:
        #     for line in f:
        #         line = line.strip().split(",")
        #         self.item_feature_dict[line[0]] = [line[1], line[2], line[3]]
        #         # ---------- Build item embedding dictionary ----------
        #         item_emb_dic = {}
        #         with torch.no_grad():
        #             for item_id, feats in self.item_feature_dict.items():
        #                 row = [
        #                     self.id2idx(item_id),
        #                     self.id2idx(feats[0]),
        #                     self.id2idx(feats[1]),
        #                     self.id2idx(feats[2]),
        #                 ]
        #                 item_tensor = torch.tensor([row], dtype=torch.long, device=self.device)
        #                 _, _, item_emb = self.model(None, item_tensor, infer_item=True)
        #                 item_emb_dic[item_id] = item_emb.squeeze().cpu().numpy()
        #
        #         # ---------- Build BallTree ----------
        #         self.item_ids = list(item_emb_dic.keys())
        #         item_emb_array = np.array(list(item_emb_dic.values()))
        #         self.tree = BallTree(item_emb_array, leaf_size=10)
        #
        #         print(f"✅ Loaded {len(self.item_ids)} item embeddings for recall index.")

    def recall(self, user_features, k=20):
        """Given user features, find top-k similar items."""
        with torch.no_grad():
            # Build tensor for user features
            row = [
                user_features["user"][0][0],
                user_features["utag1"][0][0],
                user_features["utag2"][0][0],
            ]
            user_tensor = torch.tensor([row], dtype=torch.long, device=self.device)

            user_emb, _, _ = self.model(user_tensor, None, infer_user=True)
            user_emb_np = user_emb.cpu().numpy()

            dist, ind = self.tree.query(user_emb_np, k=k)
            recall_items = [self.item_ids[i] for i in ind[0]]
        return recall_items

class Predict:
    def __init__(self, device="mps"):
        self.device = device

        # ---------- Load id2index ----------
        self.id2index = {}
        with open("../data/index") as f:
            for line in f:
                key, idx = line.strip().split("\t")
                self.id2index[key] = int(idx)

        # ---------- Load user features ----------
        self.user_feature_dict = {}
        with open("../data/raw_data/user_feature.dat", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(",")
                self.user_feature_dict[line[0]] = [line[1], line[2]]
        # Item features
        self.item_feature_dict = {}
        with open("../data/raw_data/item_feature.dat", encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(",")
                self.item_feature_dict[line[0]] = [line[1], line[2], line[3]]

        # ---------- Init online recall model ----------
        self.recaller = OnlineRecall(device=device)

    def recall(self, user_id, k=20):
        """Perform DSSM-based online recall."""
        uf = self.user_feature_dict.get(user_id, ["utag1", "utag2"])
        X = {
            "user": [[self.id2index[user_id]]],
            "utag1": [[self.id2index[uf[0]]]],
            "utag2": [[self.id2index[uf[1]]]],
        }
        return self.recaller.recall(X, k=k)

# ----------------- Flask Server -----------------
app = Flask(__name__)
pred = Predict(device="mps")

@app.route("/predict", methods=["POST"])
def infer():
    return_dict = {}
    if request.get_data() is None:
        return_dict["errcode"] = 1
        return_dict["errdesc"] = "data is None"
        return json.dumps(return_dict)

    data = json.loads(request.get_data())
    user_id = data.get("user_id", None)
    req_type = data.get("type", "recall")

    if user_id is not None:
        if req_type == "recall":
            res = pred.recall(user_id)
            print(res)
            return json.dumps(res)

# Recall
# curl http://127.0.0.1:5000/predict -d '{"user_id":"8101331121422814073","type":"recall"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"10000797410342149707","type":"recall"}'

if __name__ == "__main__":
    app.run(debug=True)