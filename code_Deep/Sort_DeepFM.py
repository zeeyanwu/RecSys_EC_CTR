import os
import json
import redis
import torch
import torch.nn as nn
from flask import Flask, request
from base_layer import FMLayer, MLP
from DeepFM_layer import DeepFM
from DeepFM_trainer import DeepFMDataset,DeepFMTrainer


class Predict(object):
    def __init__(self,device='mps'):
        self.device = device

        # load id2index mapping
        self.id2index = {}
        with open("../data/index") as f:
            for line in f:
                key, idx = line.strip().split("\t")
                self.id2index[key] = int(idx)

        # User features
        self.user_feature_dict = {}
        with open("../data/raw_data/user_feature.dat", encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(",")
                self.user_feature_dict[line[0]] = [line[1],line[2]]
        # Item features
        self.item_feature_dict = {}
        with open("../data/raw_data/item_feature.dat", encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(",")
                self.item_feature_dict[line[0]] = [line[1], line[2],line[3]]

        # set parameters
        epochs = 2
        batch_size = 100
        # feature_num = 20597  # if will base on the len(index)
        feature_num = len(self.id2index)
        embedding_size = 16
        lr = 0.01
        device = "mps"
        hidden_dims = [64, 8]
        dropout = 0.2
        use_bn = True
        user_fields = [0, 2, 3]
        item_fields = [1, 4, 5, 6]

        # Model
        self.model = DeepFM(feature_num, embedding_size, hidden_dims, dropout=0.2).to(device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Trainer
        self.trainer = DeepFMTrainer(self.model, self.loss_fn, self.optimizer, device=device)
        # load trained model
        self.trainer.load_model()

        # Redis connection
        pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=2)
        self.r = redis.Redis(connection_pool=pool)

    # Recall from Redis
    def recall(self, user):
        recall_result = self.r.get(user)
        if recall_result is None:
            return []
        return recall_result.decode().split(",")

    # Sort using PyTorch inference
    def sort(self, user, recall_result):
        """Rank the recall candidates using FM model"""
        feature_rows = []
        for item in recall_result:
            uf = self.user_feature_dict.get(user, ["utag1", "utag2"])
            itf = self.item_feature_dict.get(item, ["itag1", "itag2", "itag3"])

            row = [
                self.id2index[user],   # user id
                self.id2index[item],   # item id
                self.id2index[uf[0]],  # user tag1
                self.id2index[uf[1]],  # user tag2
                self.id2index[itf[0]], # item tag1
                self.id2index[itf[1]], # item tag2
                self.id2index[itf[2]], # item tag3
            ]
            feature_rows.append(row)

        # Convert to tensor
        features = torch.tensor(feature_rows, dtype=torch.long, device=self.device)

        # Predict scores
        with torch.no_grad():
            preds = self.model(features).squeeze().cpu().numpy()

        # Map items to predictions
        result = {recall_result[i]: float(preds[i]) for i in range(len(recall_result))}
        return result

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
            return json.dumps(res)
        if req_type == "sorted":
            res = pred.recall(user_id)
            res = pred.sort(user_id, res)
            return json.dumps(res)

    return json.dumps({"errcode": 2, "errdesc": "invalid request"})

# Recall
# curl http://127.0.0.1:5000/predict -d '{"user_id":"8101331121422814073","type":"recall"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"10000797410342149707","type":"recall"}'

# Sorted ranking
# curl http://127.0.0.1:5000/predict -d '{"user_id":"4015078819788459","type":"sorted"}'

if __name__ == "__main__":
    app.run(debug=True)