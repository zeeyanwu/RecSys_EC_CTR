"""
EC scene
电商场景深度学习模型：
deepfm, dssm
DNN = Deep Neural Network (a neural network with multiple hidden layers).

In recommendation systems, a DNN is usually a feed-forward fully connected network
 that learns nonlinear interactions among user/item/context features.

recall is about efficiency, while ranking is about accuracy.

task1: DSSM -0.88 is applied to recall
task2: deepFM -0.9196 is applied to sort
task 3: FM sorting


FMlayer:
Uses efficient formula:
0.5* ((∑v)^2−∑(v^2))
Returns [first_order, second_order] concatenated if needed.

MLP Layer
Uses nn.ModuleList for layers and optional BatchNorm1d.
Dropout applied in forward with F.dropout.
general MLP with optional BN, activation, dropout.

DeepFM Layer
– main model, taking user/item fields, embedding table, combining FM + DNN. (FMlayer + MLP)
forward output:
logit

DeepFm trainer
- model initialization: DeepFM
- loss and optimizer
- metrics tracking
- training loop
- save model: deepfm_best.pth


DSSM Layer
Each feature goes through a shared embedding.
User/item features concatenated → feed through tower → project to final embedding.
Output similarity via elementwise multiplication + sum → sigmoid.

forward output:
out, user_emb, item_emb

DSSM trianer
Deep Structured Semantic Models
- model initialization: DSSMLayer
- loss and optimizer
- metrics tracking
- training loop
- save model: dssm_best.pth

Advantages for recall:

Precompute item embeddings offline
- You can store all item_emb in memory or an index (BallTree, FAISS, etc.)
- At runtime, you only compute user_emb → fast nearest neighbor search
Efficient similarity search
- Dot product or cosine similarity is very fast
- Works even when item set is very large (millions)
Lightweight inference
- Each tower is small MLP → low latency

DSSM sacrifices some expressiveness to achieve scalability and speed, which is perfect for recall.

Ranking models (like DeepFM, Wide & Deep, DNNs, gradient boosted trees) can use all features:

Tradeoff:
- Slower to compute than DSSM → can’t handle millions of items online
- But much more precise for ordering the top items

DSSM: “deep” but shallow towers for embeddings
- DSSM uses two towers of MLPs (one for user, one for item) to map inputs to embeddings.
- Typical structure: 1–3 hidden layers per tower, often small (e.g., 64 → 32 → 8).
- Goal: produce compact embeddings for fast similarity search.
- Even though it’s technically a deep network (it has multiple layers), it’s optimized for efficiency, not expressive power.
- So in practice, DSSM is deep in architecture, but shallow in depth compared to modern DeepFM/DNN models.


DeepFM: truly “deep” for ranking
- DeepFM has a wide part + deep part:
- Wide part: memorizes feature interactions linearly
- Deep part: MLP layers capture high-order non-linear feature interactions
- Typically deeper and wider than DSSM towers: 3–5 layers, 128–512 neurons per layer, often with dropout, batch norm, etc.
- Goal: maximize ranking accuracy using all available features.

SoDeepFM is “deeper” and more expressive, but slower — not suitable for huge recall sets.


Model	Layers	                Depth	            Purpose
DSSM	1–3 MLP                 layers per tower	Shallow to moderate	Produce embeddings for recall efficiently
DeepFM	3–5+ layers, large MLP	Deeper	            Capture complex interactions for ranking / sorting

solution1
Dssm recall (offline) + DeepFm sort  (online)
recall: test_dataloader
load_model(dssm_best.pth) -> forward: out, user_emb, item_emb
-> user_emb_dict: {user_id: user_emb}
-> item_emb_dict :{item_id: item_emb}
->  item_emb -> BallTree -> nearest items index (20)
-> user_emb -> BallTree query -> item index -> item id :  [user_id, top 20 items id] save redis

sort:
-> get user_id
    -> recall: user_id -> redis loopup -> return: top items id
    -> sort: user_id + top items id ->  DeepFm forward -> {items_id : logits}



offline recall:
redis - recall
sort


Online recall:
dssm -
double_tower.py 部署 user tower 在线, item tower 离线
user- embedding - online 计算 结果在 balltree 中 不需要redis
emb_recall

online_recall:
build item embeddings, store BallTree, handle recall
-> index mappings index:id
-> trained model weights DSSm
-> item features -> build all item embeddings once
-> keeps a BallTree index for similarity search
-> build the user embedding
-> queires top-k neariest item vectors

predict : web server layer (Flash API)
"""
