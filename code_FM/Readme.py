"""
matrixcf 进行优化 fm
运用更多的特征
解决冷启动问题

raw_data:
shop.dat: ts, uid, iid, clk
user_feature.dat: uid, gender, age
item_feature.dat: iid, c1,c2,c3

code:
1. Data_process.py:
-> id to index
-> features(index), label
-> train_data, test_data (split by timestamp: history data for training, the latest data for test)
-> save slot

2. base_layer.py:
-> fm locig user_emb, item_emb(sum all feature embedding)

3. fm.py:
train loop
save model
infer

4.recall.py
user_emb, item_emb
vector server: BallTree
save to redis

5. sort.py
logits = model.forward()

"""