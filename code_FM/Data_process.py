"""
Outputs:
training set, test set, index
"""
import torch
import os
import pandas as pd
from typing import Dict, List

def get_slot(id) -> int:
    """
    Assgin a unique integer to each ID
    """
    if id not in slot:
        slot[id] = len(slot)
    return slot[id]

def load_features_dict(file_path, num_fields) ->Dict[str,list]:
    """Load user/item features from file into a dictionary."""
    feature_dict: Dict[str,List(str)] = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == num_fields:
                key, feature_values = parts[0],parts[1:num_fields]
                feature_dict[key] = feature_values
    return feature_dict

def to_example(line, user_dict,item_dict) -> Dict:
    """
    Convert one interaction line into a dictionary with slot IDs.
    """
    ts, user_id, item_id, ctr = int(line[0]),line[1],line[2],float(line[3])

    example = {
        'ts':ts,
        'uid':get_slot(user_id),
        'iid':get_slot(item_id),
        'ctr': ctr,
    }

    # Encode user features
    user_feats = user_dict.get(user_id,['utag1','utag2'])
    for tag_name, feat in zip(['utag1', 'utag2'], user_feats):
        example[tag_name] = get_slot(feat)

    # item features
    item_feats = item_dict.get(item_id, ["itag1", "itag2", "itag3"])
    for tag_name, feat in zip(["itag1", "itag2", "itag3"], item_feats):
        example[tag_name] = get_slot(feat)

    return example

def write_index(slot_map, index_path):
    with open(index_path,"w") as f:
        for key, value in slot_map.items():
            f.write(f'{key}\t{value}\n')

if __name__ == "__main__":
    # Paths
    data_path = os.path.join('..', 'data', 'raw_data')
    train_path = os.path.join('..', 'data', 'train','train.csv')
    test_path = os.path.join('..', 'data', 'test','test.csv')
    index_path = os.path.join('..', 'data','index')

    # Global slot mapping
    slot: Dict[str, int] = {}

    # load feature dictionaries
    user_feature_dict = load_features_dict(os.path.join(data_path,"user_feature.dat"), 3)
    item_feature_dict = load_features_dict(os.path.join(data_path,"item_feature.dat"), 4)

    # process and split
    data = []

    with open(os.path.join(data_path,"shop.dat"), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                features = to_example(parts, user_feature_dict, item_feature_dict)
                data.append(features)

    # Convert to DataFrame for easier train/test split
    df = pd.DataFrame(data)


    # Sort by user and timestamp
    df.sort_values(['uid', 'ts'], inplace=True)

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Split per user
    for user_id, user_df in df.groupby('uid'):
        latest_ts = user_df['ts'].max()
        # train_list.append(user_df[user_df['ts'] < latest_ts])
        # test_list.append(user_df[user_df['ts'] == latest_ts])

        train_data = pd.concat([train_data, user_df[user_df['ts'] < latest_ts]], ignore_index=True)
        test_data = pd.concat([test_data, user_df[user_df['ts'] == latest_ts]], ignore_index=True)

    # Save train/test dataset

    train_data = train_data.drop(train_data.columns[0], axis=1)
    test_data = test_data.drop(test_data.columns[0], axis=1)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(test_data)} test samples to {test_path}")

    # Save slot mapping
    write_index(slot, index_path)

    print(f"Index written to {index_path}")








