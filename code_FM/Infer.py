from FM_trainer import FMTrainer
from FM_layer import FMLayer
import torch
import torch.nn as nn

if __name__ == "__main__":
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
    model = FMLayer(feature_num, embedding_size,user_fields,item_fields).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Trainer
    trainer = FMTrainer(model,loss_fn, optimizer,device=device)

    # Load model
    trainer.load_model()

    # Example feature batch to predict
    import numpy as np
    features = np.array([
        [0,5280,2,3,39,5281,53],
        [0,4479,2,3,12,4480,182],
        [0,7903,2,3,79,7904,518]
        ])

    # Run inference
    predictions = trainer.infer(features)
    print("Predictions:", predictions)
