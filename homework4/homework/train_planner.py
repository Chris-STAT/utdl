"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
#print("Time to train")

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch 
import torch.nn as nn
import torch.utils.tensorboard as tb
import torch.optim as optim

from .models import MLPPlanner, load_model, save_model
from .datasets.road_dataset import load_data

def train_planner(
    exp_dir: str = "logs",
    model_name: str = 'MLPPlanner',
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)


    train_data = load_data(dataset_path="drive_data/drive_data/train", 
                           transform_pipeline="default", 
                           return_dataloader=True,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=2)
    
    val_data = load_data(dataset_path="drive_data/drive_data/val", 
                         transform_pipeline="default", 
                         return_dataloader=True,
                         batch_size=batch_size,
                         shuffle=False,
                        num_workers=2)
    



    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    global_step = 0
    metrics = {"train_mse": [], "val_mse": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            track_left = batch["track_left"].to(device)   # shape(B, n_track, 2)
            track_right = batch["track_right"].to(device) # shape(B, n_track, 2)
            waypoints = batch["waypoints"].to(device)     # shape(B, n_waypoints, 2)
            mask = batch["waypoints_mask"].to(device)     # shape(B, n_waypoints)

            optimizer.zero_grad()
            predicted_waypoints = model(track_left, track_right)

            loss = loss_func(predicted_waypoints[mask], waypoints[mask])

            loss.backward()
            optimizer.step()

            metrics["train_mse"].append(loss.item())

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left = batch["track_left"].to(device)   # shape(B, n_track, 2)
                track_right = batch["track_right"].to(device) # shape(B, n_track, 2)
                waypoints = batch["waypoints"].to(device)     # shape(B, n_waypoints, 2)
                mask = batch["waypoints_mask"].to(device)     # shape(B, n_waypoints)
                predicted_waypoints = model(track_left, track_right)
                loss = loss_func(predicted_waypoints[mask], waypoints[mask])
                metrics["val_mse"].append(loss.item())

        # log average train and val mse to tensorboard
        epoch_train_mse = torch.as_tensor(metrics["train_mse"]).mean().item()
        epoch_val_mse = torch.as_tensor(metrics["val_mse"]).mean().item()

        logger.add_scalar("MSE/Train", epoch_train_mse, epoch)
        logger.add_scalar("MSE/Validation", epoch_val_mse , epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_mse={epoch_train_mse:.4f} "
                f"val_mse={epoch_val_mse:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    train_planner(**vars(args))