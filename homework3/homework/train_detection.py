import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

import torch.optim as optim

from .models import ClassificationLoss, ClassificationLoss_detection, RegressionLoss, Detector, load_model, save_model
from .datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 70,
    lr: float = 1e-4,
    batch_size: int = 128,
    seed: int = 2024,
    loss_wgt: float = 0.5,
    class_wgt: list = [0.33,0.33,0.33],
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
    model.train()

    train_data = load_data("road_data/train", batch_size=batch_size, shuffle=True, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False, num_workers=2)

    # create loss function and optimizer

    class_weights = torch.tensor(class_wgt, dtype=torch.float).to(device)

    class_loss_func = ClassificationLoss_detection()
    reg_loss_func = RegressionLoss()
    # optimizer = ...
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    

    global_step = 0
    metrics = {"train_acc": [], "val_acc": [], "train_mse": [], "val_mse": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device)

            # TODO: implement training step
            # raise NotImplementedError("Training step not implemented")
            optimizer.zero_grad()
            logits, raw_depth = model(img)
            
            class_loss = class_loss_func(logits, track, weight=class_weights)
            reg_loss = reg_loss_func(raw_depth, depth)      
            
            loss =  loss_wgt*class_loss + (1-loss_wgt)*reg_loss

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == track).float().mean()
            metrics["train_acc"].append(accuracy.item())

            mse = reg_loss_func(raw_depth, depth).item()
            metrics["train_mse"].append(mse)

            global_step += 1
        
     

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in train_data:
                img = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                # TODO: compute validation accuracy
                #raise NotImplementedError("Validation accuracy not implemented")
                logits, raw_depth = model(img)
                _, predicted =  torch.max(logits, 1)
                accuracy = (predicted == track).float().mean()
                metrics["val_acc"].append(accuracy.item())

                mse = reg_loss_func(raw_depth, depth).item()
                metrics["val_mse"].append(mse)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        epoch_train_mse = torch.as_tensor(metrics["train_mse"]).mean()
        epoch_val_mse = torch.as_tensor(metrics["val_mse"]).mean()

        #raise NotImplementedError("Logging not implemented")
        logger.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        logger.add_scalar("Accuracy/Validation", epoch_val_acc, epoch)
        logger.add_scalar("MSE/Train", epoch_train_mse, epoch)
        logger.add_scalar("MSE/Validation", epoch_val_mse , epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
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
    parser.add_argument("--loss_wgt", type=float, default=0.5)
    parser.add_argument("--class_wgt", type=str, default="0.33,0.33,0.33")
    args = parser.parse_args()
    args.class_wgt = [float(w) for w in args.class_wgt.split(",")]
    train(**vars(args))