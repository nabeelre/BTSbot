import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import datetime
import wandb
import json
import time
import sys
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

from torch_utils import RandomRightAngleRotation, make_report, FlexibleDataset
import architectures_torch as architectures
import val_torch as val

# Print styling
BOLD = '\033[1m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
END = '\033[0m'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Define categories for model types based on their names
IMAGE_ONLY_MODELS = ['SwinV2']
METADATA_ONLY_MODELS = ['um_nn']
MULTIMODAL_MODELS = ['mm_SwinV2']


def sweep_train(config=None):
    with wandb.init(config=config) as run:
        run_training(run.config, run_name=run.name, sweeping=True)


def classic_train(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    run_training(config)


def perf_to_stdout(epoch, epochs, start_time, batch, batches, loss, acc, flush_stdout=True):
    sys.stdout.write(
        f"\r  {BOLD}epoch: {epoch+1}/{epochs}{END} " +
        f"t: {(time.time()-start_time):.2f}s " +
        f"[batch: {batch}/{batches}], " +
        f"{RED}train loss{END}: {loss:.5f}, " +
        f"{BLUE}train accuracy{END}: {acc:.5f}"
    )
    if flush_stdout:
        sys.stdout.flush()


def run_training(config, run_name: str = "", sweeping: bool = False):
    # Read parameters from config
    model_name = config['model_name']

    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    beta1 = config['beta_1']
    beta2 = config['beta_2']
    patience = config['patience']

    h_flip = bool(config["data_aug_h_flip"])
    v_flip = bool(config["data_aug_v_flip"])
    rot = bool(config["data_aug_rot"])

    random_state = config['random_seed']
    dataset_version = config['train_data_version']
    data_base_dir = config.get('data_base_dir', '')

    N_max_p = config.get('N_max', 100)
    N_max_n = N_max_p
    N_str = f"_N{N_max_p}"

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # /-----------------------------------/
    #    MODEL, OPTIMIZER, & LOSS SET UP
    # /-----------------------------------/

    need_triplets = (
        model_name in IMAGE_ONLY_MODELS or
        model_name in MULTIMODAL_MODELS
    )
    need_metadata = (
        model_name in METADATA_ONLY_MODELS or
        model_name in MULTIMODAL_MODELS
    )

    if not need_triplets and not need_metadata:
        print(
            f"{model_name} not categorized as image-only/metadata-only/multimodal."
        )
        exit(1)

    if need_metadata:
        metadata_cols = config.get('metadata_cols', None)
        if metadata_cols is None:
            print("Metadata columns not found in config.")
            exit(1)

    # Initialize model
    try:
        model_type = getattr(architectures, model_name)
    except AttributeError:
        print(f"{RED}Could not find model of name {model_name}{END}")
        exit(0)
    model = model_type(config).to(device)

    # Unfreeze all layers
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2)
    )

    current_lr = learning_rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.4,
        patience=patience // 2,
    )

    print(f"*** Running {model_name} with N_max_p={N_max_p}," +
          f"N_max_n={N_max_n}, and batch_size={batch_size} for epochs={epochs} ***")

    # /-----------------------------/
    #       LOAD TRAINING DATA
    # /-----------------------------/

    cand = pd.read_csv(f'{data_base_dir}data/train_cand_{dataset_version}{N_str}.csv')
    labels_tensor = torch.tensor(cand["label"].values, dtype=torch.long)

    triplets_tensor = None
    if need_triplets:
        triplets_np = np.load(
            f'{data_base_dir}data/train_triplets_{dataset_version}{N_str}.npy',
            mmap_mode='r'
        ).astype(np.float32)

        if np.any(np.isnan(triplets_np)):
            nan_trip_idxs = np.isnan(triplets_np).any(axis=(1, 2, 3))
            triplets_np = triplets_np[~nan_trip_idxs]
            # Filter cand and labels_tensor accordingly
            cand = cand.loc[~nan_trip_idxs].reset_index(drop=True)
            labels_tensor = torch.tensor(cand["label"].values, dtype=torch.long)
            print(
                f"{YELLOW}**** Null in triplets ****{END}\n"
                f"Removed {np.sum(nan_trip_idxs)} alert(s) from triplets and "
                f"corresponding cand/labels."
            )

        triplets_np = np.transpose(triplets_np, (0, 3, 1, 2))
        triplets_tensor = torch.from_numpy(triplets_np.copy())

    metadata_tensor = None
    if need_metadata:
        metadata_values = cand[metadata_cols].values
        if np.isnan(metadata_values).any():
            print(
                f"{RED}NaNs found in metadata columns after potential filtering "
                f"based on triplets.{END}"
            )
            nan_cols = cand[metadata_cols].isnull().sum()
            print(
                f"Columns with NaNs: "
                f"{nan_cols[nan_cols > 0]}{END}"
            )
            print(f"{YELLOW}Please ensure metadata is clean or implement imputation. Exiting.{END}")
            exit(1)
        metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32)

    num_bts = torch.sum(labels_tensor == 1).item()
    num_notbts = torch.sum(labels_tensor == 0).item()
    print(f'num_notbts: {num_notbts}')
    print(f'num_bts: {num_bts}')

    # Data augmentations
    transforms_list = []
    if need_triplets:  # Only add image transforms if triplets are needed
        transforms_list.append(transforms.ToDtype(torch.float32))
        if h_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if v_flip:
            transforms_list.append(transforms.RandomVerticalFlip(p=0.5))
        if rot:
            transforms_list.append(RandomRightAngleRotation())

    # Pass need_triplets and need_metadata to dataset
    dataset = FlexibleDataset(
        images=triplets_tensor,
        metadata=metadata_tensor,
        labels=labels_tensor,
        transform=transforms.Compose(transforms_list) if transforms_list else None
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    bts_weight = torch.FloatTensor([num_notbts / num_bts]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=bts_weight).to(device)

    # /-----------------------------/
    #         CONNECT WandB
    # /-----------------------------/

    if not sweeping:
        if not config.get('testing', False):
            wandb.init(project="BTSbotv2", config=config)
            run_name = wandb.run.name
        else:
            run_name = "testing"

    run_model_name = f"{model_name}_{dataset_version}{N_str}_{device}"
    model_dir = f"models/{run_model_name}/{run_name}/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # /----------------------/
    #          TRAIN
    # /----------------------/

    train_losses, train_accs, val_losses, val_accs \
        = [np.zeros(epochs) for _ in range(4)]
    run_data = {
        "run_name": run_name,
        "train_loss": train_losses,
        "train_accuracy": train_accs,
        "val_loss": val_losses,
        "val_accuracy": val_accs,
    }

    best_raw_preds, best_val_labels = None, None
    epochs_since_improvement = 0

    for epoch in range(epochs):
        # Run training data through model, compute loss and accuracy, take step
        epoch_train_loss, epoch_train_acc = train_epoch(
            dataloader, epoch, epochs, optimizer, loss_fn, model,
            need_triplets, need_metadata
        )
        train_losses[epoch] = epoch_train_loss
        train_accs[epoch] = epoch_train_acc

        # Save latest model to disk
        torch.save(model.state_dict(), os.path.join(model_dir, "latest_model.pth"))

        # Run validation data through model, compute loss and accuracy
        epoch_val_loss, epoch_val_acc, val_raw_preds, val_labels = val.run_val(
            config, model_dir, dataset_version, "latest_model.pth", bts_weight
        )
        val_losses[epoch] = epoch_val_loss
        val_accs[epoch] = epoch_val_acc
        print(
            f"\n       {BOLD}{YELLOW}" +
            f"val loss: {epoch_val_loss:.5f}, " +
            f"val accuracy: {epoch_val_acc:.5f}{END}"
        )

        # Step the learning rate scheduler (i.e. check if LR decrease needed)
        scheduler.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] != current_lr:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"       {BOLD}{BLUE}LR decreased to {current_lr}{END}")

        # If val loss improved, save model
        prev_best_val_loss = min([np.inf] + list(val_losses[:epoch]))
        if epoch_val_loss < prev_best_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(
                f"       {GREEN}" +
                f"val loss improved from {prev_best_val_loss:.5f}, saved model{END}\n"
            )
            best_raw_preds = np.copy(val_raw_preds)
            best_val_labels = np.copy(val_labels)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"       No improvement in val loss for {epochs_since_improvement} epoch(s)")
            if epochs_since_improvement >= patience:
                print(f"       {BOLD}{RED}Triggered early stopping{END}\n")
                break
            print()

        if not config.get('testing', False):
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
                "learning_rate": current_lr
            })

    # Create figure
    val_summ = val.diagnostic_fig(
        run_data={
            "type": model_name,
            "raw_preds": best_raw_preds,
            "labels": best_val_labels,
            "run_name": run_name,
            "loss": train_losses[:epoch+1],
            "accuracy": train_accs[:epoch+1],
            "val_loss": val_losses[:epoch+1],
            "val_accuracy": val_accs[:epoch+1],
        },
        run_descriptor=model_dir,
        cand_dir=f'{data_base_dir}data/val_cand_{dataset_version}{N_str}.csv'
    )

    if not config.get('testing', False):
        def F1(precision, recall):
            return 2 * precision * recall / (precision + recall + 1e-7)

        wandb.summary['ROC_AUC'] = val_summ['roc_auc']
        wandb.summary['bal_acc'] = val_summ['bal_acc']
        wandb.summary['bts_acc'] = val_summ['bts_acc']
        wandb.summary['notbts_acc'] = val_summ['notbts_acc']

        wandb.summary['alert_precision'] = val_summ['alert_precision']
        wandb.summary['alert_recall'] = val_summ['alert_recall']
        wandb.summary['alert_F1'] = F1(val_summ['alert_precision'], val_summ['alert_recall'])

        for pol_name in list(val_summ['policy_performance']):
            perf = val_summ['policy_performance'][pol_name]

            wandb.summary[pol_name+"_precision"] = perf['policy_precision']
            wandb.summary[pol_name+"_recall"] = perf['policy_recall']
            wandb.summary[pol_name+"_binned_precision"] = perf['binned_precision']
            wandb.summary[pol_name+"_binned_recall"] = perf['binned_recall']
            wandb.summary[pol_name+"_peakmag_bins"] = perf['peakmag_bins']

            wandb.summary[pol_name+"_save_dt"] = perf['med_save_dt']
            wandb.summary[pol_name+"_trigger_dt"] = perf['med_trigger_dt']

            wandb.summary[pol_name+"_F1"] = F1(perf['policy_precision'], perf['policy_recall'])
        wandb.log({"figure": wandb.Image(val_summ['fig'])})
    plt.clf()
    plt.close()

    print(BOLD+'============ Summary ============='+END)
    print(f'Best val loss: {min(val_losses[:epoch+1]):.5f}')
    print(f'Best val accuracy: {max(val_accs[:epoch+1]):.5f}')
    print(f'Model diagnostics at {model_dir}\n')

    make_report(config, f"{model_dir}/report.json", run_data)

    del triplets_tensor, triplets_np, metadata_tensor, labels_tensor, dataset
    del dataloader, model, optimizer, loss_fn, cand, run_data

    return


def train_epoch(dataloader: DataLoader, epoch: int, epochs: int,
                optimizer: optim.Optimizer, loss_fn, model,
                need_triplets: bool, need_metadata: bool):
    """
    Run one epoch of training.
    """
    epoch_start_time = time.time()
    num_batches = len(dataloader)
    data_iterator = iter(dataloader)

    all_logits = []
    all_labels = []
    all_raw_preds = []

    # Iterate of batches of training data
    for i in range(num_batches):
        # Clear gradients, run model on batch, and compute loss
        model.zero_grad()

        # Get next batch of training data
        data_items = next(data_iterator)

        images_batch, meta_batch, labels_batch = None, None, None

        if need_triplets and need_metadata:
            images_batch, meta_batch, labels_batch = data_items
            images_batch = images_batch.to(device)
            meta_batch = meta_batch.to(device)

            logits = model(image_input=images_batch, metadata_input=meta_batch)
        elif need_triplets:
            images_batch, labels_batch = data_items
            images_batch = images_batch.to(device)

            logits = model(input_data=images_batch)  # For SwinV2 like models
        elif need_metadata:
            meta_batch, labels_batch = data_items
            meta_batch = meta_batch.to(device)

            logits = model(input_data=meta_batch)  # For TabularNet like models

        labels_batch = labels_batch.unsqueeze(1).to(device).float()

        # Compute loss, backpropogate, and take step in gradient descent
        batch_train_loss = loss_fn(logits, labels_batch)
        batch_train_loss.backward()
        optimizer.step()

        # Compute scores from logits
        raw_preds = torch.sigmoid(logits)

        # Keep track of predictions and labels
        all_logits.append(logits.detach())
        all_labels.append(labels_batch.detach())
        all_raw_preds.append(raw_preds.detach())

        # Calculate batch accuracy
        preds = (raw_preds > 0.5).float()
        correct = (preds == labels_batch).float().sum()
        batch_train_acc = correct / labels_batch.shape[0]

        # Log batch quantities to stdout
        perf_to_stdout(
            epoch, epochs, epoch_start_time,
            i + 1, num_batches,
            batch_train_loss.item(), batch_train_acc.item(),
        )

    # Compute loss for all batches of val
    epoch_loss = loss_fn(
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0)
    ).item()

    # Compute accuracy for the entire epoch
    all_raw_preds = torch.cat(all_raw_preds, dim=0).squeeze().cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).squeeze().cpu().numpy()
    epoch_accuracy = np.sum((all_raw_preds > 0.5) == all_labels) / len(all_labels)

    perf_to_stdout(
        epoch, epochs, epoch_start_time,
        num_batches, num_batches,
        epoch_loss, epoch_accuracy, flush_stdout=False
    )

    return epoch_loss, epoch_accuracy


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep_id = "0h2lo6qd"
        wandb.agent(sweep_id, function=sweep_train, count=12, project="BTSbotv2")
    else:
        classic_train(sys.argv[1])
