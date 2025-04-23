import torch
import argparse
import wandb
from prefect import flow, task
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column
from preprocess_tf import TinyImageNetSplitLoader, RemappedSubset
from train_tf import train_and_evaluate_model, create_lighter_alexnet, create_bnb_model, create_anb_model, fine_tune_model, create_custom_alexnet_classifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import train_tf, inspect
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@task
def preprocess_task(train_dir: str, val_dir_fixed: str, batch_size: int = 32, num_classes_a: int = 100, seed: int = 42):
    loader = TinyImageNetSplitLoader(train_dir, val_dir_fixed, img_size=(224, 224), batch_size=batch_size, val_ratio=0.2, seed=seed)
    loader.split_classes(num_classes_a=num_classes_a)
    dataloaders_a, dataloaders_b, classes_a, classes_b = loader.create_dataloaders()

    torch.save(dataloaders_a, "tiny-imagenet-200/processed/dataloaders_a.pth")
    torch.save(dataloaders_b, "tiny-imagenet-200/processed/dataloaders_b.pth")
    loader.summary()

    return dataloaders_a, dataloaders_b


@task
def train_task(num_classes=100, lr=1e-4, weight_decay=1e-3, epochs=5, patience=3):
    # instead of trusting create_lighter_alexnet to be patched,
    # re‐inject a proper classifier on YOUR DEVICE:
    dataloaders_a = torch.load("tiny-imagenet-200/processed/dataloaders_a.pth", weights_only=False)
    dataloaders_b = torch.load("tiny-imagenet-200/processed/dataloaders_b.pth", weights_only=False)

    # --- Create base models ---
    base_A = create_lighter_alexnet(num_classes=100, use_pretrained_features=True)
    base_B = create_lighter_alexnet(num_classes=100, use_pretrained_features=True)

    # 4. Train base models
    logger.warning("\nTraining base model B...")
    base_B, train_losses_B, train_acc_B, val_losses_B, val_acc_B = train_and_evaluate_model(
        model=base_B,
        data_loader=dataloaders_b,
        dropout_rate=0.5,
        epochs=epochs,
        optimizer_type="ADAM",
        lr=0.0001,
        patience=3,
        weight_decay=1e-3
    )

    logger.warning("\nTraining base model A...")
    base_A, train_losses_A, train_acc_A, val_losses_A, val_acc_A = train_and_evaluate_model(
        model=base_A,
        data_loader=dataloaders_a,
        dropout_rate=0.5,
        epochs=epochs,
        optimizer_type="ADAM",
        lr=0.0001,
        patience=3,
        weight_decay=1e-3
    )
    # --- Save base model results ---
    base_model_results = {
        "base_A": {
            "train_losses": train_losses_A,
            "train_accuracies": train_acc_A,
            "val_losses": val_losses_A,
            "val_accuracies": val_acc_A
        },
        "base_B": {
            "train_losses": train_losses_B,
            "train_accuracies": train_acc_B,
            "val_losses": val_losses_B,
            "val_accuracies": val_acc_B
        }
    }
    torch.save(base_model_results, "results/base_model_results.pth")

    # 5. Create BnB models
    bnb_results = {}   # Before fine-tuning (BnB)
    bnb_plus_results = {}  # After fine-tuning (BnB+)

    bnb_models = [create_bnb_model(i, 100) for i in range(1, 6)]

    for idx, model in enumerate(bnb_models, 1):
        logger.warning(f"\n🏋️‍♂️ Training B{idx}B model (frozen {idx} layers)...")

        # --- First Training (BnB) ---
        model = model.to(DEVICE)
        model, train_losses, train_acc, val_losses, val_acc = train_and_evaluate_model(
            model,
            dataloaders_b,
            dropout_rate=0.5,
            epochs=epochs,
            optimizer_type="ADAM",
            lr=0.0001, 
            patience=3,
            weight_decay=1e-3
        )

        # --- Save Initial Training Results ---
        bnb_results[f"B{idx}B"] = {
            "model": model,  # (optional)
            "train_losses": train_losses,
            "train_accuracies": train_acc,
            "val_losses": val_losses,
            "val_accuracies": val_acc,
        }
        torch.save(bnb_results, "results/bnb_results.pth")

        logger.warning(f"✅ Saved B{idx}B results!")

        # --- Fine-Tuning (BnB+) ---
        logger.warning(f"\n🎯 Fine-tuning B{idx}B model...")
        model, train_losses_ft, train_acc_ft, val_losses_ft, val_acc_ft = fine_tune_model(model, dataloaders_b, epochs = epochs)

        # --- Save Fine-Tuning Results ---
        bnb_plus_results[f"B{idx}B+"] = {
            "model": model,  # (optional)
            "train_losses": train_losses_ft,
            "train_accuracies": train_acc_ft,
            "val_losses": val_losses_ft,
            "val_accuracies": val_acc_ft,
        }
        torch.save(bnb_plus_results, "results/bnb_plus_results.pth")

        logger.warning(f"✅ Saved B{idx}B+ results!")

    anb_results = {}   # Before fine-tuning (BnB)
    anb_plus_results = {}  # After fine-tuning (BnB+)

    # 7. Create AnB models
    logger.warning("\nCreating AnB models...")
    anb_models = [create_anb_model(base_A, i, 100) for i in range(1, 6)]


    for idx, model in enumerate(anb_models, 1):
        logger.warning(f"\n🏋️‍♂️ Training A{idx}B model (frozen {idx} layers)...")

        # --- First Training (BnB) ---
        model = model.to(DEVICE)
        model, train_losses, train_acc, val_losses, val_acc = train_and_evaluate_model(
            model,
            dataloaders_b,
            dropout_rate=0.5,
            epochs=epochs,
            optimizer_type="ADAM",
            lr=0.0001, 
            patience=3,
            weight_decay=1e-3
        )

        # --- Save Initial Training Results ---
        anb_results[f"A{idx}B"] = {
            "model": model,  # (optional)
            "train_losses": train_losses,
            "train_accuracies": train_acc,
            "val_losses": val_losses,
            "val_accuracies": val_acc,
        }
        torch.save(anb_results, "results/anb_results.pth")

        logger.warning(f"✅ Saved A{idx}B results!")

        # --- Fine-Tuning (BnB+) ---
        logger.warning(f"\n🎯 Fine-tuning A{idx}B model...")
        model, train_losses_ft, train_acc_ft, val_losses_ft, val_acc_ft = fine_tune_model(model, dataloaders_b, epochs = epochs)

        # --- Save Fine-Tuning Results ---
        anb_plus_results[f"A{idx}B+"] = {
            "model": model,  # (optional)
            "train_losses": train_losses_ft,
            "train_accuracies": train_acc_ft,
            "val_losses": val_losses_ft,
            "val_accuracies": val_acc_ft,
        }
        torch.save(anb_plus_results, "results/anb_plus_results.pth")
        logger.warning(f"✅ Saved B{idx}B+ results!")

    
    return base_model_results, bnb_results, bnb_plus_results, anb_results, anb_plus_results


@task
def error_analyze(base_model_results, bnb_results, bnb_plus_results, anb_results, anb_plus_results, save_path_csv="results/all_models_results.csv"):
    os.makedirs("model_results_csv", exist_ok=True)

    # Merge all into one dict
    model_results_dict = {}

    # Convert each result set to DataFrames and assign names
    def convert_to_dfs(result_dict, suffix=""):
        dfs = {}
        for model_name, result in result_dict.items():
            df = pd.DataFrame({
                "train_loss": result["train_losses"],
                "train_acc": result["train_accuracies"],
                "val_loss": result["val_losses"],
                "val_acc": result["val_accuracies"]
            })
            dfs[f"{model_name}{suffix}"] = df
        return dfs

    model_results_dict.update(convert_to_dfs({"B": base_model_results["base_B"]}))
    model_results_dict.update(convert_to_dfs(bnb_results))
    model_results_dict.update(convert_to_dfs(bnb_plus_results, suffix="p"))
    model_results_dict.update(convert_to_dfs(anb_results))
    model_results_dict.update(convert_to_dfs(anb_plus_results, suffix="p"))

    # Reset index and build combined DataFrame
    for model_name, df in model_results_dict.items():
        model_results_dict[model_name] = df.reset_index(drop=True)
    combined_df = pd.concat(model_results_dict.values(), axis=1)
    combined_df.to_csv(save_path_csv, index=False)



    # --- Scatter Plot for final validation accuracy ---
    x_positions = {
        "B": 0, "B1B": 1, "B1Bp": 1, "B2B": 2, "B2Bp": 2,
        "B3B": 3, "B3Bp": 3, "B4B": 4, "B4Bp": 4, "B5B": 5, "B5Bp": 5,
        "A1B": 1, "A1Bp": 1, "A2B": 2, "A2Bp": 2,
        "A3B": 3, "A3Bp": 3, "A4B": 4, "A4Bp": 4, "A5B": 5, "A5Bp": 5
    }

    x, y, colors, markers = [], [], [], []

    for model_name, df in model_results_dict.items():
        if model_name not in x_positions:
            continue
        last_val_acc = df.iloc[-1, 3]  # val_acc assumed at column index 3
        x.append(x_positions[model_name])
        y.append(last_val_acc)

        if model_name == "B":
            colors.append("cyan")
            markers.append("o")
        elif model_name.startswith("B"):
            if model_name.endswith("p"):
                colors.append("lightblue")
                markers.append("P")
            else:
                colors.append("darkblue")
                markers.append("s")
        elif model_name.startswith("A"):
            if model_name.endswith("p"):
                colors.append("salmon")
                markers.append("P")
            else:
                colors.append("crimson")
                markers.append("s")

    plt.figure(figsize=(12, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=100, edgecolors='black', linewidth=1.5)

    plt.xlabel("Number of Frozen Layers (n)", fontsize=14)
    plt.ylabel("Top-1 Validation Accuracy", fontsize=14)
    plt.title("Validation Accuracy vs Number of Frozen Layers", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='BaseB', markerfacecolor='cyan', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='BnB', markerfacecolor='darkblue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='P', color='w', label='BnB⁺', markerfacecolor='lightblue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='AnB', markerfacecolor='crimson', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='P', color='w', label='AnB⁺', markerfacecolor='salmon', markersize=10, markeredgecolor='black'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.xticks(range(6))
    plt.tight_layout()
    plt.show()
    plot_path = "results/final_val_accuracy_plot.png"
    plt.savefig(plot_path)
    wandb.log({"Final Accuracy Plot": wandb.Image(plot_path)})

    # Log CSV to W&B
    artifact = wandb.Artifact("model_results", type="results")
    artifact.add_file(save_path_csv)
    wandb.log_artifact(artifact)

@flow(name="AlexNet Transfer Learning Flow")
def main_flow(train_dir: str, val_dir_fixed: str, preprocess: bool = True, train: bool = True, error_analysis: bool = True,
              batch_size: int = 32, num_classes_a: int = 100, lr: float = 1e-4, weight_decay: float = 1e-3,
              patience: int = 3, epochs: int = 10):

    wandb.init(project="transfer-learning-tiny-imagenet", entity="maorblumberg-tel-aviv-university", settings=wandb.Settings(start_method="thread"))
    wandb.config.update(locals())

    if preprocess:
        dataloaders_a, dataloaders_b = preprocess_task(train_dir, val_dir_fixed, batch_size, num_classes_a)
    else:
        dataloaders_a = torch.load("tiny-imagenet-200/processed/dataloaders_a.pth")
        dataloaders_b = torch.load("tiny-imagenet-200/processed/dataloaders_b.pth")

    if train:
        base_model_results, bnb_results, bnb_plus_results, anb_results, anb_plus_results = train_task(lr, weight_decay, patience, epochs)
    if error_analysis:
        error_analyze(base_model_results, bnb_results, bnb_plus_results, anb_results, anb_plus_results)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='tiny-imagenet-200/train')
    parser.add_argument('--val_dir_fixed', type=str, default='tiny-imagenet-200/val_fixed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes_a', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--error_analysis', action='store_true', help='Perform error analysis')
    args = parser.parse_args()

    main_flow(
        train_dir=args.train_dir,
        val_dir_fixed=args.val_dir_fixed,
        preprocess=args.preprocess,
        train=args.train,
        error_analysis=args.error_analysis,
        batch_size=args.batch_size,
        num_classes_a=args.num_classes_a,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        epochs=args.epochs
    )
