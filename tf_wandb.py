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
from prefect import task
import os
import pandas as pd
import torch
import wandb
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io.export import export_png
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
    os.makedirs("results", exist_ok=True)

    # --- Fix key names ---
    def remap_plus_keys(results_dict):
        return {
            k.replace("B+", "p").replace("A+", "p"): v
            for k, v in results_dict.items()
        }

    bnb_plus_results = remap_plus_keys(bnb_plus_results)
    anb_plus_results = remap_plus_keys(anb_plus_results)

    # --- Convert to DataFrames ---
    def convert_to_dfs(result_dict):
        return {
            model_name: pd.DataFrame({
                "train_loss": result["train_losses"],
                "train_acc": result["train_accuracies"],
                "val_loss": result["val_losses"],
                "val_acc": result["val_accuracies"]
            }) for model_name, result in result_dict.items()
        }

    model_results_dict = {}
    model_results_dict.update(convert_to_dfs({"B": base_model_results["base_B"]}))
    model_results_dict.update(convert_to_dfs(bnb_results))
    model_results_dict.update(convert_to_dfs(bnb_plus_results))
    model_results_dict.update(convert_to_dfs(anb_results))
    model_results_dict.update(convert_to_dfs(anb_plus_results))

    # Save combined CSV
    for k in model_results_dict:
        model_results_dict[k] = model_results_dict[k].reset_index(drop=True)
    combined_df = pd.concat(model_results_dict.values(), axis=1)
    combined_df.to_csv(save_path_csv, index=False)

    # x-axis positions
    x_positions = {
        "B": 0, "B1B": 1, "B1p": 1, "B2B": 2, "B2p": 2,
        "B3B": 3, "B3p": 3, "B4B": 4, "B4p": 4, "B5B": 5, "B5p": 5,
        "A1B": 1, "A1p": 1, "A2B": 2, "A2p": 2,
        "A3B": 3, "A3p": 3, "A4B": 4, "A4p": 4, "A5B": 5, "A5p": 5
    }

    # Group definitions
    groups = {
        "BaseB": {"color": "cyan", "models": ["B"], "marker": "circle"},
        "BnB": {"color": "darkblue", "models": ["B1B", "B2B", "B3B", "B4B", "B5B"], "marker": "square"},
        "BnB⁺": {"color": "darkblue", "models": ["B1p", "B2p", "B3p", "B4p", "B5p"], "marker": "cross"},
        "AnB": {"color": "crimson", "models": ["A1B", "A2B", "A3B", "A4B", "A5B"], "marker": "square"},
        "AnB⁺": {"color": "crimson", "models": ["A1p", "A2p", "A3p", "A4p", "A5p"], "marker": "cross"}
    }

    # Plot setup
    p = figure(
        title="Validation Accuracy vs Number of Frozen Layers",
        x_axis_label="Number of Frozen Layers (n)",
        y_axis_label="Top-1 Validation Accuracy",
        width=900,
        height=600,
        tools="pan,wheel_zoom,box_zoom,reset"
    )

    for group_name, cfg in groups.items():
        xs, ys, labels = [], [], []
        for model_name in cfg["models"]:
            if model_name not in model_results_dict:
                continue
            x = x_positions[model_name]
            y = model_results_dict[model_name]["val_acc"].iloc[-1]
            xs.append(x)
            ys.append(y)
            labels.append(model_name)

        source = ColumnDataSource(data={"x": xs, "y": ys, "label": labels})
        glyph = getattr(p, cfg["marker"])  # circle, square, triangle
        if cfg["marker"] == "cross":
            glyph(x="x", y="y", size=12, line_color=cfg["color"], source=source, legend_label=group_name)
        else:
            glyph(x="x", y="y", size=12, fill_color=cfg["color"], line_color="black", source=source, legend_label=group_name)

        p.add_tools(HoverTool(tooltips=[("Model", "@label"), ("Val Acc", "@y{0.000}")], renderers=[p.renderers[-1]]))

    p.legend.title = "Model Type"
    p.legend.location = "top_right"
    p.legend.label_text_font_size = "10pt"
    p.grid.grid_line_alpha = 0.4
    p.toolbar.autohide = True

    # Save HTML
    html_path = "results/final_val_accuracy_plot.html"
    output_file(html_path)
    save(p)

    # Log interactive to W&B
    wandb.log({
        "Final Accuracy Interactive Plot": wandb.Html(open(html_path)),
    })

    # Log as artifact too
    artifact = wandb.Artifact("model_results", type="results")
    artifact.add_file(save_path_csv)
    artifact.add_file(html_path)
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
        dataloaders_a = torch.load("tiny-imagenet-200/processed/dataloaders_a.pth", weights_only = False)
        dataloaders_b = torch.load("tiny-imagenet-200/processed/dataloaders_b.pth", weights_only = False)

    if train:
        base_model_results, bnb_results, bnb_plus_results, anb_results, anb_plus_results = train_task(lr, weight_decay, patience, epochs)
    if error_analysis:
        if not train:
            base_model_results = torch.load("results/base_model_results.pth", weights_only = False)
            bnb_results = torch.load("results/bnb_results.pth", weights_only = False)
            bnb_plus_results = torch.load("results/bnb_plus_results.pth", weights_only = False)
            anb_results = torch.load("results/anb_results.pth", weights_only = False)
            anb_plus_results = torch.load("results/anb_plus_results.pth", weights_only = False)
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
