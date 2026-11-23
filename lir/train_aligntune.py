"""D13: AlignTune plain SFT LoRA experiment."""

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = Path("configs/d13/aligntune.json")


def load_manifest(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))

    required = [
        "model", "model_revision", "dataset", "expected_rows",
        "epochs", "learning_rate", "batch_size",
        "gradient_accumulation_steps", "max_seq_length", "seed",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing D13 configuration: {key}")

    return config


def check_dataset(config):
    path = Path(config["dataset"])
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if len(rows) != config["expected_rows"]:
        raise ValueError(
            f"Expected {config['expected_rows']} rows, got {len(rows)}"
        )

    for index, row in enumerate(rows):
        if not str(row.get("instruction", "")).strip():
            raise ValueError(f"Empty instruction at row {index}")
        if not str(row.get("response", "")).strip():
            raise ValueError(f"Empty response at row {index}")

    return len(rows)


def build_trainer(config, model_path):
    from aligntune.core.backend_factory import create_sft_trainer

    lora = config["lora"]

    return create_sft_trainer(
        model_name=str(model_path),
        dataset_name=str(Path(config["dataset"]).resolve()),
        backend="trl",
        output_dir="outputs/d13/aligntune",
        num_epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        max_seq_length=config["max_seq_length"],
        seed=config["seed"],
        data_seed=config["seed"],
        precision=config["precision"],
        use_peft=True,
        lora_r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        lora_target_modules=lora["target_modules"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        completion_only_loss=config["completion_only_loss"],
        assistant_only_loss=config["assistant_only_loss"],
        packing=config["packing"],
        curator_schema_gate=config["curator_schema_gate"],
        curator_clean=config["curator_clean"],
        curator_dedup=config["curator_dedup"],
        keep_columns=True,
        val_split_ratio=None,
        test_split_ratio=None,
        warmup_ratio=0.03,
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler="cosine",
        group_by_length=True,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        bf16=False,
        fp16=False,
        split_seed=config["seed"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    config = load_manifest(args.config)
    count = check_dataset(config)

    print(f"D13 dataset validated: {count} rows")
    print(f"Model: {config['model']}")
    print(f"Pinned revision: {config['model_revision']}")

    if args.preflight:
        print("Preflight passed. No model loaded and no training started.")
        return

    if not args.model_path:
        raise ValueError(
            "Full training requires --model-path pointing to "
            "the pinned model snapshot."
        )

    trainer = build_trainer(config, args.model_path)

    if not trainer.config.model.peft.enabled:
        raise RuntimeError("AlignTune LoRA was not enabled")

    print("AlignTune trainer configuration created.")
    print("Training has NOT started.")

    # GPU training will be enabled after the model and token-mask checks.


if __name__ == "__main__":
    main()
