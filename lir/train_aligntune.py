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
        task_type="instruction_following",
        backend="trl",
        output_dir="runs/gpu-plain-aligntune/adapter",
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


def verify_training_batch(trainer, expected_rows):
    """Verify the actual TRL training batch before GPU training."""
    from trl import SFTTrainer
    from aligntune.core.sft.training_args_builder import build_sft_config

    trainer.setup_model()
    trainer.setup_data()
    trainer.dataset = trainer.train_dataset

    dataset = trainer.dataset

    if len(dataset) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} training rows, got {len(dataset)}"
        )

    if "prompt" not in dataset.column_names:
        raise RuntimeError("Missing prompt column")

    if "completion" not in dataset.column_names:
        raise RuntimeError("Missing completion column")

    if "messages" in dataset.column_names:
        raise RuntimeError(
            "Unexpected messages column; completion masking may differ"
        )

    print(f"AlignTune dataset verified: {len(dataset)} rows")

    sft_config = build_sft_config(
        config=trainer.config,
        train_dataset=dataset,
        eval_dataset=None,
        output_dir=trainer.config.logging.output_dir,
    )

    test_trainer = SFTTrainer(
        model=trainer.model,
        args=sft_config,
        train_dataset=dataset.select_columns(
            ["prompt", "completion"]
        ),
        processing_class=trainer.tokenizer,
        peft_config=None,
    )

    prepared = test_trainer.train_dataset

    if len(prepared) != expected_rows:
        raise RuntimeError(
            f"TRL retained {len(prepared)} of {expected_rows} rows"
        )

    if "completion_mask" not in prepared.column_names:
        raise RuntimeError(
            "TRL did not create a completion_mask column"
        )

    mask = prepared[0]["completion_mask"]

    if not any(value == 0 for value in mask):
        raise RuntimeError(
            "No prompt tokens are excluded by the completion mask"
        )

    if not any(value == 1 for value in mask):
        raise RuntimeError(
            "No completion tokens are included in the loss"
        )

    print("TRL completion mask contains excluded prompt tokens.")
    print("TRL completion mask contains included response tokens.")

    example = prepared[0]
    token_ids = example["input_ids"]

    if len(mask) != len(token_ids):
        raise RuntimeError(
            "Completion mask length does not match input tokens"
        )

    first_response_token = mask.index(1)

    if any(value != 0 for value in mask[:first_response_token]):
        raise RuntimeError("Prompt masking is inconsistent")

    if any(value != 1 for value in mask[first_response_token:]):
        raise RuntimeError("Response masking is inconsistent")

    print(f"First response token index: {first_response_token}")
    print("Prompt/completion mask boundary verified.")
    prompt_text = dataset[0]["prompt"]
    completion_text = dataset[0]["completion"]

    decoded_prompt = trainer.tokenizer.decode(
        token_ids[:first_response_token],
        skip_special_tokens=False,
    )

    decoded_response = trainer.tokenizer.decode(
        token_ids[first_response_token:],
        skip_special_tokens=False,
    )

    print(f"Original prompt preview: {prompt_text[:100]!r}")
    print(f"Original response preview: {completion_text[:100]!r}")
    print(f"Decoded prompt preview: {decoded_prompt[:100]!r}")
    print(f"Decoded response preview: {decoded_response[:100]!r}")

    batch = next(iter(test_trainer.get_train_dataloader()))
    labels = batch["labels"]

    ignored = (labels == -100).sum().item()
    trained = (labels != -100).sum().item()

    if ignored == 0:
        raise RuntimeError(
            "No masked tokens found in training batch"
        )

    if trained == 0:
        raise RuntimeError(
            "No trainable tokens found in training batch"
        )

    print(f"Masked tokens: {ignored}")
    print(f"Trainable tokens: {trained}")
    print("D13 training batch verification passed.")

    return test_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-batch", action="store_true")
    args = parser.parse_args()

    config = load_manifest(args.config)
    count = check_dataset(config)

    print(f"D13 dataset validated: {count} rows")
    print(f"Model: {config['model']}")
    print(f"Pinned revision: {config['model_revision']}")

    if args.preflight:
        print("Preflight passed. No model loaded and no training started.")
        return

    from huggingface_hub import snapshot_download

    model_path = Path(args.model_path or "runs/d13/model")

    print("Downloading pinned Tiny Aya model snapshot...")

    snapshot_download(
        repo_id=config["model"],
        revision=config["model_revision"],
        local_dir=str(model_path),
    )

    print(f"Pinned model snapshot ready: {model_path}")

    trainer = build_trainer(config, model_path)

    if not trainer.config.model.peft.enabled:
        raise RuntimeError("AlignTune LoRA was not enabled")

    task_type = trainer.config.dataset.task_type
    task_name = getattr(task_type, "value", task_type)

    if task_name != "instruction_following":
        raise RuntimeError(
            f"Unexpected D13 task type: {task_name}"
        )

    if not trainer.config.train.completion_only_loss:
        raise RuntimeError(
            "D13 requires completion_only_loss=True"
        )

    print("D13 task type and completion-only settings verified.")
    print("AlignTune trainer configuration created.")

    if args.verify_batch:
        print("Starting D13 training-batch verification...")
        verify_training_batch(trainer, config["expected_rows"])
        print("Verification completed. No training started.")
        return

    print("Training has NOT started.")

    # GPU training will be enabled after the model and token-mask checks.


if __name__ == "__main__":
    main()
