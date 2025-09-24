#!/usr/bin/env python3
"""
BIFROST Training CLI

Train the BIFROST model from the command line with configurable model size,
epochs, batch size, learning rate, and more.
"""

import argparse
import json
import os
import sys
import logging
from typing import Any, Dict, Optional
import torch
from .config import create_model_config, create_training_config
from .model import create_bifrost_model
from .data.tokenizer import tokenizer
from .data.dataset import CrystalStructureDataset, create_dataloader
from .training import create_trainer


def _default_dataset_path() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    mp_dataset = os.path.join(here, "data", "mp", "mp_dataset.json")
    if os.path.exists(mp_dataset):
        return mp_dataset
    return None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the BIFROST model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick debug run on small dataset
  bifrost-train --model-size small --epochs 2 --batch-size 8 --dataset bifrost/data/mp/mp_dataset_small.json

  # Larger run
  bifrost-train --model-size base --epochs 10 --batch-size 64 --lr 2e-4 \
    --save-interval 5 --eval-interval 5 --checkpoint-dir checkpoints
        """,
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default=_default_dataset_path(),
        help="Path to training dataset JSON (default: bundled MP dataset if available)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Model size preset (default: small)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides preset)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (overrides preset)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Warmup steps (overrides preset)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["one_cycle", "cosine", "linear", "none"],
        default=None,
        help="LR scheduler type (overrides preset)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Gradient clipping value (overrides preset)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Batches between log lines (default: 100)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Epochs between checkpoints (default: 5)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Epochs between validations (default: 1)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints/)",
    )
    # TensorBoard
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: runs/)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for TensorBoard subdirectory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Dataset processing
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override model max sequence length for dataset encoding",
    )
    parser.add_argument(
        "--property-dropout",
        type=float,
        default=0.3,
        help="Probability of keeping a property in prefix (default: 0.3)",
    )
    parser.add_argument(
        "--property-removal",
        type=float,
        default=0.1,
        help="Probability to mask kept property value (default: 0.1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )

    return parser


def _load_dataset(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _merge_training_overrides(
    base: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    cfg = base.copy()
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.scheduler is not None:
        cfg["scheduler_type"] = None if args.scheduler == "none" else args.scheduler
    if args.gradient_clip is not None:
        cfg["gradient_clip"] = args.gradient_clip
    # Mixed precision flags
    if args.mixed_precision and args.no_mixed_precision:
        print(
            "Both --mixed-precision and --no-mixed-precision set; defaulting to preset"
        )
    elif args.mixed_precision:
        cfg["mixed_precision"] = True
    elif args.no_mixed_precision:
        cfg["mixed_precision"] = False
    cfg["log_interval"] = args.log_interval
    cfg["checkpoint_dir"] = args.checkpoint_dir
    # TensorBoard settings
    if args.tensorboard:
        cfg["tensorboard"] = True
    if args.tensorboard_log_dir is not None:
        cfg["tensorboard_log_dir"] = args.tensorboard_log_dir
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    return cfg


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = create_parser()
    args = parser.parse_args()

    _configure_logging()
    logger = logging.getLogger(__name__)

    if not args.dataset:
        logger.error("No dataset specified and no default dataset found.")
        logger.error("Please provide --dataset pointing to a JSON dataset.")
        sys.exit(2)

    try:
        data = _load_dataset(args.dataset)
    except Exception as e:
        logger.exception(f"Failed to load dataset from {args.dataset}: {e}")
        sys.exit(1)

    # Prepare model config
    model_config = create_model_config(args.model_size)
    model_config["vocab_size"] = tokenizer.get_vocab_size()
    if args.max_seq_len is not None:
        model_config["max_seq_len"] = args.max_seq_len

    # Build model
    model = create_bifrost_model(model_config)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Split data
    total = len(data)
    val_count = int(total * args.val_split)
    train_data = data[: total - val_count]
    val_data = data[total - val_count :] if val_count > 0 else []

    # Datasets
    max_seq_len = model_config["max_seq_len"]
    train_dataset = CrystalStructureDataset(
        train_data,
        tokenizer,
        max_seq_len=max_seq_len,
        property_dropout=args.property_dropout,
        property_removal=args.property_removal,
        curriculum_level=0,
    )
    val_dataset = (
        CrystalStructureDataset(
            val_data,
            tokenizer,
            max_seq_len=max_seq_len,
            property_dropout=args.property_dropout,
            property_removal=args.property_removal,
            curriculum_level=0,
        )
        if val_count > 0
        else None
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",  # only if using GPU
    )
    val_loader = (
        create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=device.type == "cuda",  # only if using GPU
        )
        if val_dataset is not None
        else None
    )

    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Val dataset: {len(val_dataset)}")

    # Training config
    base_train_cfg = create_training_config("default")
    train_cfg = _merge_training_overrides(base_train_cfg, args)

    # Derive total steps for scheduler (resume-aware)
    steps_per_epoch = max(1, len(train_loader))

    start_epoch_from_ckpt = 0
    if args.resume:
        try:
            # Read checkpoint metadata without constructing the trainer yet
            ckpt = torch.load(args.resume, map_location="cpu")
            # Checkpoints in this project save one-based epoch indices
            start_epoch_from_ckpt = int(ckpt.get("epoch", 0))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to read checkpoint metadata from {args.resume}: {e}"
            )

    planned_total_epochs = max(1, args.epochs + start_epoch_from_ckpt)
    planned_total_steps = steps_per_epoch * planned_total_epochs
    train_cfg["total_steps"] = planned_total_steps

    # Smart default for warmup steps if not explicitly provided
    if train_cfg.get("warmup_steps") is None:
        # Use 5% of total steps with a reasonable cap
        dynamic_warmup = max(1, min(int(planned_total_steps * 0.05), 10000))
        train_cfg["warmup_steps"] = dynamic_warmup

    logger.info(f"Training config: {train_cfg}")

    # Trainer
    trainer = create_trainer(model, train_loader, val_loader, train_cfg)

    # Optionally resume (after trainer is constructed with resume-aware scheduler)
    if args.resume:
        try:
            last_epoch = trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed from epoch {last_epoch}")
        except Exception as e:
            logger.exception(f"Failed to resume from checkpoint: {e}")
            sys.exit(1)

    # Train
    try:
        results = trainer.train(
            num_epochs=args.epochs,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

    # Summary
    if results and "final_stats" in results and results["final_stats"]:
        final = results["final_stats"].get("overall_stats", {})
        if final:
            logger.info(
                f"Final loss: {final.get('final_loss', float('nan')):.4f} | "
                f"Avg loss: {final.get('avg_loss', float('nan')):.4f}"
            )


if __name__ == "__main__":
    main()
