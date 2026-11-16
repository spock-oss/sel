"""Command-line tools for source-aligned SEL models."""

import argparse
import json
from pathlib import Path

from .heads import SELECTION_MODES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="Pack raw text and image records")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    train = subcommands.add_parser("train", help="Train or resume one epoch of SEL 27B")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--max-steps", type=int, help="Pause at this total optimizer step"
    )

    predict = subcommands.add_parser("predict", help="Score source text or an image")
    predict.add_argument("--weights", type=Path, required=True)
    text = predict.add_mutually_exclusive_group()
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    predict.add_argument("--image", type=Path)
    predict.add_argument("--query", action="append", required=True)
    predict.add_argument(
        "--mode", choices=SELECTION_MODES, help="Override the learned router"
    )
    predict.add_argument("--threshold", type=float)
    predict.add_argument("--calibrated", action="store_true")
    predict.add_argument("--environment")
    predict.add_argument("--query-first", action="store_true")

    evaluate = subcommands.add_parser(
        "evaluate", help="Evaluate a bundle on a frozen split"
    )
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")

    export = subcommands.add_parser(
        "export", help="Export LoRA and selection head from a trusted checkpoint"
    )
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    for command in (prepare, train, predict, evaluate):
        command.add_argument(
            "--base", type=Path, help="Local copy of the pinned Qwen snapshot"
        )

    args = parser.parse_args()
    if args.command == "predict":
        if not (args.text or args.text_file or args.image):
            parser.error("predict requires --text, --text-file or --image")
        if args.threshold is not None and not 0 <= args.threshold <= 1:
            parser.error("--threshold must be between 0 and 1")
        if args.mode == "linked" and len(args.query) < 2:
            parser.error("linked mode requires at least two queries")
        if args.mode == "one_coherent_answer" and len(args.query) != 1:
            parser.error("one_coherent_answer mode requires one query")
    if args.command == "train" and args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    return args


def main():
    import torch

    from .model import BASE_MODEL, build_network, load_processor, resolve_base
    from .weights import export_checkpoint, load_weights, read_config

    args = parse_args()
    torch.set_num_threads(4)
    if args.command == "export":
        export_checkpoint(args.checkpoint, args.output)
        return

    config = (
        read_config(args.weights) if args.command in {"predict", "evaluate"} else {}
    )
    model_id = config.get("base_model", BASE_MODEL)
    base = resolve_base(args.base, model_id=model_id)
    processor = load_processor(base)
    if args.command == "prepare":
        from .dataset import prepare

        print(json.dumps(prepare(args.source, args.output, processor), indent=2))
        return
    if args.command == "train":
        from .train import train

        train(
            args.data,
            args.output,
            base,
            processor,
            resume=args.resume,
            max_steps=args.max_steps,
        )
        return

    if args.command == "predict" and config.get("max_questions") == 1:
        if len(args.query) != 1 or args.mode not in (None, "dense"):
            raise SystemExit(
                "This SEL bundle accepts one question per request in dense mode."
            )
        if args.calibrated and not config.get("calibrated_thresholds"):
            raise SystemExit(
                "This SEL bundle provides fixed thresholds. Use --threshold to override."
            )
    network = build_network(base, model_id=model_id)
    config = load_weights(network, args.weights)
    if args.command == "predict":
        from .predict import predict

        threshold_key = (
            "calibrated_thresholds" if args.calibrated else "fixed_thresholds"
        )
        text = args.text_file.read_text() if args.text_file else args.text
        result = predict(
            network,
            processor,
            text=text,
            image=args.image,
            queries=args.query,
            mode=args.mode,
            threshold=args.threshold,
            thresholds=config[threshold_key],
            environment=args.environment,
            query_first=args.query_first or config.get("query_first", False),
        )
    else:
        from .dataset import open_dataset
        from .metrics import aggregate_scores
        from .train import evaluate_split

        entries, _ = open_dataset(args.data)
        rows, routing = evaluate_split(
            network, processor, args.data, entries[args.split]
        )
        result = {
            "fixed": aggregate_scores(rows, config["fixed_thresholds"]),
            "calibrated": aggregate_scores(rows, config["calibrated_thresholds"])
            if config["calibrated_thresholds"]
            else None,
            "routing": routing,
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
