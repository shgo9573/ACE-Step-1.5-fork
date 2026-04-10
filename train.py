#!/usr/bin/env python3
"""
ACE-Step Training V2 (Side-Step) -- CLI Entry Point
Modified for AUTOMATIC 1-CLICK TRAINING (No UI, No Wizard)
"""

from __future__ import annotations

import gc
import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup (before any library imports that might configure logging)
# ---------------------------------------------------------------------------

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)

try:
    _file_handler = logging.FileHandler("sidestep.log", mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_log_formatter)
    _log_handlers = [_console_handler, _file_handler]
except OSError:
    _log_handlers = [_console_handler]

logging.basicConfig(level=logging.DEBUG, handlers=_log_handlers)
logger = logging.getLogger("train")


def _has_subcommand() -> bool:
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        return True
    known = {"vanilla", "fixed", "estimate"}
    return bool(known & set(args))


def _cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _dispatch(args) -> int:
    from acestep.training_v2.cli.common import validate_paths

    if getattr(args, "preprocess", False):
        return _run_preprocess(args)

    sub = args.subcommand

    if not validate_paths(args):
        return 1

    if sub == "vanilla":
        from acestep.training_v2.cli.train_vanilla import run_vanilla
        return run_vanilla(args)

    elif sub == "fixed":
        from acestep.training_v2.cli.train_fixed import run_fixed
        return run_fixed(args)

    elif sub == "estimate":
        return _run_estimate(args)

    else:
        print(f"[FAIL] Unknown subcommand: {sub}", file=sys.stderr)
        return 1


def main() -> int:
    from acestep.training_v2.cli.common import build_root_parser
    parser = build_root_parser()

    # =======================================================================
    # 🚀 המעקף האוטומטי: אם לא כתבת כלום, הוא מריץ את ההגדרות שלך אוטומטית!
    # =======================================================================
    if not _has_subcommand():
        print("\n" + "="*50)
        print("🚀 מתחיל אימון אוטומטי (בלי שאלות, בלי ממשק)...")
        print("="*50 + "\n")
        
        # כאן אנחנו מזריקים לתוכנה את כל התשובות מראש:
        sys.argv = [
            "train.py",
            "fixed",                                # שימוש במנגנון האימון המתוקן והיציב
            "--checkpoint-dir", "checkpoints",      # תיקיית מודלי הבסיס
            "--model-variant", "turbo",             # שימוש במודל טורבו
            "--dataset-dir", "dataset/my_voice",    # התיקייה עם השירים שלך!
            "--output-dir", "output/my_model",      # לאן לשמור את התוצאה
            "--batch-size", "1"                     # מותאם לזיכרון של קאגל
        ]

    args = parser.parse_args()

    try:
        last_code = _dispatch(args)
    except Exception as exc:
        logger.exception("Unhandled error in automated run")
        print(f"[FAIL] {exc}", file=sys.stderr)
        last_code = 1
    finally:
        _cleanup_gpu()

    return last_code


# ===========================================================================
# Subcommand implementations
# ===========================================================================

def _run_preprocess(args) -> int:
    from acestep.training_v2.preprocess import preprocess_audio_files

    audio_dir = getattr(args, "audio_dir", None)
    dataset_json = getattr(args, "dataset_json", None)
    tensor_output = getattr(args, "tensor_output", None)

    if not audio_dir and not dataset_json:
        print("[FAIL] --audio-dir or --dataset-json is required.", file=sys.stderr)
        return 1
    if not tensor_output:
        print("[FAIL] --tensor-output is required.", file=sys.stderr)
        return 1

    source_label = dataset_json if dataset_json else audio_dir

    print("\n" + "=" * 60)
    print("  Preprocessing Summary")
    print("=" * 60)
    print(f"  Source:        {source_label}")
    print(f"  Output:        {tensor_output}")
    print(f"  Checkpoint:    {args.checkpoint_dir}")
    print(f"  Model variant: {args.model_variant}")
    print(f"  Max duration:  {getattr(args, 'max_duration', 240.0)}s")
    print("=" * 60)

    try:
        result = preprocess_audio_files(
            audio_dir=audio_dir,
            output_dir=tensor_output,
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            max_duration=getattr(args, "max_duration", 240.0),
            dataset_json=dataset_json,
            device=getattr(args, "device", "auto"),
            precision=getattr(args, "precision", "auto"),
        )
    except Exception as exc:
        print(f"[FAIL] Preprocessing failed: {exc}", file=sys.stderr)
        logger.exception("Preprocessing error")
        return 1
    finally:
        _cleanup_gpu()

    print(f"\n[OK] Preprocessing complete:")
    print(f"     Processed: {result['processed']}/{result['total']}")
    return 0


def _run_estimate(args) -> int:
    import json as _json
    from acestep.training_v2.estimate import run_estimation

    num_batches = getattr(args, "estimate_batches", 5) or 5
    
    try:
        results = run_estimation(
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            dataset_dir=args.dataset_dir,
            num_batches=num_batches,
            batch_size=args.batch_size,
            top_k=getattr(args, "top_k", 16) or 16,
            granularity=getattr(args, "granularity", "module") or "module",
        )
    except Exception as exc:
        print(f"[FAIL] Estimation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _cleanup_gpu()

    if not results:
        return 1

    output_path = getattr(args, "estimate_output", None) or "./estimate_results.json"
    with open(output_path, "w") as f:
        _json.dump(results, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())
