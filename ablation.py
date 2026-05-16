"""Ablation study: does visual instruction tuning (stage 2) actually help?

Evaluates the stage-1 projector-only model and the full stage-1+2 model on the
same VQAv2 subset and writes a Markdown comparison to results.md. This is the
most paper-faithful ablation: it isolates the contribution of LoRA instruction
tuning on top of a fixed, pretrained projector.

Projector-depth and LoRA-rank ablations need their own short stage-2 runs
(`train.py --projector-depth N` / `--lora-rank N` into separate --out dirs);
pass those dirs with --extra to fold them into the same table.

  python ablation.py --n 2000
  python ablation.py --extra r8=checkpoints_r8 --extra r64=checkpoints_r64
"""

import argparse

import torch

from chat import load_trained
from data_utils import build_processor
from eval import compute_vqa_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--extra", action="append", default=[],
                    help="name=ckpt_dir for an additional stage-1+2 variant")
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = build_processor()

    # name -> (ckpt_dir, use_lora)
    variants = {
        "stage 1 only (projector)": (args.ckpt, False),
        "stage 1 + 2 (projector + LoRA)": (args.ckpt, True),
    }
    for spec in args.extra:
        name, path = spec.split("=", 1)
        variants[name] = (path, True)

    rows = []
    for name, (ckpt_dir, use_lora) in variants.items():
        model = load_trained(ckpt_dir, device, use_lora=use_lora)
        acc = compute_vqa_accuracy(model, processor, device, n=args.n)
        print(f"{name}: VQAv2 acc {acc:.4f}")
        rows.append((name, acc))
        del model
        torch.cuda.empty_cache()

    lines = ["## Ablation results", "",
             f"VQAv2 soft-accuracy over {args.n} held-out validation questions.",
             "", "| model | VQAv2 accuracy |", "|---|---|"]
    lines += [f"| {name} | {acc:.4f} |" for name, acc in rows]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
