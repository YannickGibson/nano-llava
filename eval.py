"""Quantitative evaluation: VQAv2 accuracy on a held-out subset.

Uses the official VQA soft-accuracy metric: a prediction scores
min(#matching annotators / 3, 1), averaged over questions. `compute_vqa_accuracy`
is reusable - `ablation.py` imports it.

  python eval.py
  python eval.py --n 1000 --no-lora
"""

import argparse
import itertools
import re
import string

import torch

from chat import answer, load_trained
from data_utils import build_processor

VQA_PROMPT = "Answer the question using a single word or short phrase. "
ARTICLES = {"a", "an", "the"}


def normalize(text):
    """VQA-style answer normalization: lowercase, strip punctuation/articles."""
    text = text.lower().strip()
    text = "".join(c for c in text if c not in string.punctuation)
    words = [w for w in re.split(r"\s+", text) if w and w not in ARTICLES]
    return " ".join(words)


def soft_score(pred, gt_answers):
    """Official VQA accuracy for one prediction against 10 annotator answers."""
    pred = normalize(pred)
    matches = sum(pred == normalize(a) for a in gt_answers)
    return min(matches / 3.0, 1.0)


@torch.no_grad()
def compute_vqa_accuracy(model, processor, device, n=2000):
    """Stream a VQAv2 validation subset and return mean soft-accuracy."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    total, count = 0.0, 0
    for row in itertools.islice(ds, n):
        gt = [a["answer"] for a in row["answers"]]
        pred = answer(model, processor, row["image"], VQA_PROMPT + row["question"],
                      device, max_new_tokens=16)
        total += soft_score(pred, gt)
        count += 1
        if count % 200 == 0:
            print(f"  {count}/{n}  running acc {total / count:.4f}")
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.ckpt, device, use_lora=not args.no_lora)
    processor = build_processor()

    acc = compute_vqa_accuracy(model, processor, device, n=args.n)
    tag = "stage 1 only" if args.no_lora else "stage 1 + 2"
    print(f"\nVQAv2 accuracy ({tag}, n={args.n}): {acc:.4f}")

    with open(args.out, "w") as f:
        f.write("## Evaluation results\n\n")
        f.write(f"VQAv2 soft-accuracy over {args.n} held-out validation "
                f"questions.\n\n")
        f.write("| model | VQAv2 accuracy |\n|---|---|\n")
        f.write(f"| {tag} | {acc:.4f} |\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
