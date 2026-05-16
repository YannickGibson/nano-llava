"""Render a qualitative example grid: image, prompt, and model answer per row.

Pulls a handful of held-out Flickr30k images, runs the trained model on a
varied set of prompts, and writes assets/example_grid.png for the README.

  python grid.py
  python grid.py --no-lora --out assets/stage1_grid.png
"""

import argparse
import textwrap

import torch
from PIL import Image, ImageDraw, ImageFont

from chat import answer, load_trained
from data_utils import build_processor

# (prompt cycled across the sampled images).
PROMPTS = [
    "Describe this image in one sentence.",
    "What is the main subject doing?",
    "What objects can you see?",
    "What is the setting of this scene?",
    "How many people are in the image?",
    "What is the overall mood of this picture?",
]


def _load_examples(n):
    """Take `n` Flickr30k images from the held-out tail (unseen in training)."""
    from datasets import load_dataset

    from data_utils import HELDOUT_IMAGES

    ds = load_dataset("lmms-lab/flickr30k", split="test")
    start = len(ds) - HELDOUT_IMAGES
    return [ds[start + i]["image"].convert("RGB") for i in range(n)]


def save_qualitative_grid(rows, path, cell=320, scale=1):
    """Render (image, prompt, answer) rows to a single PNG.

    Each row is the image on the left and the prompt + wrapped answer on the
    right, mirroring the labeled-grid idea from the sibling DiT repo.
    """
    text_w, pad = 520, 16
    row_h = cell + 2 * pad
    canvas = Image.new("RGB", (cell + text_w + 3 * pad, row_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    bold = ImageFont.load_default(size=17)

    for i, (image, prompt, reply) in enumerate(rows):
        y0 = i * row_h
        img = image.resize((cell, cell))
        canvas.paste(img, (pad, y0 + pad))
        tx, ty = cell + 2 * pad, y0 + pad
        draw.text((tx, ty), "prompt", fill="#1a73e8", font=bold)
        ty += 24
        for line in textwrap.wrap(prompt, 60):
            draw.text((tx, ty), line, fill="black", font=font)
            ty += 22
        ty += 12
        draw.text((tx, ty), "nano-llava", fill="#188038", font=bold)
        ty += 24
        for line in textwrap.wrap(reply, 60):
            draw.text((tx, ty), line, fill="#202124", font=font)
            ty += 22
        draw.line([(0, y0), (canvas.width, y0)], fill="#e0e0e0", width=1)

    if scale != 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="assets/example_grid.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.ckpt, device, use_lora=not args.no_lora)
    processor = build_processor()

    images = _load_examples(args.n)
    rows = []
    for i, image in enumerate(images):
        prompt = PROMPTS[i % len(PROMPTS)]
        reply = answer(model, processor, image, prompt, device,
                       max_new_tokens=128)
        print(f"[{i + 1}/{len(images)}] {prompt} -> {reply}")
        rows.append((image, prompt, reply))

    save_qualitative_grid(rows, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
