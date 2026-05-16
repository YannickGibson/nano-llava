"""Run a trained nano-LLaVA on an image and a prompt.

  python chat.py --image cat.jpg --prompt "What is in this photo?"
  python chat.py --image cat.jpg --interactive
  python chat.py --image cat.jpg --prompt "Describe it." --no-lora   # stage-1 model
"""

import argparse
import os

import torch
from PIL import Image

from data_utils import build_inference_inputs, build_processor
from model import NanoLLaVA


def load_trained(ckpt_dir, device, use_lora=True, projector_depth=2):
    """Build NanoLLaVA and load stage-1 (or stage-1+2 LoRA) weights."""
    model = NanoLLaVA(projector_depth=projector_depth)
    if use_lora:
        from peft import PeftModel

        proj = os.path.join(ckpt_dir, "stage2_projector.pt")
        model.projector.load_state_dict(torch.load(proj, map_location="cpu"))
        model.llm = PeftModel.from_pretrained(
            model.llm, os.path.join(ckpt_dir, "stage2_lora")
        )
    else:
        proj = os.path.join(ckpt_dir, "stage1_projector.pt")
        model.projector.load_state_dict(torch.load(proj, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def answer(model, processor, image, prompt, device, max_new_tokens=256):
    """Generate the assistant's reply to `prompt` about `image`."""
    input_ids, attn, pixels = build_inference_inputs(
        model.tokenizer, processor, image, prompt
    )
    input_ids, attn, pixels = input_ids.to(device), attn.to(device), pixels.to(device)
    out = model.generate(input_ids, attn, pixels, max_new_tokens=max_new_tokens)
    return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--no-lora", action="store_true",
                    help="use the stage-1 projector-only model")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.ckpt, device, use_lora=not args.no_lora)
    processor = build_processor()
    image = Image.open(args.image)

    if args.interactive:
        print("nano-llava interactive chat. Ctrl-C to exit.")
        while True:
            prompt = input("\nyou> ").strip()
            if not prompt:
                continue
            print("nano-llava>", answer(model, processor, image, prompt,
                                        device, args.max_new_tokens))
    else:
        prompt = args.prompt or "Describe this image."
        print(answer(model, processor, image, prompt, device,
                     args.max_new_tokens))


if __name__ == "__main__":
    main()
