"""Two-stage training for nano-LLaVA (Liu et al., 2023).

Stage 1 - projector pretraining: CLIP and the LLM are frozen; only the MLP
projector is trained on image-caption pairs so it learns to speak the LLM's
embedding language.

Stage 2 - visual instruction tuning: the stage-1 projector is loaded, LoRA
adapters are added to the (otherwise frozen) LLM, and projector + LoRA are
trained together on instruction-following conversations.

  python train.py --stage 1
  python train.py --stage 2 --wandb
  python train.py --stage 1 --max-steps 20      # smoke test
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from data_utils import VLMCollator, build_dataset, build_processor
from model import NanoLLaVA, count_params

# Per-stage defaults; CLI flags override.
STAGE_DEFAULTS = {
    1: dict(epochs=1, batch_size=32, grad_accum=4, lr=1e-3),
    2: dict(epochs=1, batch_size=16, grad_accum=8, lr=2e-4),
}


def add_lora(model, rank):
    """Wrap the LLM in LoRA adapters and return trainable-param info."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model.llm = get_peft_model(model.llm, cfg)
    model.llm.enable_input_require_grads()   # let grads flow under checkpointing


@torch.no_grad()
def log_sample_panel(model, processor, examples, device, run, step):
    """Render the model's current answers on held-out images, log to W&B.

    The VLM analogue of the per-epoch sample grid in the sibling DiT repo:
    it turns the loss curve into something you can actually read.
    """
    import wandb

    from chat import answer
    from grid import save_qualitative_grid

    model.eval()
    rows = [(img, prompt, answer(model, processor, img, prompt, device,
                                 max_new_tokens=64))
            for img, prompt in examples]
    model.train()
    path = f"samples/step_{step:05d}.png"
    save_qualitative_grid(rows, path)
    run.log({"samples": wandb.Image(path)}, step=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--projector-depth", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap dataset size (0 = stage default)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N optimizer steps (0 = no limit)")
    ap.add_argument("--sample-every", type=int, default=200,
                    help="steps between W&B sample panels (needs --wandb)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="nano-llava")
    args = ap.parse_args()

    d = STAGE_DEFAULTS[args.stage]
    epochs = args.epochs or d["epochs"]
    batch_size = args.batch_size or d["batch_size"]
    grad_accum = args.grad_accum or d["grad_accum"]
    lr = args.lr or d["lr"]

    # Optional experiment tracking; wandb is imported only when --wandb is set.
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # -- model ----------------------------------------------------------------
    model = NanoLLaVA(projector_depth=args.projector_depth)
    if args.stage == 2:
        proj_ckpt = os.path.join(args.out, "stage1_projector.pt")
        model.projector.load_state_dict(torch.load(proj_ckpt, map_location="cpu"))
        print(f"loaded stage-1 projector from {proj_ckpt}")
        add_lora(model, args.lora_rank)
        model.gradient_checkpointing_enable()
        model.llm.config.use_cache = False
    model.projector.requires_grad_(True)     # trained in both stages
    model = model.to(device)

    trainable, total = count_params(model)
    print(f"stage {args.stage} | trainable: {trainable:.1f}M / {total:.1f}M params "
          f"| device: {device}")
    assert trainable > 0, "no trainable parameters - check LoRA target modules"

    # -- data -----------------------------------------------------------------
    processor = build_processor()
    data = build_dataset(args.stage, model.tokenizer, processor,
                         max_samples=args.max_samples or None)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        num_workers=4, collate_fn=VLMCollator(model.tokenizer),
                        pin_memory=True, drop_last=True)
    print(f"dataset: {len(data)} examples | {len(loader)} batches/epoch")

    # -- optimizer ------------------------------------------------------------
    if args.stage == 1:
        param_groups = [{"params": model.projector.parameters(), "lr": lr}]
    else:
        lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
        param_groups = [
            {"params": lora_params, "lr": lr},
            {"params": model.projector.parameters(), "lr": lr * 0.1},
        ]
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    if run is not None:
        run.summary["trainable_millions"] = trainable

    # Fixed held-out images for the periodic W&B sample panel.
    panel_examples = None
    if run is not None:
        from grid import PROMPTS, _load_examples

        os.makedirs("samples", exist_ok=True)
        panel_examples = list(zip(_load_examples(3), PROMPTS[:3]))

    # -- training loop --------------------------------------------------------
    step = 0
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
                loss = model(**batch).loss / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for g in param_groups for p in g["params"]), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                shown = loss.item() * grad_accum
                if run is not None and step % 25 == 0:
                    run.log({"loss": shown, "lr": lr, "epoch": epoch}, step=step)
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step} loss {shown:.4f}")
                if run is not None and step % args.sample_every == 0:
                    log_sample_panel(model, processor, panel_examples,
                                     device, run, step)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    # -- save -----------------------------------------------------------------
    if args.stage == 1:
        path = os.path.join(args.out, "stage1_projector.pt")
        torch.save(model.projector.state_dict(), path)
    else:
        model.llm.save_pretrained(os.path.join(args.out, "stage2_lora"))
        path = os.path.join(args.out, "stage2_projector.pt")
        torch.save(model.projector.state_dict(), path)
    print(f"done. stage {args.stage} checkpoint -> {args.out}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
