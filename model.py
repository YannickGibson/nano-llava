"""nano-LLaVA: a minimal vision-language model.

Faithful to "Visual Instruction Tuning" (Liu et al., 2023) and its LLaVA-1.5
follow-up (Liu et al., 2023): a frozen CLIP vision encoder and a frozen LLM are
bridged by a small trainable MLP projector. Image patch features are projected
into the LLM's token embedding space and spliced in where an `<image>`
placeholder sits, so the image becomes "just more tokens" in the same
self-attention stream as the text.

Only the projector (stage 1) and LoRA adapters (stage 2) are ever trained; the
~0.6B of backbone weights stay frozen, which is what keeps the whole two-stage
recipe inside a single-GPU day.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()   # quiet the one-time backbone load reports

# ----------------------------------------------------------------------------
# Backbone identifiers and the image-token contract
# ----------------------------------------------------------------------------
VISION_MODEL = "openai/clip-vit-base-patch16"   # 224px, patch 16 -> 14x14 grid
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
IMAGE_TOKEN = "<image>"
NUM_IMAGE_TOKENS = 196          # 14 * 14 patches, CLS token dropped
VISION_SELECT_LAYER = -2        # LLaVA-1.5 uses the penultimate CLIP layer


# ----------------------------------------------------------------------------
# Projector: CLIP patch dim -> LLM embedding dim
# ----------------------------------------------------------------------------
class VisionProjector(nn.Module):
    """MLP bridging CLIP patch features into the LLM token space.

    depth=1 is a bare linear map (original LLaVA); depth=2 is the LLaVA-1.5
    GELU-MLP default. `ablation.py` varies `depth` to measure its effect.
    """

    def __init__(self, in_dim, out_dim, depth=2):
        super().__init__()
        if depth == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, out_dim)]
            for _ in range(depth - 1):
                layers += [nn.GELU(), nn.Linear(out_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Full vision-language model
# ----------------------------------------------------------------------------
class NanoLLaVA(nn.Module):
    def __init__(self, projector_depth=2):
        super().__init__()
        # Frozen vision encoder.
        self.vision = CLIPVisionModel.from_pretrained(VISION_MODEL)
        self.vision.requires_grad_(False).eval()

        # LLM + tokenizer. The `<image>` placeholder is registered as a real
        # special token so it survives chat-template formatting as one id.
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [IMAGE_TOKEN]}
        )
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.requires_grad_(False)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        vision_dim = self.vision.config.hidden_size      # 768
        llm_dim = self.llm.config.hidden_size            # 896
        self.projector = VisionProjector(vision_dim, llm_dim, projector_depth)
        self.num_image_tokens = NUM_IMAGE_TOKENS

    # -- components -----------------------------------------------------------
    def encode_images(self, pixel_values):
        """CLIP pixels -> projected patch tokens, shape (B, 196, llm_dim)."""
        with torch.no_grad():
            out = self.vision(pixel_values=pixel_values, output_hidden_states=True)
        feats = out.hidden_states[VISION_SELECT_LAYER][:, 1:, :]   # drop CLS
        return self.projector(feats.to(self.projector.net[0].weight.dtype))

    def _merge(self, input_ids, pixel_values):
        """Embed text, then overwrite `<image>` slots with projected patches."""
        embeds = self.llm.get_input_embeddings()(input_ids)
        img = self.encode_images(pixel_values).to(embeds.dtype)
        mask = input_ids == self.image_token_id
        embeds = embeds.clone()
        embeds[mask] = img.reshape(-1, img.shape[-1])
        return embeds

    # -- training / inference -------------------------------------------------
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        embeds = self._merge(input_ids, pixel_values)
        return self.llm(inputs_embeds=embeds, attention_mask=attention_mask,
                        labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values,
                 max_new_tokens=256):
        embeds = self._merge(input_ids, pixel_values)
        return self.llm.generate(
            inputs_embeds=embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()


def count_params(module):
    """Return (trainable, total) parameter counts in millions."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable / 1e6, total / 1e6
