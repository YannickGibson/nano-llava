"""Datasets, prompt formatting, and the batch collator for the two stages.

Stage 1 (caption pretraining) uses Flickr30k: short "describe this image"
turns that teach the projector to map CLIP features into words.
Stage 2 (instruction tuning) uses the LLaVA-Instruct conversation mix.

Both stages share one formatting path: a chat conversation is rendered with
Qwen's chat template, the single `<image>` placeholder is expanded to 196
image tokens, and labels are masked to -100 everywhere except the assistant
turns so loss is computed only on what the model should generate.
"""

import random

import torch
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from model import IMAGE_TOKEN, NUM_IMAGE_TOKENS, VISION_MODEL

SYSTEM_PROMPT = "You are a helpful assistant that can see and describe images."
MAX_SEQ_LEN = 1024

# The last HELDOUT_IMAGES Flickr30k images are never used for training, so
# grid.py can render the qualitative examples on genuinely unseen data.
HELDOUT_IMAGES = 100

# Per-stage dataset configs (splatted into build_dataset).
STAGE_CONFIG = {
    1: dict(hf_id="lmms-lab/flickr30k", max_samples=80_000),
    2: dict(hf_id="HuggingFaceH4/llava-instruct-mix-vsft", max_samples=50_000),
}


# ----------------------------------------------------------------------------
# Conversation formatting
# ----------------------------------------------------------------------------
def _expand_image_tokens(text):
    """Replace the single `<image>` placeholder with 196 image tokens."""
    return text.replace(IMAGE_TOKEN, IMAGE_TOKEN * NUM_IMAGE_TOKENS, 1)


def _encode_chat(tokenizer, messages, add_generation_prompt=False):
    """Render a conversation with Qwen's chat template and tokenize to ids."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def tokenize_conversation(tokenizer, messages):
    """Tokenize a chat conversation, masking labels outside assistant turns.

    `messages` is a list of {"role", "content"} dicts; the user turn carries a
    single `<image>` placeholder. Returns (input_ids, labels) as long tensors.
    The assistant spans are found by diffing the rendered prefix lengths, which
    is robust to whatever Qwen's chat template emits around each turn.
    """
    messages = [
        {**m, "content": _expand_image_tokens(m["content"])} for m in messages
    ]
    full = _encode_chat(tokenizer, messages)
    labels = [-100] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = _encode_chat(tokenizer, messages[:i], add_generation_prompt=True)
        upto = _encode_chat(tokenizer, messages[: i + 1])
        for j in range(len(prefix), min(len(upto), len(full))):
            labels[j] = full[j]
    input_ids = torch.tensor(full[:MAX_SEQ_LEN], dtype=torch.long)
    labels = torch.tensor(labels[:MAX_SEQ_LEN], dtype=torch.long)
    return input_ids, labels


# ----------------------------------------------------------------------------
# Stage datasets
# ----------------------------------------------------------------------------
class Stage1Dataset(Dataset):
    """Flickr30k image-caption pairs (projector pretraining)."""

    def __init__(self, hf_id, tokenizer, processor, max_samples):
        from datasets import load_dataset

        ds = load_dataset(hf_id, split="test")          # one split, all images
        # Train on every image but the held-out tail; one (image, caption)
        # entry per caption, shuffled, then capped to max_samples.
        n_train = len(ds) - HELDOUT_IMAGES
        pairs = [(i, c) for i in range(n_train) for c in range(5)]
        random.Random(0).shuffle(pairs)
        self.pairs = pairs[:max_samples]
        self.ds, self.tokenizer, self.processor = ds, tokenizer, processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_idx, cap_idx = self.pairs[idx]
        row = self.ds[img_idx]
        captions = row["caption"]
        caption = captions[cap_idx % len(captions)].strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{IMAGE_TOKEN}\nDescribe this image."},
            {"role": "assistant", "content": caption},
        ]
        input_ids, labels = tokenize_conversation(self.tokenizer, messages)
        pixels = self.processor(images=row["image"].convert("RGB"),
                                return_tensors="pt").pixel_values[0]
        return dict(input_ids=input_ids, labels=labels, pixel_values=pixels)


def _parse_vsft_messages(raw_messages):
    """Flatten LLaVA-mix messages into {role, content} dicts with one <image>.

    Each message's content is a list of typed parts (text / image). Image
    parts become an `<image>` marker; the marker is forced into the first
    user turn exactly once.
    """
    messages, image_used = [], False
    for msg in raw_messages:
        content = msg["content"]
        if isinstance(content, str):
            text = content
        else:
            parts = []
            for p in content:
                if p.get("type") == "image":
                    parts.append(IMAGE_TOKEN)
                elif p.get("text"):
                    parts.append(p["text"])
            text = "\n".join(parts)
        if IMAGE_TOKEN in text:
            image_used = True
        messages.append({"role": msg["role"], "content": text})
    if not image_used and messages:
        first_user = next(m for m in messages if m["role"] == "user")
        first_user["content"] = f"{IMAGE_TOKEN}\n{first_user['content']}"
    return messages


class Stage2Dataset(Dataset):
    """LLaVA-Instruct conversation mix (instruction tuning)."""

    def __init__(self, hf_id, tokenizer, processor, max_samples):
        from datasets import load_dataset

        ds = load_dataset(hf_id, split="train")
        self.ds = ds.select(range(min(max_samples, len(ds))))
        self.tokenizer, self.processor = tokenizer, processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        messages = _parse_vsft_messages(row["messages"])
        # Ensure the system prompt leads the conversation.
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        input_ids, labels = tokenize_conversation(self.tokenizer, messages)
        image = row["images"][0].convert("RGB")
        pixels = self.processor(images=image,
                                return_tensors="pt").pixel_values[0]
        return dict(input_ids=input_ids, labels=labels, pixel_values=pixels)


def build_inference_inputs(tokenizer, processor, image, prompt):
    """Format one (image, prompt) pair for generation: no assistant turn yet."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _expand_image_tokens(f"{IMAGE_TOKEN}\n{prompt}")},
    ]
    ids = _encode_chat(tokenizer, messages, add_generation_prompt=True)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixels = processor(images=image.convert("RGB"),
                       return_tensors="pt").pixel_values
    return input_ids, attention_mask, pixels


def build_processor():
    return CLIPImageProcessor.from_pretrained(VISION_MODEL)


def build_dataset(stage, tokenizer, processor, max_samples=None):
    cfg = STAGE_CONFIG[stage]
    n = max_samples or cfg["max_samples"]
    cls = Stage1Dataset if stage == 1 else Stage2Dataset
    return cls(cfg["hf_id"], tokenizer, processor, n)


# ----------------------------------------------------------------------------
# Collator
# ----------------------------------------------------------------------------
class VLMCollator:
    """Right-pad a batch of variable-length conversations.

    input_ids pad with the tokenizer pad id, labels with -100, attention_mask
    with 0. pixel_values are a fixed shape, so they just stack.
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = max_len - len(ids)
            input_ids.append(torch.cat([ids, ids.new_full((pad,), self.pad_id)]))
            labels.append(torch.cat([lab, lab.new_full((pad,), -100)]))
            attn.append(torch.cat([torch.ones(len(ids), dtype=torch.long),
                                   torch.zeros(pad, dtype=torch.long)]))
        input_ids = torch.stack(input_ids)
        # Every sample must carry exactly one full image (196 image tokens).
        n_img = (input_ids == self.image_token_id).sum(dim=1)
        assert (n_img == NUM_IMAGE_TOKENS).all(), f"bad image-token counts: {n_img}"
        return dict(
            input_ids=input_ids,
            labels=torch.stack(labels),
            attention_mask=torch.stack(attn),
            pixel_values=torch.stack([b["pixel_values"] for b in batch]),
        )
