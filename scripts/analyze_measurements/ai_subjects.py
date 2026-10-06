"""Classify AI-looking subject names for analysis and publication policy.

This is a name-based heuristic, not provider metadata or a property of the
collected measurements.  A benchmark is classified as having AI subjects when
at least one subject's ``display_name`` contains one of the known keywords.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Keywords identifying an AI model/agent by name. A subject is treated as an AI
# system if any of these appears (case-insensitive substring) in its
# ``display_name``. Short/ambiguous tokens (o1, yi, phi, opt, ...) are hyphenated
# to avoid matching ordinary words. Extend this list as new model families ship.
AI_SUBJECT_KEYWORDS = [
    # OpenAI
    "gpt", "chatgpt", "davinci", "babbage", "curie", "o1-", "o3-", "o4-",
    # Anthropic
    "claude", "opus", "sonnet", "haiku", "anthropic",
    # Google
    "gemini", "gemma", "palm",
    # Meta / common open families
    "llama", "qwen", "qwq", "mistral", "mixtral", "deepseek", "falcon",
    "vicuna", "alpaca", "koala", "guanaco", "wizardlm", "openchat",
    "zephyr", "starling", "tulu", "dolphin", "hermes",
    # Mistral's newer lines
    "pixtral", "codestral", "ministral", "magistral", "devstral",
    # Other vendors / families
    "baichuan", "internlm", "grok", "kimi", "itformer",
    "glm", "chatglm", "jamba", "dbrx", "olmo", "reka",
    "minimax", "ernie", "hunyuan", "doubao", "nemotron",
    "bloom", "pythia", "stablelm", "phi-", "yi-", "moonshot",
    # Vision-language / multimodal families
    "llava", "kosmos", "cogvlm", "moondream",
    "flamingo", "idefics", "fuyu", "paligemma", "internvl", "minigpt",
    "instructblip", "qwen-vl",
    # Vision foundation / self-supervised encoders (used as classifiers/featurizers)
    "clip", "dinov2", "dino ", "siglip", "vit-",
    # Text-to-image / image-generation families
    "sdxl", "stable-diffusion", "stable diffusion", "midjourney",
    "dall-e", "dalle", "kandinsky", "dreamlike", "playground-v",
    # Classic CNN image-classifier architectures
    "resnet", "vgg", "densenet", "alexnet", "googlenet", "efficientnet",
    "inception", "mobilenet",
    # Tabular foundation / in-context-learning models (AI classifiers/regressors)
    "tabicl", "tabpfn", "tabdpt", "tabflex", "mitra", "limix",
    # LLM-based agent frameworks / agentic scaffolds (subject = the agent system,
    # typically orchestrating a generative LLM such as gpt-4o)
    "confagents", "colacare", "mdagents", "medagent", "medagents",
]


def display_name_is_ai(name: object) -> bool:
    """Return whether ``name`` contains a known AI model or agent keyword."""
    if not isinstance(name, str):
        return False
    low = name.lower()
    return any(keyword in low for keyword in AI_SUBJECT_KEYWORDS)


def subjects_include_ai(subjects: pd.DataFrame) -> bool:
    """Return whether any subject has an AI-looking ``display_name``."""
    if "display_name" not in subjects.columns:
        raise ValueError("subjects table has no 'display_name' column")
    return any(display_name_is_ai(name) for name in subjects["display_name"])


def benchmark_has_ai_subjects(benchmark_dir: str | Path) -> bool:
    """Read ``subjects.parquet`` and apply :func:`subjects_include_ai`."""
    tables_dir = Path(benchmark_dir) / "formatted_tables"
    if not tables_dir.is_dir():
        tables_dir = Path(benchmark_dir)
    subjects = pd.read_parquet(
        tables_dir / "subjects.parquet",
        columns=["display_name"],
    )
    return subjects_include_ai(subjects)
