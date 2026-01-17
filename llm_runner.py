# llm_runner.py (add/replace relevant parts)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import json
import re


@dataclass
class RunResult:
    pred: str
    raw: str


def _extract_label(raw_text: str, label_names):
    """
    Robustly extract label from LLM output.

    Supports:
    1) Pure JSON: {"label": "class_1"}
    2) JSON with extra text
    3) Plain text containing class_k
    4) Falls back to invalid
    """

    if raw_text is None:
        return "__invalid__"

    text = raw_text.strip()

    # ---------- 1. Try JSON parsing ----------
    try:
        # 有些模型会输出多余文本，这里尝试截取 JSON 子串
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict) and "label" in obj:
                label = obj["label"]
                if label in label_names:
                    return label
    except Exception:
        pass

    # ---------- 2. Regex search for class_k ----------
    m = re.search(r"class_\d+", text)
    if m:
        label = m.group(0)
        if label in label_names:
            return label

    # ---------- 3. Last resort: exact match ----------
    text_clean = text.strip().strip('"').strip("'")
    if text_clean in label_names:
        return text_clean

    # ---------- 4. Give up gracefully ----------
    return "__invalid__"



class HFInstructRunner:
    """
    本地小 Instruct LLM（推荐 Qwen2.5-0.5B-Instruct 之类）
    - 只读 prompt（LLM-only）
    - 强约束输出：尽量只输出 class_x
    """

    def __init__(
        self,
        label_names: List[str],
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",           # "cpu" / "cuda" / "auto"
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ):
        self.label_names = label_names
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch 

        self.tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        if device == "cuda":
            self.model = self.model.cuda()
        elif device == "auto":
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        self.model.eval()

    def run(self, prompt: str) -> RunResult:
        import torch

        # 给模型一个“更硬”的输出约束（尤其重要！）
        hard_constraint = ( 
            "\n\n"
            f"Candidate labels: {', '.join(self.label_names)}\n" 
            "Return ONLY one label from the candidates.\n"
            "Return format: {\"label\": \"class_k\"}\n"
        )  

        user_content = prompt.rstrip() + hard_constraint #   
        # 如果模型/Tokenizer支持 chat_template，用它（Instruct 模型更稳）
        if hasattr(self.tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a strict classifier."},
                {"role": "user", "content": user_content},
            ]
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # fallback：普通拼接（适配非 chat 模型）
            text = "You are a strict classifier.\n" + user_content + "\nAnswer:"

        inputs = self.tok(text, return_tensors="pt")
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=max(self.temperature, 1e-6),
                pad_token_id=self.tok.eos_token_id,
            )

        decoded = self.tok.decode(out[0], skip_special_tokens=True)
        # 尽量拿到“新生成部分”
        raw = decoded[len(self.tok.decode(inputs["input_ids"][0], skip_special_tokens=True)) :].strip()
        if not raw:
            raw = decoded.strip()

        pred = _extract_label(raw, self.label_names)
        return RunResult(pred=pred, raw=raw)
