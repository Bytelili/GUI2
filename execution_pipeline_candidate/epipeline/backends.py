from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .actions import ParsedAction, parse_action
from .prompts import SYSTEM_PROMPT, build_prompt


@dataclass(frozen=True)
class Prediction:
    raw_text: str
    action: ParsedAction
    elapsed_seconds: float
    prompt_tokens: int
    response_tokens: int
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


class ModelBackend(Protocol):
    paper_eligible: bool

    def predict(
        self,
        task: dict[str, Any],
        *,
        screenshot: str,
        xml_path: str,
        previous_actions: list[str],
    ) -> Prediction: ...


class ReplayModelBackend:
    """Golden-action backend used only to test the experiment machinery."""

    paper_eligible = False

    def predict(
        self,
        task: dict[str, Any],
        *,
        screenshot: str,
        xml_path: str,
        previous_actions: list[str],
    ) -> Prediction:
        del screenshot, xml_path
        target = task.get("target") if isinstance(task.get("target"), dict) else {}
        actions = [str(value) for value in target.get("actions") or []]
        raw = actions[len(previous_actions)] if len(previous_actions) < len(actions) else "finished()"
        return Prediction(raw, parse_action(raw), 0.0, 0, 0)


class LlamaFactoryBackend:
    paper_eligible = True

    def __init__(self, inference: dict[str, Any], model: dict[str, Any]):
        project_root = Path(__file__).resolve().parents[2]
        import sys

        sys.path.insert(0, str(project_root / "LLaMA-Factory" / "src"))
        try:
            from llamafactory.chat import ChatModel
        except ImportError as error:
            raise RuntimeError("LLaMA-Factory is required for the llamafactory inference backend") from error
        base_model = Path(str(inference["base_model"]))
        if not base_model.is_dir():
            raise FileNotFoundError(f"Base model directory does not exist: {base_model}")
        adapter = str(model.get("adapter") or "")
        if adapter and not (Path(adapter) / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Adapter is incomplete: {adapter}")
        args = {
            "model_name_or_path": str(base_model),
            "template": str(inference.get("template") or "qwen2_vl"),
            "finetuning_type": "lora",
            "stage": "sft",
            "infer_backend": "huggingface",
            "infer_dtype": str(inference.get("infer_dtype") or "bfloat16"),
            "trust_remote_code": True,
            "do_sample": False,
            "max_new_tokens": int(inference.get("max_new_tokens") or 128),
        }
        if adapter:
            args["adapter_name_or_path"] = adapter
        self.model = ChatModel(args)
        self.max_new_tokens = int(args["max_new_tokens"])
        self.prompt_style = str(inference.get("prompt_style") or "training_aligned")

    def predict(
        self,
        task: dict[str, Any],
        *,
        screenshot: str,
        xml_path: str,
        previous_actions: list[str],
    ) -> Prediction:
        if not screenshot or not Path(screenshot).is_file():
            return Prediction(
                "",
                parse_action(""),
                0.0,
                0,
                0,
                error=f"Missing screenshot observation: {screenshot!r}",
            )
        prompt = build_prompt(task, screenshot, xml_path, previous_actions, style=self.prompt_style)
        started = time.perf_counter()
        try:
            responses = self.model.chat(
                [{"role": "user", "content": prompt}],
                system=SYSTEM_PROMPT,
                images=[screenshot],
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
            response = responses[0]
            raw = str(response.response_text or "").strip()
            return Prediction(
                raw,
                parse_action(raw),
                time.perf_counter() - started,
                int(response.prompt_length or 0),
                int(response.response_length or 0),
            )
        except Exception as error:
            return Prediction(
                "",
                parse_action(""),
                time.perf_counter() - started,
                0,
                0,
                error=f"{type(error).__name__}: {error}",
            )


def create_model_backend(inference: dict[str, Any], model: dict[str, Any]) -> ModelBackend:
    backend = str(inference.get("backend") or "")
    if backend == "replay":
        return ReplayModelBackend()
    if backend == "llamafactory":
        return LlamaFactoryBackend(inference, model)
    raise ValueError(f"Unsupported inference backend: {backend}")
