from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_catalog import download_model, model_cache_dir
from .runtime import save_api_failure


JSON_OBJECT_PATTERN = re.compile(r'\{.*\}', re.DOTALL)


@dataclass(slots=True)
class ConceptDraft:
    spec: dict[str, str]
    request_prompt: str
    raw_response: str


class LocalConceptBackend:
    _MODELS: dict[tuple[str, str, str], tuple[Any, Any, str]] = {}

    def __init__(self, *, logs_root: Path, profile: dict[str, Any], token: str | None) -> None:
        self._logs_root = logs_root
        self._profile = profile
        self._token = token

    def _save_failure(self, stage: str, payload: dict[str, Any], error: Exception) -> None:
        save_api_failure(self._logs_root, stage, payload, error)

    @staticmethod
    def ensure_dependencies() -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as error:
            raise RuntimeError('backend=local-svg requires transformers and torch in the worker image.') from error

    def _resolve_model_dir(self) -> Path:
        repo_id = str(self._profile['repo_id'])
        target = model_cache_dir(repo_id)
        if target.exists() and any(target.rglob('*.json')):
            return target
        if not bool(self._profile.get('allowAutoDownload', True)):
            raise RuntimeError(f"Model '{repo_id}' is not downloaded. Download it first from Settings.")
        download_model(str(self._profile['preset_id']))
        return target

    def _load_model(self) -> tuple[Any, Any, str]:
        self.ensure_dependencies()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = self._resolve_model_dir()
        device = 'cuda' if self._profile.get('device') == 'cuda' and torch.cuda.is_available() else 'cpu'
        dtype_name = 'float16' if device == 'cuda' and self._profile.get('prefer_float16') else 'float32'
        cache_key = (str(model_dir), device, dtype_name)
        if cache_key in self._MODELS:
            return self._MODELS[cache_key]

        torch_dtype = torch.float16 if dtype_name == 'float16' else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), token=self._token)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=self._token,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device == 'cuda':
            model = model.to('cuda')
        model.eval()
        self._MODELS[cache_key] = (tokenizer, model, device)
        return tokenizer, model, device

    def _build_prompt(
        self,
        *,
        product_name: str,
        description: str,
        notes: str,
        requested_direction: str,
        source_summary: str | None,
        variant_index: int,
        choices: dict[str, list[str]],
    ) -> str:
        return (
            'You are a brand art director for SaaS and company logos. '\
            'Pick only from the allowed values and return one JSON object. '\
            'Prefer simple website/company logos, not mascots, not seals, not posters, not illustrations.\n\n'
            'Return JSON with exactly these keys: direction, palette_name, font_family, shape_kind, wordmark_case, weight, motif, rationale.\n'
            f"direction choices: {', '.join(choices['directions'])}\n"
            f"palette_name choices: {', '.join(choices['palettes'])}\n"
            f"font_family choices: {', '.join(choices['fonts'])}\n"
            f"shape_kind choices: {', '.join(choices['shapes'])}\n"
            'wordmark_case choices: title, upper, lower\n'
            'weight choices: semibold, bold\n'
            'motif should be 1-3 words and abstract, like signal, bridge, grid, path, link, orbit.\n'
            'rationale should be short, under 18 words, and mention why it fits a company or product logo.\n\n'
            f'product_name: {product_name}\n'
            f'description: {description}\n'
            f'notes: {notes or "none"}\n'
            f'requested_direction: {requested_direction or "broad exploration"}\n'
            f'source_candidate: {source_summary or "none"}\n'
            f'variant_index: {variant_index + 1}\n'
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        candidate = text.strip()
        if candidate.startswith('{') and candidate.endswith('}'):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        match = JSON_OBJECT_PATTERN.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _safe_choice(value: Any, allowed: list[str], fallback: str) -> str:
        text = str(value or '').strip()
        for candidate in allowed:
            if text.lower() == candidate.lower():
                return candidate
        return fallback

    @staticmethod
    def _safe_short_text(value: Any, fallback: str, *, max_words: int = 3) -> str:
        text = ' '.join(str(value or '').strip().split())
        if not text:
            return fallback
        words = text.split(' ')
        return ' '.join(words[:max_words])

    def _heuristic_fallback(
        self,
        *,
        product_name: str,
        description: str,
        requested_direction: str,
        variant_index: int,
        choices: dict[str, list[str]],
    ) -> dict[str, str]:
        basis = f'{product_name}|{description}|{requested_direction}|{variant_index}'
        index = sum(ord(char) for char in basis)
        directions = choices['directions']
        palettes = choices['palettes']
        fonts = choices['fonts']
        shapes = choices['shapes']
        tokens = [token.lower() for token in re.findall(r'[a-zA-Z]{3,}', f'{product_name} {description}')]
        motif = next((token for token in tokens if token in {'signal', 'grid', 'path', 'bridge', 'link', 'orbit', 'frame', 'fold'}), 'signal')
        return {
            'direction': requested_direction if requested_direction in directions else directions[index % len(directions)],
            'palette_name': palettes[index % len(palettes)],
            'font_family': fonts[index % len(fonts)],
            'shape_kind': shapes[index % len(shapes)],
            'wordmark_case': ['title', 'upper', 'lower'][index % 3],
            'weight': 'bold' if index % 2 == 0 else 'semibold',
            'motif': motif,
            'rationale': 'Keeps the mark simple and recognizable in a homepage header.',
        }

    def generate_spec(
        self,
        *,
        product_name: str,
        description: str,
        notes: str,
        requested_direction: str,
        source_summary: str | None,
        variant_index: int,
        choices: dict[str, list[str]],
    ) -> ConceptDraft:
        prompt = self._build_prompt(
            product_name=product_name,
            description=description,
            notes=notes,
            requested_direction=requested_direction,
            source_summary=source_summary,
            variant_index=variant_index,
            choices=choices,
        )
        payload = {
            'repo_id': self._profile.get('repo_id'),
            'device': self._profile.get('device'),
            'preset_id': self._profile.get('preset_id'),
            'prompt': prompt,
            'max_new_tokens': self._profile.get('max_new_tokens'),
            'temperature': self._profile.get('temperature'),
            'top_p': self._profile.get('top_p'),
        }
        fallback = self._heuristic_fallback(
            product_name=product_name,
            description=description,
            requested_direction=requested_direction,
            variant_index=variant_index,
            choices=choices,
        )
        try:
            tokenizer, model, device = self._load_model()
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt_text = tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': 'You are a concise brand art director.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = prompt
            inputs = tokenizer(prompt_text, return_tensors='pt')
            if device == 'cuda':
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            generation = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=int(self._profile.get('max_new_tokens') or 220),
                temperature=float(self._profile.get('temperature') or 0.85),
                top_p=float(self._profile.get('top_p') or 0.92),
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_tokens = generation[0][inputs['input_ids'].shape[-1]:]
            raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            raw_payload = self._extract_json(raw_response) or {}
            spec = {
                'direction': self._safe_choice(raw_payload.get('direction'), choices['directions'], fallback['direction']),
                'palette_name': self._safe_choice(raw_payload.get('palette_name'), choices['palettes'], fallback['palette_name']),
                'font_family': self._safe_choice(raw_payload.get('font_family'), choices['fonts'], fallback['font_family']),
                'shape_kind': self._safe_choice(raw_payload.get('shape_kind'), choices['shapes'], fallback['shape_kind']),
                'wordmark_case': self._safe_choice(raw_payload.get('wordmark_case'), ['title', 'upper', 'lower'], fallback['wordmark_case']),
                'weight': self._safe_choice(raw_payload.get('weight'), ['semibold', 'bold'], fallback['weight']),
                'motif': self._safe_short_text(raw_payload.get('motif'), fallback['motif']),
                'rationale': self._safe_short_text(raw_payload.get('rationale'), fallback['rationale'], max_words=18),
            }
            return ConceptDraft(spec=spec, request_prompt=prompt, raw_response=raw_response)
        except Exception as error:
            self._save_failure('local_concept_generation', payload, error)
            return ConceptDraft(spec=fallback, request_prompt=prompt, raw_response='')
