from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from transformers import AutoTokenizer


@dataclass
class TokenizationResult:
    input_ids: List[int]


class StreamingTokenizer:
    """Simple wrapper around Hugging Face tokenizers."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self._tokenizer.pad_token = "<|pad|>"

    @classmethod
    def from_pretrained(cls, name: str, **kwargs) -> "StreamingTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False,
            **kwargs,
        )
        return cls(tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        value = self._tokenizer.bos_token_id or self._tokenizer.cls_token_id
        if value is None:
            value = self._tokenizer.pad_token_id
        return value

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id or self._tokenizer.sep_token_id

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> List[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(self, ids: Iterable[int]) -> str:
        return self._tokenizer.decode(list(ids))
