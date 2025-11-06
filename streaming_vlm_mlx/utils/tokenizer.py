from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class TokenizerOutput:
    input_ids: List[int]
    attention_mask: List[int]


class StreamingTokenizer:
    """Adapter around any python tokenizer to keep dependencies optional."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab

    def encode(self, pieces: Sequence[str]) -> TokenizerOutput:
        tokens: List[int] = []
        for piece in pieces:
            tokens.append(self.vocab.get(piece, self.vocab["<unk>"]))
        mask = [1] * len(tokens)
        return TokenizerOutput(input_ids=tokens, attention_mask=mask)
