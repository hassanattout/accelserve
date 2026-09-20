from __future__ import annotations


class ByteTokenizer:
    """A tiny dependency-free UTF-8 tokenizer for the reference runtime.

    IDs 0, 1 and 2 are reserved for PAD/BOS/EOS. Bytes are mapped to 3..258.
    This tokenizer is deliberately simple so the runtime can be tested without
    downloading model assets. Production model adapters can replace it.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    BYTE_OFFSET = 3
    VOCAB_SIZE = 259

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        ids = [b + self.BYTE_OFFSET for b in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.BOS_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        values = [
            token_id - self.BYTE_OFFSET
            for token_id in ids
            if self.BYTE_OFFSET <= token_id < self.VOCAB_SIZE
        ]
        return bytes(values).decode("utf-8", errors="replace")
