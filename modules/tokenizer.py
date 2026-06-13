import sys
import os
import torch
import logging
import json
from pathlib import Path
from torch import Tensor
from typing import Dict, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BLOOM_MODEL_NAME, MAX_SEQ_LEN, VOCAB_FILE, VOCAB_SIZE

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class NanoJEPATokenizer:
    """
    Two-level tokenizer:
        Level 1 — BLOOM's BPE tokenizer  (250,880 tokens)
        Level 2 — Our reduced vocab      (VOCAB_SIZE tokens)

    Token ID layout (must never change after first build):
        0  = PAD   — padding, ignored by attention
        1  = BOS   — beginning of sequence
        2  = EOS   — end of sequence
        3  = (reserved, unused)
        4  = UNK   — bloom token not in our reduced vocab
        5+ = learned tokens, sorted by corpus frequency
    """

    # ── Class-level constants ──────────────────────────────────────
    # Single source of truth for the reserved-slot layout.
    # FIRST_LEARNED_ID must match the `start=` value in build_vocab().
    FIRST_LEARNED_ID: int = 5

    # Tokens that MUST survive encode→decode regardless of corpus
    # frequency.  Math operators are semantically critical but may be
    # rare in a general corpus.
    FORCED_VOCAB_TOKENS: list[str] = [

        # ── Single operators and symbols ───────────────────────
        # These tokenize 1:1 with their character in BLOOM.
        "=", "+", "-", "*", "/", "%", "$", ".",
        ",", ":", "!", "?", "(", ")", "[", "]",

        # ── Single digits ──────────────────────────────────────
        # BLOOM keeps single digits as individual tokens.
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9",

        # ── Two-digit numbers ──────────────────────────────────
        # CONFIRMED by diagnostic: "36" → single bloom token 4634
        # BLOOM merges common two-digit numbers into one token.
        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
        "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
        "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
        "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
        "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
        "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
        "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",

        # ── Three-digit numbers ────────────────────────────────
        # CONFIRMED by diagnostic: "124" → single bloom token 48061
        "100", "101", "102", "103", "104", "105",
        "110", "115", "120", "121", "122", "123", "124", "125",
        "130", "140", "150", "160", "170", "175", "180", "190",
        "200", "210", "220", "225", "230", "240", "250",
        "300", "350", "400", "450", "500", "600", "700",
        "750", "800", "900", "999",

        # ── Percentage combinations ────────────────────────────
        # CONFIRMED by diagnostic: "15%" → single bloom token 100548
        # BLOOM merges number+% into one token. Must force the
        # merged form, NOT "15" and "%" separately.
        "1%",  "2%",  "3%",  "4%",  "5%",
        "6%",  "7%",  "8%",  "9%",  "10%",
        "11%", "12%", "13%", "14%", "15%",
        "16%", "17%", "18%", "19%", "20%",
        "25%", "30%", "33%", "40%", "45%",
        "50%", "60%", "66%", "70%", "75%",
        "80%", "90%", "95%", "99%", "100%",

        # ── Decimal suffixes ───────────────────────────────────
        # CONFIRMED: ".50" → bloom token 2559 (already in vocab
        # for the test corpus, but forced here for safety).
        ".0",  ".00", ".1",  ".10",
        ".2",  ".20", ".25", ".3",
        ".4",  ".5",  ".50", ".6",
        ".7",  ".75", ".8",  ".9",
        ".99", ".95",
    ]

    # ── Initialization ─────────────────────────────────────────────
    def __init__(self) -> None:
        self._bloom_tok = AutoTokenizer.from_pretrained(BLOOM_MODEL_NAME)
        self.bloom_to_reduced: dict[int, int] = {}
        self.reduced_to_bloom: dict[int, int] = {}
        self.pad_id: int = 0
        self.bos_id: int = 1
        self.eos_id: int = 2
        self.unk_id: int = 4
        self._vocab_ready: bool = False

    # ── Vocabulary Management ──────────────────────────────────────
    def build_vocab(self, texts: list[str]) -> None:
        """
        Build a reduced vocabulary from corpus frequency, then persist
        it to VOCAB_FILE.  On subsequent calls the file is loaded
        instead of rebuilt.

        Slot layout after build:
            IDs 0-4   reserved (PAD / BOS / EOS / unused / UNK)
            IDs 5+    learned tokens, high-frequency first

        Forced tokens (FORCED_VOCAB_TOKENS) have their counts inflated
        by +10,000 so they always make the top-K cut even when rare.
        """
        if VOCAB_FILE.exists():
            logger.info(
                f"Vocab file found at {VOCAB_FILE}. "
                "Loading instead of rebuilding."
            )
            self.load_vocab()
            return

        logger.info(
            f"Building vocab from {len(texts):,} texts. "
            "This runs once and saves to disk..."
        )

        # ── Step 1: Count token frequencies ──────────────────────
        token_counts: dict[int, int] = {}
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"  Tokenizing text {i:,}/{len(texts):,}...")

            ids: list[int] = self._bloom_tok.encode(
                text,
                add_special_tokens=False   # BOS/EOS added later by encode()
            )
            for token_id in ids:
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

        # ── Step 2: Force-include critical symbol tokens ──────────
        # Tokenize each forced symbol → get its bloom ID(s) → inflate
        # count so it survives the top-K sort below.
        forced_bloom_ids: set[int] = set()
        for symbol in self.FORCED_VOCAB_TOKENS:
            symbol_ids = self._bloom_tok.encode(
                symbol, add_special_tokens=False
            )
            for sid in symbol_ids:
                forced_bloom_ids.add(sid)
                token_counts[sid] = token_counts.get(sid, 0) + 10_000

        logger.info(
            f"  Force-including {len(forced_bloom_ids)} bloom IDs "
            f"for {len(self.FORCED_VOCAB_TOKENS)} critical symbols."
        )

        # ── Step 3: Sort by frequency, keep top (VOCAB_SIZE - 5) ──
        # IDs 0-4 are reserved; learned tokens start at FIRST_LEARNED_ID.
        max_learned = VOCAB_SIZE - self.FIRST_LEARNED_ID
        sorted_tokens = sorted(
            token_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_learned]

        # ── Step 4: Build bidirectional maps ──────────────────────
        self.bloom_to_reduced = {}
        self.reduced_to_bloom = {}

        for new_id, (bloom_id, _count) in enumerate(
            sorted_tokens, start=self.FIRST_LEARNED_ID
        ):
            self.bloom_to_reduced[bloom_id] = new_id
            self.reduced_to_bloom[new_id]   = bloom_id

        # ── Step 5: Verify forced tokens survived the cut ─────────
        missing_forced = forced_bloom_ids - set(self.bloom_to_reduced.keys())
        if missing_forced:
            logger.warning(
                f"  {len(missing_forced)} forced bloom IDs did NOT make it "
                f"into vocab (VOCAB_SIZE={VOCAB_SIZE} may be too small): "
                f"{missing_forced}"
            )
        else:
            logger.info(
                f"  ✓ All {len(forced_bloom_ids)} forced symbol IDs "
                f"are in vocab."
            )

        # ── Step 6: Persist to disk ───────────────────────────────
        vocab_data = {
            "bloom_to_reduced": {
                str(k): v for k, v in self.bloom_to_reduced.items()
            },
            "reduced_to_bloom": {
                str(k): v for k, v in self.reduced_to_bloom.items()
            },
        }
        VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)

        self._vocab_ready = True
        logger.info(
            f"Vocab built: {len(self.bloom_to_reduced):,} tokens "
            f"saved to {VOCAB_FILE}"
        )

    def load_vocab(self) -> None:
        """Load a previously built vocabulary from VOCAB_FILE."""
        if not VOCAB_FILE.exists():
            raise FileNotFoundError(
                f"Vocab file not found at {VOCAB_FILE}. "
                "Run build_vocab() first."
            )

        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # JSON keys are always strings; convert back to int
        self.bloom_to_reduced = {
            int(k): v
            for k, v in vocab_data["bloom_to_reduced"].items()
        }
        self.reduced_to_bloom = {
            int(k): v
            for k, v in vocab_data["reduced_to_bloom"].items()
        }
        self._vocab_ready = True
        logger.info(
            f"Vocab loaded from {VOCAB_FILE}. "
            f"{len(self.bloom_to_reduced):,} tokens mapped."
        )

    # ── Internal Encoding Helpers ──────────────────────────────────
    def _remap_ids(
        self,
        bloom_ids: list[int],
        log_unknowns: bool = False      # ← required by Cell 5 diagnostics
    ) -> list[int]:
        """
        Map bloom token IDs → reduced vocab IDs.

        Tokens not in our reduced vocab become self.unk_id (4).

        Args:
            bloom_ids:    Raw token IDs from the BLOOM tokenizer.
            log_unknowns: When True, log the surface form of every
                          token that maps to UNK.  Keep False in
                          production — intended for coverage debugging.
        """
        reduced_ids: list[int] = []
        unk_surface_forms: list[str] = []

        for bid in bloom_ids:
            if bid in self.bloom_to_reduced:
                reduced_ids.append(self.bloom_to_reduced[bid])
            else:
                reduced_ids.append(self.unk_id)
                if log_unknowns:
                    surface = self._bloom_tok.decode([bid])
                    unk_surface_forms.append(repr(surface))

        if log_unknowns and unk_surface_forms:
            logger.debug(
                f"  _remap_ids: {len(unk_surface_forms)} UNK tokens: "
                f"{unk_surface_forms}"
            )

        return reduced_ids

    # ── Public Encoding API ────────────────────────────────────────
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int = MAX_SEQ_LEN,
    ) -> dict[str, Tensor]:
        """
        Encode a single string → fixed-length padded tensors.

        Returns:
            {
              "input_ids":      LongTensor [max_length]
              "attention_mask": LongTensor [max_length]  (1=real, 0=pad)
            }
        """
        if not self._vocab_ready:
            raise RuntimeError(
                "Vocabulary not built. "
                "Call build_vocab() or load_vocab() first."
            )

        if not text or not text.strip():
            logger.warning(
                "Empty string passed to encode(). "
                "Returning minimal BOS+EOS tensor."
            )
            ids = [self.bos_id, self.eos_id] if add_bos else [self.eos_id]
            return self._pad_and_mask(ids, max_length)

        # Step 1: BLOOM tokenization
        bloom_ids: list[int] = self._bloom_tok.encode(
            text, add_special_tokens=False
        )

        # Step 2: Remap to reduced vocab
        reduced_ids: list[int] = self._remap_ids(bloom_ids)

        # Step 3: Wrap with special tokens
        if add_bos:
            reduced_ids = [self.bos_id] + reduced_ids
        if add_eos:
            reduced_ids = reduced_ids + [self.eos_id]

        return self._pad_and_mask(reduced_ids, max_length)

    def batch_encode(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int = MAX_SEQ_LEN,
    ) -> dict[str, Tensor]:
        """
        Encode a list of strings → batched tensors.

        Returns:
            {
              "input_ids":      LongTensor [B, max_length]
              "attention_mask": LongTensor [B, max_length]
            }
        """
        if not texts:
            raise ValueError("batch_encode() received an empty list.")

        encoded_list = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos,
                        max_length=max_length)
            for text in texts
        ]

        return {
            "input_ids": torch.stack(
                [e["input_ids"] for e in encoded_list], dim=0
            ),
            "attention_mask": torch.stack(
                [e["attention_mask"] for e in encoded_list], dim=0
            ),
        }

    # ── Internal Tensor Preparation ───────────────────────────────
    def _pad_and_mask(
        self,
        ids: list[int],
        max_length: int,
    ) -> dict[str, Tensor]:
        """
        Truncate or right-pad `ids` to exactly `max_length`.

        Truncation rule: keep the first max_length tokens; force the
        last position to EOS so the model always sees a sequence end.

        Padding rule: append PAD (0) until length == max_length.
        """
        # CASE 1: Too long → truncate, preserve EOS at boundary
        if len(ids) > max_length:
            ids = ids[:max_length]
            if ids[-1] != self.eos_id:
                ids[-1] = self.eos_id

        real_length = len(ids)

        # CASE 2: Too short → right-pad with PAD token (0)
        padding_needed = max_length - real_length
        padded_ids     = ids + [self.pad_id] * padding_needed
        mask           = [1]  * real_length + [0] * padding_needed

        return {
            "input_ids":      torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask,       dtype=torch.long),
        }

    # ── Decoding API ───────────────────────────────────────────────
    def decode(self, token_ids: Union[Tensor, list[int]]) -> str:
        """
        Convert a tensor / list of reduced IDs back to a string.

        Special tokens (PAD, BOS, EOS, UNK) are stripped before
        passing the remaining bloom IDs to BLOOM's own decoder.
        """
        if isinstance(token_ids, Tensor):
            ids_list: list[int] = token_ids.tolist()
        else:
            ids_list = list(token_ids)

        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        bloom_ids = [
            self.reduced_to_bloom[rid]
            for rid in ids_list
            if rid not in special_ids and rid in self.reduced_to_bloom
        ]

        if not bloom_ids:
            return ""

        return self._bloom_tok.decode(bloom_ids, skip_special_tokens=True)

    # ── Analysis Utilities ─────────────────────────────────────────
    def compute_coverage(self, texts: list[str]) -> float:
        """
        Return the fraction of bloom tokens that map to our reduced
        vocab (i.e. are NOT mapped to UNK).

        Useful for evaluating whether VOCAB_SIZE is large enough for
        a given corpus.
        """
        if not self._vocab_ready:
            raise RuntimeError("Call build_vocab() or load_vocab() first.")

        total   = 0
        unknown = 0

        for text in texts:
            bloom_ids = self._bloom_tok.encode(text, add_special_tokens=False)
            for bid in bloom_ids:
                total += 1
                if bid not in self.bloom_to_reduced:
                    unknown += 1

        if total == 0:
            logger.warning("No tokens found in provided texts.")
            return 0.0

        coverage = (total - unknown) / total
        logger.info(
            f"Coverage: {coverage:.4%} "
            f"({total - unknown:,}/{total:,} tokens in vocab, "
            f"{unknown:,} mapped to <unk>)"
        )
        return coverage