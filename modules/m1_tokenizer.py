import torch
from transformers import AutoTokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Tokenizer:
    
    MODEL_NAME = "bigscience/bloom-560m"

    def __init__(self):

        print(f"Loading tokenizer: {self.MODEL_NAME}")

        try:
            # Load tokenizer from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
        except OSError as e:
            print(f"\nERROR: Cannot load tokenizer.")
            print(f"  Fix 1: Check internet connection")
            print(f"  Fix 2: pip install --upgrade transformers")
            raise e

        # If padding token is missing, fix it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Always add padding on the right side
        self.tokenizer.padding_side = 'right'

        self.vocab_size   = len(self.tokenizer)
        self.max_length   = config.MAX_SEQ_LEN
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        print(f"  type         : {type(self.tokenizer).__name__}")
        print(f"  vocab_size   : {self.vocab_size:,}")
        print(f"  max_length   : {self.max_length}")
        print(f"  pad_token    : '{self.tokenizer.pad_token}' "
              f"(id={self.pad_token_id})")
        print(f"  padding_side : {self.tokenizer.padding_side}")


    # -------------------------
    # Convert text → numbers
    # -------------------------
    
    def encode(self, text: str) -> dict:
   
        if not isinstance(text, str):
            raise TypeError(
                f"encode() needs a string.\n"
                f"  Got : {type(text).__name__} = {repr(text)}"
            )
        if len(text.strip()) == 0:
            raise ValueError("encode() got empty string.")

        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'raw_text':       text
        }


    # -------------------------
    # Convert numbers → text
    # -------------------------
    
    def decode(self, token_ids) -> str:
   
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if not isinstance(token_ids, list):
            raise TypeError(
                f"decode() needs tensor or list.\n"
                f"  Got : {type(token_ids).__name__}"
            )

        # remove padding tokens
        clean_ids = [
            tid for tid in token_ids
            if tid != self.pad_token_id
        ]

        return self.tokenizer.decode(
            clean_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )


     # -------------------------
    # Batch (many texts at once)
    # -------------------------
    
    def batch_encode(self, texts: list) -> dict:

        if not isinstance(texts, list):
            raise TypeError(
                f"batch_encode() needs a list.\n"
                f"  Got : {type(texts).__name__}"
            )
        if len(texts) == 0:
            raise ValueError("batch_encode() got empty list.")

        for i, item in enumerate(texts):
            if not isinstance(item, str):
                raise TypeError(
                    f"Item [{i}] must be string.\n"
                    f"  Got : {type(item).__name__} = {repr(item)}"
                )

        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'raw_texts':      list(texts)
        }


    def __repr__(self) -> str:
        return (
            f"Tokenizer(\n"
            f"  model      = {self.MODEL_NAME}\n"
            f"  vocab_size = {self.vocab_size:,}\n"
            f"  max_length = {self.max_length}\n"
            f"  pad_token  = '{self.tokenizer.pad_token}'\n"
            f"  pad_side   = {self.tokenizer.padding_side}\n"
            f")"
        )


