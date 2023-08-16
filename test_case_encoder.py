from transformers import pipeline
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast

import os

tokenizer_folder = os.path.join("tokenizer", "chords")
model_folder = os.path.join("models", "decoder")

# Create a Fill mask pipeline
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=tokenizer_folder, pad_token="<PAD>", mask_token="<MASK>")

fill_mask = pipeline(
    "fill-mask",
    model=model_folder,
    tokenizer=tokenizer
)

print(fill_mask("D(D,C#)AF# BD<MASK>B F#F#A(C#,D) (G#,A)(B,C#)G#E"))