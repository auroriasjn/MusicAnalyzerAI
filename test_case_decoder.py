from transformers import pipeline
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast

import os

tokenizer_folder = os.path.join("tokenizer", "roman_nums")
model_folder = os.path.join("models", "decoder")

# Create a Fill mask pipeline
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=tokenizer_folder, pad_token="<PAD>", mask_token="<MASK>")

fill_mask = pipeline(
    "fill-mask",
    model=model_folder,
    tokenizer=tokenizer,
)

print(fill_mask("(F) I ii7 V43 (d)[I=III] vii°6 V6 <MASK> ii°65 V (F)[i=vi] V65/V V I64 V42 I6 vi65 (V/V V7/V) V"))
print(fill_mask("(D♭) I (I6 I) IV <MASK> IV6 I V"))