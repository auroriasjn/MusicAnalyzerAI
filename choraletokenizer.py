from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    ByteLevelBPETokenizer
)
import os
from choraledataset import ChoraleDataset

SHARP_NOTES = [(str(chr(c)) + "#") for c in range(65, 72)]
FLAT_NOTES = [(str(chr(c)) + "♭") for c in range(65, 72)]
class ChoraleTokenizer:
    def __init__(self, dataset=None, is_input=True, preload=False):
        self.special_tokens = ["<UNK>", "<BOS>", "<EOS>", "<PAD>", "<MASK>"]
        if not is_input:
            minor_key_langs = ["(a♭)", "(a)", "(a#)", "(b♭)", "(b)", "(c)", "(c#)", "(d)",
                               "(d#)", "(e♭)", "(e)", "(f)", "(f#)", "(g)", "(g#)"]
            major_key_langs = ["(A♭)", "(A)",  "(B♭)", "(B)", "(C♭)", "(C)", "(C#)", "(D♭)",
                               "(D)", "(E♭)", "(E)", "(F)", "(F#)", "(G)", "(G#)"]
            self.special_tokens += minor_key_langs
            self.special_tokens += major_key_langs

        self.dataset = dataset
        self.is_input = is_input

        vocab = None
        merges = None
        if preload and self.dataset is not None:
            vocab = self.dataset.input_vocab_dict if is_input else self.dataset.output_vocab_dict

        self.tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        self.trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=5,
                        show_progress=True,
                        special_tokens=self.special_tokens,
                        continuing_subword_prefix="**")

        self.train()

        self.bos_token_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_token_id = self.tokenizer.token_to_id("<EOS>")
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS>:0 $A:0 <EOS>:0",
            pair=f"<BOS>:0 $A:0 <EOS>:0 $B:1 <EOS>:1",
            special_tokens=[
                ("<BOS>", self.bos_token_id),
                ("<EOS>", self.eos_token_id),
            ]
        )

        self.tokenizer.bos_token_id = self.bos_token_id
        self.tokenizer.eos_token_id = self.eos_token_id

    def batch_iterator(self, batch_size=16):
        if self.dataset is not None and isinstance(self.dataset, ChoraleDataset):
            for i in range(0, len(self.dataset), batch_size):
                yield self.dataset[i: i + batch_size]["translation"][0] if self.is_input \
                    else self.dataset[i: i + batch_size]["translation"][1]

    def train(self):
        self.tokenizer.train_from_iterator(self.batch_iterator(16), trainer=self.trainer)

    def save_model(self, tokenizer_folder='tokenizer', filename="tokenizer.json"):
        if not os.path.exists(tokenizer_folder):
            os.makedirs(tokenizer_folder)

        self.tokenizer.save(os.path.join(tokenizer_folder, filename))
