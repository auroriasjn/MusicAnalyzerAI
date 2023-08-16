from choraledataset import ChoraleDataset
import evaluate
import datasets

#Tokenizer
from transformers import PreTrainedTokenizerFast

#Encoder-Decoder Model
from transformers import EncoderDecoderModel

#Training
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from datasets import Dataset
import torch
import logging

# Initialization of basic hyperparameters
TRAIN_BATCH_SIZE = 16   #32    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 4    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 20        # number of epochs to train (default: 10)
VAL_EPOCHS = 1
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 100
SUMMARY_LEN = 7

batch_size=TRAIN_BATCH_SIZE
encoder_max_length=MAX_LEN
decoder_max_length=SUMMARY_LEN

mps_device = torch.device("mps:0")

# Initial preparation of Tokenizers
model_dir = 'models'
encoder_folder = 'models/decoder'
decoder_folder = 'models/decoder'
s2s_folder = 'models/final_dir'

dataset = ChoraleDataset()
c_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path='tokenizer/chords', pad_token="<PAD>", mask_token="<MASK>")
r_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path='tokenizer/roman_nums', pad_token="<PAD>", mask_token="<MASK>")

roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_folder, decoder_folder, tie_encoder_decoder=True).to(mps_device)

# set special tokens
roberta_shared.config.decoder_start_token_id = 1
roberta_shared.config.eos_token_id = 2
roberta_shared.config.pad_token_id = 3

# sensible parameters for beam search
# set decoding params
roberta_shared.config.max_length = SUMMARY_LEN
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 1
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.repetition_penalty = 3.0
roberta_shared.config.num_beams = 10
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

# load rouge for validation
rouge = datasets.load_metric("rouge")
def process_data_to_model_inputs(batch):
  # Tokenize the input and target data
  inputs = c_tokenizer(batch["translation"][0], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = r_tokenizer(batch["translation"][1], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  batch["labels"] = [[-100 if token == c_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = r_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = r_tokenizer.pad_token_id
    label_str = r_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

# Splitting the data up into train, test, and validation
data_list = []
for idx in range(len(dataset)):
    data_list.append(dataset[idx])

huggingface_dataset = Dataset.from_dict({key: [dic[key] for dic in data_list] for key in data_list[0]})
processed_data = huggingface_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["id", "translation"]
)
processed_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

train_data, val_data, test_data = torch.utils.data.random_split(processed_data, [0.8, 0.1, 0.1])

training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    # evaluate_during_training=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    #max_steps=1500, # delete for full training
    num_train_epochs = TRAIN_EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()
trainer.save_model(s2s_folder)

# Setup logging
if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )