from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast, PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments
from choraledataset import ChoraleDataset, ProcessedDataset
import torch
import os

mps_device = torch.device("mps:0")
tokenizer_folder = os.path.join("tokenizer", "roman_nums")
model_folder = os.path.join("models", "decoder")

# Set a configuration for our RoBERTa model
config = RobertaConfig(
    vocab_size=1024,
    max_position_embeddings=1024,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config).to(mps_device)
print('Num parameters: ',model.num_parameters())

tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=tokenizer_folder, pad_token="<PAD>", mask_token="<MASK>")

dataset = ChoraleDataset()
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

train_sentences = [train_data[i]["translation"][1] for i in range(len(train_data))]
val_sentences = [val_data[i]["translation"][1] for i in range(len(val_data))]
test_sentences = [test_data[i]["translation"][1] for i in range(len(test_data))]

print(train_sentences)

train_dataset = ProcessedDataset(sentences=train_sentences, tokenizer=tokenizer)
eval_dataset = ProcessedDataset(sentences=val_sentences, tokenizer=tokenizer)
test_dataset = ProcessedDataset(sentences=test_sentences, tokenizer=tokenizer)

TRAIN_BATCH_SIZE = 32   #32    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 4    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 20        # number of epochs to train (default: 10)
VAL_EPOCHS = 1
WEIGHT_DECAY = 0.005
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 100
SUMMARY_LEN = 7

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,
    evaluation_strategy = 'epoch',
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    save_steps=8192,
    save_total_limit=1,
)

# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()
trainer.save_model(model_folder)
