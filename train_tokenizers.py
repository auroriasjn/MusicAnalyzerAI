from choraledataset import ChoraleDataset
from choraletokenizer import ChoraleTokenizer

dataset = ChoraleDataset()
input_tokenizer = ChoraleTokenizer(dataset)
input_tokenizer.train()

input_tokenizer.save_model(filename="chords/tokenizer.json")

output_tokenizer = ChoraleTokenizer(dataset, is_input=False)
output_tokenizer.train()
output_tokenizer.save_model(filename="roman_nums/tokenizer.json")