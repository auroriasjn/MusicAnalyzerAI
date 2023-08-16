import utils_tokenizer as tokenizer
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import datasets


EOS_TOKEN = 0
def dict_from_list(input_list):
    temp_dict = dict.fromkeys(input_list)
    return {key: index for index, key in enumerate(temp_dict)}

class ChoraleDataset(Dataset):
    def __init__(self, key_folder='dataset', tokenize=True):
        self.key_folder = key_folder

        self.input_sentences, self.output_sentences = [], []
        self.input_words, self.output_words = [], []

        self.input_vocab_list = {}
        self.output_vocabulary = ["<EOS>"]
        self.input_vocabulary = ["<EOS>"]
        for folder in os.listdir(key_folder):
            key = os.path.join(key_folder, folder)
            key_input_vocabulary = ["<EOS>"]

            if os.path.isdir(key):
                sentence_pairs, vocab_pairs = self.load_key_folder(key)
                self.input_sentences += sentence_pairs[0]
                self.output_sentences += sentence_pairs[1]

                self.input_words += vocab_pairs[0]
                self.output_words += vocab_pairs[1]

                [key_input_vocabulary.append(chord) for sentence in vocab_pairs[0]
                 for chord in sentence if chord not in key_input_vocabulary]

                [self.input_vocabulary.append(chord) for sentence in vocab_pairs[0]
                 for chord in sentence if chord not in self.input_vocabulary]

                [self.output_vocabulary.append(roman_num) for sentence in vocab_pairs[1]
                 for roman_num in sentence if roman_num not in self.output_vocabulary]

            self.input_vocab_list[folder] = key_input_vocabulary
            self.input_vocab_list[folder].sort(reverse=True)

        self.input_vocabulary.sort(reverse=False)
        self.output_vocabulary.sort(reverse=False)

        # Creating Dictionaries
        if tokenize:
            self.input_vocab_dict = dict_from_list(self.input_vocabulary)
            self.output_vocab_dict = dict_from_list(self.output_vocabulary)
            self.key_vocab_dict = {}
            for key in self.input_vocab_list:
                self.key_vocab_dict[key] = dict_from_list(self.input_vocab_list[key])

            self.input_reverse = {v: k for k, v in self.input_vocab_dict.items()}
            self.output_reverse = {v: k for k, v in self.output_vocab_dict.items()}

            self.tokenized_input_sentences = self.convert_vocab(self.input_words, self.input_vocab_dict)
            self.tokenized_output_sentences = self.convert_vocab(self.output_words, self.output_vocab_dict)

        self.num_input_sentences = len(self.input_sentences)
        self.num_output_sentences = len(self.output_sentences)
        self.num_output_words = len(self.output_vocabulary)

    def __len__(self):
        return self.num_input_sentences

    def __getitem__(self, index):
        input = self.input_sentences[index]
        output = self.output_sentences[index]

        return {'id': index, 'translation': [input, output]}

    def load_key_folder(self, folder_name):
        key_inputs, key_outputs = [], []
        key_vocab_inputs, key_vocab_outputs = [], []

        for fil in os.listdir(folder_name):
            file_path = os.path.join(folder_name, fil)
            if not file_path.endswith('.txt'):
                continue

            fil_inputs, fil_outputs, _ = tokenizer.sentence_tokenize(file_path)
            vocab_inputs, vocab_outputs, _ = tokenizer.word_tokenize(file_path)

            key_inputs += fil_inputs
            key_outputs += fil_outputs

            key_vocab_inputs += vocab_inputs
            key_vocab_outputs += vocab_outputs

        return [[key_inputs, key_outputs], [key_vocab_inputs, key_vocab_outputs]]

    def convert_vocab(self, words, dict):
        tokenized_output_sentences = []

        for sentence in tqdm(words, desc="Tokenizing vocabulary"):
            converted_sentence = [dict[K] for K in sentence]
            tokenized_output_sentences += [converted_sentence]

        return tokenized_output_sentences

    def input_word2index(self, input_str):
        return self.input_vocabulary[input_str]

    def output_word2index(self, output_str):
        return self.output_vocabulary[output_str]

    def key_word2index(self, key, input_str):
        return self.key_vocab_dict[key][input_str]
    def input_index2word(self, input_index):
        return self.input_reverse[input_index]
    def output_index2word(self, output_index):
        return self.output_reverse[output_index]

    def decode_input(self, input):
        return [self.input_index2word(chord) for chord in input]

    def decode_output(self, output):
        return [self.output_index2word(chord) for chord in output]

    def save_input_sentences(self, output_dir='roman_nums', prefix=0):
        output_dir = str(prefix) + "_" + output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i = 0
        for sentence in self.input_sentences:
            path = os.path.join(output_dir, f"{i}.txt")
            f = open(path, 'wb')

            sentence = sentence.strip()
            f.write(sentence.encode('utf-8'))

            f.close()

            i += 1

class ProcessedDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        # or use the RobertaTokenizer from `transformers` directly.
        self.examples = []
        MAX_LEN = 8192

        # For every value in the dataframe
        for example in sentences:
            x = tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding=True)
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])



