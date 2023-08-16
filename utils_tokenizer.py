import re
import os
import sys
import numpy as np
import itertools

# Globals
input_dir = 'dataset'

# Loads in a file as outputted by MusicParse.
def file_load(filename):
    init_input, init_output, key = "", "", ""
    with open(filename, 'r') as infile:
        has_encountered_split = False
        for line in infile:
            if line.strip() == "---":
                has_encountered_split = True
                continue
            elif len(line.strip()) > 0 and line.strip()[0] == 'K':
                key = line[3:].strip()
                continue
            elif line.strip()[:2] == "M:":
                continue

            if not has_encountered_split:
                init_input += line
            else:
                init_output += line

    return [init_input, init_output, key]

# Prepares the input string for tokenization by removing some of the human readable information.
def tokenizer_prep(init_input_str, init_output_str, key):
    prepped_input, prepped_output = init_input_str, init_output_str

    # Removing |
    prepped_input = prepped_input.replace(" |", "")
    prepped_output = prepped_output.replace(" |", "")

    # Inserting end of sequence tokens at end of each long phrase
    input_measures, output_measures = prepped_input.splitlines(), prepped_output.splitlines()

    pattern = r'\([^)]+\)(?!\S)|\S+'

    prepped_input, prepped_output = [], []

    curr_sentence_key = key
    prepped_output.append(f'({curr_sentence_key})')

    for measure in zip(input_measures, output_measures):
        measure1, measure2 = measure[0][4:].strip(), measure[1][4:].strip()

        # Getting the measures split properly
        m1_list = re.findall(pattern, measure1)
        m2_list = re.findall(pattern, measure2)

        for beat_pair in itertools.zip_longest(m1_list, m2_list, fillvalue="_"):
            b1_num = re.search(r'\d+$', beat_pair[0])
            altered_inp_beat = beat_pair[0]

            # Removing beat durations from appended pair
            if b1_num is not None:
                altered_inp_beat = altered_inp_beat[1:-2]
            else:
                altered_inp_beat = altered_inp_beat[1:-1]

            if len(beat_pair[0]) > 2:
                prepped_input.append(altered_inp_beat)

            # Checking for key changes
            if '(' in beat_pair[1]:
                inner_str_inds = extract_letters_between_parentheses(beat_pair[1])
                if len(inner_str_inds) != 0:
                    inner_str_inds = inner_str_inds[0]
                    inner_str = beat_pair[1][inner_str_inds[0] + 1:inner_str_inds[1]]
                    curr_sentence_key = inner_str

            if beat_pair[1] == "_":
                # If the lengths are not equal we don't want to end it on a half cadence.
                if 'V' not in prepped_output[-1] and 'vii' not in prepped_output[-1]:
                    prepped_input.append("<EOS>")
                    prepped_output.append("<EOS>")

                    prepped_input.append("<BOS>")
                    prepped_output.append("<BOS>")

                    prepped_output.append(f'({curr_sentence_key})')
                continue
            else:
                prepped_output.append(beat_pair[1])

            # Do not end on a key change!
            if b1_num is not None and len(prepped_input) > len(m1_list) and '[' not in beat_pair[1]:
                prepped_input.append("<EOS>")
                prepped_output.append("<EOS>")

                prepped_input.append("<BOS>")
                prepped_output.append("<BOS>")

                prepped_output.append(f'({curr_sentence_key})')

    if len(prepped_input) > 1 and prepped_input[-1] != "<EOS>" and prepped_output[-1] != "<EOS>":
        prepped_input.append("<EOS>")
        prepped_output.append("<EOS>")

        prepped_input.append("<BOS>")
        prepped_output.append("<BOS>")

    return [prepped_input, prepped_output]

# Splits the prepared input and output into token arrays.
def split_into_sentences(input, output):
    np_inputs = np.array(input)
    np_outputs = np.array(output)

    inidx = np.where(np_inputs == "<EOS>")[0]
    outidx = np.where(np_outputs == "<EOS>")[0]

    split_inputs = np.split(np_inputs, inidx + 1)
    split_outputs = np.split(np_outputs, outidx + 1)

    split_inputs = [subarray.tolist() for subarray in split_inputs]
    split_outputs = [subarray.tolist() for subarray in split_outputs]

    # Removing zero
    zero_indices = [i for i in range(len(split_inputs)) if len(split_inputs[i]) < 3]
    split_inputs = [split_inputs[i][1:-1] for i in range(len(split_inputs)) if i not in zero_indices]
    split_outputs = [split_outputs[i][1:-1] for i in range(len(split_outputs)) if i not in zero_indices]

    assert len(split_inputs) == len(split_outputs)

    return [split_inputs, split_outputs]

# Wrapper function to be used by an external decode_bach.
def word_tokenize(filename):
    init_input, init_output, key = file_load(filename)
    prepped_input, prepped_output = tokenizer_prep(init_input, init_output, key)
    split_inputs, split_outputs = split_into_sentences(prepped_input, prepped_output)

    return [split_inputs, split_outputs, key]

# Wrapper function to be used by an external decode_bach.
def sentence_tokenize(filename):
    tokenized_inputs, tokenized_outputs, key = word_tokenize(filename)

    tokenized_inputs = [" ".join(phrase) for phrase in tokenized_inputs]
    tokenized_outputs = [" ".join(phrase) for phrase in tokenized_outputs]

    return [tokenized_inputs, tokenized_outputs, key]

# Helper Function
def extract_letters_between_parentheses(string):
    indices = []
    start = 0

    while True:
        opening = string.find("(", start)
        if opening == -1:
            break

        closing = string.find(")", opening + 1)
        if closing == -1:
            break

        letters = string[opening + 1:closing].strip()
        if len(letters) == 1 or len(letters) == 2:
            indices.append((opening, closing))

        start = closing + 1

    return indices

if __name__ == '__main__':
    input, output, key = sentence_tokenize(os.path.join(input_dir, sys.argv[1]))
    for phrase in zip(input, output):
        print("---")
        print(f'{key}: {phrase[0]}')
        print(f'{key}: {phrase[1]}')