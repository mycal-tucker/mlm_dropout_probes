import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


def create_sentence(n1, n2, v1, v2):
    sentence_list = ["when the", n1, v1, "the", n2, '[MASK]', v2 + '.']
    sentence = " ".join(sentence_list)
    return sentence


def create_tree(n1, n2, v1, v2):
    tree1 = '(ROOT\n' +\
                '(S\n' +\
                    '(SBAR\n' +\
                        '(WHADVP (WRB when))\n' +\
                        '(S\n' +\
                            '(NP (DT the) (NN ' + n1 + '))\n' +\
                            '(VP (VBD ' + v1 + '))))\n' +\
                '(NP (DT the) (NN ' + n2 + '))\n' +\
                '(ADVP (RB [MASK]))\n' +\
                '(VP (VBD ' + v2 + '))\n' +\
                '(. .)))\n\n'
    tree2 = '(ROOT\n' +\
                '(S\n' +\
                    '(SBAR\n' +\
                        '(WHADVP (WRB when))\n' +\
                        '(S\n' +\
                            '(NP (DT the) (NN ' + n1 + '))\n' +\
                            '(VP (VBD ' + v1 + ')\n' + \
            '(NP (DT the) (NN ' + n2 + ')))))\n' + \
            '(NP (PRP [MASK]))\n' +\
                '(VP (VBD ' + v2 + '))\n' +\
                '(. .)))\n\n'
    return tree1, tree2

def write_data(line_idx):
    for n1 in nn1:
        for n2 in nn2:
            for v1 in v1s:
                for v2 in v2s:
                    text = create_sentence(n1, n2, v1, v2)

                    np_text = tokenizer(text, return_tensors='np')
                    mask_idx = np.where(np_text.data['input_ids'] == mask_id)[1][0]


                    tokenized_text = tokenizer.encode_plus(text,
                                                           return_tensors='pt')
                    pred = model(**tokenized_text, output_hidden_states=True)
                    prediction = pred.logits
                    hidden_states = pred.hidden_states
                    np_prediction = prediction.cpu().detach().numpy()

                    overall_best = np.argmax(np_prediction[0, mask_idx])
                    best_token = tokenizer.convert_ids_to_tokens([overall_best])[0]
                    print(text, "\t", best_token)
                    candidates.add(best_token)
                    print(candidates)
                    parses = create_tree(n1, n2, v1, v2)
                    for parse in parses:
                        # Dump data into text files
                        file_mode = 'w' if line_idx == 0 else 'a'
                        with open(root_dir + 'text.txt', file_mode) as text_file:
                            text_file.write(text + '\n')
                        # Where was the masked word?
                        with open(root_dir + 'token_idxs.txt', file_mode) as token_idx_file:
                            token_idx_file.write('\t'.join([str(entry) for entry in ['npz', mask_idx] + list(candidates)]) + '\n')
                        # And write the tree itself.
                        with open(root_dir + 'text.trees', file_mode) as tree_file:
                            tree_file.write(parse)
                        line_idx += 1
    return line_idx

#
# candidates = ['suddenly', 'almost', 'had', 'you', 'she', 'i']
# candidates_ids = tokenizer.convert_tokens_to_ids(candidates)
candidates = set()

# fn_map = {'no_comma': (create_sentence(add_comma=0), create_tree(add_comma=0)),
#           'comma1': (create_sentence(add_comma=1), create_tree(add_comma=1)),
#           'comma2': (create_sentence(add_comma=2), create_tree(add_comma=2))}

root_dir = 'data/npz/'
nn1 = ['dog', 'child']
nn2 = ['vet', 'boy', 'girl']
v1s = ['scratched', 'bit']
v2s = ['ran', 'screamed', 'smiled']
mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
line_id = write_data(0)     

full_data = [[['author'], ['wrote'], ['book'], ['grew']],
                [['doctor', 'professor'], ['lectured'], ['student'], ['listened']],
                [['girls', 'boys'], ['raced'], ['kids', 'children'], ['watched', 'cheered']],
                [['people', 'spectators'], ['watched'], ['show', 'movie'], ['stopped', 'paused']],
                [['lawyers', 'judges'], ['studied', 'considered'], ['case'], ['languished', 'proceeded']],
                [['people', 'viewers'], ['notice', 'spot'], ['actor'], ['departs', 'stays']],
                [['band', 'convention'], ['left'], ['hotel', 'stalls'], ['closed']]]

for data in full_data:
    nn1, v1s, nn2, v2s = data
    line_id = write_data(line_id)