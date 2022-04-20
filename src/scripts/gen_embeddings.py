import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForQuestionAnswering

# We default to using a model trained on the cloze task. If you're interested in other models, make sure to fetch them
# here. E.g., previously, we used a qa model that had been finetuned on squad.
# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
# model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased-whole-word-masking")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# Below are options for the QA model
tokenizer = AutoTokenizer.from_pretrained('twmkn9/bert-base-uncased-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('twmkn9/bert-base-uncased-squad2')

num_layers = 13  # 25 for large models.
# Set the desired source_dir and filename here.
# source_dir = 'data/ptb/'
# source_dir = 'data/conj/'
# source_dir = 'data/npz/'
# source_dir = 'data/qa_rc/'
# source_dir = 'data/qa_npvp/'
# source_dir = 'data/qa_coord/'
source_dir = 'data/qa_rc/'
filename = 'text'
break_on_qmark = True  # If you're using questions from a QA task, this should be true.

source_file = source_dir + filename + '.txt'
# Mark the QA embeddings specially
#targ_file = source_dir + filename + '_qa' + '.hdf5'
targ_file = source_dir + filename + '.hdf5'

file1 = open(source_file, 'r')
idx = 0
hf = h5py.File(targ_file, 'w')
for line in file1:
    print()
    print("Analyzing line number", idx)
    if '?' in line and break_on_qmark:
        q_mark_idx = line.index('?')
        context = line[q_mark_idx + 2:]
        question = line[:q_mark_idx + 1]
        print("Context", context)
        print("Question", question)
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    else:
        inputs = tokenizer.encode_plus(line, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**inputs, output_hidden_states=True)
        # Debugging logs of seeing question answers, but not actually used in main logic.
        if '?' in line and break_on_qmark:  # QA input case, so print out model answer.
            answer_start = torch.argmax(model_output[0], dim=1).numpy()[0]
            answer_end = (torch.argmax(model_output[1], dim=1) + 1).numpy()[0]
            selected_tokens = inputs["input_ids"][0][answer_start:answer_end]
            print("Answer", tokenizer.decode(selected_tokens))
        embeddings = model_output[-1]
        np_embeddings = np.zeros((num_layers, embeddings[0].shape[1], embeddings[0].shape[2]))
        for layer in range(num_layers):
            np_embeddings[layer] = embeddings[layer].numpy()
        # For each of the 25 layers in the model, I have a 1 x s_length x 768 tensor.
        # Now write the embeddings to an hdf5 file for training a probe with later.
        hf.create_dataset(str(idx), data=np_embeddings)
    idx += 1
    if idx > 5000:
        print("Breaking after 5000; the only reason to have that many examples is if you are generating data to train a probe.")
        break
file1.close()
hf.close()
