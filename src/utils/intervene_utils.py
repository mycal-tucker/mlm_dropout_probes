import h5py
import torch
import numpy as np

def generate_embeddings(tokenizer, model, source_dir, filename, break_on_qmark, device=None, max_num_embeddings=5000):
    num_layers = 13  # 13 for BERT models; 6 for distilbert; 12 for roberta; 25 for large
    source_file = source_dir + filename + '.txt'
    targ_file = source_dir + filename + '.hdf5'
    file1 = open(source_file, 'r')
    idx = 0
    hf = h5py.File(targ_file, 'w')
    for line in file1:
        if '?' in line and break_on_qmark:
            q_mark_idx = line.index('?')
            context = line[q_mark_idx + 2:]
            question = line[:q_mark_idx + 1]
            inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        elif break_on_qmark and '?' not in line:
            # print("NLI embedding")
            inputs = tokenizer.encode_plus(line, add_special_tokens=True, return_tensors="pt")
        else:
            inputs = tokenizer.encode_plus(line, return_tensors='pt')
        if device is not None:
            inputs.to(device)
        with torch.no_grad():
            model_output = model(**inputs, output_hidden_states=True)
            # Debugging logs of seeing question answers, but not actually used in main logic.
            # if '?' in line and break_on_qmark:  # QA input case, so print out model answer.
            #     answer_start = torch.argmax(model_output[0], dim=1).cpu().numpy()[0]
            #     answer_end = (torch.argmax(model_output[1], dim=1) + 1).cpu().numpy()[0]
            #     selected_tokens = inputs["input_ids"][0][answer_start:answer_end]
            #     print("Answer", tokenizer.decode(selected_tokens))
            embeddings = model_output[-1]
            np_embeddings = np.zeros((num_layers, embeddings[0].shape[1], embeddings[0].shape[2]))
            for layer in range(num_layers):
                np_embeddings[layer] = embeddings[layer].cpu().numpy()
            # For each of the layers in the model, I have a 1 x s_length x 768 tensor.
            # Now write the embeddings to an hdf5 file for training a probe with later.
            hf.create_dataset(str(idx), data=np_embeddings)
        idx += 1
        # if idx % 1000 == 0:
        #     print("At idx", idx)
        if idx >= max_num_embeddings:
            print(
                "Breaking after", max_num_embeddings, "; the only reason to have that many examples is if you are generating data to train a probe.")
            break
    file1.close()
    hf.close()