import h5py
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForQuestionAnswering

from src.models.intervention_model import ClozeTail, QATail


# suite = 'conj'  # For mask
# suite = 'npz'   # For mask
suite = 'qa_coord'  # QA coordination
# suite = 'qa_npvp'   # QA npvp
# suite = 'qa_rc'     # QA relative clause
if 'qa' not in suite:
    checkpoint = 'bert-base-uncased'  # Mask
else:
    checkpoint = 'twmkn9/bert-base-uncased-squad2'  # QA

is_cloze_model = 'squad' not in checkpoint
if is_cloze_model:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
else:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)


# Return the sorted embeddings at the specified path.
def get_embeddings(path):
    embedding_hf = h5py.File(path, 'r')
    embeddings = []
    # We need to get the embeddings in a consistent order, and we know that the keys are integers, so we iterate over
    # those integers. Perhaps a little gross.
    for key in range(len(embedding_hf.keys())):
        embeddings.append(embedding_hf.get(str(key)).value)
    return embeddings


def get_cloze_output(model_output, mask_idx, do_print=False):
    np_prediction = model_output.cpu().detach().numpy()
    predictions_candidates = np_prediction[0, mask_idx, candidates_ids]
    answer_idx = np.argmax(predictions_candidates)
    overall_best = np.argmax(np_prediction[0, mask_idx])
    best_token = tokenizer.convert_ids_to_tokens([overall_best])[0]
    if do_print:
        print(f'The most likely word is "{candidates[answer_idx]}".')
        print("Overall best", best_token)
    return predictions_candidates, best_token.strip()


def get_qa_output(model_output, tokens_line):
    start = torch.argmax(model_output[0], dim=1).numpy()[0]
    end = (torch.argmax(model_output[1], dim=1) + 1).numpy()[0]
    selected_tokens = tokens_line["input_ids"][0][start:end]
    return tokenizer.decode(selected_tokens), softmax(model_output[0].numpy()), softmax(model_output[1].numpy())


def logit_to_prob(logit):
    return softmax(logit)

candidates = None
if suite == 'npz':
    candidates = ['all', 'he', 'just', 'it', 'they', 'she', 'was', 'were']
elif suite == 'conj':
    candidates = ['is', 'are', 'was', 'were', 'as']

mask_id = tokenizer.convert_tokens_to_ids("[MASK]")


def get_cloze_texts():
    text = []
    tokenized_text = []
    mask_locations = []
    with open('%stext.txt' % text_data_dir, 'r') as text_file:
        for line in text_file:
            text.append(line)
            # Grab the index of where the mask is
            np_text = tokenizer(line, return_tensors='np')
            mask_idx = np.where(np_text.data['input_ids'] == mask_id)[1][0]
            mask_locations.append(mask_idx)
            # Record the original outputs.
            tokenized = tokenizer.encode_plus(line, return_tensors='pt')
            tokenized_text.append(tokenized)
    return text, tokenized_text, mask_locations


def get_qa_texts():
    # Load the sentences so we can pull out text answers.
    text = []
    tokenized_text = []
    with open('%stext.txt' % text_data_dir, 'r') as text_file:
        for line in text_file:
            text.append(line)
            question_idx = line.index('?')
            context = line[question_idx + 1:]
            question = line[:question_idx + 2]
            tokenized = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
            tokenized_text.append(tokenized)
    return text, tokenized_text


def scaffold_embedding(original_word_embeddings):
    word_idx = 0
    num_replaced = 0
    scaffolded_updated_embeddings = np.zeros_like(original_embedding)
    for token_idx in range(original_embedding.shape[1]):
        original_token_embedding = original_embedding[0, token_idx]
        found_matching_word = False
        for possible_word_idx in range(word_idx, min(token_idx, original_word_embeddings.shape[1])):
            original_word_embedding = original_word_embeddings[0, possible_word_idx]
            if np.allclose(original_token_embedding, original_word_embedding, atol=0.1):
                found_matching_word = True
                updated_word_embedding = updated_embedding[0, possible_word_idx]
                scaffolded_updated_embeddings[0, token_idx] = updated_word_embedding
                word_idx = possible_word_idx
                num_replaced += 1
                break
        if not found_matching_word:
            scaffolded_updated_embeddings[0, token_idx] = original_token_embedding
    assert num_replaced > 1, "Didn't put in enough words. Is the layer set right?"
    return scaffolded_updated_embeddings


def run_cloze_eval(mask_idx):
    original_candidate_logits, original_best = get_cloze_output(original_output, mask_idx, do_print=False)

    # Now find the columns that correspond to the embeddings for just the words by looking at the outputs
    # when wiring through the original encodings.
    original_word_embeddings = word_embeddings[i]
    if len(original_word_embeddings.shape) == 2:
        original_word_embeddings = original_word_embeddings.reshape(1, -1, 768)
    scaffolded_updated_embeddings = scaffold_embedding(original_word_embeddings)
    updated_output = tail_model(torch.tensor(scaffolded_updated_embeddings, dtype=torch.float32))
    candidate_logits, new_best = get_cloze_output(updated_output, mask_idx, do_print=False)
    original_probs = logit_to_prob(original_candidate_logits)
    updated_probs = logit_to_prob(candidate_logits)

    all_probs = original_probs.tolist() + updated_probs.tolist()
    return all_probs, scaffolded_updated_embeddings


def run_qa_eval():
    # Now find the columns that correspond to the embeddings for just the words by looking at the outputs
    # when wiring through the original encodings.
    original_word_embeddings = word_embeddings[i]
    if len(original_word_embeddings.shape) == 2:
        original_word_embeddings = original_word_embeddings.reshape(1, -1, 768)
    scaffolded_updated_embeddings = scaffold_embedding(original_word_embeddings)
    # updated_output = tail_model(torch.tensor(scaffolded_updated_embeddings, dtype=torch.float32))
    # original_np = [logit_to_prob(original.numpy()) for original in original_output]
    # updated_np = [logit_to_prob(updated.numpy()) for updated in updated_output]
    # sentence_probs = [(original_np[0][0, j], original_np[1][0, j], updated_np[0][0, j], updated_np[1][0, j]) for j in
    #                   range(original_np[0].shape[1])]
    sentence_probs = 0
    return sentence_probs, scaffolded_updated_embeddings


text_fn = get_cloze_texts if is_cloze_model else get_qa_texts
tail_model_cls = ClozeTail if is_cloze_model else QATail
for xfact_loss in [0.3, 0.2, 0.1, 0.05]:
    for dropout_rate in [8, 9]:  # FIXME missing some cases
        for seed in range(0, 5):
            # Where is the original text.
            text_data_dir = 'data/' + suite + '/'
            # What is the root of the directories that have the updated embeddings.
            counterfactuals_dir = 'counterfactuals/' + suite + '/seed' + str(seed) + ('/dropout' if is_cloze_model else '/qa_dropout') + str(dropout_rate) + '_dist_3layer/'
            probe_type = 'depth' if 'depth' in counterfactuals_dir else 'dist'

            for layer in range(1, 13):  # FIXME
                print("xfact_loss", xfact_loss, "dropout rate", dropout_rate, "Seed", seed, "layer", layer, "suite", suite)
                experiment_dir = counterfactuals_dir + 'model_' + probe_type + str(layer) + '/'
                original_embeddings = get_embeddings('%stext.hdf5' % text_data_dir)
                word_embeddings = get_embeddings('%soriginal_words.hdf5' % experiment_dir)
                updated_word_embeddings = get_embeddings(experiment_dir + 'updated_words_xfactloss_' + str(xfact_loss) + '.hdf5')

                # Load the sentences so we can pull out original outputs
                text_data = text_fn()
                text, tokenized_text = text_data[:2]

                original_answers = set()
                with open('%stoken_idxs.txt' % text_data_dir, 'r') as token_file:
                    for line in token_file:
                        pass
                    last_line = line
                    parsed_line = last_line.split('\t')
                    for elt_idx, elt in enumerate(parsed_line):
                        if elt_idx <= 1:
                            continue
                        original_answers.add(str(elt).strip())
                if is_cloze_model:
                    candidates_ids = tokenizer.convert_tokens_to_ids(candidates)

                # Sanity check: just use the pipeline end-to-end.
                # If you're really crunched for time, you don't need to run this.
                normal_outputs = []
                for i, token_text in enumerate(tokenized_text):
                    with torch.no_grad():
                        normal_output = model(**token_text)
                        if is_cloze_model:
                            normal_output = normal_output[0]
                            mask_idxs = text_data[2]
                            get_cloze_output(normal_output, mask_idxs[i])
                        else:
                            get_qa_output(normal_output, token_text)
                        normal_outputs.append(normal_output)

                tail_model = tail_model_cls(model, layer)
                file_data = []
                updated_distances = []
                all_scaffolded_for_layer = []
                for i, updated_embedding in enumerate(updated_word_embeddings):
                    updated_embedding = updated_embedding.reshape(1, -1, 768)
                    with torch.no_grad():
                        # print('\n\n\n\n')
                        # print("\nUsing text:\t", text[i])
                        original_embedding = original_embeddings[i][layer].reshape(1, -1, 768)
                        original_output = tail_model(torch.tensor(original_embedding, dtype=torch.float32))
                        # Check that the output from original embedding matches the normal output.
                        if is_cloze_model:  # This assert doesn't work with QA models for whatever reason.
                            assert len(normal_outputs) == 0 or np.allclose(normal_outputs[i][0], original_output[0]), "Not all close!"

                        data, scaffolded_updated = run_cloze_eval(mask_idxs[i]) if is_cloze_model else run_qa_eval()
                        file_data.append(data)

                        # Just as a fun metric, calculate the distance between the original and updated embeddings.
                        update_dist = np.linalg.norm(original_embedding - scaffolded_updated)
                        updated_distances.append(update_dist)
                # Now write file_data to the file.
                with open(experiment_dir + 'updated_probs_xfactloss_' + str(xfact_loss) + '.txt', 'w') as results_file:
                    if is_cloze_model:
                        results_file.write('\t'.join(['Candidates'] + candidates + ['\n']))
                    for line in file_data:
                        results_file.write('\t'.join([str(entry) for entry in line]))
                        results_file.write('\n')
                with open(experiment_dir + 'updated_distances_xfactloss_' + str(xfact_loss) + '.txt', 'w') as dist_file:
                    if is_cloze_model:
                        dist_file.write('\t'.join(['Candidates'] + candidates + ['\n']))
                    for line in updated_distances:
                        dist_file.write(str(line) + '\n')
