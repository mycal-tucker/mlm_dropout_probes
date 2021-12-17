
import h5py

from scipy.special import softmax
import numpy as np

def get_embeddings(path):
    embedding_hf = h5py.File(path, 'r')
    embeddings = []
    # We need to get the embeddings in a consistent order, and we know that the keys are integers, so we iterate over
    # those integers. Perhaps a little gross.
    for key in range(len(embedding_hf.keys())):
        embeddings.append(embedding_hf.get(str(key)).value)
    return embeddings

def logit_to_prob(logit):
    return softmax(logit)


def scaffold_embedding(original_embeddings, original_word_embeddings, updated_word_embeddings):
    word_idx = 0
    num_replaced = 0
    scaffolded_updated_embeddings = np.zeros_like(original_embeddings)
    for token_idx in range(original_embeddings.shape[1]):
        original_token_embedding = original_embeddings[0, token_idx]
        found_matching_word = False
        for possible_word_idx in range(word_idx, min(token_idx, original_word_embeddings.shape[1])):
            original_word_embedding = original_word_embeddings[0, possible_word_idx]
            if np.allclose(original_token_embedding, original_word_embedding, atol=0.1):
                found_matching_word = True
                updated_word_embedding = updated_word_embeddings[0, possible_word_idx]
                scaffolded_updated_embeddings[0, token_idx] = updated_word_embedding
                word_idx = possible_word_idx
                num_replaced += 1
                break
        if not found_matching_word:
            scaffolded_updated_embeddings[0, token_idx] = original_token_embedding
    assert num_replaced > 1, "Didn't put in enough words. Is the layer set right?"
    return scaffolded_updated_embeddings