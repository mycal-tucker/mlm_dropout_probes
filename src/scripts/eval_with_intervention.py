import torch

import yaml
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,  BertModel, BertTokenizer
from src.utils.intervene_utils import generate_embeddings
from src.scripts.gen_counterfactuals import execute_experiment as xfact_experiment
from src.utils.eval_utils import get_embeddings, scaffold_embedding
from src.utils.squad_utils import load_parts, eval_qa
import numpy as np
# from src.models.nli_model import BERTNLIModel

from scipy.special import softmax
from src.models.intervention_model import QATail #, NLITail



def gen_embeddings(tokenizer, model, sdir, files, break_on_q, num_embeddings):
    for file, num in zip(files, num_embeddings):
        generate_embeddings(tokenizer, model, sdir, file, break_on_q, device=device, max_num_embeddings=num)


def gen_counterfactuals(config, loss_tolerance):
    xfact_experiment(config, loss_tolerance=loss_tolerance)


def run_qa_eval(original_embedding, _word_embedding, _updated_word_embedding, _device, tail_model, encoding_dim=768):
    original_word_embeddings = _word_embedding
    if len(original_word_embeddings.shape) == 2:
        original_word_embeddings = original_word_embeddings.reshape(1, -1, encoding_dim)
    if len(_updated_word_embedding.shape) == 2:
        _updated_word_embedding = _updated_word_embedding.reshape(1, -1, encoding_dim)
    scaffolded_updated_embeddings = scaffold_embedding(original_embedding, original_word_embeddings, _updated_word_embedding)
    updated_output = tail_model(torch.tensor(scaffolded_updated_embeddings, dtype=torch.float32).to(_device))
    return updated_output


def eval_nli(csv_file, preds):
    labels = ['entailment', 'contradiction', 'neutral']
    num_total = 0
    num_correct = 0
    prob_mass = 0
    all_probs = []
    with open(csv_file, 'r') as f:
        for line, pred in zip(f, preds):
            probs = softmax(pred)
            all_probs.append(probs)
            # print("Line", line)
            # print(probs)
            pred_numerical = np.argmax(pred)
            pred_label = labels[pred_numerical]
            true_label = line.split(',')[0]
            prob_mass += probs[labels.index(true_label)]
            if pred_label == true_label:
                num_correct += 1
            num_total += 1
    print("Accuracy", num_correct / num_total)
    print("Prob mass", prob_mass / num_total)
    return np.array(all_probs)


def eval_effect(model, tokenizer, text_data_dir, probe_dir, encoding_dim, xfact_loss):
    original_embeddings = get_embeddings('%stext.hdf5' % text_data_dir)
    word_embeddings = get_embeddings('%s/original_words.hdf5' % probe_dir)
    updated_word_embeddings = get_embeddings(probe_dir + '/updated_words_xfactloss_' + str(xfact_loss) + '.hdf5')

    tail_model = QATail(model, layer)
    tail_model = tail_model.to(device)
    all_originals = []
    all_updates = []
    for i, updated_embedding in enumerate(updated_word_embeddings):
        updated_embedding = updated_embedding.reshape(1, -1, encoding_dim)
        with torch.no_grad():
            original_embedding = original_embeddings[i][layer].reshape(1, -1, encoding_dim)
            tensor_embedding = torch.tensor(original_embedding, dtype=torch.float32)
            tensor_embedding = tensor_embedding.to(device)
            original_output = tail_model(tensor_embedding)
            updated_output = run_qa_eval(original_embedding, word_embeddings[i], updated_embedding, device, tail_model, encoding_dim=encoding_dim)
            # print("Original", original_output)
            # print("Updated", updated_output)
            np_originals = [output.detach().cpu().numpy() for output in original_output]
            np_originals = [np.pad(original, ((0, 0), (0, 512 - original.shape[1],)), 'constant') for original in np_originals]
            all_originals.append(np_originals)
            np_updates = [output.detach().cpu().numpy() for output in updated_output]
            np_updates = [np.pad(original, ((0, 0), (0, 512 - original.shape[1],)), 'constant') for original in np_updates]
            all_updates.append(np_updates)

    all_originals = [np.stack([elt[i] for elt in all_originals]).squeeze(1) for i in [0, 1]]
    all_updates = [np.stack([elt[i] for elt in all_updates]).squeeze(1) for i in [0, 1]]
    return all_originals, all_updates


def eval_nli_effect(model, text_data_dir, probe_dir, encoding_dim):
    original_embeddings = get_embeddings('%stext.hdf5' % text_data_dir)
    word_embeddings = get_embeddings('%s/original_words.hdf5' % probe_dir)
    updated_word_embeddings = get_embeddings('%s/updated_words.hdf5' % probe_dir)

    tail_model = NLITail(model, layer)
    tail_model = tail_model.to(device)
    all_originals = []
    all_updates = []
    for i, updated_embedding in enumerate(updated_word_embeddings):
        with torch.no_grad():
            original_embedding = original_embeddings[i][layer].reshape(1, -1, encoding_dim)
            tensor_embedding = torch.tensor(original_embedding, dtype=torch.float32)
            tensor_embedding = tensor_embedding.to(device)
            original_output = tail_model(tensor_embedding)
            updated_output = run_qa_eval(original_embedding, word_embeddings[i], updated_embedding, device, tail_model,
                                         encoding_dim=encoding_dim)
            # print("Original", original_output)
            # print("Updated", updated_output)
            o_np = original_output.detach().cpu().numpy()
            all_originals.append(o_np)
            u_np = updated_output.detach().cpu().numpy()
            all_updates.append(u_np)

    all_originals = np.stack(all_originals)
    all_updates = np.stack(all_updates)
    all_originals = all_originals.squeeze(1)
    all_updates = all_updates.squeeze(1)
    return all_originals, all_updates


def run_eval(counterfactual_config, corpus_path):
    model_checkpoint = base_checkpoint
    print("Model", model_checkpoint)
    if is_nli_model:
        hidden_dim = 512
        output_dim = 3
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BERTNLIModel(bert_model,
                             hidden_dim,
                             output_dim,
                             ).to(device)
        model.load_state_dict(torch.load(base_checkpoint))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)
    model.eval()

    gen_embeddings(tokenizer, model, sdir=eval_corpus, files=['text'], break_on_q=True, num_embeddings=[5000])

    if not is_nli_model:
        probe_save_dir = 'counterfactuals/qa_intervene/seed' + str(seed) + '/qa_dropout' + str(dropout_rate) + '_dist_3layer/model_dist' + str(layer)
    else:
        probe_save_dir = 'saved_models/nli/eval_probe_layer' + str(layer)

    # xfact_losses = [0.025, 0.05, 0.1, 0.15, 0.2]
    # xfact_losses = [0.05, 0.1, 0.2, 0.3]  # Validation
    # xfact_losses = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # Test
    xfact_losses = [0.3]  # Just for single plot.
    f1s = []
    exacts = []
    data_file = 'dist' + str(dropout_rate) + '_test_inter.txt'
    for xfact_loss in xfact_losses:
        print("Creating counterfactual embeddings")
        counterfactual_config['reporting']['root'] = probe_save_dir
        counterfactual_config['model']['model_layer'] = layer
        gen_counterfactuals(counterfactual_config, loss_tolerance=xfact_loss)

        # Lastly, evaluate the predictions when using the original embeddings vs. when using the counterfactuals.
        # Just generate model outputs for each type.
        if not is_nli_model:
            original_preds, updated_preds = eval_effect(model, tokenizer, eval_corpus, probe_save_dir, counterfactual_config['model']['hidden_dim'], xfact_loss)
        else:
            original_preds, updated_preds = eval_nli_effect(model, eval_corpus, probe_save_dir, counterfactual_config['model']['hidden_dim'])

        # For some other dataset.
        if not is_nli_model:
            datasets, trainer, tokenizer, pad_on_right, max_length, doc_stride = load_parts(
                model_checkpoint=model_checkpoint,
                datasets=corpus_path,
                num_train=2,
                num_val=1000,
                shuffle=False)
            print("Original")
            res = eval_qa(datasets, trainer, tokenizer, pad_on_right, max_length, doc_stride, logits=original_preds)
            print("Updated to loss", xfact_loss)
            res = eval_qa(datasets, trainer, tokenizer, pad_on_right, max_length, doc_stride, logits=updated_preds)
            f1s.append(res.get('f1'))
            exacts.append(res.get('exact'))
        else:
            print("Doing NLI eval!")
            print("Original")
            o_probs = eval_nli(eval_corpus + 'text.csv', original_preds)
            # print("Updated to loss", xfact_loss)
            u_probs = eval_nli(eval_corpus + 'text.csv', updated_preds)
            # print("Probs diff", u_probs - o_probs)
    with open(data_file, 'a') as f:
        f.write("Seed " + str(seed) + ' Layer ' + str(layer) + '\n')
        f.write(str(f1s) + '\n')
        f.write(str(exacts) + '\n')



if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('counterfactual_config')
    cli_args = argp.parse_args()
    counterfactual_args = yaml.load(open(cli_args.counterfactual_config))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    counterfactual_args['device'] = device


    # QA models
    base_checkpoint = 'twmkn9/bert-base-uncased-squad2'
    # NLI model
    # base_checkpoint = 'saved_models/nli/bert-nli.pt'

    is_nli_model = 'nli' in base_checkpoint
    # For NLI models.
    # eval_corpus = 'data/eval_corpora/nli/attach/'  # For attach
    # eval_corpus = 'data/eval_corpora/nli/subsequence/'  # The doctor newar the actor danced

    for layer in range(1, 13):
        for seed in range(5):
            print("Seed", seed)
            dropout_rate = 0
            eval_corpus = counterfactual_args['dataset']['corpus']['root']
            # counterfactual_args['reporting']['root'] = eval_corpus
            run_eval(counterfactual_args, eval_corpus + '/text.json')
