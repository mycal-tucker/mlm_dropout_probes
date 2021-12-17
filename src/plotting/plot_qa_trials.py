import ast
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import FormatStrFormatter
import matplotlib


def plot_trials(xfact_config, _seeds, _layers):
    _layer_offset = _layers[0]
    text_dir = xfact_config['dataset']['corpus']['root']

    # Read in the other question info as well
    corpus_types = []
    answer_lengths = []
    start_likelihoods = []
    contexts = []
    questions = []
    answers = []
    with open(text_dir + 'setup.txt', 'r') as setup_file:
        for line_idx, line in enumerate(setup_file):
            split_line = line.split('\t')
            corpus_types.append(split_line[0])
            answer_lengths.append(int(split_line[1]))
            start_likelihoods.append(float(split_line[2]))
            contexts.append(split_line[3])
            questions.append(split_line[4])
            answers.append(split_line[5])

    # Read in the token id data. We care about probability changes at specific locations, which were stored way back
    # when the corpus was generated in token_idxs.txt.
    # We care about 4 locations (determiner and noun) x (location 1 and location 2)
    det1_token_idxs = []
    nn1_token_idxs = []
    det2_token_idxs = []
    nn2_token_idxs = []
    all_token_idxs = [det1_token_idxs, nn1_token_idxs, det2_token_idxs, nn2_token_idxs]
    with open(text_dir + 'token_idxs.txt', 'r') as token_file:
        for line_idx, line in enumerate(token_file):
            if line_idx % 2 == 0:
                continue  # Have twice as many token lines as needed because the sentence were duplicated.
            split_line = line.split('\t')
            det1_token_idxs.append(int(split_line[0]))
            nn1_token_idxs.append(int(split_line[1]))
            det2_token_idxs.append(int(split_line[2]))
            nn2_token_idxs.append(int(split_line[3]))

    all_seeds_original_starts = []
    all_seeds_nn1_parse_starts = []
    all_seeds_nn2_parse_starts = []
    for seed in _seeds:
        effect_reporting_dir = 'counterfactuals/qa_coord/seed' + str(seed) + '/qa_dropout0_dist_3layer/'
        all_layers_original_starts = []
        all_layers_nn1_parse_starts = []
        all_layers_nn2_parse_starts = []
        for layer in _layers:
            update_file_base = effect_reporting_dir + 'model_dist' + str(layer) + '/updated_probs_xfactloss_' + str(xfact_loss)
            # Read in how the probabilities got updated.
            original_start_probs = []
            nn1_parse_updated_start_probs = []
            nn2_parse_updated_start_probs = []
            with open(update_file_base + '.txt', 'r') as results_file:
                for line_idx, line in enumerate(results_file):
                    split_line = line.split('\t')
                    if line_idx % 2 == 0:
                        original_start_probs.append([ast.literal_eval(data)[0] for data in split_line])
                        nn1_parse_updated_start_probs.append([ast.literal_eval(data)[2] for data in split_line])
                    else:
                        nn2_parse_updated_start_probs.append([ast.literal_eval(data)[2] for data in split_line])
            # Dump the layer-specific data into an aggregator.
            all_layers_original_starts.append(original_start_probs)
            all_layers_nn1_parse_starts.append(nn1_parse_updated_start_probs)
            all_layers_nn2_parse_starts.append(nn2_parse_updated_start_probs)
        all_seeds_original_starts.append(all_layers_original_starts)
        all_seeds_nn1_parse_starts.append(all_layers_nn1_parse_starts)
        all_seeds_nn2_parse_starts.append(all_layers_nn2_parse_starts)

    def gen_for_seed(_seed):
        # Test suite stuff is independent of seed.
        np1_starts = all_token_idxs[0]
        np1_ends = all_token_idxs[1]
        np2_starts = all_token_idxs[2]
        np2_ends = all_token_idxs[3]

        # Pull out seed-specific stuff
        _all_layers_original_starts = all_seeds_original_starts[_seed]
        _all_layers_nn1_parse_starts = all_seeds_nn1_parse_starts[_seed]
        _all_layers_nn2_parse_starts = all_seeds_nn2_parse_starts[_seed]

        np1_p1_all = []
        np1_p2_all = []
        np1_original_all = []
        for _layer in _layers:
            np1_original_probs = []
            np1_p1_probs = []
            np1_p2_probs = []
            for sentence_idx, token_limits in enumerate(zip(np1_starts, np1_ends, np2_starts, np2_ends)):
                np1_start, np1_end, np2_start, np2_end = token_limits
                s_np1_original = []
                s_np1_p1 = []
                s_np1_p2 = []
                for np1_token_idx in range(np1_start, np1_end + 1):
                    original_prob = _all_layers_original_starts[_layer - _layer_offset][sentence_idx][np1_token_idx]
                    np1_p1_prob = _all_layers_nn1_parse_starts[_layer - _layer_offset][sentence_idx][np1_token_idx]
                    np1_p2_prob = _all_layers_nn2_parse_starts[_layer - _layer_offset][sentence_idx][np1_token_idx]
                    s_np1_original.append(original_prob)
                    s_np1_p1.append(np1_p1_prob)
                    s_np1_p2.append(np1_p2_prob)
                np1_original_probs.append(np.sum(s_np1_original))
                np1_p1_probs.append(np.sum(s_np1_p1))
                np1_p2_probs.append(np.sum(s_np1_p2))
            np1_p1_all.append(np.mean(np1_p1_probs))
            np1_p2_all.append(np.mean(np1_p2_probs))
            np1_original_all.append(np.mean(np1_original_probs))

        # Plot for just this seed if you want:
        # fig, ax = plt.subplots(nrows=1, figsize=(10, 5))
        # x_axis = [i for i in _layers]
        # ax.set_title("Start prob for NP1")
        # ax.set_xlabel('Layer idx')
        # ax.errorbar(x_axis, np1_p1_all, color='red', marker='s', label='NP1 Parse')
        # ax.errorbar(x_axis, np1_p2_all, color='blue', marker='s', label='NP2 Parse')
        # ax.errorbar(x_axis, np1_original_all, color='black')
        # plt.show()
        return np1_p1_all, np1_p2_all, np1_original_all

    trials_p1 = []
    trials_p2 = []
    trials_orig = []
    for seed in _seeds:
        p1_all, p2_all, original_all = gen_for_seed(seed)
        trials_p1.append(p1_all)
        trials_p2.append(p2_all)
        trials_orig.append(original_all)
    # Now plot for all trials.

    # Calculate the actual mean difference.
    layer_cutoff = 13
    overall_p1_mean = np.mean(np.array(trials_p1)[:, :layer_cutoff])
    overall_p2_mean = np.mean(np.array(trials_p2)[:, :layer_cutoff])
    overall_original_mean = np.mean(trials_orig)
    print("Mean p1 diff", overall_p1_mean - overall_original_mean)
    print("Mean p2 diff", overall_original_mean - overall_p2_mean)
    mean_effect = overall_p1_mean - overall_p2_mean
    print("Mean effect", mean_effect)

    matplotlib.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(nrows=1, figsize=(10, 3.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    x_axis = [i for i in _layers]
    # ax.set_title("Likelihood of NP1 Start by Layer")
    ax.set_xlabel('Layer index')
    ax.set_ylabel("Prob. NP1")
    ax.errorbar(x_axis, np.mean(trials_p1, axis=0), yerr=np.std(trials_p1, axis=0), color='red', linestyle='--', label='NP1 Parse')
    ax.errorbar(x_axis, np.mean(trials_p2, axis=0), yerr=np.std(trials_p2, axis=0), color='blue', label='NP2 Parse')
    ax.errorbar(x_axis, np.mean(trials_orig, axis=0), color='green', label='Original')
    # ax.legend(loc="upper right")
    plt.tight_layout()
    plt.xlim(1, len(_layers))
    # plt.ylim(0.40, 0.70)  # Attach
    # plt.ylim(0.52, 0.56)  # NPVP
    # plt.ylim(0.65, 0.82)  # RC
    # plt.savefig('qa_net.png')
    plt.show()
    return mean_effect


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('counterfactual_config')
    cli_args = argp.parse_args()
    counterfactual_args = yaml.load(open(cli_args.counterfactual_config))
    layers = [i for i in range(1, 10)]
    seeds = [i for i in range(1)]
    xfact_loss = 0.3
    plot_trials(counterfactual_args, seeds, layers)
