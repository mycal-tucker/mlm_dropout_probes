import ast
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from matplotlib.ticker import FormatStrFormatter


def do_plotting():

    if is_npz:
        POS1 = 'Adverb'
        POS2 = 'Noun'
        parse1_label = 'Adverb'
        parse2_label = 'Noun'
        # candidates = ['all', 'he', 'just', 'it', 'they', 'she', 'was', 'were']
        parts_of_speech1 = ['all', 'just', 'was', 'were']
        parts_of_speech2 = ['he', 'she', 'they', 'it']
    else:
        POS1 = 'Plural'
        POS2 = 'Singular'
        parse1_label = 'Plural'
        parse2_label = 'Singular'
        # You need to define which words belong to which part of speech
        parts_of_speech1 = ['were', 'are', 'as']
        parts_of_speech2 = ['was', 'is']

    word_to_type = {}
    for word1 in parts_of_speech1:
        word_to_type[word1] = POS1
    for word2 in parts_of_speech2:
        word_to_type[word2] = POS2

    layer_offset = test_layers[0]  # Have to handle the zero vs. one-indexing of layers.
    candidates = None

    trials_all_layers_originals = []
    trials_all_layers_parse1 = []
    trials_all_layers_parse2 = []
    for seed in seeds:
        if is_npz:
            counterfactual_dir = 'counterfactuals/npz/seed' + str(seed) + '/dropout' + str(dropout_rate) + '_' + dist_or_depth + '_3layer/'
        else:
            counterfactual_dir = 'counterfactuals/conj/seed' + str(seed) + '/dropout' + str(dropout_rate) + '_' + dist_or_depth + '_3layer/'
        probe_type = 'model_dist' if 'dist' in counterfactual_dir else 'model_depth'
        # Read in the data about original and updated_probabilities
        all_layers_originals = []
        all_layers_parse1 = []
        all_layers_parse2 = []
        for layer in test_layers:
            # Read in how the probabilities got updated.
            original_probs = []
            p1_updated_probs = []
            p2_updated_probs = []
            with open(counterfactual_dir + probe_type + str(layer) + '/updated_probs_xfactloss_' + str(xfact_loss) + '.txt', 'r') as results_file:
                for line_idx, line in enumerate(results_file):
                    split_line = line.split('\t')
                    if line_idx == 0 and split_line[0] == 'Candidates':
                        new_candidates = split_line[1:-1]
                        candidates = [cand.strip() for cand in new_candidates]
                        continue
                    # First half is the original probability, second half is updated
                    updated = split_line[int(len(split_line) / 2):]
                    if line_idx % 2 == 1:  # Off by 1 because of candidates thing!
                        original = split_line[:int(len(split_line) / 2)]
                        original_probs.append([ast.literal_eval(data) for data in original])
                        p1_updated_probs.append([ast.literal_eval(data) for data in updated])
                    else:
                        p2_updated_probs.append([ast.literal_eval(data) for data in updated])
            # Now we have the data, so if you want to plot probabilities for a single sentence, you can by uncommenting below.
            # for i in range(2):
            #     plot_sentence_probs(i)
            # Dump the layer-specific data into an aggregator.
            all_layers_originals.append(original_probs)
            all_layers_parse1.append(p1_updated_probs)
            all_layers_parse2.append(p2_updated_probs)
        trials_all_layers_originals.append(all_layers_originals)
        trials_all_layers_parse1.append(all_layers_parse1)
        trials_all_layers_parse2.append(all_layers_parse2)

    # Now that candidates is initialized, group by part of speech.
    pos1_idxs = []
    pos2_idxs = []
    for candidate_idx, candidate in enumerate(candidates):
        tag = word_to_type.get(candidate)
        if tag is None:
            assert False, "Need to tag " + candidate + " in word_to_type"
        if tag == POS1:
            pos1_idxs.append(candidate_idx)
        elif tag == POS2:
            pos2_idxs.append(candidate_idx)
        else:
            assert False, "Bad tag for candidate " + candidate + " in word_to_type"

    # There's a bit of ugly reshuffling/grouping of the data to get it sliced up by different parts of speeches and by
    # different parses.
    original_pos1s = np.asarray(trials_all_layers_originals)[:, :, :, pos1_idxs]
    original_pos2s = np.asarray(trials_all_layers_originals)[:, :, :, pos2_idxs]
    p1_pos1s = np.asarray(trials_all_layers_parse1)[:, :, :, pos1_idxs]
    p2_pos1s = np.asarray(trials_all_layers_parse2)[:, :, :, pos1_idxs]
    p1_pos2s = np.asarray(trials_all_layers_parse1)[:, :, :, pos2_idxs]
    p2_pos2s = np.asarray(trials_all_layers_parse2)[:, :, :, pos2_idxs]
    p1_pos1_delta = p1_pos1s - original_pos1s
    p2_pos1_delta = p2_pos1s - original_pos1s
    p1_pos2_delta = p1_pos2s - original_pos2s
    p2_pos2_delta = p2_pos2s - original_pos2s

    p1_pos1_change_by_sentence = np.sum(p1_pos1_delta, axis=3)
    p2_pos1_change_by_sentence = np.sum(p2_pos1_delta, axis=3)
    p1_pos2_change_by_sentence = np.sum(p1_pos2_delta, axis=3)
    p2_pos2_change_by_sentence = np.sum(p2_pos2_delta, axis=3)



    # Plot the original and updated agregated probabilities by layer for each parse and part of speech. These are the
    # types of plots included in the main paper.
    def net_probabilities():
        matplotlib.rcParams.update({'font.size': 18})
        p1_pos1_means = []
        p2_pos1_means = []
        x_axis = test_layers
        for trial_id in seeds:
            p1_pos1_mean = np.mean(np.sum(p1_pos1s[trial_id], axis=2), axis=1)
            p2_pos1_mean = np.mean(np.sum(p2_pos1s[trial_id], axis=2), axis=1)
            original_pos1s_mean = np.mean(np.sum(original_pos1s[trial_id], axis=2), axis=1)
            p1_pos1_means.append(p1_pos1_mean)
            p2_pos1_means.append(p2_pos1_mean)

        p1_mean = np.mean(p1_pos1_means, axis=0)
        p2_mean = np.mean(p2_pos1_means, axis=0)

        mean_diff = np.mean(p1_mean) - np.mean(p2_mean)
        std_diff = np.mean(np.std(np.asarray(p1_pos1_means) - np.asarray(p2_pos1_means), axis=0))
        print("XFact loss", xfact_loss, "mean_diff", mean_diff, "std", std_diff)

        fig, ax1 = plt.subplots(nrows=1, figsize=(10, 3.1))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.errorbar(x_axis, p1_mean, yerr=np.std(p1_pos1_means, axis=0), color='red', linestyle='--', label=parse1_label + ' parse')
        ax1.errorbar(x_axis, original_pos1s_mean, color='green', label='Original')
        ax1.errorbar(x_axis, p2_mean, yerr=np.std(p1_pos1_means, axis=0), color='blue', label=parse2_label + ' parse')
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Layer index")
        ax1.set_ylabel("Prob. " + POS1)
        fig.suptitle("Likelihood of " + POS1 + " Candidates by Layer")
        plt.xlim(1, len(test_layers))
        # plt.ylim(0.29, 0.45)
        fig.tight_layout()
        # plt.savefig('net_probs.png')
        plt.show()


    net_probabilities()

if __name__ == '__main__':
    is_npz = True
    dist_or_depth = 'depth'
    text_dir = 'data/npz' if is_npz else 'data/conj/'
    dropout_rate = 9 
    test_layers = [i for i in range(1, 13)]
    seeds = [i for i in range(0, 5)]
    for xfact_loss in [0.3, 0.2, 0.1, 0.05]:
        do_plotting()

