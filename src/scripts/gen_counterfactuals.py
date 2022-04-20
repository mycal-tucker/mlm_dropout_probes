import os
from argparse import ArgumentParser

import h5py
import numpy as np
import torch
import yaml

from src.utils.training_utils import choose_task_classes, choose_dataset_class, choose_model_class, choose_probe_class


"""
Generates counterfactual embeddings using probes for each layer.
"""


# Given some config, a probe, d, and a loss function, updates the elements of the dataset to minimize the loss of the
# probe. This is the core counterfactual generation technique.
def gen_counterfactuals(args, probe, dataset, loss, loss_tolerance):
    probe_params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
    probe.load_state_dict(torch.load(probe_params_path, map_location=torch.device('cuda:0')))
    probe.eval()
    original_embeddings = None
    word_embeddings = []
    updated_embeddings = []
    test_dataloader = dataset.get_test_dataloader()
    iteration_idx = -1
    #all_tolerances = [0.3, 0.2, 0.1, 0.05]
    all_tolerances = [0.35, 0.3, 0.25, 0.20, 0.15, 0.1, 0.05]
    all_xfacts = {}
    next_loss_idx = 0
    loss_tolerance = 0.05
    for batch_idx, batch in enumerate(test_dataloader):
        iteration_idx += batch[0].shape[0]
        observation_batch, label_batch, length_batch, _ = batch
        start_embeddings = observation_batch
        true_labels = label_batch
        if args['dataset']['embeddings']['break_on_qmark']:
            q_mark_idxs = []
            for intra_batch_idx in range(batch[0].shape[0]):
                observation = test_dataloader.dataset.observations[iteration_idx]
                sentence = observation.sentence
                q_mark_idx = sentence.index('?')
                q_mark_idxs.append(q_mark_idx)
            for i, q_mark_idx in enumerate(q_mark_idxs):
                # print("dim", true_labels.shape)
                if len(true_labels.shape) == 2:  # Depth task
                    true_labels[i, :q_mark_idx + 1] = -1
                else:
                    true_labels[i, :q_mark_idx + 1] = -1
                    true_labels[i, :, :q_mark_idx + 1] = -1
        curr_embeddings = start_embeddings
        word_embeddings.extend([curr_embedding.clone() for curr_embedding in curr_embeddings])
        curr_embeddings = curr_embeddings.cuda()
        curr_embeddings.requires_grad = True
        my_optimizer = torch.optim.Adam([curr_embeddings], lr=0.00001)
        prediction_loss = 100  # Initialize the prediction loss as really high - it gets overwritten during updates.
        increment_idx = 0
        # We implement patience manually.
        smallest_loss = prediction_loss
        steps_since_best = 0
        patience = 5000
        while next_loss_idx < len(all_tolerances):
            if increment_idx % 1000 == 0:
                print("At increment", increment_idx, "loss", prediction_loss)
            if increment_idx >= 50000:
                print("Breaking because of increment index")
                break
            if prediction_loss < all_tolerances[next_loss_idx]:
                # Save this result.
                all_xfacts[all_tolerances[next_loss_idx]] = [curr_embedding.detach().clone() for curr_embedding in curr_embeddings]
                next_loss_idx += 1
            predictions = probe(curr_embeddings)
            prediction_loss, count = loss(predictions, torch.reshape(true_labels, predictions.shape), length_batch)
            prediction_loss.backward()
            my_optimizer.step()
            if prediction_loss < smallest_loss:
                steps_since_best = 0
                smallest_loss = prediction_loss
            else:
                steps_since_best += 1
                if steps_since_best == patience:
                    print("Breaking because of patience with loss", prediction_loss)
                    #if next_loss_idx < len(all_tolerances):
                    #    while next_loss_idx < len(all_tolerances):
                    #        all_xfacts[all_tolerances[next_loss_idx]] = all_xfacts[all_tolerances[next_loss_idx - 1]]
                    #        next_loss_idx += 1
                    break
            increment_idx += 1
        print("Exited grad update loop after", increment_idx, "steps with loss", prediction_loss)
        updated_embeddings.extend([curr_embedding for curr_embedding in curr_embeddings])
    if next_loss_idx < len(all_tolerances):  # Stopped early
        while next_loss_idx < len(all_tolerances):
            all_xfacts[all_tolerances[next_loss_idx]] = [curr_embedding.detach().clone() for curr_embedding in curr_embeddings]
            next_loss_idx += 1
    #return original_embeddings, word_embeddings, updated_embeddings
    return word_embeddings, all_xfacts


def execute_experiment(args, loss_tolerance):
    dataset_class = choose_dataset_class(args)
    task_class, reporter_class, loss_class = choose_task_classes(args)
    probe_class = choose_probe_class(args)
    task = task_class()
    expt_dataset = dataset_class(args, task)
    expt_probe = probe_class(args)
    expt_loss = loss_class(args)
    results_dir = args['reporting']['root']
    print("Called execute experiment with loss", loss_tolerance)
    all_losses = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # If it exists, just return that.
    if os.path.exists(results_dir + '/updated_words_xfactloss_' + str(loss_tolerance) + '.hdf5'):
        print('Already computed for loss', loss_tolerance)
        return
    print("Don't already have file", results_dir + '/updated_words_xfactloss_' + str(loss_tolerance) + '.hdf5')
    # Otherwise, create all the embeddings
    # Generate the counterfactuals by delegating to the right method.
    #_, word_only_embeddings, updated_embeddings = gen_counterfactuals(args, expt_probe, expt_dataset, expt_loss, loss_tolerance)
    word_only_embeddings, all_xfacts = gen_counterfactuals(args, expt_probe, expt_dataset, expt_loss, loss_tolerance)
    # Save the updated and original words to files.
    with torch.no_grad():
        np_word = [embedding.cpu().numpy() for embedding in word_only_embeddings]
        for key, val in all_xfacts.items():
            np_updates = [embedding.cpu().numpy() for embedding in val]
            hf = h5py.File(results_dir + '/updated_words_xfactloss_' + str(key) + '.hdf5', 'w')
            for i, embedding in enumerate(np_updates):
                hf.create_dataset(str(i), data=embedding)
        #np_updated = [embedding.cpu().numpy() for embedding in updated_embeddings]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    hf = h5py.File(results_dir + '/original_words.hdf5', 'w')
    for i, embedding in enumerate(np_word):
        hf.create_dataset(str(i), data=embedding)
    #hf = h5py.File(results_dir + '/updated_words_xfactloss_' + str(loss_tolerance) + '.hdf5', 'w')
    #for i, embedding in enumerate(np_updated):
    #    hf.create_dataset(str(i), data=embedding)


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--results-dir', default='',
                      help='Set to reuse an old results dir; '
                           'if left empty, new directory is created')
    argp.add_argument('--report-results', default=1, type=int,
                      help='Set to report results; '
                           '(optionally after training a new probe)')
    argp.add_argument('--seed', default=0, type=int,
                      help='sets all random seeds for (within-machine) reproducibility')
    cli_args = argp.parse_args()
    # if cli_args.seed:
    #     np.random.seed(cli_args.seed)
    #     torch.manual_seed(cli_args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    yaml_args = yaml.load(open(cli_args.experiment_config))

    true_reporting_root = yaml_args['reporting']['root']
    suite = yaml_args['dataset']['corpus']['root'].split('/')[-2]  # Conj vs. npz
    for dropout_rate in range(0, 10):  # FIXME
        offset = 7 if 'qa' not in true_reporting_root else 10
        drop_report_root = true_reporting_root[:offset] + str(dropout_rate) + true_reporting_root[offset + 1:]
        seeds = [i for i in range(0, 5)]
        # for xfact_loss in [0.3, 0.2, 0.1, 0.05]:
        for xfact_loss in [0.05]:
            for seed in seeds:
                curr_reporting_root = 'counterfactuals/' + suite + '/seed' + str(seed) + '/' + drop_report_root
                for layer_idx in range(1, 13):  # FIXME
                    print("Counterfactuals for dropout", dropout_rate, "suite", suite, "xfact_loss", xfact_loss, "seed", seed, "layer", layer_idx)
                    # Somewhat gross, but we override part of the config file to do a full "experiment" for each layer.
                    yaml_args['model']['model_layer'] = layer_idx
                    yaml_args['reporting']['root'] = curr_reporting_root + str(layer_idx)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    yaml_args['device'] = device
                    execute_experiment(yaml_args, xfact_loss)
