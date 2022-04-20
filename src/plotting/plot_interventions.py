import numpy as np
import ast
import matplotlib.pyplot as plt

max_dropout = 10
# Load all the data from the files
dropout_to_layer_to_f1 = np.zeros((max_dropout, 12, 5, 7))
dropout_to_layer_to_exact = np.zeros((max_dropout, 12, 5, 7))
for dropout_rate in range(0, max_dropout):
    with open('dist' + str(dropout_rate) + '_test_inter.txt', 'r') as f:
        curr_layer_idx = None
        curr_seed = None
        for line_idx, line in enumerate(f):
            if 'Layer' in line:
                layer = int(line[13:])
                curr_layer_idx = layer - 1  # Because start at layer 1
                curr_seed = int(line[5])
                continue
            data = dropout_to_layer_to_f1 if line_idx % 3 == 1 else dropout_to_layer_to_exact
            line_data = ast.literal_eval(line)
            data[dropout_rate, curr_layer_idx, curr_seed, :] = line_data

# Now do some plotting, etc, with all the data.
for layer_idx in range(12):
    fig, ax = plt.subplots(nrows=1, figsize=(5, 2.1))
    x_data = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    markers = ['o', '*', 'v', '^', 'x']
    line_idx = 0
    for dropout_rate in range(0, max_dropout, 2):
        raw = dropout_to_layer_to_f1[dropout_rate, layer_idx, :, :]
        
        ax.errorbar(x_data, np.median(raw, axis=0),
                yerr=[np.quantile(raw, 0.5, axis=0) - np.quantile(raw, 0.25, axis=0),
                      np.quantile(raw, 0.75, axis=0) - np.quantile(raw, 0.5, axis=0)],
                label=str(0.1 * dropout_rate)[:3],
                marker=markers[line_idx])
        line_idx += 1
    ax.set_ylabel('F1')
    ax.set_xlabel('Counterfactual Stopping Loss')
    ax.legend(loc='upper right')
    ax.set_ylim([81, 86])
    plt.tight_layout()
    plt.savefig('f1_vs_xfactloss_layer' + str(layer_idx) + '.png')
    ax.set_ylim([81, 86])
    plt.close()
    
    # Now plot f1 vs dropout rate for the same layer
    fig, ax = plt.subplots(nrows=1, figsize=(5, 2.1))
    x_data = [i * 0.1 for i in range(max_dropout)]
    line_idx = 0
    #for xfact_loss_idx, xfact_loss in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]):
    for xfact_loss_idx, xfact_loss in enumerate([0.05, 0.15, 0.25, 0.35]):
        raw = dropout_to_layer_to_f1[:, layer_idx, :, xfact_loss_idx]
        
        ax.errorbar(x_data, np.median(raw, axis=1),
                yerr=[np.quantile(raw, 0.5, axis=1) - np.quantile(raw, 0.25, axis=1),
                    np.quantile(raw, 0.75, axis=1) - np.quantile(raw, 0.5, axis=1)],
                label='Loss ' + str(xfact_loss),
                marker=markers[line_idx])
        line_idx += 1
    ax.set_ylabel('F1')
    ax.set_xlabel('Dropout Rate')
    ax.legend(loc='upper right')
    ax.set_ylim([81, 86])
    plt.tight_layout()
    plt.savefig('f1_vs_dropout_layer' + str(layer_idx) + '.png')
    plt.close()

    fig, ax = plt.subplots(nrows=1, figsize=(5, 2.1))
    for xfact_loss_idx, xfact_loss in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]):
        raw = dropout_to_layer_to_exact[:, layer_idx, :, xfact_loss_idx]
        ax.errorbar(x_data, np.median(raw, axis=1),
                yerr = [np.quantile(raw, 0.5, axis=1) - np.quantile(raw, 0.25, axis=1),
                    np.quantile(raw, 0.75, axis=1) - np.quantile(raw, 0.5, axis=1)],
                label='Loss' + str(xfact_loss))
    ax.set_ylabel('Exact')
    ax.set_xlabel('Dropout Rate')
    ax.legend(loc='upper right')
    ax.set_ylim([62, 70])
    plt.tight_layout()
    plt.savefig('exact_vs_dropout_layer' + str(layer_idx) + '.png')
    plt.close()

fig, ax = plt.subplots(nrows=1, figsize=(5, 2.1))
x_data = [i * 0.1 for i in range(max_dropout)]
for xfact_loss_idx, xfact_loss in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]):
    raw = dropout_to_layer_to_f1[:, :, :, xfact_loss_idx]
    ax.errorbar(x_data, np.median(raw, axis=(1, 2)), label='Loss' + str(xfact_loss))
ax.legend(loc='upper right')
ax.set_ylim(81, 84)
plt.tight_layout()
plt.savefig('f1_vs_dropout_overall.png')
plt.close()

fig, ax = plt.subplots(nrows=1, figsize=(5, 2.1))
for xfact_loss_idx, xfact_loss in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]):
    raw = dropout_to_layer_to_exact[:, :, :, xfact_loss_idx]
    ax.errorbar(x_data, np.median(raw, axis=(1, 2)), label='Loss' + str(xfact_loss))
ax.legend(loc='upper right')
ax.set_ylim([62, 70])
plt.tight_layout()
plt.savefig('exact_vs_dropout_overall.png')
plt.close()
