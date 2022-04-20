# dropout-probes
Implementation of counterfactual probing technique (with dropout) discussed in "When Does Syntax Mediate Neural Language Model Performance?
Evidence from Dropout Probes" by Tucker et al.
This code is largely similar to the causal probing techinque first introduced by Tucker et al. in "What if This Modified That? Syntactic Interventions via Counterfactual Embeddings,"
The key differences are that probes now use dropout, which is critical overcoming "false negatives."

If you find this code useful, please cite the paper.
If you have questions, feel free to email mycal@mit.edu.

## Example workflow
This README is mostly designed to walk you through a full working example from training probes to generating counterfactuals to evaluating results.
We'll work through a specific example because modifying bits of functionality should be relatively simple within the framework.

All scripts must be run from mlm_dropout_probes/

### Training probes
To generate counterfactuals, we need trained probes. This isn't the core research here, so if you have trouble, just email mycal@mit.edu. But the steps below should get you training probes easily enough.
1) Run scripts/gen_embeddings.py with ``source_dir = 'data/ptb/'`` and ``filename='ptb_test``(and dev and train),
   As a heads up, the resulting .hdf5 files are quite big, so make sure you have about 80 GB of disk space.
   Oh, and set break_on_qmark to False, because we don't want to break things up by question mark when training the probe (but we will later for QA counterfactual embeddings).
2) Run ``src/scripts/train_probes.py`` with a parameter pointing to the config file ``config/example/parse_dist_ptb.yaml``. This trains the probes and takes a few hours to run.
Note that the dropout rate is hardcoded within the probe class right now. So, make sure to update that to whatever value you want.
Trained probes, with performance metrics, are saved to the reporting destination, specified in config.
 3) If you want to plot the probe performance metrics instead of just reading the files, look at the ``plotting/plot_probe_perf.py`` script.
   It pulls out some of the basic metrics and plots them. You might have to do some editing of the file if you want the depth probe vs. distance probe, for example.
   
### Generating syntactically interesting setup.
This is how to generate the data that we'll use for counterfactuals. It's actually a lot like the steps used for training probes.


1) Run scripts/gen_cloze_suite.py. This generates some example sentences and trees in data/example.
2) Generate the conllx file by going into the stanford nlp folder somewhere and running the following command (obviously updated for your specific paths)

``sudo java -mx3g -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile ~/src/mlm_dropout_probes/data/example/text.trees -checkConnected -basic -keepPunct -conllx > ~/src/causal-probe/data/example/text.conllx``

3) Run scripts/gen_embeddings.py. This generates embeddings for each layer for each of the sentences in the specified text file and saves them to an hdf5 file.

At this point, we have the embeddings for the interesting sentences created, so in the next step, we will create counterfactuals.

### Generating the counterfactuals.
Now that we have the embeddings for the interesting sentences and the interesting parses, let's generate the counterfactual embeddings.

We'll end up saving the embeddings as files, and it makes sense to group them with the trained probes, but maybe not in the saved_models directory.

1) To copy over the trained models, you can do it by hand, or you can use the ``src/scripts/migrate_trained_probes.py`` script. It just copies over the model parameters.
2) Run ``src/scripts/gen_counterfactuals.py`` with the argument of ``config/example/counterfactual_dist_cloze.yaml``.
   This generates two .hdf5 files in the directories with the probe parameters (all under counterfactuals).
   ``updated_words.hdf5`` had the counterfactual embeddings; ``original_words.hdf5`` has the original embeddings, but only for the words that got updated.
   For QA models, for example, this means only the words in the sentence, rather that words in the question as well.

### Evaluating counterfactual behaviors.
We've saved the counterfactual embeddings to hdf5 files in the previous step. Now we want to see if it has changed the model outputs.
Evaluation is done in two steps: first we pass the counterfactuals through the model and record outputs, and then we plot these outputs.
We break the plotting up like this just because the first evaluation step can take a long time, so it's nice to have the saved files when you're playing with new plotting utils.

1) Measure the effect of counterfactual embeddings by running ``src/scripts/eval_counterfactuals.py``.
The variables near the top fo the script define directories to look up text data and the embeddings.
The script produces .txt files that save relevant metrics about the model outputs using original and updated embeddings.
2) Plot these saved outputs by running ``src/plotting/plot_counterfactual_results.py``.
You set relevant directories by directly modifying the variables for different directories.
There are lots of different types of plots to generate - the different plotting methods have comments at the top of them to say what each one does.
Based on the findings from our work, you should find a consistent effect from using the distance-based probes to generate counterfactuals for mask word prediction.


### What about "boosting performance"?
In the paper, we discuss how we can boost a QA model's performance by injecting syntactic information for non-syntactically-ambiguous sentences.
The key script for recreating those experiments is ``src/scripts/eval_with_intervention.py`` which takes the standard config file as its argument.

The script writes results to a file, which you can then plot using the ``plot_interventions.py`` script.

### What about redundancy analysis?
In the paper, we use a technique called Mutual Information Neural Estimation (MINE) from prior art to estimate syntactic redundancy in encodings.
You can recreate those results with the ``src/scripts/info_analysis.py`` script as well.

## Congratulations!
Congratulations! You made it to the end.

At this point, you've trained probes for a fixed dropout rate and measured the counterfactual effects.
You've also shown how we can boost QA model performance via interventions.

Next steps for mor rigorous evaluation would be:
1) Test on different test suites. Any of the scripts about generating suites (all under ``src/scripts``) should be useful.
2) Test with different dropout rates. The dropout rate is currently hardcoded in probe.py. Feel free to edit that. Just change the directory you save the probes to in the config so that you don't overwrite your old probes.
3) Test with different BERT-based models. This codebase supports MASK and QA models - the NLI codebase was kept separate during rapid development. In principle, however, the key ideas of (dropout) probe-based interventions can be easily adapted to any BERT-like model.
   
If you have questions or more ideas, please reach out to mycal@mit.edu.
If you find this work useful, please cite the paper.


