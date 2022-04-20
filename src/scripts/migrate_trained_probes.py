import glob
import os
import shutil

from tqdm import tqdm


"""
Utility method copies over trained models from the saved_models area into counterfactuals.
This will create the directory structure in which counterfactual embeddings will be saved, without corrupting
the saved_models directory.
"""

probe_suite = 'qa_dropout9_depth_3layer'
# probe_suite = 'qa_dropout1_dist_3layer'
# probe_suite = 'dropout4_depth_3layer'
for seed in range(0, 5):
    # for suite in ['conj', 'npz']:
    for suite in ['qa_coord', 'qa_npvp', 'qa_rc', 'qa_intervene']:
        source_dir = 'saved_models/seed%s/%s' % (seed, probe_suite)
        dest_dir = 'counterfactuals/' + suite + '/seed%s/%s' % (seed, probe_suite)

        model_prefix = 'model_dist' if 'dist' in probe_suite else 'model_depth'

        for layer_id in range(1, 13):
            probe_dir = source_dir + '/' + model_prefix + str(layer_id)
            # Find the last saved probe directory in the source area.
            print("Looking at probe dir", probe_dir)
            model_dirs = glob.glob(os.path.join(probe_dir, '*'))
            model_dir = sorted(model_dirs)[-1]

            dest = dest_dir + '/' + model_prefix + str(layer_id)
            if not os.path.exists(dest):
                os.makedirs(dest)
            else:
                print("Directory already exists at", dest)
            # Copy over the model parameters.
            try:
                shutil.copyfile(os.path.join(model_dir, 'predictor.params'), os.path.join(dest, 'predictor.params'))
            except shutil.SameFileError:
                tqdm.write('Note, the config being used is the same as that already present in the results dir')
