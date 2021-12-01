import glob
import os
import shutil

from tqdm import tqdm


"""
Utility method copies over trained models from the saved_models area into counterfactuals.
This will create the directory structure in which counterfactual embeddings will be saved, without corrupting
the saved_models directory.
"""

source_dir = 'saved_models/model_dist_3layer'
dest_dir = 'counterfactuals/model_dist_3layer'

model_prefix = 'model_dist'

for layer_id in range(1, 6):  # FIXME
    probe_dir = source_dir + '/' + model_prefix + str(layer_id)
    # Find the last saved probe directory in the source area.
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