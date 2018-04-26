from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import numpy as np

input_weights = \
    './lib/datasets/data/trained_models/28594643_model_final.pkl'
output_weights = \
    './lib/datasets/data/trained_models/28594643_model_final_w_embed.pkl'
embed_glove = \
    './lib/datasets/bbox2mask_dataset_processing/coco/glove_randn/fixed_embedding_mat_coco_glove.npy'
embed_randn = \
    './lib/datasets/bbox2mask_dataset_processing/coco/glove_randn/fixed_embedding_mat_coco_randn_0.npy'

v_glove = np.load(embed_glove)
v_glove = np.concatenate((np.zeros_like(v_glove)[:1], v_glove))  # add bg

v_randn = np.load(embed_randn)
v_randn = np.concatenate((np.zeros_like(v_randn)[:1], v_randn))  # add bg

with open(input_weights, 'rb') as f:
    weights = pickle.load(f)

assert 'blobs' in weights
weights['blobs']['v_glove'] = v_glove
weights['blobs']['v_randn_0'] = v_randn

with open(output_weights, 'wb') as f:
    pickle.dump(weights, f)
