from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np

#
# Generate configs to train COCO random splits
#

cfgs_dir = \
    './configs/bbox2mask_coco/randsplits/'
template_baseline = \
    open(os.path.join(cfgs_dir, 'template_baseline.yaml')).read()
template_clsbox_2_layer = \
    open(os.path.join(cfgs_dir, 'template_clsbox_2_layer.yaml')).read()


# Randomly split the datasets into N_A classes and
N_AB = 80
N_As = [20, 30, 40, 50, 60]
N_E = 5

np.random.seed(3)  # fix random seed for repeatibility
for n_exp in range(N_E):
    for N_A in N_As:
        name = 'E{}_A{}B{}'.format(n_exp + 1, N_A, N_AB - N_A)

        save_cfg_name = 'eval_sw/{}_baseline.yaml'.format(name)
        with open(os.path.join(cfgs_dir, save_cfg_name), 'w') as f:
            f.write(template_baseline.format(name, name, name, name))

        save_cfg_name = 'eval_sw/{}_clsbox_2_layer.yaml'.format(name)
        with open(os.path.join(cfgs_dir, save_cfg_name), 'w') as f:
            f.write(template_clsbox_2_layer.format(name, name, name, name))
