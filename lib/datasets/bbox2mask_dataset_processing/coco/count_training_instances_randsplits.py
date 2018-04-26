from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

#
# Count the number of training instances for each random split
#

json_prefix = './lib/datasets/data/coco_bbox2mask/split_{}_A_'

# Randomly split the datasets into N_A classes and
N_AB = 80
N_As = [20, 30, 40, 50, 60]
N_E = 5

pairs = []
for N_A in N_As:
    for n_exp in range(N_E):
        name = 'E{}_A{}B{}'.format(n_exp + 1, N_A, N_AB - N_A)

        with open(json_prefix.format(name) + 'instances_train2014.json') as f:
            num_train = len(json.load(f)['annotations'])
        with open(json_prefix.format(name) + 'instances_valminusminival2014.json') as f:
            num_val = len(json.load(f)['annotations'])
        num_trainval = num_train + num_val

        pairs.append((name, num_trainval))
        print('{}: #training instances = {}'.format(name, num_trainval))

print(pairs)
