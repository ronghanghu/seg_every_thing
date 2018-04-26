from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

#
# Count the number of training instances for VOC / non-VOC
#

json_prefix = './lib/datasets/data/coco_bbox2mask/split_{}_'

pairs = []
for name in ['voc', 'nonvoc']:
    with open(json_prefix.format(name) + 'instances_train2014.json') as f:
        num_train = len(json.load(f)['annotations'])
    with open(json_prefix.format(name) + 'instances_valminusminival2014.json') as f:
        num_val = len(json.load(f)['annotations'])
    num_trainval = num_train + num_val

    pairs.append((name, num_trainval))
    print('{}: #training instances = {}'.format(name, num_trainval))
