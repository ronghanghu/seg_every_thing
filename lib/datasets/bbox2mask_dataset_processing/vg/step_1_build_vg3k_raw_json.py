from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
from collections import Counter


N_class = 3000  # keep the top 3000 classes
raw_data_dir = './lib/datasets/data/vg/annotations/'
output_dir = './lib/datasets/data/vg3k_bbox2mask/'

# ---------------------------------------------------------------------------- #
# Load raw VG annotations and collect top-frequent synsets
# ---------------------------------------------------------------------------- #

with open(raw_data_dir + 'image_data.json') as f:
    raw_img_data = json.load(f)
with open(raw_data_dir + 'objects.json') as f:
    raw_obj_data = json.load(f)

# collect top frequent synsets
all_synsets = [
    synset for img in raw_obj_data
    for obj in img['objects'] for synset in obj['synsets']]
synset_counter = Counter(all_synsets)
top_synsets = [
    synset for synset, _ in synset_counter.most_common(N_class)]

# ---------------------------------------------------------------------------- #
# build raw "categories"
# ---------------------------------------------------------------------------- #

categories = [
    {'id': (n + 1), 'name': synset} for n, synset in enumerate(top_synsets)]
synset2cid = {c['name']: c['id'] for c in categories}

# ---------------------------------------------------------------------------- #
# build "image"
# ---------------------------------------------------------------------------- #

images = [
    {'id': img['image_id'],
     'width': img['width'],
     'height': img['height'],
     'file_name': img['url'].replace('https://cs.stanford.edu/people/rak248/', ''),
     'coco_id': img['coco_id']}
    for img in raw_img_data]

# ---------------------------------------------------------------------------- #
# build "annotations"
# ---------------------------------------------------------------------------- #

annotations = []
skip_count_1, skip_count_2, skip_count_3 = 0, 0, 0
for img in raw_obj_data:
    for obj in img['objects']:
        synsets = obj['synsets']
        if len(synsets) == 0:
            skip_count_1 += 1
        elif len(synsets) > 1:
            skip_count_2 += 1
        elif synsets[0] not in synset2cid:
            skip_count_3 += 1
        else:
            cid = synset2cid[synsets[0]]
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            area = obj['w'] * obj['h']
            ann = {'id': obj['object_id'],
                   'image_id': img['image_id'],
                   'category_id': cid,
                   'segmentation': [],
                   'area': area,
                   'bbox': bbox,
                   'iscrowd': 0}
            annotations.append(ann)

# ---------------------------------------------------------------------------- #
# Save to json file
# ---------------------------------------------------------------------------- #

with open(output_dir + 'instances_vg3k_raw.json', 'w') as f:
    json.dump(
        {'images': images,
         'annotations': annotations,
         'categories': categories}, f)
