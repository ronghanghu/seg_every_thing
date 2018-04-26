from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json


vg3k_raw_json_file = './lib/datasets/data/vg3k_bbox2mask/instances_vg3k_raw.json'
coco_minival_json_file = './lib/datasets/data/coco/instances_minival2014.json'

output_dir = './lib/datasets/data/vg3k_bbox2mask/'

# ---------------------------------------------------------------------------- #
# Load raw VG annotations and COCO annotations
# ---------------------------------------------------------------------------- #

with open(vg3k_raw_json_file) as f:
    dataset_vg3k = json.load(f)
with open(coco_minival_json_file) as f:
    dataset_coco = json.load(f)

vg3k_cls_names = [c['name'] for c in dataset_vg3k['categories']]
coco_cls_names = [c['name'] for c in dataset_coco['categories']]

# collect the coco minival image ids
coco_minival_imgids = [img['id'] for img in dataset_coco['images']]

# ---------------------------------------------------------------------------- #
# Map COCO class names to VG
# ---------------------------------------------------------------------------- #

# Step 1: identify the common class names in the two dataset
#
# clean VG class names before aligning with COCO:
#   e.g. "truck.n.01" -> "truck"
vg3k_cls_names_clean = [
    name[:-5].replace('_', ' ') for name in vg3k_cls_names]
# reverse mapping cleaned names ("truck") to the original name ("truck.n.01")
#
# build dictionary from back to forth so that the more frequent class is kept
# for duplicated names
vg3k_clean2original = {
    n1: n2 for n1, n2 in zip(vg3k_cls_names_clean[::-1], vg3k_cls_names[::-1])}
common_cls_names = set(vg3k_cls_names_clean) & set(coco_cls_names)
name_map_coco = {
    name_coco: vg3k_clean2original[name_coco] for name_coco in common_cls_names}

# Step 2: manually mapping the rest of classes
name_map_coco.update(
    {'cell phone': 'cellular_telephone.n.01',
     'couch': 'sofa.n.01',
     'dining table': 'dinner_table.n.01',
     'donut': 'doughnut.n.02',
     'fire hydrant': 'water_faucet.n.01',  # this is how hydrant is labeled in VG
     'hair drier': 'hand_blower.n.01',
     'handbag': 'bag.n.04',
     'hot dog': 'hotdog.n.01',
     'orange': 'orange.n.01',
     'potted plant': 'plant.n.01',
     'remote': 'remote_control.n.01',
     'skis': 'ski.n.01',
     'sports ball': 'ball.n.01',
     'stop sign': 'sign.n.02',
     'suitcase': 'bag.n.06',  # this is how suitcase is labeled in VG
     'teddy bear': 'teddy.n.01',
     'tv': 'television.n.01',
     'wine glass': 'wineglass.n.01'})

# Make sure all COCO classes are mapped to VG
assert len(name_map_coco) == len(coco_cls_names)

# ---------------------------------------------------------------------------- #
# Build a joint category list of VG+COCO (put 80 COCO classes at beginning)
# ---------------------------------------------------------------------------- #

# Keep the 80 COCO classes at the beginning of the list for easy weight slicing
joint_cls_names_coco = [name_map_coco[n] for n in coco_cls_names]
joint_cls_names_vg3k_minus_coco = sorted(
    set(vg3k_cls_names) - set(joint_cls_names_coco))
joint_cls_names = joint_cls_names_coco + joint_cls_names_vg3k_minus_coco

categories_joint = [
    {'id': n_c + 1, 'name': name} for n_c, name in enumerate(joint_cls_names)]
joint_cls_name2id = {c['name']: c['id'] for c in categories_joint}
# Append '.coco' to COCO classes (to distinguish them)
for c in categories_joint:
    if c['name'] in name_map_coco.values():
        c['name'] += '.coco'

# Build a map from old VG class ids to new class ids
cid_vg3k2joint = {
    c['id']: joint_cls_name2id[c['name']] for c in dataset_vg3k['categories']}

# ---------------------------------------------------------------------------- #
# Align VG3k dataset with COCO (map old class ids to new ones)
# ---------------------------------------------------------------------------- #

for ann in dataset_vg3k['annotations']:
    ann['category_id'] = cid_vg3k2joint[ann['category_id']]
dataset_vg3k['categories'] = categories_joint

# ---------------------------------------------------------------------------- #
# Split into train, val and test (where val & test are not in COCO trainval35k)
# ---------------------------------------------------------------------------- #

# Split: val (those in COCO minival)
images_val = [
    img for img in dataset_vg3k['images']
    if img['coco_id'] in coco_minival_imgids]
imgids_val = {img['id'] for img in images_val}

# Take 5000 images (not in COCO) for test
images_traintest = [
    img for img in dataset_vg3k['images']
    if img['coco_id'] not in coco_minival_imgids]
# Split: test
images_test = [
    img for img in images_traintest
    if img['coco_id'] is None][-5000:]
imgids_test = {img['id'] for img in images_test}
# Split: train
images_train = [
    img for img in images_traintest
    if img['id'] not in imgids_test]
imgids_train = {img['id'] for img in images_train}
print('number of train, val, test images: {}, {}, {}'.format(
    len(images_train), len(images_val), len(images_test)))

# Save the dataset splits
annotations_train = [
    ann for ann in dataset_vg3k['annotations']
    if ann['image_id'] in imgids_train]
annotations_val = [
    ann for ann in dataset_vg3k['annotations']
    if ann['image_id'] in imgids_val]
annotations_test = [
    ann for ann in dataset_vg3k['annotations']
    if ann['image_id'] in imgids_test]

# ---------------------------------------------------------------------------- #
# Save to json file
# ---------------------------------------------------------------------------- #

dataset_vg3k_train = {
    'images': images_train,
    'annotations': annotations_train,
    'categories': categories_joint}
dataset_vg3k_val = {
    'images': images_val,
    'annotations': annotations_val,
    'categories': categories_joint}
dataset_vg3k_test = {
    'images': images_test,
    'annotations': annotations_test,
    'categories': categories_joint}

with open(output_dir + 'instances_vg3k_cocoaligned_train.json', 'w') as f:
    json.dump(dataset_vg3k_train, f)
with open(output_dir + 'instances_vg3k_cocoaligned_val.json', 'w') as f:
    json.dump(dataset_vg3k_val, f)
with open(output_dir + 'instances_vg3k_cocoaligned_test.json', 'w') as f:
    json.dump(dataset_vg3k_test, f)
