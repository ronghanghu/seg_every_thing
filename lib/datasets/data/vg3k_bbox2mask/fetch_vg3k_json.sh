#!/bin/bash
wget -O ./lib/datasets/data/vg3k_bbox2mask/instances_vg3k_cocoaligned_train.json \
  https://people.eecs.berkeley.edu/~ronghang/projects/seg_every_thing/data/vg3k_bbox2mask/instances_vg3k_cocoaligned_train.json
wget -O ./lib/datasets/data/vg3k_bbox2mask/instances_vg3k_cocoaligned_val.json \
  https://people.eecs.berkeley.edu/~ronghang/projects/seg_every_thing/data/vg3k_bbox2mask/instances_vg3k_cocoaligned_val.json
wget -O ./lib/datasets/data/vg3k_bbox2mask/instances_vg3k_cocoaligned_test.json \
  https://people.eecs.berkeley.edu/~ronghang/projects/seg_every_thing/data/vg3k_bbox2mask/instances_vg3k_cocoaligned_test.json
