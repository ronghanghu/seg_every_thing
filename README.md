# Learning to Segment Every Thing

This repository contains the code for the following paper:

* R. Hu, P. Dollár, K. He, T. Darrell, R. Girshick, *Learning to Segment Every Thing*. in CVPR, 2018. ([PDF](https://arxiv.org/pdf/1711.10370.pdf))
```
@inproceedings{hu2018learning,
  title={Learning to Segment Every Thing},
  author={Hu, Ronghang and Dollár, Piotr and He, Kaiming and Darrell, Trevor and Girshick, Ross},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

Project Page: http://ronghanghu.com/seg_every_thing

Note: this repository is built upon the Detectron codebase for object detection and segmentation (https://github.com/facebookresearch/Detectron), based on Detectron commit [3c4c7f67d37eeb4ab15a87034003980a1d259c94](https://github.com/facebookresearch/Detectron/commit/3c4c7f67d37eeb4ab15a87034003980a1d259c94). Please see [`README_DETECTRON.md`](README_DETECTRON.md) for details.

## Installation

The installation procedure follows Detectron.

Please find installation instructions for Caffe2 and Detectron in [`INSTALL.md`](INSTALL.md).

**Note: all the experiments below run on 8 GPUs on a single machine.** If you have less than 8 GPU available, please modify the yaml config files according to the [linear scaling rule](https://arxiv.org/pdf/1706.02677). For example, if you only have 4 GPUs, set `NUM_GPUS` to 4, downscale `SOLVER.BASE_LR` by 0.5x and multiply `SOLVER.STEPS` and `SOLVER.MAX_ITER` by 2x.

## Part 1: Controlled Experiments on the COCO dataset

In this work, we explore our approach in two settings. First, we use the COCO dataset to simulate the partially supervised instance segmentation task as a means of establishing quantitative results on a dataset with high-quality annotations and evaluation metrics. Specifically, we split the full set of COCO categories into a subset with mask annotations and a complementary subset for which the system has access to only bounding box annotations. Because the COCO dataset involves only a small number (80) of semantically well-separated classes, quantitative evaluation is precise and reliable.

In our experiments, we split COCO into either  
* VOC Split: 20 PASCAL-VOC classes v.s. 60 non-PASCAL-VOC classes. We experiment with 1) *VOC -> non-VOC*, where set A={VOC} and 2) *non-VOC -> VOC*, where set A={non-VOC}.  
* Random Splits: randomly partitioned two subsets A and B of the 80 COCO classes.  

and experiment with two training setups:
* Stage-wise training, where first a Faster R-CNN detector is trained and kept frozen, and then the mask branch (including the weight transfer function) is added later.
* End-to-end training, where the RPN, the box head, the mask head and the weight transfer function are trained together.

Please refer to Section 4 of [our paper](https://arxiv.org/pdf/1711.10370.pdf) for details on the COCO experiments.

**COCO Installation**: To run the COCO experiments, first download the COCO dataset and install it according to the [dataset guide](lib/datasets/data/README.md).

### Evaluation

The following experiments correspond to the results in Section 4.2 and Table 2 of [our paper](https://arxiv.org/pdf/1711.10370.pdf).

To run the experiments:
1. Split the COCO dataset into VOC / non-VOC classes:  
`python2 lib/datasets/bbox2mask_dataset_processing/coco/split_coco_dataset_voc_nonvoc.py`.  
2. Set the training split using `SPLIT` variable:  
  * To train on *VOC -> non-VOC*, where set A={VOC}, use `export SPLIT=voc2nonvoc`.
  * To train on *non-VOC -> VOC*, where set A={non-VOC}, use `export SPLIT=nonvoc2voc`.

Then use `tools/train_net.py` to run the following yaml config files for each experiment with ResNet-50-FPN backbone or ResNet-101-FPN backbone.

Please follow the instruction in [GETTING_STARTED.md](GETTING_STARTED.md) to train with the config files. The training scripts automatically test the trained models and print the bbox and mask APs on the VOC ('coco_split_voc_2014_minival') and non-VOC splits ('coco_split_nonvoc_2014_minival').

Using ResNet-50-FPN backbone:  
1. Class-agnostic (baseline): `configs/bbox2mask_coco/${SPLIT}/eval_e2e/e2e_baseline.yaml`  
2. **MaskX R-CNN (ours, tansfer+MLP)**: `configs/bbox2mask_coco/${SPLIT}/eval_e2e/e2e_clsbox_2_layer_mlp_nograd.yaml`  
3. Fully-supervised (oracle): `configs/bbox2mask_coco/oracle/e2e_mask_rcnn_R-50-FPN_1x.yaml`  

Using ResNet-101-FPN backbone:  
1. Class-agnostic (baseline): `configs/bbox2mask_coco/${SPLIT}/eval_e2e_R101/e2e_baseline.yaml`  
2. **MaskX R-CNN (ours, tansfer+MLP)**: `configs/bbox2mask_coco/${SPLIT}/eval_e2e_R101/e2e_clsbox_2_layer_mlp_nograd.yaml`  
3. Fully-supervised (oracle): `configs/bbox2mask_coco/oracle/e2e_mask_rcnn_R-101-FPN_1x.yaml`  

### Ablation Study

This section runs ablation studies on the VOC Split (20 PASCAL-VOC classes v.s. 60 non-PASCAL-VOC classes) using ResNet-50-FPN backbone. The results correspond to Section 4.1 and Table 1 of [our paper](https://arxiv.org/pdf/1711.10370.pdf).

To run the experiments:
1. (If you haven't done so in the above section) Split the COCO dataset into VOC / non-VOC classes:  
`python2 lib/datasets/bbox2mask_dataset_processing/coco/split_coco_dataset_voc_nonvoc.py`.  
2. For Study 1, 2, 3 and 5, download the pre-trained Faster R-CNN model with ResNet-50-FPN by running  
`bash lib/datasets/data/trained_models/fetch_coco_faster_rcnn_model.sh`.  
(Alternatively, you can train it yourself using `configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml` and copy it to `lib/datasets/data/trained_models/28594643_model_final.pkl`.)  
3. For Study 1, add the GloVe and random embeddings of the COCO class names to the Faster R-CNN weights with  
`python2 lib/datasets/bbox2mask_dataset_processing/coco/add_embeddings_to_weights.py`.  
4. Set the training split using `SPLIT` variable:  
  * To train on *VOC -> non-VOC*, where set A={VOC}, use `export SPLIT=voc2nonvoc`.  
  * To train on *non-VOC -> VOC*, where set A={non-VOC}, use `export SPLIT=nonvoc2voc`.  

Then use `tools/train_net.py` to run the following yaml config files for each experiment.

#### Study 1: Ablation on the input to the weight transfer function (Table 1a)  
* transfer w/ randn: `configs/bbox2mask_coco/${SPLIT}/ablation_input/randn_2_layer.yaml`  
* transfer w/ GloVe: `configs/bbox2mask_coco/${SPLIT}/ablation_input/glove_2_layer.yaml`  
* transfer w/ cls: `configs/bbox2mask_coco/${SPLIT}/ablation_input/cls_2_layer.yaml`  
* transfer w/ box: `configs/bbox2mask_coco/${SPLIT}/ablation_input/box_2_layer.yaml`  
* **transfer w/ cls+box**: `configs/bbox2mask_coco/${SPLIT}/eval_sw/clsbox_2_layer.yaml`  
* *class-agnostic (baseline)*: `configs/bbox2mask_coco/${SPLIT}/eval_sw/baseline.yaml`  
* *fully supervised (oracle)*: `configs/bbox2mask_coco/oracle/mask_rcnn_frozen_features_R-50-FPN_1x.yaml`  

#### Study 2: Ablation on the structure of the weight transfer function (Table 1b)  
* transfer w/ 1-layer, none: `configs/bbox2mask_coco/${SPLIT}/ablation_structure/clsbox_1_layer.yaml`  
* transfer w/ 2-layer, ReLU: `configs/bbox2mask_coco/${SPLIT}/ablation_structure/relu/clsbox_2_layer_relu.yaml`  
* **transfer w/ 2-layer, LeakyReLU**: same as 'transfer w/ cls+box' in Study 1  
* transfer w/ 3-layer, ReLU: `configs/bbox2mask_coco/${SPLIT}/ablation_structure/relu/clsbox_3_layer_relu.yaml`  
* transfer w/ 3-layer, LeakyReLU: `configs/bbox2mask_coco/${SPLIT}/ablation_structure/clsbox_3_layer.yaml`  

#### Study 3: Impact of the MLP mask branch (Table 1c)  
* class-agnostic: same as 'class-agnostic (baseline)' in Study 1  
* class-agnostic+MLP: `configs/bbox2mask_coco/${SPLIT}/ablation_mlp/baseline_mlp.yaml`  
* transfer: same as 'transfer w/ cls+box' in Study 1  
* **transfer+MLP**: `configs/bbox2mask_coco/${SPLIT}/ablation_mlp/clsbox_2_layer_mlp.yaml`  

#### Study 4: Ablation on the training strategy (Table 1d)  
* class-agnostic + sw: same as 'class-agnostic (baseline)' in Study 1  
* transfer + sw: same as 'transfer w/ cls+box' in Study 1  
* class-agnostic + e2e: `configs/bbox2mask_coco/${SPLIT}/eval_e2e/e2e_baseline.yaml`  
* transfer + e2e: `configs/bbox2mask_coco/${SPLIT}/ablation_e2e_stopgrad/e2e_clsbox_2_layer.yaml`  
* **transfer + e2e + stopgrad**: `configs/bbox2mask_coco/${SPLIT}/ablation_e2e_stopgrad/e2e_clsbox_2_layer_nograd.yaml`  

#### Study 5: Comparison of random A/B splits (Figure 3)

*Note: this ablation study takes a HUGE amount of computation power. It consists of 50 training experiments (= 5 trials * 5 class-number in set A (20/30/40/50/60) * 2 settings (ours/baseline) ), and each training experiment takes approximately 9 hours to complete on 8 GPUs.*

Before running Study 5:
1. Split the COCO dataset into random class splits (**This should take a while**):  
`python2 lib/datasets/bbox2mask_dataset_processing/coco/split_coco_dataset_randsplits.py`.  
2. Set the training split using `SPLIT` variable (e.g. `export SPLIT=E1_A20B60`). The split has the format `E%d_A%dB%d` for example, `E1_A20B60` is trial No. 1 with 20 random classes in set A and 60 random classes in set B. There are 5 trials (E1 to E5), with 20/30/40/50/60 random classes in set A (A20B60 to A60B20), yielding altogether 25 splits from `E1_A20B60` to `E5_A60B20`.

Then use `tools/train_net.py` to run the following yaml config files for each experiment.
* class-agnostic (baseline): `configs/bbox2mask_coco/randsplits/eval_sw/${SPLIT}_baseline.yaml`  
* tansfer w/ cls+box, 2-layer, LeakyReLU: `configs/bbox2mask_coco/randsplits/eval_sw/${SPLIT}_clsbox_2_layer.yaml`  

## Part 2: Large-scale Instance Segmentation on the Visual Genome dataset

In our second setting, we train a large-scale instance segmentation model on 3000 categories using the Visual Genome (VG) dataset. On the Visual Genome dataset, set A (w/ mask data) is the 80 COCO classes, while set B (w/o mask data, only bbox) is the remaining Visual Genome classes that are not in COCO.

Please refer to Section 5 of [our paper](https://arxiv.org/pdf/1711.10370.pdf) for details on the Visual Genome experiments.

### Inference

To run inference, download the pre-trained final model weights by running:  
`bash lib/datasets/data/trained_models/fetch_vg3k_final_model.sh`  
(Alternatively, you may train these weights yourself following the training section below.)

Then, use `tools/infer_simple.py` for prediction. *Note: due to the large number of classes and the model loading overhead, prediction on the first image can take a while.*

Using ResNet-50-FPN backbone:
```
python2 tools/infer_simple.py \
    --cfg configs/bbox2mask_vg/eval_sw/runtest_clsbox_2_layer_mlp_nograd.yaml \
    --output-dir /tmp/detectron-visualizations-vg3k \
    --image-ext jpg \
    --thresh 0.5 --use-vg3k \
    --wts lib/datasets/data/trained_models/33241332_model_final_coco2vg3k_seg.pkl \
    demo_vg3k
```

Using ResNet-101-FPN backbone:
```
python2 tools/infer_simple.py \
    --cfg configs/bbox2mask_vg/eval_sw_R101/runtest_clsbox_2_layer_mlp_nograd_R101.yaml \
    --output-dir /tmp/detectron-visualizations-vg3k-R101 \
    --image-ext jpg \
    --thresh 0.5 --use-vg3k \
    --wts lib/datasets/data/trained_models/33219850_model_final_coco2vg3k_seg.pkl \
    demo_vg3k
```

### Training

**Visual Genome Installation**: To run the Visual Genome experiments, first download the Visual Genome dataset and install it according to the [dataset guide](lib/datasets/data/README.md). Then download the converted Visual Genome json dataset files (in COCO-format) by running:  
`bash lib/datasets/data/vg3k_bbox2mask/fetch_vg3k_json.sh`.  
(Alternatively, you may build the COCO-format json dataset files yourself using the scripts in `lib/datasets/bbox2mask_dataset_processing/vg/`)

Here, we adopt the stage-wise training strategy as mentioned in Section 5 of [our paper](https://arxiv.org/pdf/1711.10370.pdf). First in Stage 1, a Faster R-CNN detector is trained on all the 3k Visual Genome classes (set A+B). Then in Stage 2, the mask branch (with the weight transfer function) is added and trained on the mask data of the 80 COCO classes (set A). Finally, the mask branch is applied on all 3k Visual Genome classes (set A+B).

Before training on the mask data of the 80 COCO classes (set A) in Stage 2, a "surgery" is done to convert the 3k VG detection weights to 80 COCO detection weights, so that the mask branch only predicts mask outputs of the 80 COCO classes (as the weight transfer function only takes as input 80 classes) to save GPU memory. After training, another "surgery" is done to convert the 80 COCO detection weights back to the 3k VG detection weights.

To run the experiments, use `tools/train_net.py` to run the following yaml config files for each experiment with ResNet-50-FPN backbone or ResNet-101-FPN backbone.

Using ResNet-50-FPN backbone:  
1. Stage 1 (bbox training on 3k VG classes): run `tools/train_net.py` with `configs/bbox2mask_vg/eval_sw/stage1_e2e_fast_rcnn_R-50-FPN_1x_1im.yaml`  
2. Weights "surgery" 1: convert 3k VG detection weights to 80 COCO detection weights:  
`python2 tools/vg3k_training/convert_coco_seg_to_vg3k.py --input_model /path/to/model_1.pkl --output_model /path/to/model_1_vg3k2coco_det.pkl`  
where `/path/to/model_1.pkl` is the path to the final model trained in Stage 1 above.
3. Stage 2 (mask training on 80 COCO classes): run `tools/train_net.py` with `configs/bbox2mask_vg/eval_sw/stage2_cocomask_clsbox_2_layer_mlp_nograd.yaml`  
**IMPORTANT**: when training Stage 2, set `TRAIN.WEIGHTS` to `/path/to/model_1_vg3k2coco_det.pkl` (the output of `convert_coco_seg_to_vg3k.py`) in `tools/train_net.py`.  
4. Weights "surgery" 2: convert 80 COCO detection weights back to 3k VG detection weights:  
`python2 tools/vg3k_training/convert_vg3k_det_to_coco.py --input_model /path/to/model_2.pkl --output_model /path/to/model_2_coco2vg3k_seg.pkl`  
where `/path/to/model_2.pkl` is the path to the final model trained in Stage 2 above. The output `/path/to/model_2_coco2vg3k_seg.pkl` can be used for VG 3k instance segmentation.

Using ResNet-101-FPN backbone:  
1. Stage 1 (bbox training on 3k VG classes): run `tools/train_net.py` with `configs/bbox2mask_vg/eval_sw_R101/stage1_e2e_fast_rcnn_R-101-FPN_1x_1im.yaml`  
2. Weights "surgery" 1: convert 3k VG detection weights to 80 COCO detection weights:  
`python2 tools/vg3k_training/convert_coco_seg_to_vg3k.py --input_model /path/to/model_1.pkl --output_model /path/to/model_1_vg3k2coco_det.pkl`  
where `/path/to/model_1.pkl` is the path to the final model trained in Stage 1 above.
3. Stage 2 (mask training on 80 COCO classes): run `tools/train_net.py` with `configs/bbox2mask_vg/eval_sw_R101/stage2_cocomask_clsbox_2_layer_mlp_nograd_R101.yaml`  
**IMPORTANT**: when training Stage 2, set `TRAIN.WEIGHTS` to `/path/to/model_1_vg3k2coco_det.pkl` (the output of `convert_coco_seg_to_vg3k.py`) in `tools/train_net.py`.  
4. Weights "surgery" 2: convert 80 COCO detection weights back to 3k VG detection weights:  
`python2 tools/vg3k_training/convert_vg3k_det_to_coco.py --input_model /path/to/model_2.pkl --output_model /path/to/model_2_coco2vg3k_seg.pkl`  
where `/path/to/model_2.pkl` is the path to the final model trained in Stage 2 above. The output `/path/to/model_2_coco2vg3k_seg.pkl` can be used for VG 3k instance segmentation.

(Alternatively, you may skip Stage 1 and Weights "surgery" 1 by directly downloading the pre-trained VG 3k detection weights by running `bash lib/datasets/data/trained_models/fetch_vg3k_faster_rcnn_model.sh`, and leaving `TRAIN.WEIGHTS` to the specified values in the yaml configs in Stage 2.)
