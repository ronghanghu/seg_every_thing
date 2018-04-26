# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg
from utils.c2 import const_fill
from utils.c2 import gauss_fill
from utils.net import get_group_gn
import modeling.ResNet as ResNet
import utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #


def concat_cls_score_bbox_pred(model):
    # flatten 'bbox_pred_w'
    # bbox_pred_w has shape (324, 1024), where 324 is (81, 4) memory structure
    # reshape to (81, 4 * 1024)
    model.net.Reshape(
        'bbox_pred_w', ['bbox_pred_w_flat', '_bbox_pred_w_oldshape'],
        shape=(model.num_classes, -1))
    cls_score_bbox_pred, _ = model.net.Concat(
        ['cls_score_w', 'bbox_pred_w_flat'],
        ['cls_score_bbox_pred', '_cls_score_bbox_pred_split_info'], axis=1)
    return cls_score_bbox_pred


def bbox2mask_weight_transfer(model, class_embed, dim_in, dim_h, dim_out):
    bbox2mask_type = str(cfg.MRCNN.BBOX2MASK.TYPE)

    def _mlp_activation(model, inputs, outputs):
        if cfg.MRCNN.BBOX2MASK.USE_LEAKYRELU:
            model.net.LeakyRelu(inputs, outputs, alpha=0.1)
        else:
            model.net.Relu(inputs, outputs)

    if (not bbox2mask_type) or bbox2mask_type == '1_layer':
        mask_w_flat = model.FC(
            class_embed, 'mask_fcn_logits_w_flat', dim_in, dim_out,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif bbox2mask_type == '2_layer':
        mlp_l1 = model.FC(
            class_embed, 'bbox2mask_mlp_l1', dim_in, dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        _mlp_activation(model, mlp_l1, mlp_l1)
        mask_w_flat = model.FC(
            mlp_l1, 'mask_fcn_logits_w_flat', dim_h, dim_out,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif bbox2mask_type == '3_layer':
        mlp_l1 = model.FC(
            class_embed, 'bbox2mask_mlp_l1', dim_in, dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        _mlp_activation(model, mlp_l1, mlp_l1)
        mlp_l2 = model.FC(
            mlp_l1, 'bbox2mask_mlp_l2', dim_h, dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        _mlp_activation(model, mlp_l2, mlp_l2)
        mask_w_flat = model.FC(
            mlp_l2, 'mask_fcn_logits_w_flat', dim_h, dim_out,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
    else:
        raise ValueError('unknown bbox2mask_type {}'.format(bbox2mask_type))

    # mask_w has shape (num_cls, dim_out, 1, 1)
    mask_w = model.net.ExpandDims(
        mask_w_flat, 'mask_fcn_logits_w', dims=[2, 3])
    return mask_w


def cls_agnostic_mlp_branch(model, blob_in, dim_in, num_cls, dim_h=1024):
    fc_mask_head_type = str(cfg.MRCNN.MLP_MASK_BRANCH_TYPE)
    dim_out = 1 * cfg.MRCNN.RESOLUTION**2

    if (not fc_mask_head_type) or fc_mask_head_type == '1_layer':
        raw_mlp_branch = model.FC(
            blob_in, 'mask_mlp_logits_raw', dim_in, dim_out,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif fc_mask_head_type == '2_layer':
        mlp_l1 = model.FC(
            blob_in, 'fc_mask_head_mlp_l1', dim_in, dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Relu(mlp_l1, mlp_l1)
        raw_mlp_branch = model.FC(
            mlp_l1, 'mask_mlp_logits_raw', dim_h, dim_out,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    elif fc_mask_head_type == '3_layer':
        mlp_l1 = model.FC(
            blob_in, 'fc_mask_head_mlp_l1', dim_in, dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Relu(mlp_l1, mlp_l1)
        mlp_l2 = model.FC(
            mlp_l1, 'fc_mask_head_mlp_l2', dim_h, dim_h,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.net.Relu(mlp_l2, mlp_l2)
        raw_mlp_branch = model.FC(
            mlp_l2, 'mask_mlp_logits_raw', dim_h, dim_out,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    else:
        raise ValueError('unknown fc_mask_head_type {}'.format(fc_mask_head_type))

    mlp_branch, _ = model.net.Reshape(
        raw_mlp_branch,
        ['mask_mlp_logits_reshaped', '_mask_mlp_logits_raw_old_shape'],
        shape=(-1, 1, cfg.MRCNN.RESOLUTION, cfg.MRCNN.RESOLUTION))
    if num_cls > 1:
        mlp_branch = model.net.Tile(
            mlp_branch, 'mask_mlp_logits_tiled', tiles=num_cls, axis=1)

    return mlp_branch


def add_mask_rcnn_outputs(model, blob_in, dim):
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1

    if cfg.MRCNN.BBOX2MASK.BBOX2MASK_ON:
        # Use weight transfer function iff BBOX2MASK_ON is True
        # Decide the input to the of weight transfer function
        #   - Case 1) From a pre-trained embedding vector (e.g. GloVe)
        #   - Case 2) From the detection weights in the box head
        if cfg.MRCNN.BBOX2MASK.USE_PRETRAINED_EMBED:
            # Case 1) From a pre-trained embedding vector (e.g. GloVe)
            class_embed = cfg.MRCNN.BBOX2MASK.PRETRAINED_EMBED_NAME
            class_embed_dim = cfg.MRCNN.BBOX2MASK.PRETRAINED_EMBED_DIM
            # This parameter is meant to be initialized from a pretrained model
            # instead of learned from scratch. Hence, the default init is HUGE
            # to cause NaN loss so that the error will not pass silently.
            model.AddParameter(model.param_init_net.GaussianFill(
                [], class_embed, shape=[num_cls, class_embed_dim], std=1e12))
            # Pretrained embedding should be fixed during training (it doesn't
            # make sense to update them)
            model.StopGradient(class_embed, class_embed + '_no_grad')
            class_embed = class_embed + '_no_grad'
        else:
            # Case 2) From the detection weights in the box head
            #   - Subcase a) using cls+box
            #   - Subcase b) using cls
            #   - Subcase c) using box
            # where 'cls' is RoI classification weights 'cls_score_w'
            # and 'box' is bounding box regression weights 'bbox_pred_w'
            if (cfg.MRCNN.BBOX2MASK.INCLUDE_CLS_SCORE and
                    cfg.MRCNN.BBOX2MASK.INCLUDE_BBOX_PRED):
                # Subcase a) using cls+box
                concat_cls_score_bbox_pred(model)
                class_embed = 'cls_score_bbox_pred'
                class_embed_dim = 1024 + 4096
            elif cfg.MRCNN.BBOX2MASK.INCLUDE_CLS_SCORE:
                # Subcase b) using cls
                class_embed = 'cls_score_w'
                class_embed_dim = 1024
            elif cfg.MRCNN.BBOX2MASK.INCLUDE_BBOX_PRED:
                # Subcase c) using box; 'bbox_pred_w' need to be flattened
                model.net.Reshape(
                    'bbox_pred_w', ['bbox_pred_w_flat', '_bbox_pred_w_oldshape'],
                    shape=(model.num_classes, -1))
                class_embed = 'bbox_pred_w_flat'
                class_embed_dim = 4096
            else:
                raise ValueError(
                    'At least one of cfg.MRCNN.BBOX2MASK.INCLUDE_CLS_SCORE and '
                    'cfg.MRCNN.BBOX2MASK.INCLUDE_BBOX_PRED needs to be True')
            # Stop the mask gradient to the detection weights if specified
            if cfg.MRCNN.BBOX2MASK.STOP_DET_W_GRAD:
                model.StopGradient(class_embed, class_embed + '_no_grad')
                class_embed = class_embed + '_no_grad'

        # Use weights transfer function to predict mask weights
        mask_w = bbox2mask_weight_transfer(
            model, class_embed, dim_in=class_embed_dim, dim_h=dim, dim_out=dim)
        # Mask prediction with predicted mask weights (no bias term)
        fcn_branch = model.net.Conv(
            [blob_in, mask_w], 'mask_fcn_logits', kernel=1, pad=0, stride=1)
    else:
        # Not using weights transfer function
        if cfg.MRCNN.USE_FC_OUTPUT:
            assert not cfg.MRCNN.JOINT_FCN_MLP_HEAD
            blob_out = model.FC(
                blob_in, 'mask_fcn_logits', dim,
                num_cls * cfg.MRCNN.RESOLUTION**2,
                weight_init=('GaussianFill', {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:
            # If using class-agnostic mask, scale down init to avoid NaN loss
            init_filler = (
                cfg.MRCNN.CONV_INIT if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill')
            fcn_branch = model.Conv(
                blob_in, 'mask_fcn_logits', dim, num_cls, 1, pad=0, stride=1,
                weight_init=(init_filler, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.}))

    # Add a complementary MLP branch if specified
    if cfg.MRCNN.JOINT_FCN_MLP_HEAD:
        # Use class-agnostic MLP branch, and class-aware FCN branch
        mlp_branch = cls_agnostic_mlp_branch(
            model, blob_in, dim_in=dim * cfg.MRCNN.RESOLUTION**2, num_cls=num_cls)
        blob_out = model.net.Add([mlp_branch, fcn_branch], 'mask_logits')
    elif not cfg.MRCNN.USE_FC_OUTPUT:
        blob_out = fcn_branch

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out


# def add_mask_rcnn_outputs(model, blob_in, dim):
#     """Add Mask R-CNN specific outputs: either mask logits or probs."""
#     num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1
#
#     if cfg.MRCNN.USE_FC_OUTPUT:
#         # Predict masks with a fully connected layer (ignore 'fcn' in the blob
#         # name)
#         blob_out = model.FC(
#             blob_in,
#             'mask_fcn_logits',
#             dim,
#             num_cls * cfg.MRCNN.RESOLUTION**2,
#             weight_init=gauss_fill(0.001),
#             bias_init=const_fill(0.0)
#         )
#     else:
#         # Predict mask using Conv
#
#         # Use GaussianFill for class-agnostic mask prediction; fills based on
#         # fan-in can be too large in this case and cause divergence
#         fill = (
#             cfg.MRCNN.CONV_INIT
#             if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
#         )
#         blob_out = model.Conv(
#             blob_in,
#             'mask_fcn_logits',
#             dim,
#             num_cls,
#             kernel=1,
#             pad=0,
#             stride=1,
#             weight_init=(fill, {'std': 0.001}),
#             bias_init=const_fill(0.0)
#         )
#
#         if cfg.MRCNN.UPSAMPLE_RATIO > 1:
#             blob_out = model.BilinearInterpolation(
#                 'mask_fcn_logits', 'mask_fcn_logits_up', num_cls, num_cls,
#                 cfg.MRCNN.UPSAMPLE_RATIO
#             )
#
#     if not model.train:  # == if test
#         blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')
#
#     return blob_out


def add_mask_rcnn_losses(model, blob_mask):
    """Add Mask R-CNN specific losses."""
    loss_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_mask, 'masks_int32'],
        'loss_mask',
        scale=model.GetLossScale() * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_mask])
    model.AddLosses('loss_mask')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #

def mask_rcnn_fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up4convs_gn(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return mask_rcnn_fcn_head_v1upXconvs_gn(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 2
    )


def mask_rcnn_fcn_head_v1upXconvs(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        current = model.Conv(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v1upXconvs_gn(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_mask_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        current = model.ConvGN(
            current,
            '_mask_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            group_gn=get_group_gn(dim_inner),
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v0upshare(model, blob_in, dim_in, spatial_scale):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    # Since box and mask head are shared, these must match
    assert cfg.MRCNN.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(
            ['res5_2_sum', 'roi_has_mask_int32'],
            ['_[mask]_res5_2_sum_sliced']
        )
    else:  # re-compute at test time
        blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
            model,
            blob_in,
            dim_in,
            spatial_scale
        )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=const_fill(0.0)
    )
    model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def mask_rcnn_fcn_head_v0up(model, blob_in, dim_in, spatial_scale):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
        model,
        blob_in,
        dim_in,
        spatial_scale
    )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def add_ResNet_roi_conv5_head_for_masks(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for predicting masks."""
    model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_pool5',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    stride_init = int(cfg.MRCNN.ROI_XFORM_RESOLUTION / 7)  # by default: 2

    s, dim_in = ResNet.add_stage(
        model,
        '_[mask]_res5',
        '_[mask]_pool5',
        3,
        dim_in,
        2048,
        512,
        dilation,
        stride_init=stride_init
    )

    return s, 2048
