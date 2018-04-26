import yaml
import sys
from ast import literal_eval


def _split_str(old_str):
    return str(tuple(old_str.split(':')))


def upgrade_cfg_file(file_path):
    with open(file_path, 'r') as f:
        cfg = yaml.load(f)

    if 'MODEL' in cfg:
        model_roi_head = cfg['MODEL'].pop('ROI_HEAD', None)
        if model_roi_head:
            cfg['FAST_RCNN']['ROI_BOX_HEAD'] = model_roi_head

    if 'MRCNN' in cfg:
        mrcnn_mask_head_name = cfg['MRCNN'].pop('MASK_HEAD_NAME', None)
        if mrcnn_mask_head_name:
            cfg['MRCNN']['ROI_MASK_HEAD'] = mrcnn_mask_head_name

    if 'TRAIN' in cfg:
        train_dataset = cfg['TRAIN'].pop('DATASET', None)
        if train_dataset:
            cfg['TRAIN']['DATASETS'] = _split_str(train_dataset)

        train_proposal_file = cfg['TRAIN'].pop('PROPOSAL_FILE', None)
        if train_proposal_file:
            cfg['TRAIN']['PROPOSAL_FILES'] = _split_str(train_proposal_file)

    if 'TEST' in cfg:
        test_scales = cfg['TEST'].pop('SCALES', None)
        if test_scales:
            test_scales = literal_eval(test_scales)
            assert isinstance(test_scales, tuple) and len(test_scales) == 1
            assert isinstance(test_scales[0], int)
            cfg['TEST']['SCALE'] = test_scales[0]

        test_dataset = cfg['TEST'].pop('DATASET', None)
        if test_dataset:
            cfg['TEST']['DATASETS'] = _split_str(test_dataset)

        test_proposal_file = cfg['TEST'].pop('PROPOSAL_FILE', None)
        if test_proposal_file:
            cfg['TEST']['PROPOSAL_FILES'] = _split_str(test_proposal_file)

    with open(file_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    upgrade_cfg_file(sys.argv[1])
