from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', default='')
    parser.add_argument('--output_model', default='')
    args = parser.parse_args()
    return args


def main(input_model, output_model):
    assert '.pkl' in input_model
    with open(input_model, 'rb') as f:
        weights = pickle.load(f)
    assert weights['blobs']['cls_score_w'].shape == (81, 1024)
    assert weights['blobs']['cls_score_b'].shape == (81,)
    assert weights['blobs']['bbox_pred_w'].shape == (81 * 4, 1024)
    assert weights['blobs']['bbox_pred_b'].shape == (81 * 4,)
    weights['blobs']['cls_score_w'] = weights['blobs']['cls_score_w_jcvbackup'].copy()
    weights['blobs']['cls_score_b'] = weights['blobs']['cls_score_b_jcvbackup'].copy()
    weights['blobs']['bbox_pred_w'] = weights['blobs']['bbox_pred_w_jcvbackup'].copy()
    weights['blobs']['bbox_pred_b'] = weights['blobs']['bbox_pred_b_jcvbackup'].copy()
    with open(output_model, 'wb') as f:
        pickle.dump(weights, f)


if __name__ == '__main__':
    args = parse_args()
    main(args.input_model, args.output_model)
