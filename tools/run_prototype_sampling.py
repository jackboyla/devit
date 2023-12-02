import torch
from pprint import pprint
import numpy as np
import os.path as osp
import fire
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from detectron2.data import MetadataCatalog, DatasetCatalog
from collections import defaultdict, Counter

import lib.data.fewshot
import lib.data.ovdshot
import lib.data.lvis
from lib.prototype_learner import PrototypeLearner

"""
dataset = {
    'labels': [],
    'class_tokens': [],
    'avg_patch_tokens': [], 
    'image_id': [],
    'boxes': [],
    'areas': [],
    'skip': 0
}
"""

def accuracy_score(match):
    return (match.sum() / len(match)).item()


def main(inp, 
        num_prototypes=10,
        token_type='pat', # pat, cls
        momentum=0.002, 
        epochs=30,
        batch_size=512,
        queue_size=8192,
        normalize='yes',
        device=0,
        save='yes',
        save_tokens='no',
        oneshot_sample_pool=30
        ):
    oneshot = 'novel_oneshot_' in inp
    kwargs = locals()
    pprint(kwargs)
    if device != 'cpu': device = int(device)
    print('load and preprocess dataset')
    dataset = torch.load(inp)
    dname = osp.basename(inp).split('.')[0]

    prototypes = 

    if 'label_names' in dataset:
        thing_classes = dataset['label_names']
    else:
        DatasetCatalog.get(dname)  # just to register some metas
        meta = MetadataCatalog.get(dname)
        thing_classes = meta.thing_classes

    labels = torch.as_tensor(dataset['labels'])
    num_classes = len(torch.unique(labels))

    dct = {'prototypes': prototypes.values.cpu().view(num_classes, num_prototypes, -1),}

    # for one-shot, `label_names` will have duplicate values, and will need to be
    # dedup when loading
    dct['label_names'] = [thing_classes[ci] for ci in range(num_classes)]
    print(f'`label_names` are set to {dct["label_names"]}')


    out_p = 'sampled_prototypes'
    torch.save(dct, out_p)

if __name__ == "__main__":
    main()