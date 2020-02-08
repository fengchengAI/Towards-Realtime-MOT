import os.path as osp
import numpy as np
import os
from glob import glob


tid_curr = 0
tid_last = -1
root_path = "/home/feng/Github/PycharmProjects/data/MOT16/train"
sequence_paths = glob(os.path.join(root_path, '*'))
sequence_paths = sorted(sequence_paths, key = lambda x: x.split('/')[-1])

for sequence_path in sequence_paths:

    seq_info = open(osp.join(sequence_path, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    image_path = glob(os.path.join(sequence_path, 'img1', '*'))
    image_path = sorted(image_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    seq_label_root = osp.join(sequence_path,"label")
    if not osp.exists(seq_label_root):
        os.makedirs(seq_label_root)

    label_path = [x.replace('.jpg', '.txt').replace('img1','labels_with_ids') for x in image_path]

    gt_txt = osp.join(sequence_path, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:  # label == 1是行人,此项目只对行人进行多目标跟踪
            continue  # 多此一举吧
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

gen_data_cfg = "/home/feng/Github/PycharmProjects/Towards-Realtime-MOT/data/mot16.train"

img_paths = glob(os.path.join(root_path, '*',"img1","*"))
img_paths = sorted(img_paths, key=lambda x: (x.split('/')[-3],int(x.split('/')[-1].split('.')[0])))

with open(gen_data_cfg, 'w') as f:
    for i in img_paths:
        f.write(i+'\n')

