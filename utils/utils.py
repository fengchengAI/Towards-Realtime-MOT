import glob
import random
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.manual_seed_all(seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))



def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.0004 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def scale_coords(img_size, coords, img0_shape):
    """

    :param img_size:  cfg中固定的图像大小
    :param coords:  detection时通过nms后的box
    :param img0_shape:  仅仅通过缩放后的图像尺寸
    :return:
    """
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """Returns the IoU of two bounding boxes
    box1 是一个meshgrid和预设anchor，所有这个函数就是计算每个预设的anchor在gt_boxes上的iou
    :param box1: （nA x nGh x nGw） x 4,其中最后一个纬度的前两个是（2, nGh, nGw）的meshgrid重复的，后两个是由（nA x 2）预设anchor重复生成的
    :param box2: gt_boxes
    :param x1y1x2y2:是坐标的排序方式，如果False则为cx,cy,w,h，
    :return:  (nA x nGh x nGw) x M    (M是gt_boxes的数量)
    """

    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle

    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)  # shape (m, n)

    '''
    torch.max(b1_x1.unsqueeze(1), b2_x1)
    在numpy下类似于
    max(b1_x1.repeat(m),dim=1),b1_x1.repeat(n,dim=0))
    即将b1_x1作为一列重复n列，将b2_x1作为一行重复m行，
    两个重复的结果都是(m,n)维度的，然后进行求max
    '''
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    #b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """test_emb用
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nGh, nGw, 2)  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nGh, nGw, 2)
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0)  # nC = number of classes
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1)
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long()
        t = t[:,[0,2,3,4,5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        #gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG
        gxy, gwh = t[: , 1:3].clone() , t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gi = torch.clamp(gxy[:, 0], min=0, max=nGw -1).long()
        gj = torch.clamp(gxy[:, 1], min=0, max=nGh -1).long()

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        #gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
        #gi, gj = gxy.long().t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative
            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.60:
                continue
        
        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid



def build_targets_thres(target, anchor_wh, nA, nC, nGh, nGw):
    """训练用
    :param target: [class] [identity] [x_center] [y_center] [width] [height]
    :param anchor_wh:  [1, self.nA, 1, 1, 2] 是cfg中预设的，仅仅进行了缩放，（该anchor与当前yolo输入的尺寸大小比例与原始anchor大小在原图中的比例一致
    :param nA: 4
    :param nC: 1
    :param nGh: yolo输入层的大小
    :param nGw:
    :return:
    tconf, 预设anchor对应gt的标签，如： 0， 1 ， -1
    tbox, 预设anchor的前景box对应gt_box的偏差
    tid 预设anchor对应gt的id
    """
    ID_THRESH = 0.5
    FG_THRESH = 0.5  # 这是什么标志,应该是前景吧
    BG_THRESH = 0.4  # 背景
    nB = len(target)  # number of images in batch

    assert(len(anchor_wh)==nA)
    tbox = torch.zeros(nB, nA, nGh, nGw, 4)  # batch size, anchors, grid size
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0)
    tid = torch.LongTensor(nB, nA, nGh, nGw,1).fill_(-1)
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long()
        t = t[:,[0,2,3,4,5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw - 1)
        gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh - 1)

        gt_boxes = torch.cat([gxy, gwh], dim=1)
        '''
        gxy = [x,2], gwh = [x,2], gt_boxes = [x,4], x为当前图片的label的id数
        4: (xc, yc, w, h)
        '''

        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)  # 这个应该看看，体现算法思想  shape (nA x nGh x nGw） x 4 是预设的anchor
        anchor_list = anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        iou_pdist = bbox_iou(anchor_list, gt_boxes)                                      # Shape (nA x nGh x nGw) x M (M是gt_boxes的数量）
        iou_max, max_gt_index = torch.max(iou_pdist, dim=1)                              # Shape (nA x nGh x nGw), both
        #  找出每个预设anchor所匹配得最大iou的gt_box的值和索引

        iou_map = iou_max.view(nA, nGh, nGw)       
        gt_index_map = max_gt_index.view(nA, nGh, nGw)

        #nms_map = pooling_nms(iou_map, 3)
        
        id_index = iou_map > ID_THRESH
        fg_index = iou_map > FG_THRESH                                                    
        bg_index = iou_map < BG_THRESH 
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        tconf[b][fg_index] = 1  # 前景
        tconf[b][bg_index] = 0  # 背景
        tconf[b][ign_index] = -1  # 忽视

        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        TrueSum = len(gt_index_map[fg_index])  # TrueSum并没有实际意义只是方便下面注释

        '''
        gt_index_map和id_index的shape相同，id_index是布尔值，选择gt_index_map对应布尔值为1的值作为一个list，该list的值即gt_index_map是iou_max的索引
        通过该索引可以找到gt_id_list
        eg: 
        id_index.shape==（4,10,18）
        gt_index_map[id_index].shape==（TrueSum）
        t_id.shape==（28） 
        gt_id_list = t_id[gt_index_map[id_index]]==（TrueSum）
        说明： 在该图像中有28个box_id即t_id，有（4,10,18）个预设anchor，只有70个满足条件了即gt_index_map[id_index].shape=70，而这70个是gt_index_map的值，故可以通过这TrueSum个gt_index_map
        来得到70个gt_id_list
        即 gt_box_list，gt_id_list 表示满足条件的gt_box及其对应的id和box
        '''

        #print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
        if torch.sum(fg_index) > 0:
            # tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1)
            tid[b][id_index] = gt_id_list.unsqueeze(1)  # tid对应的是(nB, nA, nGh, nGw）个预设anchor所对应的最匹配的前景gt_id
            fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index] 
            delta_target = encode_delta(gt_box_list, fg_anchor_list)  # gt_box_list 和 fg_anchor_list 的shape一致(TrueSum, 4)

            tbox[b][fg_index] = delta_target
    return tconf, tbox, tid

def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    #xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw) # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    # mesh主要形成anchor的cx和cy，anchor_offset_mesh形成anchor的w和h，可以看这个函数的下面两个语句
    # anchor_list = anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)           # Shpae (nA x nGh x nGw) x 4
    # 可以理解到anchor_list的物理意义，即在nGh x nGw大小的像素点中，每个像素点有nA个预设size的anchor，每个anchor有四个左边，故shape为(nA x nGh x nGw x） 4
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    """gt_box和anchor的偏差
    :param gt_box_list: 满足前景阈值的预设anchor对应的gt_box
    :param fg_anchor_list: 满足前景阈值的预设anchor
    :return:
    """
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    '''
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors) 
    anchor_mesh = anchor_mesh.permute(0,2,3,1).contiguous()              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB,1,1,1,1)
    pred_list = decode_delta(delta_map.view(-1,4), anchor_mesh.view(-1,4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map


def pooling_nms(heatmap, kernel=1):
    pad = (kernel -1 ) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return keep * heatmap

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, method='standard'):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]  # nB 个None的list
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        
        # Non-maximum suppression
        if method == 'standard':
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == 'fast':
            nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres)
        else:
            raise ValueError('Invalid NMS type!')
        det_max = pred[nms_indices]        

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output

def fast_nms(boxes, scores, iou_thres:float=0.5, top_k:int=200, second_threshold:bool=False, conf_thres:float=0.5,
             self=None):
    '''
    Vectorized, approximated, fast NMS, adopted from YOLACT:
    https://github.com/dbolya/yolact/blob/master/layers/functions/detection.py
    The original version is for multi-class NMS, here we simplify the code for single-class NMS
    '''
    scores, idx = scores.sort(0, descending=True)
    
    idx = idx[:top_k].contiguous()
    scores = scores[:top_k]
    num_dets = idx.size()

    boxes = boxes[idx, :]

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=0)

    keep = (iou_max <= iou_thres)

    if second_threshold:
        keep *= (scores > self.conf_thresh)

    return idx[keep]



@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]



def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)




def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)

    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def plot_results():
    # Plot YOLO training results file 'results.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v1.txt')

    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'mAP', 'Recall', 'Precision']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T  # column 11 is mAP
        x = range(1, results.shape[1])
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.plot(x, results[i, x], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()
