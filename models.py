from collections import OrderedDict
import torch.nn as nn
from utils.parse_config import *
from utils.utils import *
import math

try:
    from utils.syncbn import SyncBN
    batch_norm=SyncBN #nn.BatchNorm2d
except ImportError:
    batch_norm=nn.BatchNorm2d

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    这个函数只是构造模块,即构造每层网络的nn.Sequential()，并没有操作,对网络层数的关系操作在Darknet的forward函数中。
    实际操作在DarkNet的forward函数中
    """
    hyperparams = module_defs.pop(0)
    '''
    hyperparams 为
    'batch': '16', 'subdivisions': '1', 'width': '608', 'height': '1088', 'channels': '3', 
    'momentum': '0.9', 'decay': '0.0005', 'angle': '0', 'saturation': '1.5', 'exposure': '1.5', 
    'hue': '.1', 'learning_rate': '0.001', 'burn_in': '1000', 'max_batches': '500200',
    'policy': 'steps', 'steps': '400000,450000', 'scales': '.1,.1'
    'cfg' = cfg_path
    'nID' = nID
    '''
    output_filters = [int(hyperparams['channels'])]  # 每层的通道数，方便进行卷积操作的输入通道等
    module_list = nn.ModuleList()  # 是一个list每个元素都是每层网络的nn.ModuleList
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                after_bn = batch_norm(filters)
                modules.add_module('batch_norm_%d' % i, after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1. 
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight) 
                nn.init.zeros_(after_bn.bias)
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])  # 这里可以解释yolov3.cfg中的那些负数了.是倒叙的意思
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            """
            [yolo]:
                    mask = 4,5,6,7 
                    anchors = 8,24, 11, 34, 16,48, 23,68, 32,96, 45,135, 64,192, 90,271, 128,384, 180,540, 256,640, 512, 640              
                    anchors是预设的anchor大小,每两个表示一个anchor,长和宽 ,anchor是相对于cfg的width和height
                    classes=1
                    num=12
                    jitter=.3
                    ignore_thresh = .7
                    truth_thresh = 1
                    random=1
            """
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  # 两两合并
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes  此项目为1,因为是对行人做的跟踪
            img_size = (int(hyperparams['width']),int(hyperparams['height']))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, int(hyperparams['nID']), 
                                   int(hyperparams['embedding_dim']), img_size, yolo_layer_count)
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """route和shortcut层的占位符"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, nID, nE, img_size, yolo_layer):
        """
        :param anchors:
        :param nC: 1
        :param nID: 数据集的id个数，也就是训练的分类个数
        :param nE:  embedding_dim
        :param img_size: yolov3.cfg中的width和height参数
        :param yolo_layer: 当前yolo层数
        """
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer  # 一共三层,没实际操作
        #nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (4)
        self.nC = nC  # number of classes (1)
        self.nID = nID  # number of identities
        self.emb_dim = nE
        #self.shift = [1, 3, 5]
        self.img_size = img_size

        self.SmoothL1Loss  = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15*torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85*torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3*torch.ones(1))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nID-1) if self.nID>1 else 1


    def forward(self, p_cat, targets=None, classifier=None, test_emb=False):
        """
        :param p_cat: 输入
        :param targets: [class] [identity] [x_center] [y_center] [width] [height]
        :param classifier: 分类器 nn.Linear(self.emb_dim, nID) self.emb_dim为cfg参数
        :param test_emb:  test_emb标识
        :return:
        训练时： x, *losses = YOLOLayer(x, targets, self.classifier)
        test_emb:  x = YOLOLayer(x, self.img_size, targets, self.classifier, self.test_emb)
        测试时：x = YOLOLayer(x)
        """

        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]  # p_cat ==[nSample , C , H , W]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        create_grids(self, self.img_size, nGh, nGw)
        '''有4个值
        self.stride = img_size[0]/nGw
        self.grid_xy (1, 1, nGh, nGw, 2)
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh  (1, self.nA, 1, 1, 2)
        '''

        if p.is_cuda:
            self.grid_xy = self.grid_xy.cuda()  # (1, 1, nGh, nGw, 2)
            self.anchor_wh = self.anchor_wh.cuda()  # (1, self.nA, 1, 1, 2)

        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction: p ==[nB,4,nGh, nGw,6]
        # 为什么+5 (因为p的通道数为24,self.nA是4,self.nA是yolov3中的mark个数,self.nC为1,故为5)   
        p_emb = p_emb.permute(0,2,3,1).contiguous()  # [nSample , H , W , C]
        p_box = p[..., :4]  # [nB,4,nGh, nGw,4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf [nB,2,nGh, nGw,4]

        # Training
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec, self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec, self.nA, self.nC, nGh, nGw)
            #tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0  # tconf>0为前景

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            # nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf = FT([0]), FT([0])
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze()
            emb_mask, _ = mask.max(1)  # emb_mask 变为[32, 10, 18]

            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids, _ = tids.max(1)  # 每个像素点有四个不同size的anchor，这里选择其中一个 ，则shape[32, 4, 10, 18, 1]，变为[32, 10, 18, 1]



            tids = tids[emb_mask]  # tids  == (nM,1)

            # embedding = [emb_mask].contiguous()
            embedding = p_emb[emb_mask].contiguous()

            embedding = self.emb_scale * F.normalize(embedding)
            # nI = emb_mask.sum().float()
            
            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim+1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt
            
            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()  # classifier是一个线性分类器nn.Linear(self.emb_dim, nID) ，self.emb是固定参数，nID是行人id总和
                lid = self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = torch.exp(-self.s_r)*lbox + torch.exp(-self.s_c)*lconf + torch.exp(-self.s_id)*lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:,1,...].unsqueeze(-1)  # p_conf的dim=1维度上是conf信息

            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1,self.nA,1,1,1).contiguous(), dim=-1)
            p_cls = torch.zeros(nB,self.nA,nGh,nGw,1)               # Temp 这个不用理解
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)


            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))  # self.anchor_vec.to(p)会将self.anchor_vec数据类型与p一致
            p[..., :4] *= self.stride


            return p.view(nB, -1, p.shape[-1])  # (nB, 4*W*H, C)


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_dict, nID=0, test_emb=False):
        super(Darknet, self).__init__()
        if isinstance(cfg_dict, str):
            cfg_dict = parse_model_cfg(cfg_dict)  # 将yolov3.cfg文件解读文一个带有参数的字典集合(list)
        self.module_defs = cfg_dict 
        self.module_defs[0]['nID'] = nID
        self.img_size = [int(self.module_defs[0]['width']), int(self.module_defs[0]['height'])]
        self.emb_dim = int(self.module_defs[0]['embedding_dim'])
        self.hyperparams, self.module_list = create_modules(self.module_defs) # 这个函数只是构造模块,并没有操作
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb  # 这是一个什么的标志?????
        
        self.classifier = nn.Linear(self.emb_dim, nID) if nID>0 else None
        # 训练时会在在dataset中加载出nID，并传入model中
        # 推理时为0

    def forward(self, x, targets=None, targets_len=None):
        """训练时targets和targets_len都不为空
        :param x: img 经过处理过的img，主要为缩放填充，和一些数据增强
        :param targets: [class] [identity] [x_center] [y_center] [width] [height]
        :param targets_len: 批次
        :return:
        """
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        #img_size = x.shape[-1]
        layer_outputs = []
        output = []  # yolo层的输出，训练时为损失，

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x, *losses = module[0](x, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x = module[0](x, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3 
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values()))
        elif self.test_emb:
            return torch.cat(output, 0)

        return torch.cat(output, 1)


def shift_tensor_vertically(t, delta):
    # t should be a 5-D tensor (nB, nA, nH, nW, nC)
    res = torch.zeros_like(t)
    if delta >= 0:
        res[:,:, :-delta, :, :] = t[:,:, delta:, :, :]
    else:
        res[:,:, -delta:, :, :] = t[:,:, :delta, :, :]
    return res 

def create_grids(self, img_size, nGh, nGw):
    # create_grids函数中的
    """
    :param self:
    :param img_size:  cfg.json中的固定尺寸 eg：(1088, 608)
    :param nGh: 当前yolo前一层Tensor的H
    :param nGw: 当前yolo前一层Tensor的W
    :return:
    """
    self.stride = img_size[0]/nGw
    assert self.stride == img_size[1] / nGh, \
            "{} v.s. {}/{}".format(self.stride, img_size[1], nGh)

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0,1).view((1, 1, nGh, nGw)).float()
    #grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)
    '''
    torch.arange(nGw) = [0,1,2,...]
    torch.arange(nGw).repeat((nGh, 1)) = [[0,1,2,...],[0,1,2,...],[0,1,2,...]] 即将一个序列按行排列
    torch.arange(nGw).repeat((nGh, 1)).view((nGh, nGw)) = [[0,0,0,...],[1,1,1,...],[2,2,2,...]] 
    则grid_x是按行排列的,grid_y是按列排列的(因为grid_y有个transpose(0,1)操作)
    torch.stack((grid_x, grid_y), 4)是将 grid_x, grid_y进行拼接,拼接的维度在第4维度上,(torch维度从0开始的)
    grid_xy 维度为(1, 1, nGh, nGw, 2)
    grid_xy 的意义为在【nGh, nGw】的"地图"上的坐标点
    '''

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


def save_weights(self, path, cutoff=-1):
    """
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
