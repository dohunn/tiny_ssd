from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

import os
import sys
import time
import argparse
import numpy as np
import pickle
import json
import cv2

# from ssd import build_ssd
from ssd_test import build_ssd
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BDDDetection, BaseTransform
from data import VOC_CLASSES as voc_labelmap
from data import BDD_CLASSES as bdd_labelmap
sys.path.append('/home/junkyu/SSD2')
from cocomini import CocoMiniDetection

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

coco_labelmap = ('apple','orange')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--dataset_type', default='VOC', type=str,
                    help='Choose dataset type between VOC and COCO and BDD')
parser.add_argument('--tiny', default=False, type=str2bool,
                    help='Choose Tiny SSD or Original SSD')
parser.add_argument('--box_size_change', default=False, type=str2bool,
                    help='Chosse 4-6 or 3-5')
parser.add_argument('--minmax', default=False, type=str2bool,
                    help='Chosse min or max when aspect ratio is 1')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

bdd_annopath = '/home/coin/datasets/BDD100K/labels/bdd100k_labels_images_val.json'
voc_annopath = os.path.join('/home/coin/datasets/VOCdevkit', 'VOC2012', 'Annotations', '%s.xml')
imgpath = os.path.join('/home/coin/datasets/VOCdevkit', 'VOC2012', 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join('/home/coin/datasets/VOCdevkit', 'VOC2007', 'ImageSets',
                        #   'Main', '{:s}.txt')
# YEAR = '2007'
# devkit_path = '/home/coin/datasets/VOCdevkit' + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
# set_type = 'val'

current_path = '/home/dohun/Code/PythonCode/Paper_Models/SSD/ssd.pytorch'
# imagsetpath는 voc만 사용
imgsetpath = '/home/dohun/Code/PythonCode/Paper_Models/SSD/ssd.pytorch/ImageSets'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def voc_parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        # obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text) - 1,
                              float(bbox.find('ymin').text) - 1,
                              float(bbox.find('xmax').text) - 1,
                              float(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

# val.json 파일을 받는다
def bdd_parse_rec(frames):
    recs = {}
    for frame in frames: # 이미지
        objects = []
        for label in frame['labels']: # 바운딩 박스
            if 'box2d' not in label: # bounding box가 없는 경우 제외
                continue
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            obj_struct = {}
            obj_struct['name'] = label['category']
            obj_struct['bbox'] = [xy['x1'], xy['y1'], xy['x2'], xy['y2']]
            objects.append(obj_struct)
        recs[frame['name']]
    return recs

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # print(mrec.shape, mrec)
        # print(mpre.shape, mpre)

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eval_net(classname,
             classidx,
             dect_bboxes,
             annopath,
             imagesetfile,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if args.dataset_type == 'VOC':
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'voc_annots.pkl')
        # print(cachefile)
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                # print(annopath % (imagename))
                recs[imagename] = voc_parse_rec(annopath % (imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

    elif args.dataset_type == 'BDD':
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'bdd_annots.pkl')
        
        if not os.path.isfile(cachefile):
            # load annots
            frames = json.load(open(annopath, 'r'))
            recs = bdd_parse_rec(frames)

            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)
    
    class_recs = {}
    npos = 0

    if args.dataset_type == 'VOC':
        for i, imagename in enumerate(recs.keys()):
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            npos = npos + len(bbox)
            class_recs[i] = {'bbox': bbox,
                             'det': det}

    elif args.dataset_type == 'BDD':
        for i, imagename in enumerate(recs.keys()):
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            npos = npos + len(bbox)
            class_recs[i] = {'bbox': bbox,
                             'det': det}

    else:
        for i, rec in enumerate(recs):
            R = [[r for r in re] for re in rec if re[4] == classidx]
            bbox = np.array([[r for r in re] for re in R])
            det = [False] * len(R)
            npos = npos + len(bbox)
            class_recs[i] = {'bbox': bbox,
                             'det': det}

    image_ids = list()

    confidence = np.array([float(box[-1]) for img_box in dect_bboxes
                                            for box in img_box])
    BB = np.array([[float(z) for z in box[:-1]] for img_box in dect_bboxes
                                                    for box in img_box])
    for img_idx, bboxes in enumerate(dect_bboxes):
            for bbox in bboxes:
                image_ids.append(img_idx)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]           

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # image_ids는 중복된 이미지 이름을 가지고 있어서 class_recs는
        # 중복된 인덱스에 여러번 접근된다
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])

            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                    (BBGT[:, 2] - BBGT[:, 0]) *
                    (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            # jmax는 gt 바운딩 박스 중 IOU가 가장 큰 바운딩 박스의 차례
            # 를 가리키므로 이미 사용했다는 지표로 사용한다

            if ovmax >= ovthresh:
                # print("여기는 IOU가 0.5보다 더 큽니다")
                # 0.5보다 IOU가 클 경우
                # 일단 difficult는 필요가 없을 거 같다
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # npos는 바운딩 박스의 개수를 말하는 것 같다
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)

    return rec, prec, ap

def test_net(net, dataset, transform, top_k, im_size=300, thresh=0.05, labelmap=1):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        # print(im.size())

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        # print(detections.size())
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)): #batch, 21, 200, 5
            dets = detections[0, j, :] # dets is 200, 5
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            # dets[:, 0]에서 0보다 큰 것을 취한다
            # 만약 클래스 스코어가 0보다 크지 않다면 bbox 정보가 모두 False
            # 크다면 bbox 정보는 모두 True
            # dets.size(0) is 200, mask is 200. 5

            dets = torch.masked_select(dets, mask).view(-1, 5)
            # dets은 mask와 비교한 결과로 클래스 스코어가 0 보다 크지 않으면 continue 
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)

            # all_boxes의 j는 클래스 개수의 인덱스 i는 이미지 개수의 인덱스 
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    return all_boxes

if __name__ == '__main__':
    # load data and net
    print(args.dataset_type)
    if args.dataset_type == 'VOC':
        # 2012, test가 될 예정
        dataset = VOCDetection('/home/coin/datasets/VOCdevkit', [('2012', 'test')],
                                    BaseTransform(300, dataset_mean),
                                    VOCAnnotationTransform())
        # dataset = VOCDetection('/home/coin/datasets/VOCdevkit', [('2007', set_type)],
        #                             BaseTransform(300, dataset_mean),
        #                             VOCAnnotationTransform())
        labelmap = voc_labelmap
        # path_val_pkl = '/home/coin/datasets/VOCdevkitVOC2007/annotations_cache/annots.pkl'
        image_setpath = os.path.join(imgsetpath, 'test.txt')
        annopath = voc_annopath

    elif args.dataset_type == 'BDD':
        dataset = BDDDetection(root='/home/coin/datasets/BDD100K', type_sets='val', 
                                transform=BaseTransform(300, dataset_mean))
        labelmap = bdd_labelmap
        annopath = bdd_annopath
    
    else:
        dataset = CocoMiniDetection(root='/home/junkyu/SSD2/cocomini',split='val',
                                        transform = BaseTransform(300, dataset_mean))
        labelmap = coco_labelmap
        trained_model = '/home/junkyu/SSD2/weights/ssdf300_49500.pth'
        path_val_pkl = '/home/junkyu/SSD2/cocomini/annotations_val.pkl'

    num_classes = len(labelmap) + 1
    net = build_ssd('test', 300, num_classes,  tiny=args.tiny, 
                    box_size_change=args.box_size_change, minmax=args.minmax)

    trained_model = os.path.join('weights', args.trained_model)
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    print('Finished loading model!')
    
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    all_boxes = test_net(net, dataset, BaseTransform(net.size, dataset_mean), args.top_k, 300,
                        thresh=args.confidence_threshold, labelmap=labelmap)

    # pkl 파일이 보관되어 있는 경로(voc - bdd 공통사항)
    cachedir = os.path.join(current_path, 'annotations_cache')

    aps = []
    # all_box 길이는 이미지 개수
    for cls_idx, all_box in enumerate(all_boxes[1:]):
        rec, prec, ap = eval_net(labelmap[cls_idx], cls_idx, all_box,
                                annopath, image_setpath, cachedir,
                                ovthresh=0.5, use_07_metric=False)
        aps += [ap]
        print('AP for {} = {}'.format(labelmap[cls_idx], ap))
    print('Mean AP = {}'.format(np.mean(aps)))