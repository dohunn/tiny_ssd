import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

import json
import PIL.Image as Image

label_json = {'train':'bdd100k_labels_images_train.json',
                'val':'bdd100k_labels_images_val.json'}

BDD_CLASSES = (
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", 
    "train", "truck"
)

# note: if you used our download scripts, this should be right
BDD_ROOT = osp.join('/home/coin/datasets/', 'BDD100K')

# dict 말고 list 형식으로 바꾸기
def label2dict(frames):
    img_boxes = dict()
    for frame in frames:
        boxes = list()
        for label in frame['labels']:
            if 'box2d' not in label: # bounding box가 없는 경우 제외
                continue
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            boxes.append([xy['x1'], xy['y1'], xy['x2'], xy['y2'], label['category']])
            
        img_boxes[frame['name']] = boxes
    return img_boxes

def label2img(frames):
    images = list()
    for frame in frames:
        images.append(frame['name'])
    return images

class BDDDetection(data.Dataset):
    """BDD(Berkely Driving Dataset) Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 type_sets, 
                 transform=None,
                 dataset_name='BDD100K'):
        self.root = root # '/home/coin/datasets/BDD100K'
        self.type_sets = type_sets # 'train' or 'val'
        self.name = dataset_name
        self.transform = transform # dataset transform!

        self._labelpath = osp.join(root, 'labels', label_json[type_sets]) # train.json, val.json
        self._imgpath = osp.join(root, 'images/100k', type_sets) # alright
        
        # train val path를 입력 받으면 그에 해당하는 label과 img들을 알아내야한다
        self.label = json.load(open(self._labelpath, 'r'))
        self.id2bboxes = label2dict(self.label) # img id -> bounding boxes(xmin,ymin,xmax,ymax,category(str))
        self.ids = label2img(self.label) # img id list

        self.class_to_ind = dict(zip(BDD_CLASSES, range(len(BDD_CLASSES))))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        # image 1개 당 여러개의 바운 딩 박스정보를 포함한다, 그러니 img_id 아래에 여러개의 바운딩 박스 정보가 존재해야한다
        _target = self.id2bboxes[img_id]
        # pil->numpy로 대체 
        img = Image.open(osp.join(self._imgpath, img_id))
        img = np.array(img)
        height, width, channels = img.shape

        target = list()
        for i, bbox in enumerate(_target):
            bboxes = list()
            for j, pt in enumerate(bbox):
                if j == 4:
                    label_idx = self.class_to_ind[pt]
                    bboxes.append(label_idx)
                else:
                    pt = pt / width if i % 2 == 0 else pt / height
                    bboxes.append(pt)
            target += [bboxes]

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            # img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
