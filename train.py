from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
# from ssd import build_ssd
from ssd_test import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'BDD', 'COCO'],
                    type=str, help='VOC or BDD or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--fine_tune', default=None, type=str,
                    help='Checkpoint state_dict file to fine tuning trained model')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--tiny', default=False, type=str2bool,
                    help='Choose Tiny SSD or Original SSD')
parser.add_argument('--box_size_change', default=False, type=str2bool,
                    help='Chosse 4-6 or 3-5')
parser.add_argument('--minmax', default=False, type=str2bool,
                    help='Chosse min or max when aspect ratio is 1') 
                    # min = False, max = True
#iteration 추가하기?
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,6,7,8'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,8,9'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,5,9'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,6,7,9'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    else:
        print('Berkeley DeepDrive')
        cfg = bdd
        dataset = BDDDetection(root='/home/coin/datasets/BDD100K', type_sets='train', 
                               transform=SSDAugmentation(cfg['min_dim'],
                                                        MEANS))

    if args.tiny:
        if args.box_size_change:
            if args.lr == 1e-3:
                print('learning rate 1e-3')
                if args.minmax:
                    save_model_name = 'weights/tiny_ssd300_' + 'batch' + str(args.batch_size) + '_lr3_max_'
                else:
                    save_model_name = 'weights/tiny_ssd300_' + 'batch' + str(args.batch_size) + '_lr3_min_'
            else:
                print('learning rate 1e-4')
                if args.minmax:
                    save_model_name = 'weights/tiny_ssd300_' + 'batch' + str(args.batch_size) + '_lr4_max_'
                else:
                    save_model_name = 'weights/tiny_ssd300_' + 'batch' + str(args.batch_size) + '_lr4_min_'
        else:
            if args.lr == 1e-3:
                print('learning rate 1e-3')
                save_model_name = 'weights/tiny_param_ssd300_' + 'batch' + str(args.batch_size) + '_lr3_'
            else:
                print('learning rate 1e-4')
                save_model_name = 'weights/tiny_param_ssd300_' + 'batch' + str(args.batch_size) + '_lr4_'
    else:
        if args.box_size_change:
            if args.lr == 1e-3:
                print('learning rate 1e-3')
                if args.minmax:
                    save_model_name = 'weights/tiny_size_ssd300_' + 'batch' + str(args.batch_size) + '_lr3_max_'
                else:
                    save_model_name = 'weights/tiny_size_ssd300_' + 'batch' + str(args.batch_size) + '_lr3_min_'
            else:
                print('learning rate 1e-4')
                if args.minmax:
                    save_model_name = 'weights/tiny_size_ssd300_' + 'batch' + str(args.batch_size) + '_lr4_max_'
                else:
                    save_model_name = 'weights/tiny_size_ssd300_' + 'batch' + str(args.batch_size) + '_lr4_min_'
        else:
            if args.lr == 1e-3:
                print('learning rate 1e-3')
                save_model_name = 'weights/ssd300_' + 'batch' + str(args.batch_size) + '_lr3_'
            else:
                print('learning rate 1e-4')
                save_model_name = 'weights/ssd300_' + 'batch' + str(args.batch_size) + '_lr4_'

    if args.fine_tune:
        # voc에서 학습한 모델 로드!
        cfg = voc
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], tiny=args.tiny, 
                        box_size_change=args.box_size_change, minmax=args.minmax)
        trained_model = os.path.join('weights', args.fine_tune)
        ssd_net.load_state_dict(torch.load(trained_model))
        
        # bdd에 맞춘 모델 설계
        cfg = bdd
        new_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], tiny=args.tiny, 
                        box_size_change=args.box_size_change, minmax=args.minmax)

        # Origin SSD일 경우
        # base-network만 freezing
        # if args.tiny:
        #     for k, v in enumerate(ssd_net.vgg):
        #         if k == 21 or k == 33:
        #             for param in v.parameters():
        #                 param.requires_grad = True
        #         else:
        #             for param in v.parameters():
        #                 param.requires_grad = False
        # else:
        #     for k, v in enumerate(ssd_net.vgg):
        #         if k == 13 or k == 23:
        #             for param in v.parameters():
        #                 param.requires_grad = True
        #         else:
        #             for param in v.parameters():
        #                 param.requires_grad = False
        new_net.vgg = ssd_net.vgg
        vgg_parameters = count_parameters(ssd_net.vgg)
        print('vgg parameters : ', vgg_parameters)

        # extras 중 feature map과 관련된 부분을 제외하고 고정
        # for k, extra in enumerate(ssd_net.extras):
        #     if k % 2 == 0:
        #         for param in extra.parameters():
        #             param.requires_grad = False
        new_net.extras = ssd_net.extras

        new_net.loc = ssd_net.loc
        net = new_net

        save_model_name += 'fine'

        if args.box_size_change:
            net.conf[0] = nn.Conv2d(512, cfg['num_classes'] * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[1] = nn.Conv2d(1024, cfg['num_classes'] * 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[2] = nn.Conv2d(512, cfg['num_classes'] * 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[3] = nn.Conv2d(256, cfg['num_classes'] * 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[4] = nn.Conv2d(256, cfg['num_classes'] * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[5] = nn.Conv2d(256, cfg['num_classes'] * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            net.conf[0] = nn.Conv2d(512, cfg['num_classes'] * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[1] = nn.Conv2d(1024, cfg['num_classes'] * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[2] = nn.Conv2d(512, cfg['num_classes'] * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[3] = nn.Conv2d(256, cfg['num_classes'] * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[4] = nn.Conv2d(256, cfg['num_classes'] * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            net.conf[5] = nn.Conv2d(256, cfg['num_classes'] * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        print('fine tuning!')

    else:
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], tiny=args.tiny, 
                        box_size_change=args.box_size_change, minmax=args.minmax)
        net = ssd_net
        print('just training!')

    # if args.resume:
    #     ssd_net.load_state_dict(torch.load('/home/dohun/Code/PythonCode/Paper_Models/SSD/ssd.pytorch/weights/tiny_ssd300_batch32_lr3_max_10000_test.pth'))
    #     print('load 성공!')
    #     print('Resuming training, loading {}...'.format(args.resume))
    #     # ssd_net.load_weights(args.resume)

    if not args.resume:
        # initialize newly added layers' weights with xavier method
        if args.fine_tune:
            net.conf.apply(weights_init)
            print('Initializing weights... when fine tuning')
        else:
            net.extras.apply(weights_init)
            net.loc.apply(weights_init)
            net.conf.apply(weights_init)
            print('Initializing weights...')

    
    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.cuda:
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    epoch_iteration = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size

    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print('epoch_size', epoch_size)

    step_index = 0

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma=0.1)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # default max_iter is 120000 when batch size is 32, epoch_size 517
    # 80000(1e-3), 10000(1e-4), 120000(1e-5)
    # default max_iter is 60000 when batch size is 64, epoch_size 258
    # 40000(1e-3), 50000(1e-4), 60000(1e-5)

    # create batch iterator
    # epoch = 0
    # if args.batch_size == 32:
    #     if args.fine_tune:
    #         print('fine')
    #         max_iter = cfg['f_max_iter']
    #         lr_steps = [int(step) for step in cfg['f_lr_steps']]
    #         print('max_iter', max_iter)
    #         print('lr_steps', lr_steps)
    #     else:
    #         print('!fine')
    #         max_iter = cfg['max_iter']
    #         lr_steps = [int(step) for step in cfg['lr_steps']]
    #         print('max_iter', max_iter)
    #         print('lr_steps', lr_steps)
    # elif args.batch_size == 64:
    #     # batch_size = 64, 2020.01.01.15:21 돌린다
    #     max_iter = cfg['max_iter']
    #     lr_steps = [int(step) for step in cfg['lr_steps']]
    #     print('max_iter', max_iter)
    #     print('lr_steps', lr_steps)

    lr_steps = [int(step) for step in cfg['lr_steps']]
    max_iter = cfg['f_max_iter']
    # max_iter, lr_steps = return_iter(args.batch_size, args.fine_tune, cfg)

    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, max_iter):
        if iteration != 0 and ((iteration + 1) % epoch_size == 0):
            print('epoch {} || Localize Loss : {} Confidece Loss : {}'.format(epoch+1, 
                    loc_loss/epoch_iteration, conf_loss/epoch_iteration))
            loc_loss = 0
            conf_loss = 0
            epoch_iteration = 0
            epoch += 1

        if iteration in lr_steps:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
            # print(images.dtype, images.size())
            # print(targets[0].dtype, targets[0].size(), targets[0])
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        epoch_iteration += 1

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            #requires_grad=True
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        # scheduler.step()

        if iteration % 100 == 0:
            # current_lr = scheduler.get_lr()
            print('timer: %.4f sec.' % (t1 - t0))
            print('iteration : {} || Localize Loss : {} Confidence Loss : {}'.format(iteration, 
                    loss_l.item(), loss_c.item()))
            # print('current learning rate : {}'.format(current_lr))
            for param_group in optimizer.param_groups:
                print('current learning rate : {}'.format(param_group['lr']))

        if iteration != 0 and iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), save_model_name + repr(iteration) + '_.pth')

    torch.save(ssd_net.state_dict(), save_model_name + '_final.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    train()