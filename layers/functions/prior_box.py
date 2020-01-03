from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg, box_size_change, minmax):
        super(PriorBox, self).__init__()
        self.box_size_change = box_size_change
        if box_size_change:
            print('bboxes 8732 -> 6792, minmax :', minmax)
        else:
            print('bboxed 8732!')
        self.minmax = minmax
        self.image_size = cfg['min_dim'] 
        # 300
        # print(cfg)
        
        # number of priors for feature map location (either 4 or 6)
        # 위는 3과 5로 바꾸고 싶다

        self.num_priors = len(cfg['aspect_ratios'])
        # 6
        self.variance = cfg['variance'] or [0.1] # 이건 무엇인가..
        # [0.1, 0.2]
        self.feature_maps = cfg['feature_maps']
        # [38, 19, 10, 5, 3, 1] image size, 문제 없음
        self.min_sizes = cfg['min_sizes'] # 이건 무엇인가..
        # [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes'] 
        # [60, 111, 162, 213, 264, 315]
        # 아마 min-max size는 각 피처맵에서 생성하는 바운딩 박스의 크기를 말하는 것 같다
        self.steps = cfg['steps']
        # [8, 16, 32, 64, 100, 300]
        # steps * feature_maps ~~ 300, 300, 300, 300, 300, 300
        self.aspect_ratios = cfg['aspect_ratios']
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = cfg['clip']
        # True
        self.version = cfg['name']
        # VOC
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            # k는 0, 1, 2, 3, 4, 5, | f는 38, 19, 10, 5, 3, 1
            # 즉 k는 feature map의 인덱스이고 f는 feature map의 size

            for i, j in product(range(f), repeat=2):
                # i와 j는 중첩반복문이라고 생각하면 된다
                # ex) i=0, j[0 ~ 37] -> i[0 ~ 37], j[0 ~ 37]

                f_k = self.image_size / self.steps[k]
                # image_size는 항상 300 /연산을 8, 16, 32, 64, 100, 300와 함께
                # 그 결과인 f_k는 37.5, 18.75, 9.375, 4.6875, 3, 1

                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1)) # 요기가 작은 부분
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))

                # 그 동안은 작은 부분을 취하고 있었다 이번에는 보통을 취한다
                if self.box_size_change == False:
                    mean += [cx, cy, s_k, s_k] # - 8732개일때(ratio = 1일때)
                    mean += [cx, cy, s_k_prime, s_k_prime] # - 8732개일때(ratio = 1일때 작은 부분)
                else:
                    if self.minmax:
                        # print('normal size when aspect ratio is 1')
                        mean += [cx, cy, s_k, s_k]
                    else:
                        # print('min size when aspect ratio is 1')
                        mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        # print(output.size())
        if self.clip:
            output.clamp_(max=1, min=0)
        # print(output.size())
        return output