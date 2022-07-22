import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim

import math


class PairwiseDistance(Function):
    '''
        ### This class was ABANDONEND!

        Norm-p distance between x1 and x2. Initialize with p.
        Inherited from Function module.'''
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        # The sizes of $x1 and $x2 should be identical.
        assert x1.size() == x2.size()

        # In case that $x1 and $x2 are totally identical,
        # then $out will be zeros.
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)

        return torch.pow(out + eps, 1. / self.norm)

def cosine_similarity(x1, x2, eps = 1.e-7):
    '''
        Measure the cosine similarity of two vectors.
        
        args:       x1, x2 
                    - x1 and x2 both should have unit norm and have
                        the same size.
                    - both should have size of (batch_size, 1, x1.size(2)==512)
                    - the embeding size should be (1, 512)

        returns:    (x1)^T·x2
                    - (batch_size, 1)

        Usage Example:

        x1 = torch.FloatTensor([[[1,2,3]], [[1,1,1]]])
        # print(x1.size())
        # print(torch.norm(x1, p=2, dim=2, keepdim=True))
        x1 = x1/torch.norm(x1, p=2, dim=-1, keepdim=True)
        # print(torch.norm(x1, p=2, dim=2, keepdim=True))
        x2 = torch.FloatTensor([[[-1,-2,-3]], [[0,1,0]]])
        x2 = x2/torch.norm(x2, p=2, dim=-1, keepdim=True)
        print(cosine_similarity(x1, x2)) '''
    # eps = 1.e-7
    # print('x1 size', x1.size())
    norm_x1 = torch.norm(x1, p=2, dim=-1, keepdim=True)
    norm_x2 = torch.norm(x2, p=2, dim=-1, keepdim=True)
    # print('norm x1 size', norm_x1.size())
    # print(norm_x1[:2])
    assert (torch.sum(torch.abs(norm_x1-1.), axis = 0)/x1.size(0) < eps), \
            "x1 has l2-norm of {} > {}".format(torch.sum(torch.abs(norm_x1-1.), axis = 0)/x1.size(0), eps)
    assert (torch.sum(torch.abs(norm_x2-1.), axis = 0)/x2.size(0) < eps), \
            "x2 has l2-norm of {} > {}".format(torch.sum(torch.abs(norm_x2-1.), axis = 0)/x2.size(0), eps)
    assert x1.size() == x2.size()


    return torch.sum(torch.mul(x1, x2), axis = -1)


class TripletMarginLoss(Function):
    '''
        ### This class is MODIFIED to use cosine similarity.
        ### And to use a correct loss calculation.

        Triplet loss function.
        Inherited from Function module.
        
        Usage Example:

        def tounitnorm(x):
            return x/torch.norm(x, p=2, dim=-1, keepdim=True)

        tm = TripletMarginLoss(0.1)
        anchor = torch.FloatTensor([[[1,0,0]], [[0,1,0]], [[0,0,1]]])
        anchor = tounitnorm(anchor)
        positive = torch.FloatTensor([[[1,1,0]], [[0,-1,0]], [[0,0,-1]]])
        positive = tounitnorm(positive)
        negative = torch.FloatTensor([[[1,0,1]], [[0,1,-1]], [[0,0,1]]])
        negative = tounitnorm(negative)

        tm.forward(anchor, positive, negative)
        >>> tensor(1.3357)
        '''
        
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        # self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        # d_p = self.pdist.forward(anchor, positive)
        # d_n = self.pdist.forward(anchor, negative)

        # Here d_p and d_n will have the size of (batch_size, 1)
        a_p = cosine_similarity(anchor, positive)
        a_n = cosine_similarity(anchor, negative)

        ##? According to the paper,
        ##? here should be: self.margin + d_n - d_p
        # dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        dist_hinge = torch.clamp(self.margin + a_n - a_p, min=0.0)
        loss = torch.mean(dist_hinge)

        return loss


class ReLU(nn.Hardtanh):
    '''
        Activation function: The Clipped Rectified Linear (ReLU) function
            sigma(x) = min{max{x, 0}, 20}

        Inplace version if $inplace is set True.

        args:   inplace
                - default False
    '''
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    '''
        An implementation of ResBlock.
    '''

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, inplanes, planes, number):

        super(ResNet, self).__init__()
        self.resblocks = []
        self.number = number

        for i in range(self.number):
            self.resblocks.append(ResBlock(inplanes, planes))

        self.net = nn.Sequential(*self.resblocks)

    def forward(self, x):

        out = self.net(x)

        return out



class DeepSpeakerModel(nn.Module):
    def __init__(self, embedding_size, num_classes, feature_dim = 64):
        super(DeepSpeakerModel, self).__init__()

        self.embedding_size = embedding_size

        # Layer No. 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        ## philipperemy adds a ReLU here
        self.relu1 = ReLU()
        self.res1 = ResNet(64, 64, 3)

        # Layer No. 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        ## philipperemy adds a ReLU here
        self.relu2 = ReLU()
        self.res2 = ResNet(128, 128, 3)

        # Layer No. 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        ## philipperemy adds a ReLU here
        self.relu3 = ReLU()
        self.res3 = ResNet(256, 256, 3)

        # Layer No. 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        ## philipperemy adds a ReLU here
        self.relu4 = ReLU()
        self.res4 = ResNet(512, 512, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))


        if feature_dim == 64:
            self.fc = nn.Linear(512 * 4, self.embedding_size)
        elif feature_dim == 40:
            self.fc = nn.Linear(256 * 5, self.embedding_size)

        self.classifier = nn.Linear(self.embedding_size, num_classes)



    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        # print('conv1: ', x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.res1(x)

        # print('conv2: ', x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.res2(x)

        # print('conv3: ', x.size())
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.res3(x)

        # print('conv4: ', x.size())
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.res4(x)

        # print('avg: ', x.size())
        # Average
        x = self.avgpool(x)
        # print('fc: ', x.size())
        x = x.view(x.size(0), -1)

        # print('Affine: ', x.size())
        # Affine
        x = self.fc(x)

        # print('ln: ', x.size())
        # ln
        self.features = self.l2_norm(x)

        # print('output: ', x.size())
        ##? is it necessary???
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        # alpha=10
        # self.features = self.features*alpha

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return res


def create_optimizer(model, lr, wd, step_size):
    '''
        Setup an optimizer suggested in the paper.
        "In both stages, we use synchronous SGD with 0.99 momentum, with 
        a linear decreasing learning rate schedule from 0.05 to 0.005."

        input:      step_size
                    - approx total 5.3 times decay from 0.05 to 0.005
    '''
    optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.99, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                        step_size=step_size, gamma=0.65, verbose=True)
    
    return optimizer, scheduler