# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def smart_lossFunction(lossName:str, *args, **kwargs):
    """
    返回优化器对象, 示例:
    loss = smart_lossFunction('CrossEntropyLoss')
    loss = smart_lossFunction('FocalLoss', 7)
    loss = smart_lossFunction('FocalLoss', class_num=7)
    """
    if len(args) == 0 and len(kwargs) == 0:
        loss = eval(lossName)()
    else:
        loss = eval(lossName)(*args, **kwargs)
    return loss

CrossEntropyLoss = nn.CrossEntropyLoss

# class FocalLoss(nn.Module):
#     """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

#     def __init__(
#         self,
#     ):
#         """Initializer for FocalLoss class with no parameters."""
#         super().__init__()

#     @staticmethod
#     def forward(pred, label, gamma=1.5, alpha=0.25):
#         """Calculates and updates confusion matrix for object detection/classification tasks."""
#         # label = label.view(-1,1)
#         # if len(pred.shape) > 1 and pred.shape[1] > 1:
#         #     pred = pred.argmax(axis=1)
#         #     pred = pred.type(torch.float)
#         #     label = label.type(torch.float)
#         #     pred = pred.reshape(pred.shape[0], 1)
#         #     label = label.reshape(label.shape[0], 1)
#         # print(pred.dtype, '\n', label.dtype)

#         loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
#         # p_t = torch.exp(-loss)
#         # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

#         # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
#         pred_prob = pred.sigmoid()  # prob from logits
#         p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
#         modulating_factor = (1.0 - p_t) ** gamma
#         loss *= modulating_factor
#         if alpha > 0:
#             alpha_factor = label * alpha + (1 - label) * (1 - alpha)
#             loss *= alpha_factor
#         return loss.mean(1).sum()

 
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
 
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss