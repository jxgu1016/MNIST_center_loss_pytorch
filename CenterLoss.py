import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_buffer('loss_weight', Variable(torch.Tensor([loss_weight]), requires_grad=False))
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.register_parameter('centers', self.centers) # no need to register manually. See nn.Module.__setattr__(...)
        self.register_buffer('grad_centers', Variable(torch.zeros(self.centers.size()), requires_grad=False))
        self.register_buffer('counts', Variable(torch.ones(self.centers.size(0)), requires_grad=False))
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, y, feat):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        return self.centerlossfunction(feat, y, self.centers, self.loss_weight, self.grad_centers, self.counts)


class CenterlossFunction(Function):
    '''
    All the input should be Variables and the last three(loss_weight, grad_centers, counts) must 
    be registered in the nn.Module class for automatically transformation from cpu to gpu when
    calling CenterLoss.cuda()
    '''
    @staticmethod
    def forward(ctx, feature, label, centers, loss_weight, grad_centers, counts):
        ctx.save_for_backward(feature, label, centers, loss_weight, grad_centers, counts)
        centers_pred = centers.index_select(0, label.long())
        return (feature - centers_pred).pow(2).sum(1).sum(0) * loss_weight  / 2.0
        # a bug that torch.sum() not return a Tensor. 
        # See "https://discuss.pytorch.org/t/torch-sum-not-return-a-tensor/15200/5"

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, loss_weight, grad_centers, counts = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long()) # Eq. 3

        # init every iteration
        counts = counts.data.fill_(1)
        grad_centers = grad_centers.data.fill_(0)
        # print counts, grad_centers

        # Eq. 4 || need optimization !!
        for i in range(feature.size(0)):
            j = int(label[i].data[0])
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])
        # print counts
        grad_centers = Variable(grad_centers/counts.view(-1, 1))

        return grad_feature * loss_weight, None, grad_centers, None, None, None # grad_centers need to mul loss_weight???


def main():
    ct = CenterLoss(10,2)
    ct = ct.cuda()
    print list(ct.parameters())
    print ct.centers.grad
    y = Variable(torch.Tensor([0,0,2,1]).cuda())
    # print y
    feat = Variable(torch.zeros(4,2).cuda(),requires_grad=True)
    # print feat
    out = ct(y,feat)
    out.backward()
    print ct.centers.grad
    print feat.grad

# def grad_check():
#     from torch.autograd import gradcheck
#     input = (Variable(torch.randn(4,2),requires_grad=True), Variable(torch.Tensor([0,0,0,1]), requires_grad=False), Variable(torch.randn(5,2),requires_grad=True))
#     # Of course it will fail. I just want to try this function
#     test = gradcheck(CenterlossFunction.apply, input, eps=1e-6, atol=1e-4)
#     print test

if __name__ == '__main__':
    main()
    # grad_check()
