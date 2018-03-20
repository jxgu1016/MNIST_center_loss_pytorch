import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.register_parameter('centers', self.centers) # no need to register manually. See nn.Module.__setattr__(...)
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, y, feat):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        return self.centerlossfunction(feat, y, self.centers)


class CenterlossFunction(Function):
    @staticmethod
    def forward(ctx, feature, label, centers,):
        ctx.save_for_backward(feature, label, centers)
        centers_pred = centers.index_select(0, label.long())
        # print centers_pred
        return(feature - centers_pred).pow(2).sum(1).sum(0) 
        # a bug that torch.sum() not return a Tensor. 
        # See "https://discuss.pytorch.org/t/torch-sum-not-return-a-tensor/15200/5"

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long())
        # print 'grad_feature:', grad_feature
        grad_centers = torch.zeros(centers.size())
        counts = torch.ones(centers.size(0))
        if feature.data.type() != 'torch.FloatTensor':
            grad_centers = grad_centers.cuda()
            counts = counts.cuda()
        for i in range(feature.size(0)):
            j = int(label[i].data[0])
            # print j
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])
        # print counts
        grad_centers = Variable(grad_centers/counts.view(-1, 1))
        return grad_feature, None, grad_centers


def main():
    ct = CenterLoss(10,2)
    # ct = ct.cuda()
    print list(ct.parameters())

    print ct.centers.grad

    y = Variable(torch.Tensor([0,0,2,1]))#.cuda())
    # print y
    feat = Variable(torch.zeros(4,2),requires_grad=True)
    # print feat

    out = ct(y,feat)
    out.backward()
    print ct.centers.grad
    print feat.grad

def test_function():
    centers = Variable(torch.randn(10,2).cuda(),requires_grad=True)
    print centers
    feat = Variable(torch.zeros(4,2).cuda(),requires_grad=True)
    y = Variable(torch.Tensor([0,0,2,1]).cuda())
    centerlossfunction = CenterlossFunction.apply
    loss = centerlossfunction(feat, y, centers)
    print loss
    loss.backward()
    print feat.grad
    print centers.grad



if __name__ == '__main__':
    main()
    # test_function()
