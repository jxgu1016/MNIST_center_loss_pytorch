import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim，loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.register_parameter('centers', self.centers)
        self.use_cuda = False

    def forward(self, y, feat):
        centers_pred = self.centers.index_select(0, y.long())
        count = np.ones(self.num_classes)

        batch_size = feat.size()[0]
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size()[1]))

        if self.use_cuda:
            y = y.cpu()
        y_np = y.data.numpy().astype(int)
        for n in y_np:
            count[n] += 1
        count_center = Variable(torch.Tensor(count[y_np]))
        if self.use_cuda:
            count_center = count_center.cuda()
        diff = feat - centers_pred
        loss = 1 / 2.0 *(diff.pow(2).sum(1) / count_center).sum() * self.loss_weight

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))



def main():
    ct = CenterLoss(10,2)
    print list(ct.parameters())

    print ct.centers.grad

    y = Variable(torch.Tensor((0,0,0,1)))
    # print y
    feat = Variable(torch.zeros(4,2),requires_grad=True)
    # print feat

    out = ct(y,feat)
    out.backward()
    print ct.centers.grad



if __name__ == '__main__':
    main()