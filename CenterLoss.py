import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.register_parameter('centers', self.centers)

    def forward(self, y, feat):
        centers_pred = self.centers.index_select(0, y.long())
        # print centers_pred.size()
        # print feat
        difference = feat - centers_pred
        loss = difference.pow(2).sum() / (2 * y.size()[0])
        return loss




def main():
    from torch.autograd import Variable
    ct = CenterLoss(10,2)
    for p in ct.parameters():
        print p

    print ct.centers

    # y = Variable(torch.range(0,9),requires_grad=True)
    # print y
    # feat = Variable(torch.randn(5,2),requires_grad=True)
    # print feat



if __name__ == '__main__':
    main()