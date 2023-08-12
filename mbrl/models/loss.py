import torch.nn.modules.loss


class NLLLoss(torch.nn.modules.loss._Loss):
    """
    Specialized NLL loss used to predict both mean (the actual function) and the variance of the input data.
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(NLLLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, net_output, target):
        assert net_output.dim() == 3
        assert net_output.size(2) == 2
        mean = net_output[:, :, 0]
        var = net_output[:, :, 1]
        reduction = 'mean'
        # reduction = 'sum'
        ret = 0.5 * torch.log(var) + 0.5 * ((mean - target) ** 2) / var
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret
