import torch
import torch.nn as nn

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        logits = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(logits)
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * target_logits + (1 - one_hot) * logits
        output *= self.s
        loss = self.ce(output, labels)
        return loss
