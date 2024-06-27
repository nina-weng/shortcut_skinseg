import torch.nn as nn
import torch

'''
Note: preds.shape = [BS,n_classes,size[0],size[1]]
tragets.shape = [BS,1,size[0],size[1]]
'''

class DiceLoss(nn.Module):
    def __init__(self,smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets,reduction='mean'):
        assert torch.max(preds) <=1.0 and torch.min(preds) >=0.0

        
        if preds.shape[1] == 2:
            preds = preds[:, 1, :, :].unsqueeze(1)  # foreground

        if len(preds.shape) == 3:
            preds = preds.unsqueeze(0)
            targets = targets.unsqueeze(0)

        assert preds.shape == targets.shape

        preds = preds.contiguous()
        targets = targets.contiguous()    
        
        intersection = (preds * targets).sum(dim=2).sum(dim=2)
    
        loss = (1 - ((2. * intersection + self.smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + self.smooth)))
    

        if reduction == 'mean':
            loss= loss.mean()
        elif reduction == 'each':
            loss= loss
        else:
            raise NotImplemented

        return loss

class CombinedLoss(nn.Module):
    '''
    combined dice and bce
    '''
    def __init__(self, weights=[0.5,0.5]):
        super(CombinedLoss, self).__init__()
        self.weights=weights
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets, reduction='mean'):
        '''
        inputs should be in the range[0,1]
        '''
        assert torch.max(preds) <=1.0 and torch.min(preds) >=0.0

        if preds.shape[1] == 2:
            preds = preds[:, 1, :, :].unsqueeze(1)  # foreground

        assert preds.shape == targets.shape

        bs = preds.shape[0]


        loss_dice = self.dice_loss(preds, targets,reduction=reduction)
        loss_bce = self.bce_loss(preds, targets).reshape((bs,-1))
        if reduction == 'mean': loss_bce = loss_bce.mean()
        elif reduction == 'each': loss_bce = loss_bce.mean(dim=1).unsqueeze(1)
        else: raise NotImplemented

        assert loss_dice.shape == loss_bce.shape, f'{loss_dice.shape=},{loss_bce.shape=}'

        loss = self.weights[0] * loss_bce + \
               self.weights[1] * loss_dice


        return loss



def dice_loss(pred, target, smooth = 1., reduction = 'mean'):

    if pred.shape[1] == 2:
        pred = pred[:, 1, :, :].unsqueeze(1)  # foreground

    assert pred.shape == target.shape

    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'each':
        return loss.mean(dim=1)
    else:
        raise NotImplemented