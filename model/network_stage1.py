from torch import nn
from model.utils import *

class N2N(nn.Module):
    '''
    Noise model as in Noise2Noise
    '''
    def __init__(
        self,
        predenoiser,
        Lambda = 1,
    ):
        super().__init__()
        self.predenoiser = predenoiser
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.Lambda = Lambda


    @torch.no_grad()
    def denoise(self, x_in):
        return self.predenoiser(x_in['Y'])

    def p_losses(self, x_in):
        image1 = x_in['X']
        image2 = x_in['Y']

        inputs_pred1 = self.predenoiser(image1)
        loss1 = self.criterion(inputs_pred1, image2)
        labels_pred1 = self.predenoiser(image2)
        loss2 = self.criterion(labels_pred1, image1)
        loss3 = self.criterion(labels_pred1, inputs_pred1)

        loss = (loss1 + loss2 + self.Lambda * loss3) / (2 + self.Lambda)    # NOTE: 回归模型损失函数：N2N重建损失+一致性损失
        return dict(total_loss=loss)



    def forward(self, x, *args, **kwargs):
        return self.p_losses(x)
