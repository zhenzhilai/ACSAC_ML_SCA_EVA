import pytorch_ssim
import pytorch_msssim
from pytorch_msssim.ssim import MS_SSIM
import torch

class SCA_Loss(torch.nn.Module):
    def __init__(self):
        super(SCA_Loss, self).__init__()
        self.ssim = pytorch_ssim.SSIM(window_size=7)
        # self.mssim = MS_SSIM(data_range=1, win_size=7)
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
    
    def loss_function(self, logits, image):
        # mix_loss = 0.15 * self.l1(logits, image) + 0.85 * (1 - self.ssim(logits, image))
        # mix_loss = 0.15 * self.l1(logits, image) + 0.85 * (1 - self.mssim(logits, image, data_range=1, win_size=7))
        # mix_loss = 0.14 * self.l1(logits, image) + 0.86 * (1 - self.mssim(logits, image))
        mix_loss =  (1 - self.ssim(logits, image))
        # mix_loss =  self.mse(logits, image)
        return mix_loss
