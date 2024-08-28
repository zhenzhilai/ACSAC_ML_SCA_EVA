from pytorch_msssim import ssim
#import pytorch_ssim



class SsimLoss:
    def __call__(self, a, b):
        #return 1 - pytorch_ssim.ssim(a, b, window_size=8)
        return 1 - ssim(a, b, data_range=1, size_average=True)
