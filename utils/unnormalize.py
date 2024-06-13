import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class ToNumpy(object):
    def __call__(self, tensor):
        tensor = tensor.permute(1,2,0)
        tensor = tensor * 255
        return tensor.numpy()
    
class PIL2Numpy(object):
    def __call__(self, pil):
        return np.array(pil)