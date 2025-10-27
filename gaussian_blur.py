from imports import *

class GaussianBlur:
    """Gaussian blur for curriculum learning"""
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        if self.sigma == 0:
            return img
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (0, 0),
                                   sigmaX=self.sigma, sigmaY=self.sigma)
        return Image.fromarray(blurred)