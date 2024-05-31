import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k
    
class GaussialBlurOperator():
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
    

def read_img(img_path, read_alpha=False):
    img = imageio.imread(img_path)
    img = Image.fromarray(img)
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    if read_alpha:
        img = img[:, :, 3:] / 255.0
    else:
        img = img[:, :, :3] / 255.0

    img = torch.from_numpy(img).to(0).float()
    return img

if __name__=="__main__":
    from PIL import Image
    import imageio

    operator = GaussialBlurOperator(33, 3.0, 0)
    img = read_img("/home/chenxi/code/ml-hypersim/downloads/ai_001_001/images/scene_cam_00_final_preview/frame.0000.diffuse_reflectance.jpg")
    img = img[:256, :256]
    img_blurred = operator.forward(img[None].permute(0,3,1,2).cuda())[0].permute(1,2,0)
    img_out = torch.cat([img, img_blurred], dim=1)

    Image.fromarray((img_out.detach().cpu().numpy()*255).astype(np.uint8)).save("dbg/blurred.png")
