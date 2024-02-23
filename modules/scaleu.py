import torch
import torch.nn as nn
import torch.fft as fft


def Fourier_filter(x_in, threshold, scale):
    x = x_in
    B, C, H, W = x.shape

    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)


class ScaleU(nn.Module):
    def __init__(self, enable_se_scaleu=True, b_size=1280, s_size=1):
        super(ScaleU, self).__init__()
        self.scaleu_b = nn.Parameter(torch.zeros(b_size))
        self.scaleu_s = nn.Parameter(torch.zeros(s_size))
        self.enable_se_scaleu = enable_se_scaleu

    def forward(self, h, hs_, transformer_options={}):
      h = h.to(torch.float32)
      hs_ = hs_.to(torch.float32)
      b = torch.tanh(self.scaleu_b) + 1
      s = torch.tanh(self.scaleu_s) + 1
      if self.enable_se_scaleu:
          hidden_mean = h.mean(1).unsqueeze(1) # B,1,H,W 
          B = hidden_mean.shape[0]
          hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) # B,1
          hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True) # B,1
          # duplicate the hidden_mean dimension 1 to C
          hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3) # B,1,H,W
          b = torch.einsum('c,bchw->bchw', b-1, hidden_mean) + 1.0 # B,C,H,W
          h = torch.einsum('bchw,bchw->bchw', h, b)
      else:      
          h = torch.einsum('bchw,c->bchw', h, b)
      
      hs_ = Fourier_filter(hs_, threshold=1, scale=s)
      return h.to(torch.float16), hs_.to(torch.float16)
