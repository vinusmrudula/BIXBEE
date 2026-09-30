# Standalone inference for NTRO 26142 (Sentinel-2 4x SR + MC Dropout).
# Needs only: torch, numpy (matplotlib optional for the PNG).
# Usage: python sr_inference.py --input lr.npy --model ntro26142_swinir_x4_final.pth --out result --passes 5
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)

def window_reverse(w, ws, H, W):
    B = int(w.shape[0] / (H * W / ws / ws))
    x = w.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads, drop):
        super().__init__()
        self.ws, self.heads = ws, heads
        self.scale = (dim // heads) ** -0.5
        self.rpb_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer('rel_index', rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
    def forward(self, x, mask=None):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rpb_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
        attn = self.attn_drop(attn.softmax(-1))
        x = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj_drop(self.proj(x))

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio, drop):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self._masks = {}
    def get_mask(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._masks:
            img = torch.zeros(1, H, W, 1, device=device)
            cnt = 0
            sl = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
            for h in sl:
                for w in sl:
                    img[:, h, w, :] = cnt
                    cnt += 1
            mw = window_partition(img, self.ws).view(-1, self.ws * self.ws)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            self._masks[key] = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)
        return self._masks[key]
    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        use_shift = self.shift > 0 and min(H, W) > self.ws
        if use_shift:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))
            mask = self.get_mask(H, W, x.device)
        else:
            mask = None
        w = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        w = self.attn(w, mask)
        x = window_reverse(w.view(-1, self.ws, self.ws, C), self.ws, H, W)
        if use_shift:
            x = torch.roll(x, (self.shift, self.shift), (1, 2))
        x = shortcut + x.reshape(B, L, C)
        return x + self.mlp(self.norm2(x))

class RSTB(nn.Module):
    def __init__(self, dim, depth, heads, ws, mlp_ratio, drop):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2, mlp_ratio, drop)
                                     for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    def forward(self, x, H, W):
        res = x
        for b in self.blocks:
            x = b(x, H, W)
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x).flatten(2).transpose(1, 2)
        return res + x

class SwinIR(nn.Module):
    def __init__(self, in_ch=4, embed=96, depths=(4, 4, 4, 4), heads=(6, 6, 6, 6), ws=8,
                 mlp_ratio=2.0, drop=0.1, scale=4):
        super().__init__()
        self.ws, self.scale = ws, scale
        self.conv_first = nn.Conv2d(in_ch, embed, 3, 1, 1)
        self.layers = nn.ModuleList([RSTB(embed, d, h, ws, mlp_ratio, drop) for d, h in zip(depths, heads)])
        self.norm = nn.LayerNorm(embed)
        self.conv_after = nn.Conv2d(embed, embed, 3, 1, 1)
        self.conv_before = nn.Sequential(nn.Conv2d(embed, 64, 3, 1, 1), nn.LeakyReLU(0.2, True))
        up = []
        for _ in range(int(math.log2(scale))):
            up += [nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*up)
        self.conv_last = nn.Conv2d(64, in_ch, 3, 1, 1)
        self.apply(self._init)
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = (self.ws - H % self.ws) % self.ws, (self.ws - W % self.ws) % self.ws
        xp = F.pad(x, (0, pw, 0, ph), mode='replicate') if (ph or pw) else x
        Hp, Wp = xp.shape[2:]
        base = F.interpolate(xp, scale_factor=self.scale, mode='bicubic', align_corners=False)
        f = self.conv_first(xp)
        t = f.flatten(2).transpose(1, 2)
        for layer in self.layers:
            t = layer(t, Hp, Wp)
        t = self.norm(t).transpose(1, 2).reshape(B, -1, Hp, Wp)
        f = self.conv_after(t) + f
        out = self.conv_last(self.upsample(self.conv_before(f))) + base
        return out[:, :, :H * self.scale, :W * self.scale]

def set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)

def enable_mc_dropout(model):
    # eval mode everywhere, but keep Dropout layers active (Monte Carlo Dropout)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

import numpy as np

def as_chw(arr, num_ch):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[0] == num_ch:
        return arr
    if arr.shape[-1] == num_ch:
        return arr.transpose(2, 0, 1)
    raise ValueError(f'Cannot find {num_ch} channels in array of shape {arr.shape}')

def load_model(path, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = SwinIR(**ck['cfg'])
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    return model, ck['norm']

def _tile_starts(n, tile, stride):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, stride))
    s.append(n - tile)
    return sorted(set(s))

def _feather(n, ramp, floor=0.05):
    r = min(ramp, n // 2)
    v = torch.ones(n)
    if r > 0:
        t = (torch.arange(r).float() + 0.5) / r
        t = floor + (1 - floor) * 0.5 * (1 - torch.cos(math.pi * t))
        v[:r] = t
        v[-r:] = torch.flip(t, [0])
    return v

@torch.no_grad()
def super_resolve(model, norm, lr, mc_passes=5, tile=64, overlap=16, batch_tiles=4, fp16=True):
    # lr: numpy array (C,H,W) in the ORIGINAL physical units. Returns dict of numpy arrays.
    # mc_passes=0 -> deterministic (dropout off). mc_passes>=2 -> Monte Carlo Dropout.
    device = next(model.parameters()).device
    s = model.scale
    assert tile % model.ws == 0, 'tile must be a multiple of the window size (8)'
    assert 0 <= overlap < tile
    lo = torch.tensor(norm['lr_lo'], dtype=torch.float32).view(1, -1, 1, 1)   # INPUT uses LR stats
    hi = torch.tensor(norm['lr_hi'], dtype=torch.float32).view(1, -1, 1, 1)
    x = torch.from_numpy(np.nan_to_num(np.asarray(lr, dtype=np.float32))).unsqueeze(0)
    x = (x - lo) / (hi - lo)
    _, C, H, W = x.shape
    ph, pw = max(0, tile - H), max(0, tile - W)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
    Hp, Wp = x.shape[2:]
    coords = [(y, xx) for y in _tile_starts(Hp, tile, tile - overlap) for xx in _tile_starts(Wp, tile, tile - overlap)]
    win = (_feather(tile * s, overlap * s)[:, None] * _feather(tile * s, overlap * s)[None, :])
    acc_m = torch.zeros(C, Hp * s, Wp * s)
    acc_s = torch.zeros(C, Hp * s, Wp * s)
    acc_w = torch.zeros(Hp * s, Wp * s)
    if mc_passes and mc_passes > 0:
        enable_mc_dropout(model)
    else:
        model.eval()
    use_amp = fp16 and device.type == 'cuda'
    for i in range(0, len(coords), batch_tiles):
        cb = coords[i:i + batch_tiles]
        xb = torch.cat([x[:, :, y:y + tile, xx:xx + tile] for y, xx in cb]).to(device)
        n_pass = mc_passes if (mc_passes and mc_passes > 0) else 1
        preds = []
        for _ in range(n_pass):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                preds.append(model(xb).float())
        P = torch.stack(preds)
        mean = P.mean(0)
        std = P.std(0, correction=1) if n_pass > 1 else torch.zeros_like(mean)
        mean, std = mean.cpu(), std.cpu()
        for j, (y, xx) in enumerate(cb):
            ys, xs = slice(y * s, (y + tile) * s), slice(xx * s, (xx + tile) * s)
            acc_m[:, ys, xs] += mean[j] * win
            acc_s[:, ys, xs] += std[j] * win
            acc_w[ys, xs] += win
    model.eval()
    mean = (acc_m / acc_w)[:, :H * s, :W * s]
    std = (acc_s / acc_w)[:, :H * s, :W * s]
    hlo = torch.tensor(norm['hr_lo'], dtype=torch.float32).view(-1, 1, 1)     # OUTPUT uses HR stats
    hhi = torch.tensor(norm['hr_hi'], dtype=torch.float32).view(-1, 1, 1)
    rng = hhi - hlo
    out = mean * rng + hlo
    vmin = torch.tensor(norm['vmin'], dtype=torch.float32).view(-1, 1, 1)
    vmax = torch.tensor(norm['vmax'], dtype=torch.float32).view(-1, 1, 1)
    out = torch.max(torch.min(out, vmax), vmin)
    return dict(mean=out.numpy().astype(np.float32),
                std=(std * rng).numpy().astype(np.float32),
                uncertainty=std.mean(0).numpy().astype(np.float32))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Sentinel-2 4x super-resolution with MC Dropout uncertainty')
    ap.add_argument('--input', required=True, help='LR .npy file, shape (C,H,W) or (H,W,C), original units')
    ap.add_argument('--model', required=True, help='path to the final .pth')
    ap.add_argument('--out', default='sr_out', help='output prefix')
    ap.add_argument('--passes', type=int, default=5, help='MC Dropout passes (0 = deterministic)')
    ap.add_argument('--tile', type=int, default=64)
    ap.add_argument('--overlap', type=int, default=16)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    model, norm = load_model(a.model, a.device)
    lr = as_chw(np.load(a.input), norm['num_ch']).astype(np.float32)
    res = super_resolve(model, norm, lr, mc_passes=a.passes, tile=a.tile, overlap=a.overlap)
    np.save(a.out + '_sr.npy', res['mean'])
    np.save(a.out + '_std.npy', res['std'])
    np.save(a.out + '_uncertainty.npy', res['uncertainty'])
    print('SR image      :', a.out + '_sr.npy', res['mean'].shape, '(C,H*4,W*4, float32, HR / NAIP-style 0-255 units)')
    print('Per-band std  :', a.out + '_std.npy', res['std'].shape)
    print('Uncertainty   :', a.out + '_uncertainty.npy', res['uncertainty'].shape, '(mean std over bands, normalized)')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imsave(a.out + '_uncertainty.png', res['uncertainty'], cmap='inferno')
        print('Heatmap image :', a.out + '_uncertainty.png')
    except Exception as e:
        print('Heatmap png skipped:', e)
