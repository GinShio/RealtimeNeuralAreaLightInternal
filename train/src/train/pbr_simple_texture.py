import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import json
import os
import math
import threading
import time
import numpy as np
import slangpy as spy
from tqdm import tqdm
import OpenEXR
import Imath
import wandb

torch.set_float32_matmul_precision("high")


module = spy.TorchModule.load_from_file(device="cuda", path="shaders/pbr_brdf.slang")


class NormalDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        with open(os.path.join(base_dir, "data_gen_config.json"), "r") as f:
            self.config = json.load(f)

        self.batch_size = self.config["batch_size"]
        self.first_phase_shard_size = self.config["first_phase_shard_size"]
        self.first_phase_shard_count = self.config["first_phase_shard_count"]

        self.files = [
            os.path.join(base_dir, f"first_phase_data.shard-{i}.bin")
            for i in range(self.first_phase_shard_count)
        ]

        self.shard_index = 0
        self.sample_index = 0
        self.current_shard = None
        self.next_shard = None
        self.prefetch_thread = None

    def __iter__(self):
        self.shard_index = 0
        self.sample_index = 0
        self._load_next_shard()
        return self

    def _start_prefetch(self):
        next_index = (self.shard_index + 1) % len(self.files)
        filepath = self.files[next_index]

        def load():
            self.next_shard = np.fromfile(filepath, dtype=np.float16).reshape(-1, 26)

        self.prefetch_thread = threading.Thread(target=load)
        self.prefetch_thread.start()

    def _load_next_shard(self):
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
            self.current_shard = self.next_shard
            self.next_shard = None
            self.prefetch_thread = None
        else:
            filepath = self.files[self.shard_index]
            self.current_shard = np.fromfile(filepath, dtype=np.float16).reshape(-1, 26)

        self.sample_index = 0
        self.shard_index = (self.shard_index + 1) % len(self.files)
        self._start_prefetch()

    def __next__(self):
        if self.current_shard is None or self.sample_index + self.batch_size > len(
            self.current_shard
        ):
            self._load_next_shard()

        batch = self.current_shard[
            self.sample_index : self.sample_index + self.batch_size
        ]
        self.sample_index += self.batch_size

        material = torch.tensor(batch[:, 0:8], dtype=torch.float32)
        wo = torch.tensor(batch[:, 8:11], dtype=torch.float32)
        v1 = torch.tensor(batch[:, 11:14], dtype=torch.float32)
        v2 = torch.tensor(batch[:, 14:17], dtype=torch.float32)
        v3 = torch.tensor(batch[:, 17:20], dtype=torch.float32)
        v4 = torch.tensor(batch[:, 20:23], dtype=torch.float32)
        D = torch.tensor(batch[:, 23:26], dtype=torch.float32)

        return material, wo, v1, v2, v3, v4, D


class SecondPhaseDataset:
    def __init__(self, base_dir, batch_size=1):
        self.base_dir = base_dir
        self.batch_size = batch_size

        with open(os.path.join(base_dir, "data_gen_config.json"), "r") as f:
            self.config = json.load(f)

        self.second_phase_shard_size = self.config["second_phase_shard_size"]
        self.second_phase_shard_count = self.config["second_phase_shard_count"]
        self.texture_size = self.config["texture_size"]

        self.shard_paths = [
            os.path.join(base_dir, f"second_phase_data.shard-{i}.bin")
            for i in range(self.second_phase_shard_count)
        ]

        # calculate total pixel size
        self.texture_total_pixel_size = 0
        width = self.texture_size
        while width > 0:
            self.texture_total_pixel_size += width * width
            width //= 2

        self.shard_index = 0
        self.sample_index = 0
        self.current_shard = None
        self.next_shard = None
        self.prefetch_thread = None

    def __iter__(self):
        self.shard_index = 0
        self.sample_index = 0
        self._load_next_shard()
        return self

    def _start_prefetch(self):
        if self.shard_index < len(self.shard_paths):
            path = self.shard_paths[self.shard_index]

            def load():
                self.next_shard = np.fromfile(path, dtype=np.float16).reshape(
                    -1,
                    self.texture_total_pixel_size,
                    18,
                )

            self.prefetch_thread = threading.Thread(target=load)
            self.prefetch_thread.start()

    def _load_next_shard(self):
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
            self.current_shard = self.next_shard
            self.next_shard = None
            self.prefetch_thread = None
        else:
            if self.shard_index >= len(self.shard_paths):
                self.current_shard = None
                return
            path = self.shard_paths[self.shard_index]
            self.current_shard = np.fromfile(path, dtype=np.float16).reshape(
                -1, self.texture_total_pixel_size, 18
            )

        self.sample_index = 0
        self.shard_index += 1
        self._start_prefetch()

    def __next__(self):
        if self.current_shard is None:
            raise StopIteration

        if self.sample_index >= self.current_shard.shape[0]:
            self._load_next_shard()
            if self.current_shard is None:
                raise StopIteration

        sample = self.current_shard[self.sample_index]
        wo = torch.tensor(sample[:, 0:3], dtype=torch.float32)
        v1 = torch.tensor(sample[:, 3:6], dtype=torch.float32)
        v2 = torch.tensor(sample[:, 6:9], dtype=torch.float32)
        v3 = torch.tensor(sample[:, 9:12], dtype=torch.float32)
        v4 = torch.tensor(sample[:, 12:15], dtype=torch.float32)
        D = torch.tensor(sample[:, 15:18], dtype=torch.float32)

        self.sample_index += 1
        return wo, v1, v2, v3, v4, D


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def transform_frame_function(transform_input, wo, v1, v2, v3, v4):
    # transform_input: (B, 12)
    B = transform_input.shape[0]
    result = []
    for i in range(2):
        normal = transform_input[:, i * 6 + 0 : i * 6 + 3]  # (B, 3)
        tangent = transform_input[:, i * 6 + 3 : i * 6 + 6]  # (B, 3)
        bitangent = torch.cross(normal, tangent, dim=-1)  # (B, 3)

        TBN = torch.stack([tangent, bitangent, normal], dim=-1)  # (B, 3, 3)
        TBN = TBN.transpose(1, 2)  # (B, 3, 3)

        wo_tbn = torch.bmm(TBN, wo.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        v1_tbn = torch.bmm(TBN, v1.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        v2_tbn = torch.bmm(TBN, v2.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        v3_tbn = torch.bmm(TBN, v3.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        v4_tbn = torch.bmm(TBN, v4.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        result.append(wo_tbn)
        result.append(v1_tbn)
        result.append(v2_tbn)
        result.append(v3_tbn)
        result.append(v4_tbn)

    return torch.cat(result, dim=-1)  # (B, 30)


class Decoder(nn.Module):
    def __init__(self, texture_size):
        super().__init__()

        self.maxMipLevel = int(math.log2(texture_size))

        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(8 + 30, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 3 + 2 * 9 + 9 + 9)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, latent, wo, v1, v2, v3, v4):
        tf_input = self.tanh(self.fc1(latent))  # (B, 12)
        tf_output = transform_frame_function(tf_input, wo, v1, v2, v3, v4)  # (B, 30)
        x = torch.cat([latent, tf_output], dim=-1)  # (B, 38)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.fc8(x)

        D = x[:, :3]  # (B, 3)
        uv = x[:, 3:3 + 2 * 9].reshape(-1, 9, 2)  # (B, 9, 2)
        miplevel = x[:, 3 + 2 * 9:3 + 2 * 9 + 9]  # (B, 9)
        weight = x[:, 3 + 2 * 9 + 1:]  # (B, 9)

        D = torch.exp(D - 3.0)
        uv = torch.sigmoid(uv)
        mipLevel = torch.sigmoid(miplevel) * self.maxMipLevel
        weight = torch.sigmoid(weight)

        return D, uv, mipLevel, weight


def calculate_influence_edge(
    width: int,
    uv: torch.Tensor,             # (1, N, 2)
    sample_uv: torch.Tensor,      # (B, 2)
    sample_miplevel: torch.Tensor # (B, 1)
) -> torch.Tensor:
    """
    EDGE (CLAMP_TO_EDGE) モードでの寄与係数をブロードキャストで求める。
    戻り値: (B, N, 1)
    ─────────────────────────────────────────────────────────
    * ベース幅は power-of-two.
    * 0.5-texel オフセット規約 (中心 = (i+0.5)/size).
    * trilinear: lod0 = floor(lod), lod1 = lod0+1, weight = frac(lod).
    """
    assert (width & (width - 1)) == 0, "width は 2 の冪のみを想定"

    # ── 前処理 ──────────────────────────────────────────────
    max_level = int(math.log2(width))
    B, N, _   = uv.shape
    device, dt = uv.device, uv.dtype

    # LOD → (lod0, lod1, frac)
    lod_f = sample_miplevel.squeeze(-1).clamp(0, max_level)      # (B,)
    lod0  = lod_f.floor().to(torch.long)                         # (B,)
    lod1  = torch.clamp(lod0 + 1, max=max_level)                 # (B,)
    frac  = (lod_f - lod0)                                       # (B,)

    levels = torch.stack((lod0, lod1), dim=1)                    # (B, 2)
    w_lod  = torch.stack((1.0 - frac, frac), dim=1)              # (B, 2)
    # lod0==lod1 (整数 LOD or max_level) の場合は (1, 0) に修正
    same   = (levels[:, 0] == levels[:, 1]).unsqueeze(1)         # (B,1)
    w_lod  = torch.where(same, torch.tensor([1., 0.], device=device), w_lod)

    # 各 LOD の解像度 (float)
    size = (width / (2.0 ** levels.float())).to(dt)              # (B, 2)

    # ── サンプル座標を texel 空間へ ─────────────────────────
    samp = sample_uv.unsqueeze(1).repeat(1, 2, 1)                # (B, 2, 2)
    texel = samp * size.unsqueeze(-1) - 0.5                      # (B, 2, 2)
    base  = texel.floor()                                        # (B, 2, 2)
    frac_uv = texel - base                                       # (B, 2, 2)

    i0, j0 = base[..., 0], base[..., 1]                          # (B, 2)
    fu, fv = frac_uv[..., 0], frac_uv[..., 1]                    # (B, 2)

    # ── 4 近傍 (dx,dy)=(0,0)(1,0)(0,1)(1,1) を展開 ──────────
    dx = torch.tensor([0, 1, 0, 1], device=device, dtype=dt)     # (4,)
    dy = torch.tensor([0, 0, 1, 1], device=device, dtype=dt)

    ix = (i0.unsqueeze(-1) + dx).clamp(0, size.unsqueeze(-1) - 1)  # (B, 2, 4)
    iy = (j0.unsqueeze(-1) + dy).clamp(0, size.unsqueeze(-1) - 1)  # (B, 2, 4)

    u_min, v_min = ix / size.unsqueeze(-1), iy / size.unsqueeze(-1)
    u_max, v_max = (ix + 1.0) / size.unsqueeze(-1), (iy + 1.0) / size.unsqueeze(-1)

    # bilinear ウェイト (B, 2, 4)
    w_bilinear = torch.stack((
        (1 - fu) * (1 - fv),  # (0,0)
        fu * (1 - fv),        # (1,0)
        (1 - fu) * fv,        # (0,1)
        fu * fv               # (1,1)
    ), dim=-1)

    # 合成ウェイト (B, 2, 4)
    w_total = w_lod.unsqueeze(-1) * w_bilinear

    # ── uv 群との包含テスト (完全ブロードキャスト) ────────────
    uv_u, uv_v = uv[..., 0], uv[..., 1]                          # (B, N)
    uv_u = uv_u.unsqueeze(1).unsqueeze(2)                        # (B,1,1,N)
    uv_v = uv_v.unsqueeze(1).unsqueeze(2)

    def inside(x, lo, hi):
        return (x >= lo.unsqueeze(-1)) & (x < hi.unsqueeze(-1))   # (B,2,4,N)

    mask = inside(uv_u, u_min, u_max) & inside(uv_v, v_min, v_max)
    influence = (mask.to(dt) * w_total.unsqueeze(-1)).sum(dim=(1, 2))  # (B,N)

    return influence.unsqueeze(-1)                               # (B, N, 1)


def calculate_D(
        baseColor,  # (B, 3)
        metallic,  # (B, 1)
        roughness,  # (B, 1)
        normal,  # (B, 3)
        wo,  # (B, 3)
        v1,  # (B, 3)
        v2,  # (B, 3)
        v3,  # (B, 3)
        v4,  # (B, 3)
        uv,  # (N, 2)
):
    """
    brdfにG項を掛けた値を計算する
    """
    # v1, v2, v3, v4をuvで補間してLposを計算
    Lpos = (
        uv[:, 0].unsqueeze(-1) * v1.unsqueeze(1) +
        uv[:, 1].unsqueeze(-1) * v2.unsqueeze(1) +
        (1 - uv[:, 0]).unsqueeze(-1) * v3.unsqueeze(1) +
        (1 - uv[:, 1]).unsqueeze(-1) * v4.unsqueeze(1)
    )  # (N, B, 3)
    # 正規化してwiを計算
    wi = F.normalize(Lpos, p=2, dim=-1)  # (N, B, 3)
    return module.PbrBRDF(baseColor, metallic, roughness, normal, wo, wi)


def write_exr(filename, data):  # data: H x W x C (float16)
    height, width, channels = data.shape
    assert channels == 4, "Only 4-channel EXR writing supported"

    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header["channels"] = dict(R=half_chan, G=half_chan, B=half_chan, A=half_chan)

    out = OpenEXR.OutputFile(filename, header)
    r = data[:, :, 0].astype(np.float16).tobytes()
    g = data[:, :, 1].astype(np.float16).tobytes()
    b = data[:, :, 2].astype(np.float16).tobytes()
    a = data[:, :, 3].astype(np.float16).tobytes()
    out.writePixels({"R": r, "G": g, "B": b, "A": a})
    out.close()


def read_exr(filename, width, height, channels=("R", "G", "B", "A")):
    exr_file = OpenEXR.InputFile(filename)

    # float16
    pt = Imath.PixelType(Imath.PixelType.HALF)

    data = []
    for c in channels:
        raw = exr_file.channel(c, pt)
        arr = np.frombuffer(raw, dtype=np.float16).reshape(height, width)
        data.append(arr)

    return np.stack(data, axis=-1)


def save_model_as_json(model, path):
    def layer_to_json(layer):
        return {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "weights": layer.weight.detach().cpu().numpy().flatten().tolist(),
            "bias": layer.bias.detach().cpu().numpy().flatten().tolist(),
        }

    model_json = {
        "model": {
            "layers": [
                layer_to_json(model.fc1),
                layer_to_json(model.fc2),
                layer_to_json(model.fc3),
                layer_to_json(model.fc4),
                layer_to_json(model.fc5),
                layer_to_json(model.fc6),
                layer_to_json(model.fc7),
                layer_to_json(model.fc8),
            ],
        }
    }

    # mkdir
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_json, f, indent=2)


def log1p4(x):
    x = torch.log1p(x)
    x = torch.log1p(x)
    x = torch.log1p(x)
    x = torch.log1p(x)
    return x

def log1p4Inv(x):
    x = torch.expm1(x)
    x = torch.expm1(x)
    x = torch.expm1(x)
    x = torch.expm1(x)
    return x


def train_first_phase(
    data_dir, output_dir, num_steps, lr, log_interval=100, device="cuda"
):
    with open(os.path.join(data_dir, "data_gen_config.json"), "r") as f:
        config = json.load(f)
    texture_size = config["texture_size"]

    encoder = Encoder().to(device)
    decoder = Decoder(texture_size).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min= lr / 10
    )
    D_loss_fn = nn.L1Loss()
    influence_loss_fn = nn.L2Loss()

    # roughness_clipped_data = RoughnessClippedDataset(data_dir, num_steps=num_steps // 5)
    normal_data = NormalDataset(data_dir)

    data = iter(normal_data)

    W = texture_size
    H = texture_size

    # u座標とv座標を生成（それぞれ0〜1の範囲）
    u = torch.linspace(0, 1, W)
    v = torch.linspace(0, 1, H)

    # メッシュグリッドを作成
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # 形状：(H, W)

    # uuとvvをチャンネル次元にまとめる（H, W, 2）
    uv_map = torch.stack((uu, vv), dim=-1)  # 形状：(1024, 1024, 2)

    # reshapeして(1024*1024, 2)へ変形
    uv_sample = uv_map.reshape(-1, 2)  # 形状：(1048576, 2)

    for step in tqdm(range(num_steps), desc="First Phase Training"):
        try:
            material, wo, v1, v2, v3, v4, D = next(data)
        except StopIteration:
            data = iter(normal_data)
            material, wo, v1, v2, v3, v4, D = next(data)

        material = material.to(device)
        wo = wo.to(device)
        v1 = v1.to(device)
        v2 = v2.to(device)
        v3 = v3.to(device)
        v4 = v4.to(device)
        D = D.to(device)

        latent = encoder(material)
        D_pred, uv, mipLevel, weight = decoder(
            latent,
            wo,
            F.normalize(v1, p=2, dim=-1),
            F.normalize(v2, p=2, dim=-1),
            F.normalize(v3, p=2, dim=-1),
            F.normalize(v4, p=2, dim=-1),
        )

        D_pred_log = log1p4(D_pred)
        D_loss = D_loss_fn(D_pred_log, D)

        influence_pred = torch.zeros_like(uv_sample)
        for i in range(9):
            influence_pred += calculate_influence_edge(
                texture_size,
                uv_sample.unsqueeze(0).to(device),  # (1, 1048576, 2)
                uv[:, i, :].unsqueeze(0).to(device),  # (B, 2)
                mipLevel[:, i].unsqueeze(1).to(device)  # (B, 1)
            ) * weight[:, i].unsqueeze(1).to(device)

        baseColor = material[:, :3]
        metallic = material[:, 3:4]
        roughness = material[:, 4:5]
        normal = material[:, 5:8] * 2 - 1
        normal = F.normalize(normal, p=2, dim=-1)
        influence_target = calculate_D(
            baseColor,
            metallic,
            roughness,
            normal,
            wo,
            v1,
            v2,
            v3,
            v4,
            uv_sample.unsqueeze(0).to(device),  # (1, 1048576, 2)
        ) / log1p4Inv(D)

        influence_loss = influence_loss_fn(influence_pred, influence_target)

        loss = D_loss + influence_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0 or step == num_steps - 1:
            # Log to Weights & Biases
            DMean = D.mean().item()
            DVar = D.var().item()
            DMax = D.max().item()
            nonZeroDCount = (D != 0).sum().item()
            nonZeroDMean = D[D != 0].mean().item() if nonZeroDCount > 0 else 0.0
            nonZeroDVar = D[D != 0].var().item() if nonZeroDCount > 0 else 0.0
            wandb.log(
                {
                    "step": step,
                    "1st phase/loss": loss.item(),
                    "1st phase/D_mean": DMean,
                    "1st phase/D_var": DVar,
                    "1st phase/D_max": DMax,
                    "1st phase/non_zero_D_count": nonZeroDCount,
                    "1st phase/non_zero_D_mean": nonZeroDMean,
                    "1st phase/non_zero_D_var": nonZeroDVar,
                    "1st phase/learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(output_dir, "encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(output_dir, "decoder.pth"))


def generate_latent_texture(data_dir, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(data_dir, "data_gen_config.json"), "r") as f:
        config = json.load(f)

    texture_size = config["texture_size"]
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(os.path.join(output_dir, "encoder.pth")))
    encoder.eval()

    texture_total_pixel_size = 0
    mip_sizes = []
    width = texture_size
    while width > 0:
        mip_sizes.append(width)
        texture_total_pixel_size += width * width
        width //= 2

    material_path = os.path.join(data_dir, "second_phase_data.material.bin")
    material_data = np.fromfile(material_path, dtype=np.float16).reshape(
        texture_total_pixel_size, 8
    )
    material_tensor = torch.tensor(material_data, dtype=torch.float32).to(device)

    # save pre fine tuning models and latent textures
    output_dir_pre = os.path.join(output_dir, "pre")

    # Save decoder as JSON
    decoder = Decoder()
    decoder.load_state_dict(torch.load(os.path.join(output_dir, "decoder.pth")))
    json_path = os.path.join(output_dir_pre, "network.json")
    save_model_as_json(decoder, json_path)

    with torch.no_grad():
        latent_tensor = encoder(material_tensor).cpu().numpy()

    offset = 0
    for level, size in enumerate(mip_sizes):
        count = size * size
        mip = latent_tensor[offset : offset + count].reshape(size, size, 8)
        offset += count

        # Output 2 exr files
        mip_f16_a = mip[:, :, :4].astype(np.float16)  # (H, W, 4)
        mip_f16_b = mip[:, :, 4:].astype(np.float16)  # (H, W, 4)

        write_exr(
            os.path.join(output_dir_pre, f"latent-texture-0.mip{level}.exr"),
            mip_f16_a,
        )
        write_exr(
            os.path.join(output_dir_pre, f"latent-texture-1.mip{level}.exr"),
            mip_f16_b,
        )


def train_second_phase(
    data_dir,
    output_dir,
    num_steps,
    lr,
    log_interval=100,
    save_interval=1000,
    device="cuda",
):
    with open(os.path.join(data_dir, "data_gen_config.json"), "r") as f:
        config = json.load(f)
    texture_size = config["texture_size"]
    texture_total_pixel_size = 0
    mip_sizes = []
    width = texture_size
    while width > 0:
        mip_sizes.append(width)
        texture_total_pixel_size += width * width
        width //= 2

    # Load latent texture from all mip levels and concatenate
    latent_list = []
    for level in range(len(mip_sizes)):
        width = mip_sizes[level]
        latent_a = read_exr(
            os.path.join(output_dir, "pre", f"latent-texture-0.mip{level}.exr"),
            width,
            width,
        ).astype(np.float32)
        latent_b = read_exr(
            os.path.join(output_dir, "pre", f"latent-texture-1.mip{level}.exr"),
            width,
            width,
        ).astype(np.float32)
        latent = np.concatenate([latent_a, latent_b], axis=-1).reshape(-1, 8)
        latent_list.append(latent)

    latent_texture = torch.nn.Parameter(
        torch.tensor(np.concatenate(latent_list, axis=0), dtype=torch.float32).to(
            device
        )
    )

    material_path = os.path.join(data_dir, "second_phase_data.material.bin")
    material_data = np.fromfile(material_path, dtype=np.float16).reshape(
        texture_total_pixel_size, 8
    )
    material_tensor = torch.tensor(material_data, dtype=torch.float32).to(device)


    decoder = Decoder(texture_size).to(device)
    decoder.load_state_dict(torch.load(os.path.join(output_dir, "decoder.pth")))
    optimizer = torch.optim.Adam(list(decoder.parameters()) + [latent_texture], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min= lr / 10
    )
    D_loss_fn = nn.L1Loss()
    influence_loss_fn = nn.L2Loss()

    second_data = SecondPhaseDataset(data_dir)
    data = iter(second_data)

    W = texture_size
    H = texture_size

    # u座標とv座標を生成（それぞれ0〜1の範囲）
    u = torch.linspace(0, 1, W)
    v = torch.linspace(0, 1, H)

    # メッシュグリッドを作成
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # 形状：(H, W)

    # uuとvvをチャンネル次元にまとめる（H, W, 2）
    uv_map = torch.stack((uu, vv), dim=-1)  # 形状：(1024, 1024, 2)

    # reshapeして(1024*1024, 2)へ変形
    uv_sample = uv_map.reshape(-1, 2)  # 形状：(1048576, 2)

    for step in tqdm(range(num_steps), desc="Second Phase Training"):
        try:
            wo, v1, v2, v3, v4, D = next(data)
        except StopIteration:
            data = iter(second_data)
            wo, v1, v2, v3, v4, D = next(data)

        wo = wo.to(device)
        v1 = v1.to(device)
        v2 = v2.to(device)
        v3 = v3.to(device)
        v4 = v4.to(device)
        D = D.to(device)

        latent = latent_texture

        D_pred, uv, mipLevel, weight = decoder(
            latent,
            wo,
            F.normalize(v1, p=2, dim=-1),
            F.normalize(v2, p=2, dim=-1),
            F.normalize(v3, p=2, dim=-1),
            F.normalize(v4, p=2, dim=-1),
        )

        D_pred_log = log1p4(D_pred)
        D_loss = D_loss_fn(D_pred_log, D)

        influence_pred = torch.zeros_like(uv_sample)
        for i in range(9):
            influence_pred += calculate_influence_edge(
                texture_size,
                uv_sample.unsqueeze(0).to(device),  # (1, 1048576, 2)
                uv[:, i, :].unsqueeze(0).to(device),  # (B, 2)
                mipLevel[:, i].unsqueeze(1).to(device)  # (B, 1)
            ) * weight[:, i].unsqueeze(1).to(device)

        baseColor = material_tensor[:3].unsqueeze(0)  # (1, 3)
        metallic = material_tensor[3:4].unsqueeze(0)  # (1, 1)
        roughness = material_tensor[4:5].unsqueeze(0)  # (1, 1)
        normal = material_tensor[5:8].unsqueeze(0)  # (1, 3)
        normal = F.normalize(normal * 2 - 1, p=2, dim=-1)
        influence_target = calculate_D(
            baseColor,
            metallic,
            roughness,
            normal,
            wo,
            v1,
            v2,
            v3,
            v4,
            uv_sample.unsqueeze(0).to(device),  # (1, 1048576, 2)
        ) / log1p4Inv(D)

        influence_loss = influence_loss_fn(influence_pred, influence_target)

        loss = D_loss + influence_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            # Log to Weights & Biases
            wandb.log(
                {
                    "step": step,
                    "2nd phase/loss": loss.item(),
                }
            )

        if step % save_interval == 0:
            # make directory for this step
            step_output_dir = os.path.join(output_dir, f"step_{step}")
            os.makedirs(step_output_dir, exist_ok=True)

            # Save decoder weights
            weights_path = os.path.join(step_output_dir, f"decoder.pth")
            torch.save(decoder.state_dict(), weights_path)

            # Save decoder as JSON
            json_path = os.path.join(step_output_dir, "network.json")
            save_model_as_json(decoder, json_path)

            # Save mipmaps
            latent = latent_texture.detach().cpu().numpy()
            offset = 0
            for level, size in enumerate(mip_sizes):
                count = size * size
                mip = latent[offset : offset + count].reshape(size, size, 8)
                offset += count

                mip_f16_a = mip[:, :, :4].astype(np.float16)
                mip_f16_b = mip[:, :, 4:].astype(np.float16)

                write_exr(
                    os.path.join(step_output_dir, f"latent-texture-0.mip{level}.exr"),
                    mip_f16_a,
                )
                write_exr(
                    os.path.join(step_output_dir, f"latent-texture-1.mip{level}.exr"),
                    mip_f16_b,
                )

    # Log to Weights & Biases
    wandb.log(
        {
            "step": step,
            "2nd phase/loss": loss.item(),
        }
    )

    # Save decoder weights
    weights_path = os.path.join(output_dir, f"decoder.pth")
    torch.save(decoder.state_dict(), weights_path)

    # Save decoder as JSON
    json_path = os.path.join(output_dir, "network.json")
    save_model_as_json(decoder, json_path)

    # Save mipmaps
    latent = latent_texture.detach().cpu().numpy()
    offset = 0
    for level, size in enumerate(mip_sizes):
        count = size * size
        mip = latent[offset : offset + count].reshape(size, size, 8)
        offset += count

        mip_f16_a = mip[:, :, :4].astype(np.float16)
        mip_f16_b = mip[:, :, 4:].astype(np.float16)

        write_exr(
            os.path.join(output_dir, f"latent-texture-0.mip{level}.exr"),
            mip_f16_a,
        )
        write_exr(
            os.path.join(output_dir, f"latent-texture-1.mip{level}.exr"),
            mip_f16_b,
        )


def format_duration(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    mins, sec = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}min {sec}s {millis}ms"


# training MLP for PBR simple
def train(steps):
    data_dir = "data/pbr-simple"
    output_dir = "output/pbr-simple"

    lr_first = 1e-3
    lr_second = 1e-4

    config = {
        "steps": steps,
        "lr_first": lr_first,
        "lr_second": lr_second,
        "layer_count": 8,
        "hidden_size": 64,
    }

    wandb.init(project="Realtime Neural Area Light", config=config)

    start = time.time()

    # first phase
    train_first_phase(
        data_dir=data_dir,
        output_dir=output_dir,
        num_steps=steps,
        lr=lr_first,
        device="cuda",
    )

    # generate latent texture
    generate_latent_texture(data_dir, output_dir, device="cuda")

    # second phase
    train_second_phase(
        data_dir=data_dir,
        output_dir=output_dir,
        num_steps=steps // 10,
        lr=lr_second,
        log_interval=100,
        save_interval=steps // 100,
        device="cuda",
    )

    end = time.time()
    elapsed = end - start
    print(f"Training completed in {format_duration(elapsed)} seconds.")

    wandb.finish()
