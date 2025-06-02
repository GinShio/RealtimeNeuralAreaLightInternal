import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import json
import os
import threading
import time
import numpy as np
from tqdm import tqdm
import OpenEXR
import Imath
import wandb

torch.set_float32_matmul_precision("high")


class MollifiedDataset:
    def __init__(self, base_dir, num_steps):
        self.base_dir = base_dir
        self.num_steps = num_steps

        with open(os.path.join(base_dir, "data_gen_config.json"), "r") as f:
            self.config = json.load(f)

        self.batch_size = self.config["batch_size"]
        self.first_phase_shard_size = self.config["first_phase_shard_size"]
        self.mollified_shard_count = self.config["mollification_shard_count"]

        self.files = [
            os.path.join(base_dir, f"first_phase_data-mollified.shard-{i}.bin")
            for i in range(self.mollified_shard_count)
        ]

        self.total_samples = num_steps * self.batch_size
        self.sample_limit_per_shard = self.total_samples // self.mollified_shard_count

        self.shard_index = 0
        self.sample_index = 0
        self.sample_count = 0
        self.current_shard = None
        self.next_shard = None
        self.prefetch_thread = None
        self.sample_count = 0

    def __iter__(self):
        self.shard_index = 0
        self.sample_index = 0
        self.sample_count = 0
        self._load_next_shard()
        return self

    def _start_prefetch(self):
        if self.shard_index < len(self.files):
            path = self.files[self.shard_index]
            def load():
                self.next_shard = np.fromfile(path, dtype=np.float32).reshape(-1, 17)
            self.prefetch_thread = threading.Thread(target=load)
            self.prefetch_thread.start()

    def _load_next_shard(self):
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
            self.current_shard = self.next_shard
            self.next_shard = None
            self.prefetch_thread = None
        else:
            if self.shard_index >= len(self.files):
                self.current_shard = None
                return
            path = self.files[self.shard_index]
            self.current_shard = np.fromfile(path, dtype=np.float32).reshape(-1, 17)

        self.sample_index = 0
        self.sample_count = 0
        self.shard_index += 1
        self._start_prefetch()

    def __next__(self):
        if self.current_shard is None:
            raise StopIteration

        if self.sample_count >= self.sample_limit_per_shard:
            self._load_next_shard()
            if self.current_shard is None:
                raise StopIteration

        if self.sample_index + self.batch_size > len(self.current_shard):
            self.sample_index = 0

        batch = self.current_shard[
            self.sample_index : self.sample_index + self.batch_size
        ]
        self.sample_index += self.batch_size
        self.sample_count += self.batch_size

        material = torch.tensor(batch[:, 0:8], dtype=torch.float32)
        wi = torch.tensor(batch[:, 8:11], dtype=torch.float32)
        wo = torch.tensor(batch[:, 11:14], dtype=torch.float32)
        brdf = torch.tensor(batch[:, 14:17], dtype=torch.float32)

        return material, wi, wo, brdf



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
            self.next_shard = np.fromfile(filepath, dtype=np.float32).reshape(-1, 17)

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
            self.current_shard = np.fromfile(filepath, dtype=np.float32).reshape(-1, 17)

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
        wi = torch.tensor(batch[:, 8:11], dtype=torch.float32)
        wo = torch.tensor(batch[:, 11:14], dtype=torch.float32)
        brdf = torch.tensor(batch[:, 14:17], dtype=torch.float32)

        return material, wi, wo, brdf


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

        self.second_material_data_component_size = 8
        self.second_shard_data_component_size = 9

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
                self.next_shard = np.fromfile(path, dtype=np.float32).reshape(
                    -1,
                    self.texture_total_pixel_size,
                    self.second_shard_data_component_size,
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
            self.current_shard = np.fromfile(path, dtype=np.float32).reshape(
                -1, self.texture_total_pixel_size, self.second_shard_data_component_size
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

        sample = self.current_shard[self.sample_index]  # (N, 9)
        wi = torch.tensor(sample[:, 0:3], dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
        wo = torch.tensor(sample[:, 3:6], dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
        brdf = torch.tensor(sample[:, 6:9], dtype=torch.float32).unsqueeze(
            0
        )  # (1, N, 3)

        self.sample_index += 1
        return wi, wo, brdf


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


def transform_frame_function(transform_input, wi, wo):
    # transform_input: (B, 12)
    B = transform_input.shape[0]
    result = []
    for i in range(2):
        normal = transform_input[:, i * 6 + 0 : i * 6 + 3]  # (B, 3)
        tangent = transform_input[:, i * 6 + 3 : i * 6 + 6]  # (B, 3)
        bitangent = torch.cross(normal, tangent, dim=-1)  # (B, 3)

        TBN = torch.stack([tangent, bitangent, normal], dim=-1)  # (B, 3, 3)
        TBN = TBN.transpose(1, 2)  # (B, 3, 3)

        wi_tbn = torch.bmm(TBN, wi.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        wo_tbn = torch.bmm(TBN, wo.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        result.append(wi_tbn)
        result.append(wo_tbn)

    return torch.cat(result, dim=-1)  # (B, 12)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(8 + 12, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent, wi, wo):
        tf_input = self.tanh(self.fc1(latent))  # (B, 12)
        tf_output = transform_frame_function(tf_input, wi, wo)  # (B, 12)
        x = torch.cat([latent, tf_output], dim=-1)  # (B, 20)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.sigmoid(self.fc5(x))  # (B, 3)


def log1p4(x):
    for _ in range(4):
        x = torch.log1p(x)
    return x


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
            ],
        }
    }

    # mkdir
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_json, f, indent=2)


def train_first_phase(
    data_dir, output_dir, num_steps=10000, lr=1e-3, log_interval=100, device="cuda"
):
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min= lr / 10
    )
    loss_fn = nn.L1Loss()

    # mollified_data = MollifiedDataset(data_dir, num_steps=num_steps // 15)
    mollified_data = MollifiedDataset(data_dir, num_steps=num_steps)
    normal_data = NormalDataset(data_dir)

    data = iter(mollified_data)
    for i in range(100):
        next(data)

    data = iter(mollified_data)
    phase = "mollified"

    for step in tqdm(range(num_steps), desc="First Phase Training"):
        try:
            material, wi, wo, brdf = next(data)
        except StopIteration:
            data = iter(normal_data)
            material, wi, wo, brdf = next(data)

        material = material.to(device)
        wi = wi.to(device)
        wo = wo.to(device)
        brdf = brdf.to(device)
        brdf_log = log1p4(brdf)

        latent = encoder(material)
        pred = decoder(latent, wi, wo)
        loss = loss_fn(pred, brdf_log)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0 or step == num_steps - 1:
            # Log to Weights & Biases
            wandb.log(
                {
                    "step": step,
                    "1st phase/loss": loss.item(),
                    "phase": phase,
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
    material_data = np.fromfile(material_path, dtype=np.float32).reshape(
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
    num_steps=1000,
    lr=2e-4,
    log_interval=100,
    save_interval=10000,
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

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(os.path.join(output_dir, "decoder.pth")))
    optimizer = torch.optim.Adam(list(decoder.parameters()) + [latent_texture], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min= lr / 10
    )
    loss_fn = nn.L1Loss()

    second_data = SecondPhaseDataset(data_dir)
    data = iter(second_data)

    for step in tqdm(range(num_steps), desc="Second Phase Training"):
        try:
            wi, wo, brdf = next(data)
        except StopIteration:
            data = iter(second_data)
            wi, wo, brdf = next(data)

        wi = wi.squeeze(0).to(device)  # (N, 3)
        wo = wo.squeeze(0).to(device)  # (N, 3)
        brdf = brdf.squeeze(0).to(device)  # (N, 3)
        brdf_log = log1p4(brdf)

        latent = latent_texture  # (N, 8)

        pred = decoder(latent, wi, wo)  # (N, 3)
        loss = loss_fn(pred, brdf_log)

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


# training MLP for DisneyBRDF
def train(steps):
    data_dir = "data/disney-rtnam"
    output_dir = "output/disney-rtnam"

    wandb.init(project="Realtime Neural Area Light")

    start = time.time()

    # first phase
    train_first_phase(data_dir, output_dir, num_steps=steps, device="cuda")

    # generate latent texture
    generate_latent_texture(data_dir, output_dir, device="cuda")

    # second phase
    train_second_phase(
        data_dir=data_dir,
        output_dir=output_dir,
        num_steps=steps // 2,
        log_interval=100,
        save_interval=steps // 100,
        device="cuda",
    )

    end = time.time()
    elapsed = end - start
    print(f"Training completed in {format_duration(elapsed)} seconds.")

    wandb.finish()
