import os
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
import slangpy as spy

device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.joinpath("shaders").absolute(),
    ]
)

disneyModule = spy.Module.load_from_file(device, "disney-rtxns.slang")
DisneyBRDF = disneyModule.Disney


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)


def data_gen(
    total_samples=200000,
    seed=42,
    train_ratio=0.95,
):
    np.random.seed(seed)
    base_dir = "data/disney-rtxns"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    n_train = int(total_samples * train_ratio)
    n_test = total_samples - n_train

    def gen_and_save(n_samples, out_dir):
        file_idx = 0
        buffer_inputs = []
        buffer_outputs = []
        buffer_bytes = 0
        generated = 0
        # gen_batch_size = 4096
        gen_batch_size = 1 << 16
        chunk_size_bytes = 1024 * 1024  # 1MB

        while generated < n_samples:
            batch_n = min(gen_batch_size, n_samples - generated)
            # random parameters
            n_dot_l = np.random.rand(batch_n, 1)
            n_dot_v = np.random.rand(batch_n, 1)
            n_dot_h = np.random.rand(batch_n, 1)
            l_dot_h = np.random.rand(batch_n, 1)
            roughness = np.random.rand(batch_n, 1)

            # merge all inputs into a single array
            inputs = np.concatenate(
                [
                    n_dot_l.astype(np.float32),
                    n_dot_v.astype(np.float32),
                    n_dot_h.astype(np.float32),
                    l_dot_h.astype(np.float32),
                    roughness.astype(np.float32),
                ],
                axis=1,
            )  # shape: (batch_n, 5)

            # Calculate BRDF using SlangPy
            outputs = DisneyBRDF(
                n_dot_l.squeeze().astype(np.float32),
                n_dot_v.squeeze().astype(np.float32),
                n_dot_h.squeeze().astype(np.float32),
                l_dot_h.squeeze().astype(np.float32),
                roughness.squeeze().astype(np.float32),
                _result="numpy",
            )
            outputs = np.array(outputs)  # shape: (batch_n, 3)

            buffer_inputs.append(torch.from_numpy(inputs).float())
            buffer_outputs.append(torch.from_numpy(outputs).float())
            buffer_bytes += inputs.nbytes + outputs.nbytes
            generated += batch_n

            # if buffer size exceeds chunk size, save to file
            if buffer_bytes >= chunk_size_bytes:
                all_inputs = torch.cat(buffer_inputs, dim=0)
                all_outputs = torch.cat(buffer_outputs, dim=0)
                out_path = os.path.join(out_dir, f"data.{file_idx}.pt")
                torch.save({"inputs": all_inputs, "outputs": all_outputs}, out_path)
                print(f"Saved {out_path} ({len(all_inputs)} samples)")
                file_idx += 1
                buffer_inputs = []
                buffer_outputs = []
                buffer_bytes = 0

        # save remaining data
        if buffer_inputs:
            all_inputs = torch.cat(buffer_inputs, dim=0)
            all_outputs = torch.cat(buffer_outputs, dim=0)
            out_path = os.path.join(out_dir, f"data.{file_idx}.pt")
            torch.save({"inputs": all_inputs, "outputs": all_outputs}, out_path)
            print(f"Saved {out_path} ({len(all_inputs)} samples)")

    gen_and_save(n_train, train_dir)
    gen_and_save(n_test, test_dir)


class DisneyBRDFDataset(Dataset):
    """
    load chunked data from files
    """

    def __init__(self, test=False):
        base_dir = "data/disney-rtxns"
        data_dir = os.path.join(base_dir, "test" if test else "train")
        self.file_list = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".pt")
            ]
        )
        self.chunk_indices = []
        self.length = 0
        self._build_index()

    def _build_index(self):
        self.chunk_sizes = []
        for f in self.file_list:
            data = torch.load(f)
            n = len(data["inputs"])
            self.chunk_sizes.append(n)
            self.chunk_indices.append(self.length)
            self.length += n

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for i, start_idx in enumerate(self.chunk_indices):
            if idx < start_idx + self.chunk_sizes[i]:
                local_idx = idx - start_idx
                data = torch.load(self.file_list[i])
                return data["inputs"][local_idx], data["outputs"][local_idx]
        raise IndexError("Index out of range")
