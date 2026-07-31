import os
import logging
import gc
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
from torchvision import transforms
from data.util import gen_diagonal_img, transform
from skimage.exposure import match_histograms


class Loader:
    def __init__(self, dataroot, phase = None,use_mmap=False,
                 norm_method = None,
                 chunk_size = 1, window_size = None,
                 memmap_dir = None,
                 time_seq = True):
        """
        Args:
            dataroot: TIFF file path or directory
            norm_method: normalization method (snorm/vnorm/wnorm/hnorm)
            chunk_size: Number of samples loaded in blocks (balancing memory and IO efficiency)
            memmap_dir: Memory mapped cache directory
        """
        self.img_paths = self._get_tiff_paths(dataroot)
        self.phase = phase
        self.norm_method = norm_method
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.memmap_dir = memmap_dir if memmap_dir is not None else self._default_memmap_dir(dataroot)
        self.use_memmap = use_mmap
        self.need_sub = (self.phase == 'train')
        self.time_seq = time_seq

        os.makedirs(self.memmap_dir, exist_ok=True)
        self.mmap_paths = {
            'main': os.path.join(self.memmap_dir, 'data_memmap.bin'),
            'sub1': os.path.join(self.memmap_dir, 'data_memmap_sub1.bin'),
            'sub2': os.path.join(self.memmap_dir, 'data_memmap_sub2.bin'),
        }

        self.shape, self.padded_shape = self._scan_metadata()

        if self.use_memmap:
            os.makedirs(self.memmap_dir, exist_ok=True)
            self._process()
        else:
            self.data_main, self.data_sub1, self.data_sub2 = self._load_all_data()

    def _default_memmap_dir(self, dataroot):
        if os.path.isdir(dataroot):
            return os.path.join(dataroot, 'cache')
        if os.path.isfile(dataroot):
            parent_dir = os.path.dirname(dataroot)
            return os.path.join(parent_dir, 'cache')
        raise ValueError(f"Invalid dataroot for memmap_dir: {dataroot}")

    def _out_img(self):
        if self.use_memmap:
            return self.memmap_files['main'], self.memmap_files.get('sub1', None), self.memmap_files.get('sub2', None)
        else:
            return self.data_main, self.data_sub1, self.data_sub2

    def _load_all_data(self):
        logger = logging.getLogger(self.phase)
        logger.info(f"[!] Loading all data into memory at one time")

        full_data = np.stack([tiff.imread(p) for p in self.img_paths])
        full_data = self._apply_norm(self._dim_align(full_data))
        main = self._pad_chunk(full_data)

        if self.need_sub:
            sub1, sub2 = gen_diagonal_img(full_data)
            sub1 = self._pad_chunk(sub1)
            sub2 = self._pad_chunk(sub2)
        else:
            sub1 = sub2 = np.array(1)

        return map(self._to_float32, (main, sub1, sub2))

    def _get_tiff_paths(self, dataroot):
        if isinstance(dataroot, list):
            return [p for p in dataroot if p.endswith('.tif')]
        elif os.path.isdir(dataroot):
            return sorted([
                os.path.join(root, f)
                for root, _, files in os.walk(dataroot)
                for f in files if f.endswith('.tif')
            ])
        elif os.path.isfile(dataroot) and dataroot.endswith('.tif'):
            return [dataroot]
        else:
            raise ValueError(f"Invalid dataroot: {dataroot}")

    def _scan_metadata(self):
        sample = tiff.imread(self.img_paths[0])
        shape = list(sample.shape)

        if len(shape) == 2:
            shape = [1, 1] + shape
        elif len(shape) == 3:
            shape = [1] + shape if not self.time_seq else [shape[0], 1, shape[1], shape[2]]

        ori_shape = (len(self.img_paths),) + tuple(shape)
        shape[0] += 2
        shape[1] += 2
        padded_shape = (len(self.img_paths),) + tuple(shape)

        return ori_shape, padded_shape

    def _load_chunk(self, start_idx):
        end_idx = min(start_idx + self.chunk_size, len(self.img_paths))
        chunk = np.stack([tiff.imread(p) for p in self.img_paths[start_idx:end_idx]])
        chunk = self._apply_norm(self._dim_align(chunk))

        # chunk_main = self._apply_norm(chunk)
        chunk_main = self._pad_chunk(chunk)

        if self.need_sub:
            sub1, sub2 = gen_diagonal_img(chunk)
            sub1 = self._pad_chunk(sub1)
            sub2 = self._pad_chunk(sub2)
        else:
            sub1 = sub2 = np.array(1)

        return map(self._to_float32, (chunk_main, sub1, sub2))

    def _process(self):
        logger = logging.getLogger(self.phase)

        required_keys = ['main']
        if self.need_sub:
            required_keys.extend(['sub1', 'sub2'])

        mmap_exists = all(os.path.isfile(self.mmap_paths[key]) for key in required_keys)

        if mmap_exists:
            logger.info(f"[!] memmp file exists in: {self.memmap_dir}, loaded")
            self._load_memmaps(mode='r+')
        else:
            self._load_memmaps(mode='w+')
            for i in range(0, len(self.img_paths), self.chunk_size):
                main, sub1, sub2 = self._load_chunk(i)
                self.memmap_files['main'][i:i + self.chunk_size] = main
                if self.need_sub:
                    self.memmap_files['sub1'][i:i + self.chunk_size] = sub1
                    self.memmap_files['sub2'][i:i + self.chunk_size] = sub2
            gc.collect()

    def _load_memmaps(self, mode='r+'):
        self.memmap_files = {}
        self.memmap_files['main'] = np.memmap(
            self.mmap_paths['main'], dtype='float32', mode=mode, shape=self.padded_shape)
        if self.need_sub:
            self.memmap_files['sub1'] = np.memmap(
                self.mmap_paths['sub1'], dtype='float32', mode=mode, shape=self.padded_shape)
            self.memmap_files['sub2'] = np.memmap(
                self.mmap_paths['sub2'], dtype='float32', mode=mode, shape=self.padded_shape)

    def _dim_align(self, chunk):
        if chunk.ndim == 3:
            chunk = chunk[:, None, None]
        elif chunk.ndim == 4:
            chunk = chunk[:, None] if not self.time_seq else chunk[:, :, None]
        return chunk

    def _apply_norm(self, chunk):
        """chunk normlization"""
        norm_funcs = {
            'snorm': self._slice_norm,
            'vnorm': self._volume_norm,
            'wnorm': lambda x: self._sliding_window_norm(x, self.window_size),
            'hnorm': self._hnorm
        }
        return norm_funcs[self.norm_method](chunk)

    @staticmethod
    def _slice_norm(chunk):

        min_val = chunk.min(axis=(-2, -1), keepdims=True)
        max_val = chunk.max(axis=(-2, -1), keepdims=True)
        chunk = (chunk - min_val) / (max_val - min_val + 1e-8)

        return chunk

    @staticmethod
    def _volume_norm(chunk):
        min_val = chunk.min(axis=(-3,-2,-1), keepdims=True)
        max_val = chunk.max(axis=(-3,-2,-1), keepdims=True)
        return (chunk - min_val) / (max_val - min_val + 1e-8)

    @staticmethod
    def _sliding_window_norm(chunk, window_size):
        assert chunk.ndim == 5
        eps = 1e-6
        normalized = np.zeros_like(chunk, dtype=np.float32)
        N, T = chunk.shape[:2]

        for n in range(N):
            for t in range(T):
                start = max(0, t - window_size // 2)
                end = min(T, t + window_size // 2)
                window = chunk[n, start:end]
                min_val = window.min()
                max_val = window.max()
                denom = max(max_val - min_val, eps)
                normalized[n, t] = (chunk[n, t] - min_val) / denom

        return normalized

    @staticmethod
    def _hnorm(chunk):
        assert chunk.ndim == 5, 'correct shape: n,t,z,h,w'
        n, t, z, h, w = chunk.shape

        for i in range(n):
            ref = chunk[i, 0]  # shape: (z, h, w)

            for j in range(1, t):
                for k in range(z):
                    chunk[i, j, k] = match_histograms(chunk[i, j, k], ref[k])

        min_val = np.min(chunk, axis=(-3,-2,-1), keepdims=True)  # shape: (n, t, 1, 1, 1)
        max_val = np.max(chunk, axis=(-3,-2,-1), keepdims=True)
        chunk = (chunk - min_val) / (max_val - min_val + 1e-8)

        return chunk


    def _pad_chunk(self, chunk):
        chunk = np.pad(chunk, ((0, 0), (1, 1), (1, 1), (0, 0), (0, 0)))

        chunk[:, 0] = chunk[:, 2]
        chunk[:, -1] = chunk[:, -3]

        chunk[:, :, 0] = chunk[:, :, 2]
        chunk[:, :, -1] = chunk[:, :, -3]
        return chunk

    def _to_float32(self,arr):
        return arr.astype(np.float32, copy=False)

    def cleanup(self):
        for path in self.mmap_paths.values():
            if os.path.isfile(path):
                os.remove(path)
                print(f'[!] found existed memmap file: {path}, deleted')


class BaseDataset(Dataset):

    def __init__(self, dataroot, phase='train', val_volume_idx=0, val_slice_idx=9, val_frame_idx=None,
                 gt_path=None, norm_method=None, use_mmap=False, chunk_size=1, window_size=5,
                 memmap_dir=None, time_seq=True):
        # Common parameters
        self.phase = phase
        self.gt_path = gt_path
        self.norm_method = norm_method
        self.window_size = window_size
        self.use_mmap = use_mmap
        self.logger = logging.getLogger(phase)
        self.loader_config = dict(
            use_mmap=use_mmap,
            norm_method=norm_method,
            chunk_size=chunk_size,
            window_size=window_size,
            time_seq=time_seq,
            memmap_dir=memmap_dir
        )

        # Initialize data loader
        self.loader = Loader(dataroot, phase, **self.loader_config)
        self.images, self.images_sub1, self.images_sub2 = self.loader._out_img()

        self.images_shape = self.loader.shape

        self.logger.info(f"[!] {self.phase} data shape: {self.images_shape}")

        # Validation setup
        self._init_validation(val_volume_idx, val_slice_idx, val_frame_idx)

    def _init_validation(self, val_volume_idx, val_slice_idx, val_frame_idx):
        if self.phase == 'val':
            if self.gt_path is not None:
                self.gt_loader = Loader(self.gt_path, self.phase, **self.loader_config)
                self.gt_images, _, _ = self.gt_loader._out_img()

            self.val_volume_idx = self._set_val_index(val_volume_idx, self.images_shape[0])
            self.val_frame_idx = self._set_val_index(val_frame_idx, self.images_shape[1])
            self.val_slice_idx = self._set_val_index(val_slice_idx, self.images_shape[2])

            self.val_shape = (
                len(self.val_volume_idx),
                len(self.val_frame_idx),
                len(self.val_slice_idx),
                self.images_shape[-2],
                self.images_shape[-1]
            )
            self.logger.info(f"[!] Actual validation data shape: {self.val_shape}")

    def __len__(self):
        if self.phase == 'train' or self.phase == 'test':
            return self.images_shape[0] * self.images_shape[1] * self.images_shape[2]
        elif self.phase == 'val':
            return np.prod([len(x) for x in [self.val_volume_idx, self.val_frame_idx, self.val_slice_idx]])

    def _get_idx(self, index):
        shape = self.val_shape if self.phase == 'val' else self.images_shape
        frame, slice = shape[1], shape[2]
        volume_idx = index // (frame * slice)
        frame_idx = (index % (frame * slice)) // slice
        slice_idx = index % slice

        if self.phase == 'val':
            volume_idx = self.val_volume_idx[volume_idx]
            frame_idx = self.val_frame_idx[frame_idx]
            slice_idx = self.val_slice_idx[slice_idx]

        return volume_idx, frame_idx, slice_idx

    def _set_val_index(self, value, max_range):
        if value == 'all':
            return range(max_range)
        elif isinstance(value, int):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            return [int(value)]
    def _base_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__")


class S1Dataset(BaseDataset):
    """Dataset for Stage 1: Basic Noise2Noise training"""

    def __init__(self, lr_flip=0.5, **kwargs):
        super().__init__(**kwargs)
        self.lr_flip = lr_flip

        # Stage-specific transforms
        if self.phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(lr_flip),
                transforms.RandomHorizontalFlip(lr_flip),
                # transforms.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transforms = self._base_transforms()

    def __getitem__(self, index):
        volume_idx, frame_idx, slice_idx = self._get_idx(index)

        if self.phase == 'train':
            sub1 = self.images_sub1[volume_idx, frame_idx + 1, slice_idx + 1]
            sub2 = self.images_sub2[volume_idx, frame_idx + 1, slice_idx + 1]

            raw_input = self.transforms(np.stack([sub1, sub2], axis=-1))
            ret = dict(X=raw_input[0:1], Y=raw_input[1:2])
        else:
            img = self.images[volume_idx, frame_idx + 1, slice_idx + 1]
            raw_input = self.transforms(img[:,:,None])
            ret = dict(Y=raw_input, X=raw_input)

            if self.gt_path is not None:
                gt = self.gt_images[volume_idx, frame_idx + 1, slice_idx + 1]
                ret['gt'] = self.transforms(gt[:,:,None])

        return ret


class S2Dataset(BaseDataset):
    """Dataset for Stage 2: Enhanced training with spatio-temporal conditions"""

    def __init__(self, time_cond=True, space_cond=True, global_cond=False,
                 predenoise_path=None, match_file=None, **kwargs):
        super().__init__(**kwargs)
        self.time_cond = time_cond
        self.space_cond = space_cond
        self.global_cond = global_cond
        self.predenoise_path = predenoise_path

        # Load pre-denoised data
        if predenoise_path:
            prdn_loader = Loader(predenoise_path, self.phase, **self.loader_config)
            self.prdn_images, _, _ = prdn_loader._out_img()

        # Load stage2 matching state
        self.matched_state = self._parse_match_file(match_file) if match_file else None


    def _parse_match_file(self, file_path):
        if file_path is None:
            raise ValueError(
                "Stage 2 requires 'match_file', but got None. "
                "Please provide a valid adaptive matching file or let train_s2.py generate it automatically."
            )

        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Adaptive matching file not found: {file_path}\n"
                f"Please run adaptive matching first or check the path in the config."
            )

        if os.path.getsize(file_path) == 0:
            raise ValueError(
                f"Adaptive matching file is empty: {file_path}\n"
                f"Please run adaptive matching first. In the standard pipeline, train_s2.py will generate it automatically."
            )

        results = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('_')
                volume_idx, frame_idx, slice_idx, t = int(info[0]), int(info[1]), int(info[2]), int(info[3])
                if volume_idx not in results:
                    results[volume_idx] = {}
                if frame_idx not in results[volume_idx]:
                    results[volume_idx][frame_idx] = {}
                results[volume_idx][frame_idx][slice_idx] = t

        if not results:
            raise ValueError(
                f"No valid adaptive matching entries were parsed from: {file_path}\n"
                f"Please regenerate the match file."
            )
        return results

    def __getitem__(self, index):
        volume_idx, frame_idx, slice_idx = self._get_idx(index)
        rand_num = self._get_random_num()

        # Base sample
        if self.phase == 'train':
            noisy_sub1 = self.images_sub1[volume_idx, frame_idx + 1, slice_idx + 1]
            noisy_sub2 = self.images_sub2[volume_idx, frame_idx + 1, slice_idx + 1]
            sample = dict(
                X=transform(noisy_sub1, self.phase, rand_num),
                Y=transform(noisy_sub2, self.phase, rand_num)
            )
        else:
            noisy = self.images[volume_idx, frame_idx + 1, slice_idx + 1]
            noisy_trans = transform(noisy, self.phase, rand_num)
            sample = dict(Y=noisy_trans, X=noisy_trans)

        # Add conditions
        sample = self._add_conditions(sample, volume_idx, frame_idx, slice_idx, rand_num)

        # Add extra info
        return self._add_extra_info(sample, volume_idx, frame_idx, slice_idx, rand_num)

    def _get_random_num(self):
        rs = np.random.RandomState()
        return rs.uniform(0, 1, (2, 1))

    def _add_conditions(self, sample, v_idx, f_idx, s_idx, rand_num):
        # ============================================================
        # Time condition ablation mode
        #
        # "normal": 使用原始前后时间帧，原版 DUAL 行为
        # "zero"  : 保留 time_cond1/time_cond2 两个通道，但内容置为 0
        # "copy"  : 保留 time_cond1/time_cond2 两个通道，但复制当前输入帧
        #
        # 注意：
        # 不要在 JSON 里把 time_cond 改成 false。
        # 这里保留 time_cond=true，只改变条件通道的内容，
        # 这样 Stage 2 仍然是 4 通道输入，可以加载原来的 checkpoint。
        # ============================================================
        TIME_COND_MODE = "normal"  # 可选："normal"(默认) / "zero" / "copy"

        # Time conditions
        if self.time_cond:
            if TIME_COND_MODE == "normal":
                t_cond1 = self.images[v_idx, f_idx, s_idx + 1]
                t_cond2 = self.images[v_idx, f_idx + 2, s_idx + 1]
                sample.update({
                    "time_cond1": transform(t_cond1, self.phase, rand_num),
                    "time_cond2": transform(t_cond2, self.phase, rand_num)
                })

            elif TIME_COND_MODE == "zero":
                sample.update({
                    "time_cond1": torch.zeros_like(sample["Y"]),
                    "time_cond2": torch.zeros_like(sample["Y"])
                })

            elif TIME_COND_MODE == "copy":
                sample.update({
                    "time_cond1": sample["Y"].clone(),
                    "time_cond2": sample["Y"].clone()
                })

            else:
                raise ValueError(
                    f"Unknown TIME_COND_MODE: {TIME_COND_MODE}. "
                    "Choose from 'normal', 'zero', or 'copy'."
                )

        # Space conditions
        if self.space_cond:
            s_cond1 = self.images[v_idx, f_idx + 1, s_idx]
            s_cond2 = self.images[v_idx, f_idx + 1, s_idx + 2]
            sample.update({
                "space_cond1": transform(s_cond1, self.phase, rand_num),
                "space_cond2": transform(s_cond2, self.phase, rand_num)
            })

        # Global condition
        if self.global_cond:
            global_frame = self.images[v_idx, 1, s_idx + 1]
            sample["global_cond"] = transform(global_frame, self.phase, rand_num)

        return sample

    def _add_extra_info(self, sample, v_idx, f_idx, s_idx, rand_num):
        if self.phase == 'val' and self.gt_path:
            sample["gt"] = transform(self.gt_images[v_idx, f_idx + 1, s_idx + 1], self.phase, rand_num)

        if self.matched_state:
            try:
                matched_t = self.matched_state[v_idx][f_idx][s_idx]
            except KeyError:
                raise KeyError(
                    f"Missing adaptive matching state for sample "
                    f"(volume={v_idx}, frame={f_idx}, slice={s_idx}).\n"
                    f"Please check whether the match file is complete and matches the current validation/training dataset."
                )

            sample["matched_state"] = torch.tensor(
                matched_t,
                dtype=torch.float32
            )

        if self.predenoise_path:
            pre_denoise = self.prdn_images[v_idx, f_idx + 1, s_idx + 1]
            sample["pre_denoised"] = transform(pre_denoise, self.phase, rand_num)

        return sample
