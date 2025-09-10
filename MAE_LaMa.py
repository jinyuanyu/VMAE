import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image 
#from simple_lama_inpainting import SimpleLama
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
# from pytorch_msssim import SSIM

class Datasets(Dataset):
    """自定义数据集"""
    def __init__(self, data_dir: str, max_seq_len: int = 8,ocean_mask_path: str = None):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化到-1，1
        ])
        self.img_list, self.fname_list = self._load_data()
        self.mask_list = self._load_mask()
        self.lama_init = self._load_lama_init()
        if self.img_list:
            self.img_size = self.img_list[0].size[::-1]
        else:
            raise ValueError("No images found in directory:", data_dir)
        #* 添加海洋掩码
        if ocean_mask_path is not None:
            ocean_mask = Image.open(ocean_mask_path).convert('L')
            self.ocean_mask = transforms.ToTensor()(ocean_mask).float()  # [0,1]范围
        else:
            raise ValueError("Ocean mask path must be provided if ocean_mask is required.")

    def _load_data(self):
        img_list = []
        fname_list = []
        for frame_file in sorted(os.listdir(self.data_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.data_dir, frame_file)
                image = Image.open(frame_path)#.convert('L')
                img_list.append(image)
                fname = int(frame_file.split('.')[0].split('_')[-1])
                fname_list.append(fname)
        return img_list, fname_list
    def _load_mask(self):
        mask_list = []
        for frame_file in sorted(os.listdir('E:/lama/mask_img')):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join('E:/lama/mask_img', frame_file)
                mask = Image.open(frame_path).convert('L')
                mask_list.append(mask)  # 转换为0-1范围的浮点数
        mask_list = np.array([np.array(mask) for mask in mask_list])
        mask_list = torch.tensor(mask_list / 255, dtype=torch.float32).unsqueeze(1)
        return mask_list
    def _load_lama_init(self):
        lama_init = []
        for frame_file in sorted(os.listdir('E:/lama/inpainted_img/lama_init')):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join('E:/lama/inpainted_img/lama_init', frame_file)
                image = Image.open(frame_path)#.convert('L')
                lama_init.append(image)
        return lama_init

    
    def __len__(self):
        return len(self.img_list) // self.max_seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = min(start_idx + self.max_seq_len, len(self.img_list))
        frames = self.img_list[start_idx:end_idx]
        fnames = self.fname_list[start_idx:end_idx]

        lama_init = self.lama_init[:]
        if self. data_dir == 'E:/lama/masked_img/test_img/':
            mask = self._generate_random_mask()
        elif self.data_dir == 'E:/lama/jet_S2_Daily_Mosaic/' and self.ocean_mask is not None:
            mask = self.mask_list
            mask = mask[start_idx:end_idx]
        else:
            raise ValueError('重新设置此路径下数据的掩码位置')
        if len(frames) < self.max_seq_len:
            frames += [frames[-1]] * (self.max_seq_len - len(frames))
        frames = [self.transform(frame) for frame in frames]
        fnames  = torch.tensor(fnames, dtype=torch.int64)
        
        #*读入lama初始值
        lama_init = [self.transform(frame) for frame in lama_init]
        lama_init = torch.stack(lama_init, dim=0)

        video = torch.stack(frames, dim=0)
        masked_video = video * (1 - mask)
        
        return {
            'video': video,
            'masked': masked_video,
            'mask': mask,
            'times': fnames,
            'lama_init': lama_init,
            'ocean_mask': self.ocean_mask  # 添加海洋掩码
        }
        
    def _generate_random_mask(self):
        mask_type = np.random.randint(0, 3)
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        if mask_type == 0:
            num_clouds = np.random.randint(1, 4)
            for _ in range(num_clouds):
                size_range = (
                    max(10, int(min(self.img_size)*0.1)),
                    int(min(self.img_size)*0.4)
                )
                cloud_patch = self._generate_cloud_patch(size_range)
                if cloud_patch is not None:
                    h, w = cloud_patch.shape
                    h_start = np.random.randint(0, self.img_size[0]-h)
                    w_start = np.random.randint(0, self.img_size[1]-w)
                    t_start = np.random.randint(0, self.max_seq_len-1)
                    t_end = min(t_start + np.random.randint(1, 4), self.max_seq_len)
                    mask[t_start:t_end, 0, h_start:h_start+h, w_start:w_start+w] = \
                        torch.from_numpy(cloud_patch).float()
        
        elif mask_type == 1:
            num_clouds = np.random.randint(5, 16)
            for _ in range(num_clouds):
                size_range = (
                    max(15, int(min(self.img_size)*0.05)),
                    int(min(self.img_size)*0.2)
                )
                cloud_patch = self._generate_cloud_patch(size_range)
                if cloud_patch is not None:
                    h, w = cloud_patch.shape
                    h_start = np.random.randint(0, self.img_size[0]-h)
                    w_start = np.random.randint(0, self.img_size[1]-w)
                    t = np.random.randint(0, self.max_seq_len)
                    mask[t, 0, h_start:h_start+h, w_start:w_start+w] = \
                        torch.from_numpy(cloud_patch).float()
        
        else:
            for t in range(self.max_seq_len):
                if np.random.rand() < 0.5:
                    num_clouds = np.random.randint(1, 4)
                    for _ in range(num_clouds):
                        size_range = (
                            max(15, int(min(self.img_size)*0.05)),
                            int(min(self.img_size)*0.3)
                        )
                        cloud_patch = self._generate_cloud_patch(size_range)
                        if cloud_patch is not None:
                            h, w = cloud_patch.shape
                            h_start = np.random.randint(0, self.img_size[0]-h)
                            w_start = np.random.randint(0, self.img_size[1]-w)
                            mask[t, 0, h_start:h_start+h, w_start:w_start+w] = \
                                torch.from_numpy(cloud_patch).float()
        
        return mask

    def _generate_cloud_patch(self, size_range):
        size = np.random.randint(size_range[0], size_range[1] + 1)
        canvas_size = size * 3
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        num_circles = np.random.randint(3, 7)
        for _ in range(num_circles):
            cx = canvas_size // 2 + np.random.randint(-size//2, size//2)
            cy = canvas_size // 2 + np.random.randint(-size//2, size//2)
            radius = np.random.randint(size//4, size//2)
            cv2.circle(canvas, (cx, cy), radius, 1, -1)
        
        blur_size = min(11, size//5)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(canvas, (blur_size, blur_size), 0)
        
        _, cloud_patch = cv2.threshold(blurred, 0.3, 1, cv2.THRESH_BINARY)
        cloud_patch = cloud_patch.astype(np.uint8)
        
        kernel_size = max(3, size//10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cloud_patch = cv2.dilate(cloud_patch, kernel, iterations=1)
        cloud_patch = cv2.erode(cloud_patch, kernel, iterations=1)
        
        rows, cols = np.where(cloud_patch)
        if len(rows) == 0:
            return None
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        cropped = cloud_patch[min_row:max_row+1, min_col:max_col+1]
        
        scale_factor = np.random.uniform(0.8, 1.2)
        new_h = max(1, int(cropped.shape[0] * scale_factor))
        new_w = max(1, int(cropped.shape[1] * scale_factor))
        
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            _, resized = cv2.threshold(resized, 0.5, 1, cv2.THRESH_BINARY)
            return resized.astype(np.uint8)
        
        return cropped
#* 验证集多模式掩码
class Datasets_inference(Dataset):
    """自定义数据集"""
    def __init__(self, data_dir: str, max_seq_len: int = 8, ocean_mask_path: str = None, 
                 mask_type: str = "random", split_ocean_land: int = 999, mask_ratio: float = 0.5):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            max_seq_len: 最大序列长度
            ocean_mask_path: 海洋掩码路径
            mask_type: 掩码类型，可选 "random", "cloud", "strip", "mixed"
            mask_ratio: 掩码缺失比例 (0.0-1.0)
        """
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.split_ocean_land = split_ocean_land
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化到-1，1
        ])
        
        self.img_list, self.fname_list = self._load_data()
        self.mask_list = self._load_mask()
        self.lama_init = self._load_lama_init()
        
        if self.img_list:
            self.img_size = self.img_list[0].size[::-1]  # (height,width)
        else:
            raise ValueError("No images found in directory:", data_dir)
        
        #* 添加海洋掩码
        if ocean_mask_path is not None:
            ocean_mask = Image.open(ocean_mask_path).convert('L')
            # 调整海洋掩码尺寸与输入图像一致
            if hasattr(self, 'img_size'):
                ocean_mask = ocean_mask.resize((self.img_size[0], self.img_size[1]))
            self.ocean_mask = transforms.ToTensor()(ocean_mask).float()  # [0,1]范围
        else:
            self.ocean_mask = None

    def _load_data(self):
        img_list = []
        fname_list = []
        for frame_file in sorted(os.listdir(self.data_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.data_dir, frame_file)
                image = Image.open(frame_path)
                img_list.append(image)
                fname = int(frame_file.split('.')[0].split('_')[-1])
                fname_list.append(fname)
        return img_list, fname_list
    
    def _load_mask(self):
        mask_list = []
        mask_dir = 'E:/lama/mask_img'
        if os.path.exists(mask_dir):
            for frame_file in sorted(os.listdir(mask_dir)):
                if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                    frame_path = os.path.join(mask_dir, frame_file)
                    mask = Image.open(frame_path).convert('L')
                    mask_list.append(mask)
            if mask_list:
                mask_list = np.array([np.array(mask) for mask in mask_list])
                mask_list = torch.tensor(mask_list / 255, dtype=torch.float32).unsqueeze(1)
                return mask_list
        return None
    
    def _load_lama_init(self):
        lama_init = []
        lama_dir = 'E:/lama/inpainted_img/lama_init'
        if os.path.exists(lama_dir):
            for frame_file in sorted(os.listdir(lama_dir)):
                if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                    frame_path = os.path.join(lama_dir, frame_file)
                    image = Image.open(frame_path)
                    lama_init.append(image)
        return lama_init

    def __len__(self):
        return len(self.img_list) // self.max_seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = min(start_idx + self.max_seq_len, len(self.img_list))
        frames = self.img_list[start_idx:end_idx]
        fnames = self.fname_list[start_idx:end_idx]

        lama_init = self.lama_init[:] if self.lama_init else []
        
        # 根据mask_type参数生成掩码
        if self.mask_type == "predefined" and self.mask_list is not None:
            mask = self.mask_list[start_idx:end_idx]
        else:
            mask = self._generate_mask_by_type(self.mask_type, self.mask_ratio)
        
        if len(frames) < self.max_seq_len:
            frames += [frames[-1]] * (self.max_seq_len - len(frames))
        
        frames = [self.transform(frame) for frame in frames]
        fnames = torch.tensor(fnames, dtype=torch.int64)
        
        #* 读入lama初始值
        if lama_init:
            lama_init = [self.transform(frame) for frame in lama_init]
            lama_init = torch.stack(lama_init, dim=0)
        else:
            lama_init = torch.zeros((self.max_seq_len, 3, *self.img_size[::-1])) if self.img_list else torch.zeros((self.max_seq_len, 3, 224, 224))

        video = torch.stack(frames, dim=0)
        masked_video = video * (1 - mask)
        # if self.split_ocean_land == 1 and self.ocean_mask is not None:
        #     masked_video = masked_video * self.ocean_mask
        # elif self.split_ocean_land == 0 and self.ocean_mask is not None:
        #     masked_video = masked_video * (1 - self.ocean_mask)
        return {
            'video': video,
            'masked': masked_video,
            'mask': mask,
            'times': fnames,
            'lama_init': lama_init,
            'ocean_mask': self.ocean_mask if self.ocean_mask is not None else torch.zeros((1, *self.img_size[::-1]))
        }

    def _generate_mask_by_type(self, mask_type: str, mask_ratio: float = 0.5):
        """根据掩码类型生成掩码（添加薄云类型）"""
        if mask_type == "random":
            return self._generate_thin_cloud_mask(mask_ratio)  # 用薄云替代
        elif mask_type == "thin_cloud":
            return self._generate_thin_cloud_mask(mask_ratio)
        elif mask_type == "cloud":
            return self._generate_cloud_mask(mask_ratio)
        elif mask_type == "strip":
            return self._generate_strip_mask(mask_ratio)
        elif mask_type == "mixed":
            return self._generate_mixed_mask(mask_ratio)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def _generate_random_mask(self, mask_ratio: float):
        """使用薄云掩码替代原来的随机掩码"""
        return self._generate_thin_cloud_mask(mask_ratio)

    def _generate_cloud_mask(self, mask_ratio: float):
        """生成云状掩码（修复版本）"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        total_pixels = self.img_size[0] * self.img_size[1]
        target_masked_pixels = int(total_pixels * mask_ratio)
        
        for t in range(self.max_seq_len):
            current_masked = 0
            attempts = 0
            max_attempts = 100  # 防止无限循环
            
            while current_masked < target_masked_pixels and attempts < max_attempts:
                # 生成云块大小：相对于图像尺寸的比例
                cloud_size = np.random.randint(
                    min(100, self.img_size[0]*mask_ratio), 
                    min(200, self.img_size[1]*mask_ratio)
                )
                
                cloud_patch = self._generate_cloud_patch((cloud_size, cloud_size))
                if cloud_patch is None:
                    attempts += 1
                    continue
                    
                h, w = cloud_patch.shape
                if h >= self.img_size[0] or w >= self.img_size[1]:
                    attempts += 1
                    continue
                    
                h_start = np.random.randint(0, self.img_size[0] - h)
                w_start = np.random.randint(0, self.img_size[1] - w)
                
                # 计算新增的掩码像素
                new_cloud = torch.from_numpy(cloud_patch).float()
                existing_region = mask[t, 0, h_start:h_start+h, w_start:w_start+w]
                new_pixels = torch.sum(new_cloud * (1 - existing_region)).item()
                
                # 应用云块
                mask[t, 0, h_start:h_start+h, w_start:w_start+w] = torch.max(
                    existing_region, new_cloud
                )
                
                current_masked += new_pixels
                attempts += 1
        
        return mask
    
    def _generate_strip_mask(self, mask_ratio: float):
        """生成条带掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        strip_width = max(1, int(self.img_size[1] * 0.05))  # 条带宽度为图像宽度的5%
        num_strips = int(self.img_size[1] * mask_ratio / strip_width)
        
        for t in range(self.max_seq_len):
            for _ in range(num_strips):
                strip_pos = np.random.randint(0, self.img_size[1] - strip_width)
                orientation = np.random.choice(['horizontal', 'vertical'])
                
                if orientation == 'horizontal':
                    mask[t, 0, :, strip_pos:strip_pos+strip_width] = 1
                else:
                    mask[t, 0, strip_pos:strip_pos+strip_width, :] = 1
        
        return mask

    def _generate_mixed_mask(self, mask_ratio: float):
        """生成混合掩码（更新版本）"""
        # 分配比例
        cloud_ratio = mask_ratio * 0.5      # 50% 厚云
        thin_cloud_ratio = mask_ratio * 0.3  # 30% 薄云  
        strip_ratio = mask_ratio * 0.2       # 20% 条带
        
        cloud_mask = self._generate_cloud_mask(cloud_ratio)
        thin_cloud_mask = self._generate_thin_cloud_mask(thin_cloud_ratio)
        strip_mask = self._generate_strip_mask(strip_ratio)
        
        # 组合掩码
        combined_mask = torch.clamp(cloud_mask + thin_cloud_mask + strip_mask, 0, 1)
        return combined_mask

    def _generate_cloud_patch(self, size_range):
        """生成单个云块（修复版本）"""
        if len(size_range) != 2:
            size = size_range[0] if hasattr(size_range, '__len__') else size_range
        else:
            size = np.random.randint(size_range[0], size_range[1] + 1)
        
        # 确保画布大小合适
        canvas_size = max(size * 2, 50)  # 最小画布大小
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        # 生成多个重叠的圆形
        num_circles = np.random.randint(3, 8)
        for _ in range(num_circles):
            cx = canvas_size // 2 + np.random.randint(-size//3, size//3)
            cy = canvas_size // 2 + np.random.randint(-size//3, size//3)
            radius = np.random.randint(size//6, size//2)
            cv2.circle(canvas, (cx, cy), radius, 1, -1)
        
        # 高斯模糊
        blur_size = max(5, size//8)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(canvas, (blur_size, blur_size), 0)
        
        # 阈值化
        _, cloud_patch = cv2.threshold(blurred, 0.2, 1, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel_size = max(3, size//15)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cloud_patch = cv2.morphologyEx(cloud_patch, cv2.MORPH_CLOSE, kernel)
        
        # 裁剪到有效区域
        rows, cols = np.where(cloud_patch > 0)
        if len(rows) == 0:
            return None
            
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        cropped = cloud_patch[min_row:max_row+1, min_col:max_col+1]
        
        return cropped.astype(np.uint8)
    def _generate_thin_cloud_mask(self, mask_ratio: float):
        """生成薄云掩码 - 替代random掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        for t in range(self.max_seq_len):
            # 使用Perlin噪声或简化的噪声模拟薄云
            cloud_mask = self._generate_thin_cloud_pattern(mask_ratio)
            mask[t, 0] = torch.from_numpy(cloud_mask).float()
        
        return mask
    def _generate_thin_cloud_pattern(self, mask_ratio: float):
        """生成薄云图案"""
        h, w = self.img_size
        
        # 方法1：多尺度噪声组合
        cloud_pattern = np.zeros((h, w), dtype=np.float32)
        
        # 大尺度云团
        large_scale = self._generate_noise_layer(h, w, scale=0.01, octaves=3)
        # 中尺度纹理  
        medium_scale = self._generate_noise_layer(h, w, scale=0.05, octaves=2)
        # 小尺度细节
        small_scale = self._generate_noise_layer(h, w, scale=0.1, octaves=1)
        
        # 组合不同尺度
        cloud_pattern = (0.6 * large_scale + 0.3 * medium_scale + 0.1 * small_scale)
        
        # 归一化到[0,1]
        cloud_pattern = (cloud_pattern - cloud_pattern.min()) / (cloud_pattern.max() - cloud_pattern.min())
        
        # 应用阈值获得指定比例的掩码
        threshold = np.percentile(cloud_pattern, (1 - mask_ratio) * 100)
        thin_cloud_mask = (cloud_pattern > threshold).astype(np.float32)
        
        # 可选：添加一些随机性和边缘平滑
        if np.random.random() > 0.5:
            # 高斯模糊边缘
            kernel_size = max(3, min(h, w) // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1
            thin_cloud_mask = cv2.GaussianBlur(thin_cloud_mask, (kernel_size, kernel_size), 0)
            thin_cloud_mask = (thin_cloud_mask > 0.5).astype(np.float32)
        
        return thin_cloud_mask
    def _generate_noise_layer(self, h, w, scale=0.05, octaves=1):
        """生成噪声层（简化的Perlin噪声替代）"""
        # 由于没有noise库，使用简化方法模拟
        noise = np.random.randn(max(1, int(h * scale)), max(1, int(w * scale)))
        
        # 上采样并平滑
        noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 多次平滑模拟octaves效果
        for _ in range(octaves):
            kernel_size = max(3, min(h, w) // 50)
            if kernel_size % 2 == 0:
                kernel_size += 1
            noise_resized = cv2.GaussianBlur(noise_resized, (kernel_size, kernel_size), 0)
        
        return noise_resized


class PatchEmbedding(nn.Module):
    """Patch嵌入层"""
    def __init__(self, img_size_h: int = 224, img_size_w: int = 224, 
                 patch_size: int = 16, in_channels: int = 3, use_lama_init: bool = False,
                 embed_dim: int = 768, use_mask_channel: bool = True):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.embed_dim = embed_dim
        self.use_lama_init = use_lama_init
        self.use_mask_channel = use_mask_channel
        conv_in_channels = in_channels# + 1 if use_mask_channel else in_channels
        self.projection = nn.Conv2d(
            conv_in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        
    def forward(self, x, mask=None):
        B, T, C, or_H, or_W = x.shape
        
        if or_H != self.img_size_h or or_W != self.img_size_w:
            x = F.interpolate(
                x.view(-1, C, or_H, or_W),
                size=(self.img_size_h, self.img_size_w),
                mode='bilinear', align_corners=False
            )
        x = x.view(B, T, C, self.img_size_h, self.img_size_w)
        
        if self.use_mask_channel and mask is not None:
            mask = F.interpolate(
                mask.view(-1, 1, or_H, or_W),
                size=(self.img_size_h, self.img_size_w),
                mode='nearest'
            )
            mask = mask.view(B, T, 1, self.img_size_h, self.img_size_w)

            x_with_mask = torch.cat([x, mask], dim=2)
            x_with_mask = rearrange(x_with_mask, 'b t c h w -> (b t) c h w')
            x_embedded = self.projection(x_with_mask)
            x_embedded = rearrange(x_embedded, '(b t) d n_h n_w -> b t (n_h n_w) d', b=B, t=T)
        else:
            x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
            x_embedded = self.projection(x_reshaped)
            x_embedded = rearrange(x_embedded, '(b t) d n_h n_w -> b t (n_h n_w) d', b=B, t=T)
            
        return x_embedded

class TemporalAttention(nn.Module):
    """仅时间维度的注意力机制，时间编码作为偏置加入"""
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.scale = self.head_dim ** -0.5
        
        # 时间位置编码作为可学习参数
        self.temporal_bias = nn.Parameter(torch.randn(1, num_heads, max_seq_len, max_seq_len))
        
        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x形状: (B, N, T, D)"""
        B, N, T, D = x.shape
        
        # 合并批次和空间维度: (B*N, T, D)
        x_flat = x.reshape(B*N, T, D)
        
        # 线性变换
        q = self.q_proj(x_flat).view(B*N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, H, T, d)
        k = self.k_proj(x_flat).view(B*N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, H, T, d)
        v = self.v_proj(x_flat).view(B*N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B*N, H, T, d)
        
        # 计算注意力分数 (B*N, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加时间偏置 (广播到B*N)
        attn_scores = attn_scores + self.temporal_bias[:, :, :T, :T]
        
        # 注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力
        context = torch.matmul(attn_probs, v)  # (B*N, H, T, d)
        context = context.transpose(1, 2).reshape(B*N, T, D)  # (B*N, T, D)
        
        # 输出投影
        output = self.out_proj(context)
        
        # 恢复原始形状: (B, N, T, D)
        output = output.view(B, N, T, D)
        return output

class PatchDecoder(nn.Module):
    def __init__(self, img_size_h: int, img_size_w: int, patch_size: int, 
                 embed_dim: int, out_channels: int = 3):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim, out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            ),
            nn.Tanh() # 使用Tanh匹配输入范围[-1,1]
            # nn.Sigmoid()
        )
        # 尺寸调整层（如果需要）
        self.resize = None
        if (self.num_patches_h * patch_size != img_size_h) or \
           (self.num_patches_w * patch_size != img_size_w):
            self.resize = nn.Upsample(
                size=(img_size_h, img_size_w),
                mode='bilinear',
                align_corners=False
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N_D = x.shape
        D = N_D // (self.num_patches_h * self.num_patches_w)
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, D)
        x = rearrange(x, 'b t n_h n_w d -> (b t) d n_h n_w')
        output = self.decoder(x)
        # 调整尺寸（如果需要）
        if self.resize is not None:
            output = self.resize(output)
        output = rearrange(output, '(b t) c h w -> b t c h w', b=B)
        return output

class LamaInpaintingModule(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.to_PIL = transforms.ToPILImage()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        self.denormalize = transforms.Normalize(
            mean=[-1.0, -1.0, -1.0],  # -mean/std
            std=[2.0, 2.0, 2.0]       # 1/std
        )
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
        self.lama = SimpleLama()
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        repaired_images = []
        for i in range(batch_size):
            for j in range(x.shape[1]):
                img_tensor = self.denormalize(x[i,j]).cpu()
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                # img_array = img_tensor.permute(1, 2, 0).numpy()
                # plt.imshow(img_array)
                # plt.axis('off')  # 关闭坐标轴
                # plt.show()
                mask_tensor = mask[i,j,0].cpu()
                img_pil = self.to_PIL(img_tensor)
                mask_array = (mask_tensor.numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_array, mode='L')
                repaired_pil = self.lama(img_pil, mask_pil)
                if repaired_pil.size != img_pil.size:
                    repaired_pil = repaired_pil.resize(img_pil.size, Image.BILINEAR)  # 正确调用
                repaired_tensor = self.to_tensor(repaired_pil)  # 转为[0,1]
                repaired_tensor = self.normalize(repaired_tensor)  # 转为[-1,1]
                repaired_images.append(repaired_tensor)
        repaired_batch = torch.stack(repaired_images).to(self.device)
        repaired_batch = repaired_batch.view(batch_size, x.shape[1], -1, repaired_batch.shape[-2], repaired_batch.shape[-1])
        return repaired_batch

class VideoCompletionModel(nn.Module):
    def __init__(self, img_size_h: int, img_size_w: int, patch_size: int, 
                embed_dim: int, num_heads: int, max_seq_len: int, 
                use_lama_init: bool = False,use_ocean_prior: bool = False,
                freeze_backbone: bool = False,fine_tune_layers:list =None,
                use_mask_channel: bool = False, out_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=out_channels,
            embed_dim=embed_dim,
            use_mask_channel=use_mask_channel,
            use_lama_init=use_lama_init
        )
        
        # 使用时间注意力层替代原位置编码
        self.temporal_attention = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.decoder = PatchDecoder(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=out_channels
        )
        
        # 多层时间注意力
        self.atten_layers = nn.ModuleList([
            TemporalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout
            ) for _ in range(3)
        ])

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.use_lama_init = use_lama_init
        self.use_ocean_prior = use_ocean_prior
        #* LaMa修复模块（仅在use_lama_init=True时使用）
        if self.use_lama_init:
            self.lama_module = LamaInpaintingModule(device=next(self.parameters()).device)
            
            
        # 掩码更新模块：基于MAE结果学习更好的掩码
        if self.use_ocean_prior:
            self.mask_update_layer = nn.Sequential(
            nn.Conv2d(out_channels+1, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, 1, kernel_size=1),
            nn.Sigmoid()  # 生成更新的掩码权重
        )
        else:    
            self.mask_update_layer = nn.Sequential(
                nn.Conv2d(out_channels, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim//2, 1, kernel_size=1),
                nn.Sigmoid()  # 生成更新的掩码权重
        )
        #* 添加海洋先验控制
        self.use_ocean_prior = use_ocean_prior
        #* 添加微调控制参数
        self.freeze_backbone = freeze_backbone
        self.fine_tune_layers = fine_tune_layers or []
        if self.freeze_backbone:
            self.set_freeze_status()# 初始化后设置参数冻结状态

    def enhance_lama_input_with_mae(self, mae_reconstructed: torch.Tensor, original_input: torch.Tensor, mask: torch.Tensor, ocean_mask: torch.Tensor) -> tuple:
        """
        使用MAE重构结果来增强LaMa的输入和掩码
        
        Args:
            mae_reconstructed: MAE重构结果 (B, T, C+1, H, W)
            original_input: 原始输入 (B, T, C, H, W) 
            mask: 原始掩码 (B, T, 1, H, W)
        
        Returns:
            enhanced_input: 增强后的输入 (B, T, C, H, W)
            updated_mask: 更新后的掩码 (B, T, 1, H, W)
        """
        B, T, C_plus, H, W = mae_reconstructed.shape
        C = C_plus - 1  # RGB通道数
        
        updated_masks = []
        
        for t in range(T):
            mae_frame = mae_reconstructed[:, t, :C]  # 取RGB通道 (B, C, H, W)
            mae_mask = mae_reconstructed[:, t, C:]   # 取掩码通道 (B, 1, H, W)
            original_frame = original_input[:, t]    # (B, C, H, W)
            mask_frame = mask[:, t]                  # (B, 1, H, W)
            
            # 1. 生成更新的掩码权重
            if self.use_ocean_prior and ocean_mask is not None:
                # 计算MAE重构质量（与原图的相似度作为质量指标）
                mae_quality = 1.0 - torch.abs(mae_frame - original_frame).mean(dim=1, keepdim=True)
                mae_quality = torch.sigmoid(mae_quality * 10 - 5)  # 转换为0-1的质量分数
                
                mask_update_weight = self.mask_update_layer(torch.cat((mae_reconstructed[:, t], ocean_mask), dim=1))
                
                # 高质量区域减少掩码，低质量区域保持掩码
                ocean_weight = (ocean_mask > 0.5).float()
                updated_mask_frame = (
                    mask_frame * (1 - mask_update_weight * mae_quality * 0.7) * (1 - ocean_weight * 0.3) +
                    mae_mask * mask_update_weight * (1 - mae_quality) * 0.5 * (1 + ocean_weight * 0.5) +
                    ocean_weight * 0.1
                ).clamp(0, 1)
            else:
                mae_quality = 1.0 - torch.abs(mae_frame - original_frame).mean(dim=1, keepdim=True)
                mae_quality = torch.sigmoid(mae_quality * 10 - 5)
                
                mask_update_weight = self.mask_update_layer(mae_reconstructed[:, t])
                updated_mask_frame = mask_frame * (1 - mask_update_weight * mae_quality * 0.5) + \
                                   mae_mask * mask_update_weight * (1 - mae_quality) * 0.3

            updated_masks.append(updated_mask_frame)
        
        updated_mask = torch.stack(updated_masks, dim=1)      # (B, T, 1, H, W)
        
        return updated_mask

    def iterative_mae_lama_refinement(self, mae_reconstructed: torch.Tensor, original_input: torch.Tensor, mask: torch.Tensor, ocean_mask: torch.Tensor) -> torch.Tensor:
        """
        迭代式MAE-LaMa协作优化
        
        Args:
            mae_reconstructed: MAE重构结果 (B, T, C+1, H, W)
            original_input: 原始输入 (B, T, C, H, W)
            mask: 原始掩码 (B, T, 1, H, W)
        
        Returns:
            final_result: 最终重构结果 (B, T, C+1, H, W)
        """
        
        # 第一步：使用MAE结果增强LaMa输入
        updated_mask = self.enhance_lama_input_with_mae(
            mae_reconstructed, original_input, mask, ocean_mask
        )
        
        with torch.no_grad():
            # 分离梯度以避免numpy转换错误
            updated_mask_detached = updated_mask.detach()
            lama_results = self.lama_module(mae_reconstructed, updated_mask_detached)  # (B, T, C, H, W)
        
        # 将LaMa结果重新连接到计算图
        lama_results = lama_results.detach().requires_grad_(True)
        
        return lama_results

    def set_freeze_status(self):
        #* 首先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        #* 解冻指定层
        for name, param in self.named_parameters():
            # 解冻骨干网络
            if not self.freeze_backbone and ('patch_embedding' in name or 'temporal_attention' in name):
                param.requires_grad = True
            # 解冻指定微调层
            for layer_name in self.fine_tune_layers:
                if layer_name in name:
                    param.requires_grad = True
            # 确保这些层总是可训练
            if 'mask_update_layer' in name or 'decoder' in name or 'lama_module' in name:
                param.requires_grad = True
        #* 打印可训练参数信息
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable}/{total} ({100.*trainable/total:.2f}%)")
    def forward_mae_only(self, x: torch.Tensor, mask: torch.Tensor = None, ocean_mask: torch.Tensor = None) -> torch.Tensor:
        """仅执行MAE部分，返回重构结果（不经过LaMa处理）"""
        B, T, C, H, W = x.shape
        
        # 1. Patch嵌入 (B, T, N, D)
        x_embedded = self.patch_embedding(x, mask)
        
        # 2. 重新排列为 (B, N, T, D) 以便空间位置作为批次维度
        x_pos = rearrange(x_embedded, 'b t n d -> b n t d')
        
        # 3. 时间注意力机制
        attn_output = self.temporal_attention(x_pos)
        x_pos = x_pos + attn_output  # 残差连接
        x_pos = self.norm1(x_pos)
        
        # 4. 多层时间注意力
        for layer in self.atten_layers:
            attn_output = layer(x_pos)
            x_pos = x_pos + attn_output
            x_pos = self.norm1(x_pos)
            
            # 前馈网络
            ff_output = self.ffn(x_pos)
            x_pos = x_pos + ff_output
            x_pos = self.norm2(x_pos)
        
        # 5. 恢复原始形状 (B, T, N, D)
        x_out = rearrange(x_pos, 'b n t d -> b t (n d)')
        
        # 6. 解码器
        mae_reconstructed = self.decoder(x_out)  # (B, T, C+1, H, W)
        
        return mae_reconstructed
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, ocean_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C, H, W = x.shape
        
        # 1. Patch嵌入 (B, T, N, D)
        x_embedded = self.patch_embedding(x, mask)
        
        # 2. 重新排列为 (B, N, T, D) 以便空间位置作为批次维度
        x_pos = rearrange(x_embedded, 'b t n d -> b n t d')
        
        # 3. 时间注意力机制
        attn_output = self.temporal_attention(x_pos)
        x_pos = x_pos + attn_output  # 残差连接
        x_pos = self.norm1(x_pos)
        
        # 4. 多层时间注意力
        for layer in self.atten_layers:
            attn_output = layer(x_pos)
            x_pos = x_pos + attn_output
            x_pos = self.norm1(x_pos)
            
            # 前馈网络
            ff_output = self.ffn(x_pos)
            x_pos = x_pos + ff_output
            x_pos = self.norm2(x_pos)
        
        # 5. 恢复原始形状 (B, T, N, D)
        x_out = rearrange(x_pos, 'b n t d -> b t (n d)')
        
        # 6. 解码器
        mae_reconstructed = self.decoder(x_out)  # (B, T, C+1, H, W)
        
        # 7. 如果启用LaMa，则进行协作式重构
        if self.use_lama_init:
            # 使用改进的协作策略
            final_result = self.iterative_mae_lama_refinement(mae_reconstructed, x, mask, ocean_mask).to(x.device)
            return final_result
        else:
            return mae_reconstructed

#* 训练函数
def train(model, dataloader, optimizer, device, criterion, epochs=10, pretrained_path=None):
    #TODO 加载预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        
        # 过滤出可加载的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        # 只优化可训练参数
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=1e-2)
    else:
        print("Training from scratch")
    #* 原有训练循环
    model.train()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            video = batch['video'].to(device)
            masked_video = batch['masked'].to(device)
            mask = batch['mask'].to(device)
            ocean_mask= batch['ocean_mask'].to(device) 

            optimizer.zero_grad()

            if model.use_lama_init:
                # 获取MAE重构结果（包含RGB+掩码预测）
                mae_output = model.forward_mae_only(masked_video, mask, ocean_mask)  # 新增方法
                reconstructed = model(masked_video, mask, ocean_mask)  # 完整流程
                
                # 分离RGB和掩码通道
                mae_rgb = mae_output[:, :, :3]  # RGB通道
                mae_mask_pred = mae_output[:, :, 3:]  # 掩码预测通道
                
                # 1. MAE重构损失（在掩码区域）
                mae_loss = criterion(mae_rgb * mask, video * mask)
                
                # 2. 掩码预测损失（让模型学会预测哪些区域需要修复）
                # 真实掩码作为监督信号
                # mask_loss = F.binary_cross_entropy(mae_mask_pred, mask.float())
                
                # 3. 最终输出损失（LaMa处理后的结果）
                final_loss = criterion(reconstructed * mask, video * mask)
                
                # 4. 组合损失
                total_batch_loss = 125*final_loss + 125*mae_loss #+ 0.2 * mask_loss

                
            else:
                reconstructed = model(masked_video, mask, ocean_mask)
                if reconstructed.shape[2] != video.shape[2]:
                    video = torch.cat([video, mask], dim=2)
                total_batch_loss = criterion(255*reconstructed * mask, 255*video * mask)
            
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def inference_with_pretrained(out_channels,model_path, data_dir, model, dataset, dataloader, input_seq_len=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    sample = next(iter(dataloader))
    video = sample['video'].to(device)
    masked_video = sample['masked'].to(device)
    mask = sample['mask'].to(device)
    times = sample['times']
    lama = sample['lama_init'].to(device)
    ocean_mask = sample['ocean_mask'].to(device)

    with torch.no_grad():
        output = model(masked_video, mask,ocean_mask)
        # output = model(masked_video, None)  # 无掩码
        if output.shape[2] in (1, 3):
            outputRGB = output[:, :, :, :]  # 单通道或三通道输出
        else:
            outputRGB = output[:, :, :out_channels-1, :]  # 只保留RGB通道
    # print(mask.shape,output.shape)
    # 组合未缺失和重构区域
    combined = outputRGB * mask + video * (1 - mask)
    
    # 反归一化
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1).cpu() # 归一化到[0,1]
        # return torch.clamp(img, 0, 1).cpu()

    #* 计算指标（使用整幅图但在mask区域计算）
    def calculate_metrics(original, reconstructed, mask):
        """
        original/reconstructed: torch.Tensor (C,H,W) 或 (H,W)
        mask: torch.Tensor (H,W) bool
        return: ssim, psnr, mae  (全部在 mask 区域计算)
        """
        # 1. 确保输入保持原始形状
        original_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        
        # 2. 统一处理通道顺序
        if original_np.ndim == 3 and original_np.shape[0] in [1, 3]:  # (C,H,W)
            original_np   = np.moveaxis(original_np, 0, -1)
            recon_np      = np.moveaxis(recon_np, 0, -1)
            mask_np       = np.squeeze(mask_np, axis=0)
            channel_axis  = -1
        else:                                   # 单通道
            original_np   = np.squeeze(original_np, axis=0)
            recon_np      = np.squeeze(recon_np, axis=0)
            mask_np       = np.squeeze(mask_np, axis=0)
            channel_axis  = None
        
        # 3. 只保留 mask 区域
        orig_masked = original_np[mask_np]
        recon_masked = recon_np[mask_np]
        #* 也在整图上计算，但非mask区域置0
        orig_masked_img = original_np.copy()
        recon_masked_img = recon_np.copy()
        orig_masked_img[~mask_np] = 0
        recon_masked_img[~mask_np] = 0
        if orig_masked.size == 0:
            return 0.0, 0.0, 0.0

        # 4. MAE (在0-255范围计算)
        mae = float(np.mean(np.abs(orig_masked - recon_masked))) * 255.0

        # 5. PSNR
        psnr_val = float(psnr(recon_masked_img, orig_masked_img, data_range=1.0))  # 注意data_range=1.0

        # 6. SSIM - 使用整幅图但在mask区域计算
        win_size = min(7, min(original_np.shape[0], original_np.shape[1]) - 1 or 3)
        ssim_val = float(
            ssim(
                orig_masked_img,
                recon_masked_img,
                win_size=win_size,
                data_range=1.0,  # 图像范围[0,1]
                channel_axis=channel_axis
            )
        )
        return ssim_val, psnr_val, mae

    # 可视化第t帧
    t = 2
    plt.figure(figsize=(15, 5))

    # 原始帧
    plt.subplot(1, 3, 1)
    original_img = unnorm(video[0, t])
    plt.imshow(original_img.permute(1, 2, 0))
    plt.title(f'Original Frame\nTime: {times[0, t].item()}')
    plt.axis('off')

    # 带掩码的帧
    plt.subplot(1, 3, 2)
    masked_img = unnorm(masked_video[0, t])
    plt.imshow(masked_img.permute(1, 2, 0))
    plt.title('Masked Frame')
    plt.axis('off')

    # 重建帧
    plt.subplot(1, 3, 3)
    # recon_img = unnorm(outputRGB[0, t])
    recon_img = unnorm(combined[0, t])
    plt.imshow(recon_img.permute(1, 2, 0))
    plt.title(f'Reconstructed Frame')
    plt.axis('off')

    
    plt.tight_layout()
    plt.savefig('inference_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 在循环中正确计算每帧指标
    print("\n逐帧评估指标（仅mask区域）:")
    print(f"{'Time':<10}{'SSIM':<10}{'PSNR (dB)':<12}{'MAE':<8}")
    print("-" * 40)

    for t in range(input_seq_len):
        # 获取当前帧的数据
        original = unnorm(video[0, t])
        reconstructed = unnorm(combined[0, t])  # 使用组合后的完整图像
        
        # 计算当前帧的指标
        ssim_val, psnr_val, mae = calculate_metrics(
            original, 
            reconstructed, 
            mask[0, t].cpu()
        )
        
        # 保存重建结果（保持原始形状）
        if reconstructed.shape[0] == 1:  # 单通道
            recon_img = Image.fromarray((reconstructed.squeeze(0).numpy() * 255).astype(np.uint8), 'L')
        else:  # 多通道
            recon_img = Image.fromarray((reconstructed.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        os.makedirs('./inpainted_VMAE', exist_ok=True)
        recon_img.save(f'./inpainted_VMAE/reconstructed_frame_{times[0, t].item():04d}.png')
        
        # 打印当前帧的指标
        print(f"{times[0, t].item():<10}{ssim_val:.4f}{'':<2}{psnr_val:.2f}{'':<5}{mae:.4f}")

