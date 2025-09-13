import torch
import numpy as np
import time
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import warnings
from scipy.interpolate import CubicSpline, interp1d
from sklearn.ensemble import RandomForestRegressor
from pykrige.ok import OrdinaryKriging
import copy
from MAE_LaMa import*

def run_experiments_for_imagery(model, experiment_config, out_channels=3, input_seq_len=8):
    """
    专注于输出影像的实验运行函数
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # 实验配置映射表
    experiment_mapping = {
        'missing_types': ('缺失类型对比', 'exp1_missing_types/{}'),
        'missing_ratios': ('缺失比例对比', 'exp2_missing_ratios/{}percent'),
    }
    
    # 只处理配置中实际存在的实验
    available_experiments = set(experiment_config.keys()) & set(experiment_mapping.keys())
    
    if not available_experiments:
        print("没有找到可用的实验配置")
        return {}
    
    print(f"找到 {len(available_experiments)} 个可用实验: {list(available_experiments)}")
    
    for exp_key in available_experiments:
        exp_data = experiment_config[exp_key]
        if not exp_data:
            print(f"跳过 {exp_key}（配置为空）")
            continue
            
        exp_name, path_template = experiment_mapping[exp_key]
        print("="*60)
        print(f"开始实验: {exp_name}")
        
        processed_count = 0
        
        for condition, dataloader in exp_data.items():
            if dataloader is None:
                print(f"  跳过 {condition}（数据加载器为None）")
                continue
                
            try:
                print(f"  正在处理 {condition}...")
                # 生成保存路径
                save_path = path_template.format(condition)
                process_and_save_imagery(model, dataloader, out_channels, input_seq_len, device, save_path)
                
                processed_count += 1
                print(f"  {condition} 处理完成")
                
            except Exception as e:
                print(f"  处理 {condition} 时出错: {e}")
        
        print(f"  {exp_name} 完成，处理了 {processed_count}/{len(exp_data)} 个条件")
    
    print("所有实验完成！影像已保存到相应目录")
    
    return {}


def process_and_save_imagery(model, dataloader, out_channels, input_seq_len, device, save_path):
    """处理并保存所有算法的输出影像"""
    os.makedirs(f'experiment_results/{save_path}', exist_ok=True)
    
    # 只处理第一个样本
    for i, sample in enumerate(dataloader):
        if i >= 1:  # 只处理一个样本
            break
            
        video = sample['video'].to(device)
        masked_video = sample['masked'].to(device)
        mask = sample['mask'].to(device)
        ocean_mask = sample.get('ocean_mask', torch.zeros_like(mask)).to(device)

        # 模型推理
        start_time = time.time()
        with torch.no_grad():
            output = model(masked_video, mask, ocean_mask)
            if output.shape[2] in (1, 3):
                outputRGB = output[:, :, :, :]
            else:
                outputRGB = output[:, :, :out_channels-1, :]
        
        # 计算各种方法的结果
        methods_results = calculate_all_methods(masked_video, mask, device, outputRGB, video)
        
        # 保存原始视频和掩码
        save_video_and_mask(video, masked_video, mask, save_path)
        
        # 保存所有方法的结果
        save_all_methods_results(methods_results, save_path)
        
        # 生成对比图
        generate_comparison_figures(methods_results, video, mask, save_path)


def calculate_all_methods(masked_video, mask, device, model_output, original_video):
    """计算所有插值方法的结果"""
    methods = {}
    
    # 预训练模型输出
    methods["simplelama"] = model_output
    
    # DINEOF
    try:
        methods["DINEOF"] = apply_dineof_improved(masked_video, mask, device)
    except Exception as e:
        print(f"DINEOF计算失败: {e}")
        methods["DINEOF"] = masked_video.clone()
    
    # Kriging
    try:
        methods["Kriging"] = apply_kriging_interpolation(masked_video, mask, device)
    except Exception as e:
        print(f"Kriging计算失败: {e}")
        methods["Kriging"] = masked_video.clone()
    
    # IDW
    try:
        methods["IDW"] = apply_idw_interpolation(masked_video, mask, device)
    except Exception as e:
        print(f"IDW计算失败: {e}")
        methods["IDW"] = masked_video.clone()
    
    # Nearest Neighbor
    try:
        methods["Nearest Neighbor"] = apply_nearest_neighbor_simple(masked_video, mask, device)
    except Exception as e:
        print(f"Nearest Neighbor计算失败: {e}")
        methods["Nearest Neighbor"] = masked_video.clone()
    
    # Random Forest
    try:
        methods["Random Forest"] = apply_random_forest_interpolation(masked_video, mask, device, original_video)
    except Exception as e:
        print(f"Random Forest计算失败: {e}")
        methods["Random Forest"] = masked_video.clone()
    
    return methods


def apply_dineof_improved(masked_data, mask, device, max_modes=10, n_iter=20, tolerance=1e-6):
    """改进的DINEOF实现"""
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    for b in range(B):
        for c in range(C):
            # 获取该通道的时空数据 [T, H, W]
            channel_data = masked_data[b, :, c, :, :].cpu().numpy()
            channel_mask = mask[b, :, 0, :, :].cpu().numpy() > 0.5
            
            # 重塑为二维矩阵 [时间, 空间]
            spatial_shape = (H, W)
            data_2d = channel_data.reshape(T, -1)  # [T, H*W]
            mask_2d = channel_mask.reshape(T, -1)  # [T, H*W]
            
            # 初始填充：使用时间均值填充缺失值
            initial_guess = initialize_dineof(data_2d, mask_2d)
            
            # DINEOF迭代过程
            reconstructed = dineof_iteration(initial_guess, mask_2d, max_modes, n_iter, tolerance)
            
            # 重塑回原始形状
            reconstructed_3d = reconstructed.reshape(T, H, W)
            result[b, :, c, :, :] = torch.from_numpy(reconstructed_3d).to(device)
    
    return result


def initialize_dineof(data_2d, mask_2d):
    """DINEOF初始化"""
    initialized = data_2d.copy()
    
    # 对每个空间点，用时间均值填充缺失值
    for j in range(data_2d.shape[1]):
        if np.any(mask_2d[:, j]):
            valid_values = data_2d[~mask_2d[:, j], j]
            if len(valid_values) > 0:
                time_mean = np.mean(valid_values)
                initialized[mask_2d[:, j], j] = time_mean
            else:
                # 如果所有时间点都缺失，使用全局均值
                initialized[mask_2d[:, j], j] = np.nanmean(data_2d)
    
    # 处理剩余的NaN值
    initialized = np.nan_to_num(initialized, nan=np.nanmean(data_2d))
    return initialized


def dineof_iteration(initial_data, mask, max_modes=10, n_iter=20, tolerance=1e-6):
    """DINEOF核心迭代过程"""
    current_guess = initial_data.copy()
    prev_rmse = float('inf')
    
    for iteration in range(n_iter):
        # 1. 对当前猜测进行SVD分解
        U, s, Vt = np.linalg.svd(current_guess, full_matrices=False)
        
        # 2. 交叉验证确定最优模态数量（简化版）
        optimal_modes = find_optimal_modes(current_guess, mask, U, s, Vt, max_modes)
        
        # 3. 使用选定模态重建数据
        reconstructed = U[:, :optimal_modes] @ np.diag(s[:optimal_modes]) @ Vt[:optimal_modes, :]
        
        # 4. 保持已知值不变，只更新缺失值
        current_guess[mask] = reconstructed[mask]
        
        # 5. 检查收敛性
        rmse = calculate_rmse(reconstructed, current_guess, mask)
        if abs(prev_rmse - rmse) < tolerance:
            break
        
        prev_rmse = rmse
    
    return current_guess


def find_optimal_modes(data, mask, U, s, Vt, max_modes):
    """通过交叉验证确定最优EOF模态数量（简化版）"""
    total_variance = np.sum(s ** 2)
    explained_variance = np.cumsum(s ** 2) / total_variance
    
    # 找到解释方差超过95%的最小模态数
    optimal_modes = np.argmax(explained_variance >= 0.95) + 1
    return min(optimal_modes, max_modes)


def calculate_rmse(reconstructed, original, mask):
    """计算均方根误差"""
    diff = reconstructed - original
    return np.sqrt(np.mean(diff[mask] ** 2))


def apply_kriging_interpolation(masked_data, mask, device):
    """应用克里金插值"""
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    for b in range(B):
        for t in range(T):
            for c in range(C):
                frame = masked_data[b, t, c].cpu().numpy()
                mask_frame = mask[b, t, 0].cpu().numpy() > 0.5
                
                # 获取已知点的坐标和值
                known_coords = np.argwhere(~mask_frame)
                known_values = frame[~mask_frame]
                
                # 获取未知点的坐标
                unknown_coords = np.argwhere(mask_frame)
                
                if len(known_coords) > 0 and len(unknown_coords) > 0:
                    try:
                        # 创建克里金插值模型
                        ok = OrdinaryKriging(
                            known_coords[:, 0], 
                            known_coords[:, 1], 
                            known_values,
                            variogram_model='linear',
                            verbose=False,
                            enable_plotting=False
                        )
                        
                        # 预测未知点的值
                        z, ss = ok.execute('points', unknown_coords[:, 0], unknown_coords[:, 1])
                        
                        # 将预测值填充到结果中
                        for idx, coord in enumerate(unknown_coords):
                            frame[coord[0], coord[1]] = z[idx]
                    except Exception as e:
                        print(f"克里金插值出错: {e}")
                        # 如果克里金失败，使用最近邻作为备用
                        tree = KDTree(known_coords)
                        distances, indices = tree.query(unknown_coords, k=1)
                        for j, coord in enumerate(unknown_coords):
                            frame[coord[0], coord[1]] = frame[known_coords[indices[j]][0], known_coords[indices[j]][1]]
                
                result[b, t, c] = torch.from_numpy(frame).to(device)
    
    return result


def apply_idw_interpolation(masked_data, mask, device, power=2):
    """应用反距离加权插值"""
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    for b in range(B):
        for t in range(T):
            for c in range(C):
                frame = masked_data[b, t, c].cpu().numpy()
                mask_frame = mask[b, t, 0].cpu().numpy() > 0.5
                
                known_coords = np.argwhere(~mask_frame)
                known_values = frame[~mask_frame]
                unknown_coords = np.argwhere(mask_frame)
                
                if len(known_coords) > 0 and len(unknown_coords) > 0:
                    # 为每个未知点计算IDW
                    for coord in unknown_coords:
                        # 计算到所有已知点的距离
                        distances = np.sqrt(np.sum((known_coords - coord) ** 2, axis=1))
                        
                        # 避免除以零
                        distances[distances == 0] = 1e-10
                        
                        # 计算权重
                        weights = 1 / (distances ** power)
                        
                        # 计算加权平均值
                        weighted_sum = np.sum(weights * known_values)
                        weight_sum = np.sum(weights)
                        
                        frame[coord[0], coord[1]] = weighted_sum / weight_sum
                
                result[b, t, c] = torch.from_numpy(frame).to(device)
    
    return result


def apply_nearest_neighbor_simple(masked_data, mask, device):
    """简化的最近邻插值"""
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    for b in range(B):
        for t in range(T):
            for c in range(C):
                frame = masked_data[b, t, c].cpu().numpy()
                mask_frame = mask[b, t, 0].cpu().numpy() > 0.5
                
                known_coords = np.argwhere(~mask_frame)
                unknown_coords = np.argwhere(mask_frame)
                
                if len(known_coords) > 0 and len(unknown_coords) > 0:
                    tree = KDTree(known_coords)
                    distances, indices = tree.query(unknown_coords, k=1)
                    for j, coord in enumerate(unknown_coords):
                        frame[coord[0], coord[1]] = frame[known_coords[indices[j]][0], known_coords[indices[j]][1]]
                
                result[b, t, c] = torch.from_numpy(frame).to(device)
    
    return result


def apply_random_forest_interpolation(masked_data, mask, device, original_video=None):
    """应用随机森林插值"""
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    # 使用第一帧训练随机森林模型
    for b in range(B):
        for c in range(C):
            # 获取当前通道的所有帧
            channel_frames = masked_data[b, :, c].cpu().numpy()  # [T, H, W]
            channel_mask = mask[b, :, 0].cpu().numpy() > 0.5  # [T, H, W]
            
            # 使用第一帧训练模型
            train_frame = channel_frames[0]
            train_mask = channel_mask[0]
            
            # 获取训练数据
            train_coords = np.argwhere(~train_mask)
            train_values = train_frame[~train_mask]
            
            if len(train_coords) > 0:
                # 训练随机森林模型
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(train_coords, train_values)
                
                # 对所有帧应用模型
                for t in range(T):
                    frame = channel_frames[t].copy()
                    mask_frame = channel_mask[t]
                    
                    # 获取需要预测的坐标
                    predict_coords = np.argwhere(mask_frame)
                    
                    if len(predict_coords) > 0:
                        # 使用随机森林预测缺失值
                        predicted_values = rf.predict(predict_coords)
                        
                        # 填充预测值
                        for idx, coord in enumerate(predict_coords):
                            frame[coord[0], coord[1]] = predicted_values[idx]
                    
                    result[b, t, c] = torch.from_numpy(frame).to(device)
    
    return result


def save_video_and_mask(video, masked_video, mask, save_path):
    """保存原始视频和掩码"""
    os.makedirs(f'experiment_results/{save_path}/original', exist_ok=True)
    
    # 反归一化函数
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)
    
    # 保存原始视频帧
    B, T, C, H, W = video.shape
    for t in range(T):
        frame = unnorm(video[0, t]).cpu()
        if frame.shape[0] == 1:  # 单通道
            img = Image.fromarray((frame.squeeze(0).numpy() * 255).astype(np.uint8), 'L')
        else:  # 多通道
            img_array = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, 'RGB')
        img.save(f'experiment_results/{save_path}/original/frame_{t:02d}.png')
    
    # 保存掩码视频帧
    os.makedirs(f'experiment_results/{save_path}/masked', exist_ok=True)
    for t in range(T):
        frame = unnorm(masked_video[0, t]).cpu()
        if frame.shape[0] == 1:  # 单通道
            img = Image.fromarray((frame.squeeze(0).numpy() * 255).astype(np.uint8), 'L')
        else:  # 多通道
            img_array = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, 'RGB')
        img.save(f'experiment_results/{save_path}/masked/frame_{t:02d}.png')
    
    # 保存掩码
    os.makedirs(f'experiment_results/{save_path}/mask', exist_ok=True)
    for t in range(T):
        mask_frame = mask[0, t, 0].cpu().numpy()  # 取第一个通道
        img = Image.fromarray((mask_frame * 255).astype(np.uint8), 'L')
        img.save(f'experiment_results/{save_path}/mask/frame_{t:02d}.png')


def save_all_methods_results(methods_results, save_path):
    """保存所有方法的结果"""
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)
    
    for method_name, result in methods_results.items():
        method_dir = f'experiment_results/{save_path}/{method_name}'
        os.makedirs(method_dir, exist_ok=True)
        
        B, T, C, H, W = result.shape
        for t in range(T):
            frame = unnorm(result[0, t]).cpu()
            if frame.shape[0] == 1:  # 单通道
                img = Image.fromarray((frame.squeeze(0).numpy() * 255).astype(np.uint8), 'L')
            else:  # 多通道
                img_array = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, 'RGB')
            img.save(f'{method_dir}/frame_{t:02d}.png')


def generate_comparison_figures(methods_results, video, mask, save_path):
    """生成对比图"""
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)
    
    os.makedirs(f'experiment_results/{save_path}/comparison', exist_ok=True)
    
    # 选择中间帧进行比较
    T = video.shape[1]
    t = T // 2
    
    # 获取原始帧和掩码帧
    original_frame = unnorm(video[0, t]).cpu()
    mask_frame = mask[0, t, 0].cpu().numpy()
    
    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flat
    
    # 原始图像
    if original_frame.shape[0] == 1:
        axes[0].imshow(original_frame.squeeze(0), cmap='gray')
    else:
        axes[0].imshow(original_frame.permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 掩码图像
    masked_frame = unnorm(video[0, t] * (1 - mask[0, t])).cpu()
    if masked_frame.shape[0] == 1:
        axes[1].imshow(masked_frame.squeeze(0), cmap='gray')
    else:
        axes[1].imshow(masked_frame.permute(1, 2, 0))
    axes[1].set_title('Masked')
    axes[1].axis('off')
    
    # 各方法结果
    methods = list(methods_results.keys())
    for i, method in enumerate(methods):
        result_frame = unnorm(methods_results[method][0, t]).cpu()
        if result_frame.shape[0] == 1:
            axes[i+2].imshow(result_frame.squeeze(0), cmap='gray')
        else:
            axes[i+2].imshow(result_frame.permute(1, 2, 0))
        axes[i+2].set_title(method)
        axes[i+2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'experiment_results/{save_path}/comparison/frame_{t:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_experiment_config(model, out_channels=3,
                             dataloader_thin_cloud=None, dataloader_thick_cloud=None,
                             dataloader_strip=None, dataloader_mixed=None,
                             dataloader_10percent=None, dataloader_20percent=None,
                             dataloader_30percent=None, dataloader_40percent=None,
                             dataloader_50percent=None, dataloader_60percent=None):
    """创建实验配置并运行实验"""
    
    experiment_config = {
        'missing_types': {
            'thin_cloud': dataloader_thin_cloud,
            'thick_cloud': dataloader_thick_cloud,
            'strip': dataloader_strip,
            'mixed': dataloader_mixed
        },
        'missing_ratios': {
            10: dataloader_10percent,
            20: dataloader_20percent,
            30: dataloader_30percent,
            40: dataloader_40percent,
            50: dataloader_50percent,
            60: dataloader_60percent
        }
    }
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load('fine_tuned_model.pth'))
        print("模型权重加载成功")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        print("使用随机初始化的模型进行测试")
    
    # 运行所有实验
    results = run_experiments_for_imagery(model, experiment_config, out_channels)
    print("实验影像已保存到 experiment_results/ 目录")
    
    return results


# 主函数
def main():
    # 这里需要根据您的实际数据加载器进行配置
    # 示例代码，需要替换为实际的数据加载器
    # 模型初始化
    data_dir = 'E:/lama/jet_S2_Daily_Mosaic/'
    ocean_mask_path = 'E:/lama/S2_Daily_Mosaic_Masked/mask.png'
    input_seq_len = 8
    out_channels = 4
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model = VideoCompletionModel(
        img_size_h=2170,
        img_size_w=1376,
        patch_size=4,
        embed_dim=96,
        num_heads=4,
        max_seq_len=input_seq_len,
        use_mask_channel=True,
        use_lama_init=False,
        out_channels=out_channels,
        dropout=0.2,
        freeze_backbone=True,
        fine_tune_layers=['decoder', 'mask_update_layer']
    ).to(device)
    
    # 加载预训练权重（如果存在）
    try:
        model.load_state_dict(torch.load('fine_tuned_model.pth', map_location=device))
        print("成功加载预训练权重")
    except FileNotFoundError:
        print("未找到预训练权重，使用随机初始化")
    
    # 准备数据加载器
    print("准备数据加载器...")
    
    # 不同缺失比例的数据加载器
    dataloaders = {}
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        dataset = Datasets_inference(
            data_dir=data_dir,
            ocean_mask_path=ocean_mask_path,
            mask_type="cloud",
            mask_ratio=ratio
        )
        dataloaders[f'dataloader_{int(ratio*100)}percent'] = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 不同缺失类型的数据加载器
    for mask_type in ["thin_cloud", "strip", "mixed"]:
        dataset = Datasets_inference(
            data_dir=data_dir,
            ocean_mask_path=ocean_mask_path,
            mask_type=mask_type,
            mask_ratio=0.3
        )
        dataloaders[f'dataloader_{mask_type}'] = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 厚云使用30%缺失比例
    dataloaders['dataloader_thick_cloud'] = dataloaders['dataloader_30percent']
    
    # 创建实验配置
    experiment_config = {
        'missing_types': {
            'thin_cloud': dataloaders['dataloader_thin_cloud'],
            'thick_cloud': dataloaders['dataloader_thick_cloud'],
            'strip': dataloaders['dataloader_strip'],
            'mixed': dataloaders['dataloader_mixed']
        },
        'missing_ratios': {
            10: dataloaders['dataloader_10percent'],
            20: dataloaders['dataloader_20percent'],
            30: dataloaders['dataloader_30percent'],
            40: dataloaders['dataloader_40percent'],
            50: dataloaders['dataloader_50percent'],
            60: dataloaders['dataloader_60percent']
        }
    }
    
    
    
    # 运行实验
    results = run_experiments_for_imagery(model, experiment_config, out_channels=3)


if __name__ == "__main__":
    main()
