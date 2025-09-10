import torch
import numpy as np
import time
import json
import os
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def run_four_experiments(model, experiment_config, out_channels=3, input_seq_len=8):
    """
    选择性运行实验，支持动态配置
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    all_experiment_results = {}
    
    # 实验配置映射表：配置键 -> (实验名称, 保存路径模板)
    experiment_mapping = {
        'missing_types': ('缺失类型对比', 'exp1_missing_types/{}'),
        'missing_ratios': ('缺失比例对比', 'exp2_missing_ratios/{}percent'),
        'terrain_types': ('地形类型对比', 'exp3_terrain_types/{}'),
        'time_intervals': ('时间间隔对比', 'exp4_time_intervals/{}days'),
        'cloud_types': ('云类型对比', 'exp5_cloud_types/{}'),  # 自定义实验
        'seasons': ('季节对比', 'exp6_seasons/{}')  # 自定义实验
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
        
        all_experiment_results[exp_key] = {}
        processed_count = 0
        
        for condition, dataloader in exp_data.items():
            if dataloader is None:
                print(f"  跳过 {condition}（数据加载器为None）")
                continue
                
            try:
                print(f"  正在处理 {condition}...")
                results = calculate_experiment_metrics(model, dataloader, out_channels, input_seq_len, device)
                all_experiment_results[exp_key][condition] = results
                
                # 生成保存路径
                save_path = path_template.format(condition)
                save_experiment_results(results, save_path)
                
                processed_count += 1
                print(f"  {condition} 处理完成")
                
            except Exception as e:
                print(f"  处理 {condition} 时出错: {e}")
                all_experiment_results[exp_key][condition] = {'error': str(e)}
        
        print(f"  {exp_name} 完成，处理了 {processed_count}/{len(exp_data)} 个条件")
    
    # 保存结果
    if all_experiment_results:
        save_comprehensive_results(all_experiment_results)
        generate_analysis_report(all_experiment_results)
        print("所有实验完成！")
    else:
        print("没有成功处理任何实验")
    
    return all_experiment_results

def calculate_experiment_metrics(model, dataloader, out_channels, input_seq_len, device):
    """计算单个实验的指标"""
    # 处理多个批次以获得更可靠的统计结果
    all_metrics = []
    total_inference_time = 0
    batch_count = 0
    
    for i, sample in enumerate(dataloader):
        if i >= 5:  # 限制处理的批次数量以节省时间
            break
            
        video = sample['video'].to(device)
        masked_video = sample['masked'].to(device)
        mask = sample['mask'].to(device)
        times = sample.get('times', None)
        ocean_mask = sample.get('ocean_mask', torch.zeros_like(mask)).to(device)

        # 模型推理
        start_time = time.time()
        with torch.no_grad():
            output = model(masked_video, mask, ocean_mask)
            if output.shape[2] in (1, 3):
                outputRGB = output[:, :, :, :]
            else:
                outputRGB = output[:, :, :out_channels-1, :]
        
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        batch_count += 1
        
        combined = outputRGB * mask + video * (1 - mask)
        
        # 计算经典方法
        methods_results = calculate_classical_methods(masked_video, mask, device)
        methods_results["Proposed"] = combined
        
        # 计算指标
        metrics = calculate_all_metrics(video, masked_video, mask, methods_results, input_seq_len)
        all_metrics.append(metrics)
    
    # 聚合多批次结果
    aggregated_metrics = aggregate_batch_metrics(all_metrics)
    
    return {
        'metrics': aggregated_metrics,
        'methods_results': {k: v.cpu() for k, v in methods_results.items()},
        'video': video.cpu(),
        'masked_video': masked_video.cpu(),
        'mask': mask.cpu(),
        'times': times,
        'avg_inference_time': total_inference_time / batch_count if batch_count > 0 else 0
    }

def aggregate_batch_metrics(all_metrics):
    """聚合多批次的指标结果"""
    if not all_metrics:
        return {'avg_metrics': {}, 'frame_metrics': {}}
    
    # 获取所有方法名称
    method_names = list(all_metrics[0]['avg_metrics'].keys())
    metric_names = ['SSIM', 'PSNR', 'MAE', 'TCC']
    
    aggregated_avg = {}
    for method in method_names:
        aggregated_avg[method] = {}
        for metric in metric_names:
            values = [batch_metrics['avg_metrics'][method][metric] for batch_metrics in all_metrics]
            aggregated_avg[method][metric] = float(np.mean(values))
    
    # 对于frame_metrics，取第一个批次的结果作为示例
    aggregated_frame = all_metrics[0]['frame_metrics']
    
    return {
        'avg_metrics': aggregated_avg,
        'frame_metrics': aggregated_frame
    }

def calculate_classical_methods(masked_video, mask, device):
    """计算经典插值方法"""
    methods = {}
    
    # DINEOF - 改进实现
    try:
        methods["DINEOF"] = apply_dineof_improved(masked_video, mask, device)
    except Exception as e:
        print(f"DINEOF计算失败: {e}")
        methods["DINEOF"] = masked_video.clone()
    
    # Nearest Neighbor
    try:
        methods["Nearest Neighbor"] = apply_nearest_neighbor_simple(masked_video, mask, device)
    except Exception as e:
        print(f"Nearest Neighbor计算失败: {e}")
        methods["Nearest Neighbor"] = masked_video.clone()
    
    # Spline Interpolation
    try:
        methods["Spline"] = apply_spline_interpolation(masked_video, mask, device)
    except Exception as e:
        print(f"Spline计算失败: {e}")
        methods["Spline"] = masked_video.clone()
    
    return methods

def apply_dineof_improved(masked_data, mask, device, max_modes=10, n_iter=20, tolerance=1e-6):
    """
    改进的DINEOF实现
    基于经验正交函数(EOF)的时空数据插值方法
    """
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
    """DINEOF初始化：使用时间均值填充缺失值"""
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
    """
    DINEOF核心迭代过程
    """
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
    """
    通过交叉验证确定最优EOF模态数量（简化版）
    """
    # 在实际DINEOF中，这会涉及复杂的交叉验证过程
    # 这里使用简化版本：选择解释方差超过95%的模态
    
    total_variance = np.sum(s ** 2)
    explained_variance = np.cumsum(s ** 2) / total_variance
    
    # 找到解释方差超过95%的最小模态数
    optimal_modes = np.argmax(explained_variance >= 0.95) + 1
    return min(optimal_modes, max_modes)

def calculate_rmse(reconstructed, original, mask):
    """计算均方根误差"""
    diff = reconstructed - original
    return np.sqrt(np.mean(diff[mask] ** 2))


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

def apply_spline_interpolation(masked_data, mask, device):
    """
    时序样条插值：对每个像素的时间序列独立进行样条插值
    避免空间插值的内存问题，更适合时间序列数据
    """
    from scipy.interpolate import CubicSpline, interp1d
    import warnings
    
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    # 忽略插值相关的警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    for b in range(B):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    # 获取单个像素的时间序列 [T]
                    pixel_series = masked_data[b, :, c, h, w].cpu().numpy()
                    mask_series = mask[b, :, 0, h, w].cpu().numpy() > 0.5
                    
                    # 只有在这个像素有时间缺失时才进行插值
                    if np.any(mask_series) and not np.all(mask_series):
                        # 有效时间点（已知值）
                        valid_times = np.where(~mask_series)[0]
                        valid_values = pixel_series[~mask_series]
                        
                        # 缺失时间点
                        missing_times = np.where(mask_series)[0]
                        
                        if len(valid_times) >= 4:  # 三次样条至少需要4个点
                            try:
                                # 使用三次样条插值（更平滑）
                                cs = CubicSpline(valid_times, valid_values, 
                                               bc_type='natural', extrapolate=True)
                                interpolated_values = cs(missing_times)
                                
                                # 确保插值结果在合理范围内
                                valid_range = (np.min(valid_values), np.max(valid_values))
                                interpolated_values = np.clip(interpolated_values, 
                                                            valid_range[0] - 0.1 * (valid_range[1] - valid_range[0]),
                                                            valid_range[1] + 0.1 * (valid_range[1] - valid_range[0]))
                                
                                # 更新结果
                                for i, t in enumerate(missing_times):
                                    result[b, t, c, h, w] = torch.tensor(interpolated_values[i]).to(device)
                                    
                            except Exception as e:
                                # 如果三次样条失败，使用线性插值
                                if len(valid_times) >= 2:
                                    try:
                                        interp_func = interp1d(valid_times, valid_values, 
                                                             kind='linear',
                                                             bounds_error=False,
                                                             fill_value="extrapolate")
                                        interpolated_values = interp_func(missing_times)
                                        for i, t in enumerate(missing_times):
                                            result[b, t, c, h, w] = torch.tensor(interpolated_values[i]).to(device)
                                    except:
                                        # 如果线性插值也失败，使用最近时间点的值
                                        for t in missing_times:
                                            nearest_time = valid_times[np.argmin(np.abs(valid_times - t))]
                                            result[b, t, c, h, w] = result[b, nearest_time, c, h, w]
                        
                        elif len(valid_times) >= 2:
                            # 点太少，使用线性插值
                            try:
                                interp_func = interp1d(valid_times, valid_values, 
                                                     kind='linear',
                                                     bounds_error=False,
                                                     fill_value="extrapolate")
                                interpolated_values = interp_func(missing_times)
                                for i, t in enumerate(missing_times):
                                    result[b, t, c, h, w] = torch.tensor(interpolated_values[i]).to(device)
                            except:
                                # 使用最近时间点的值
                                for t in missing_times:
                                    nearest_time = valid_times[np.argmin(np.abs(valid_times - t))]
                                    result[b, t, c, h, w] = result[b, nearest_time, c, h, w]
                        
                        else:
                            # 只有一个有效点，使用该点的值
                            for t in missing_times:
                                result[b, t, c, h, w] = result[b, valid_times[0], c, h, w]
    
    warnings.resetwarnings()
    return result

def calculate_all_metrics(video, masked_video, mask, methods_results, input_seq_len):
    """计算所有指标"""
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1).cpu()
    
    metrics = {name: {"SSIM": [], "PSNR": [], "MAE": []} for name in methods_results.keys()}
    tcc_values = {}
    
    # 逐帧计算指标
    for t in range(min(input_seq_len, video.shape[1])):
        original_frame = unnorm(video[0, t])
        mask_frame = mask[0, t].cpu()
        
        for name, result in methods_results.items():
            if t < result.shape[1]:
                result_frame = unnorm(result[0, t])
                ssim_val, psnr_val, mae = calculate_single_metrics(original_frame, result_frame, mask_frame)
                
                metrics[name]["SSIM"].append(ssim_val)
                metrics[name]["PSNR"].append(psnr_val)
                metrics[name]["MAE"].append(mae)
    
    # 计算TCC
    for name, result in methods_results.items():
        tcc_values[name] = calculate_tcc_fast(
            video[0].cpu(),
            result[0].cpu(),
            mask[0].cpu()
        )
    
    # 计算平均值
    avg_metrics = {}
    for name in methods_results.keys():
        if metrics[name]["SSIM"]:  # 确保有数据
            avg_metrics[name] = {
                "SSIM": float(np.mean(metrics[name]["SSIM"])),
                "PSNR": float(np.mean(metrics[name]["PSNR"])),
                "MAE": float(np.mean(metrics[name]["MAE"])),
                "TCC": float(tcc_values[name])
            }
        else:
            avg_metrics[name] = {"SSIM": 0.0, "PSNR": 0.0, "MAE": 0.0, "TCC": 0.0}
    
    return {'frame_metrics': metrics, 'avg_metrics': avg_metrics}

def calculate_single_metrics(original, reconstructed, mask):
    """计算单帧指标 - 修复维度问题"""
    original_np = original.numpy()
    recon_np = reconstructed.numpy()
    
    # 处理mask维度
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask_np = mask.squeeze(0).numpy().astype(bool)
    elif mask.dim() == 2:
        mask_np = mask.numpy().astype(bool)
    else:
        mask_np = mask.numpy().astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
    
    # 确保所有数组维度匹配
    if original_np.ndim == 3:  # [C, H, W]
        C, H, W = original_np.shape
        if mask_np.shape != (H, W):
            print(f"维度不匹配: original {original_np.shape}, mask {mask_np.shape}")
            return 0.0, 0.0, 0.0
    
    # MAE计算 - 只在mask区域计算
    if np.any(mask_np):
        if original_np.ndim == 3:
            mae = float(np.mean(np.abs(original_np - recon_np)[:, mask_np]))
        else:
            mae = float(np.mean(np.abs(original_np - recon_np)[mask_np]))
        mae *= 255.0
    else:
        mae = 0.0
    
    # PSNR和SSIM
    try:
        if original_np.ndim == 3:
            psnr_val = psnr(original_np, recon_np, data_range=1.0)
            ssim_val = ssim(original_np, recon_np, data_range=1.0, channel_axis=0)
        else:
            psnr_val = psnr(original_np, recon_np, data_range=1.0)
            ssim_val = ssim(original_np, recon_np, data_range=1.0)
    except Exception as e:
        print(f"PSNR/SSIM计算错误: {e}")
        psnr_val, ssim_val = 0.0, 0.0
    
    return float(ssim_val), float(psnr_val), mae

def calculate_tcc_fast(original_seq, reconstructed_seq, mask_seq):
    """快速计算TCC - 时间相关系数"""
    try:
        original_seq = original_seq.numpy()
        reconstructed_seq = reconstructed_seq.numpy()
        mask_seq = mask_seq.numpy()
        
        if original_seq.ndim != 4:
            return 0.0
        
        T, C, H, W = original_seq.shape
        total_corr = 0.0
        total_count = 0
        
        # 随机采样像素以加速计算
        sample_size = min(1000, H * W)
        pixel_indices = np.random.choice(H * W, sample_size, replace=False)
        
        for c in range(C):
            orig_2d = original_seq[:, c, :, :].reshape(T, -1)
            recon_2d = reconstructed_seq[:, c, :, :].reshape(T, -1)
            mask_2d = mask_seq[:, 0, :, :].reshape(T, -1) > 0.5
            
            for idx in pixel_indices:
                if np.any(mask_2d[:, idx]):  # 该像素有缺失
                    orig_ts = orig_2d[:, idx]
                    recon_ts = recon_2d[:, idx]
                    
                    if np.std(orig_ts) > 1e-10 and np.std(recon_ts) > 1e-10:
                        try:
                            corr = np.corrcoef(orig_ts, recon_ts)[0, 1]
                            if not np.isnan(corr):
                                total_corr += corr
                                total_count += 1
                        except:
                            continue
        
        return total_corr / total_count if total_count > 0 else 0.0
    except Exception as e:
        print(f"TCC计算错误: {e}")
        return 0.0

def generate_analysis_report(all_results):
    """生成分析报告"""
    report_path = 'experiment_results/analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("遥感数据插补实验分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 实验1分析
        if 'missing_types' in all_results and all_results['missing_types']:
            f.write("实验1: 不同缺失类型对比分析\n")
            f.write("-" * 40 + "\n")
            f.write("科学问题: 模型对不同缺失模式的适应性和鲁棒性\n")
            f.write("实际意义: 验证模型在复杂缺失场景下的通用性\n\n")
            
            for missing_type, result in all_results['missing_types'].items():
                f.write(f"{missing_type}:\n")
                if 'metrics' in result and 'avg_metrics' in result['metrics']:
                    for method, metrics in result['metrics']['avg_metrics'].items():
                        f.write(f"  {method}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.2f}, TCC={metrics['TCC']:.4f}\n")
                f.write("\n")
        
        # 实验2分析
        if 'missing_ratios' in all_results and all_results['missing_ratios']:
            f.write("实验2: 不同缺失比例对比分析\n")
            f.write("-" * 40 + "\n")
            f.write("科学问题: 模型性能随缺失严重程度的变化规律\n")
            f.write("实际意义: 评估模型对高缺失率的容忍度\n\n")
            
            for ratio, result in all_results['missing_ratios'].items():
                f.write(f"{ratio}%缺失:\n")
                if 'metrics' in result and 'avg_metrics' in result['metrics']:
                    for method, metrics in result['metrics']['avg_metrics'].items():
                        f.write(f"  {method}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.2f}, TCC={metrics['TCC']:.4f}\n")
                f.write("\n")
        
        # 实验3分析
        if 'terrain_types' in all_results and all_results['terrain_types']:
            f.write("实验3: 不同地形类型对比分析\n")
            f.write("-" * 40 + "\n")
            f.write("科学问题: 模型在不同地理环境下的泛化能力\n")
            f.write("实际意义: 验证全球遥感应用的有效性\n\n")
            
            for terrain, result in all_results['terrain_types'].items():
                f.write(f"{terrain}:\n")
                if 'metrics' in result and 'avg_metrics' in result['metrics']:
                    for method, metrics in result['metrics']['avg_metrics'].items():
                        f.write(f"  {method}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.2f}, TCC={metrics['TCC']:.4f}\n")
                f.write("\n")
        
        # 实验4分析
        if 'time_intervals' in all_results and all_results['time_intervals']:
            f.write("实验4: 不同时间间隔对比分析\n")
            f.write("-" * 40 + "\n")
            f.write("科学问题: 模型对时间分辨率的敏感性\n")
            f.write("实际意义: 适应不同卫星重访周期的能力\n\n")
            
            for interval, result in all_results['time_intervals'].items():
                f.write(f"{interval}天间隔:\n")
                if 'metrics' in result and 'avg_metrics' in result['metrics']:
                    for method, metrics in result['metrics']['avg_metrics'].items():
                        f.write(f"  {method}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.2f}, TCC={metrics['TCC']:.4f}\n")
                f.write("\n")

def save_experiment_results(results, save_path):
    """保存单个实验结果"""
    os.makedirs(f'experiment_results/{save_path}', exist_ok=True)
    
    # 保存指标数据
    with open(f'experiment_results/{save_path}/metrics.json', 'w') as f:
        json.dump({
            'avg_metrics': results['metrics']['avg_metrics'],
            'frame_metrics': {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} 
                            for k, v in results['metrics']['frame_metrics'].items()}
        }, f, indent=4)
    
    # 保存样本图像
    save_sample_images(results, save_path)

def save_sample_images(results, save_path):
    """保存样本图像"""
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)
    
    os.makedirs(f'experiment_results/{save_path}/images', exist_ok=True)
    
    # 保存第一帧的对比图像
    t = 0
    if 'methods_results' in results:
        for name, result in results['methods_results'].items():
            if result.shape[1] > t:
                img_data = unnorm(result[0, t])
                if img_data.shape[0] == 1:
                    img = Image.fromarray((img_data.squeeze(0).numpy() * 255).astype(np.uint8), 'L')
                else:
                    img_array = (img_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if img_array.shape[2] == 3:
                        img = Image.fromarray(img_array, 'RGB')
                    else:
                        img = Image.fromarray(img_array[:,:,0], 'L')
                img.save(f'experiment_results/{save_path}/images/{name}_frame{t}.png')

def save_comprehensive_results(all_results):
    """保存综合结果"""
    comprehensive_data = {}
    
    for exp_name, exp_data in all_results.items():
        comprehensive_data[exp_name] = {}
        
        for condition, results in exp_data.items():
            if 'metrics' in results and 'avg_metrics' in results['metrics']:
                comprehensive_data[exp_name][condition] = results['metrics']['avg_metrics']
    
    with open('experiment_results/comprehensive_results.json', 'w') as f:
        json.dump(comprehensive_data, f, indent=4)

# 使用示例
def create_experiment_config(model, out_channels=3,
                             dataloader_thin_cloud=None, dataloader_thick_cloud=None,
                             dataloader_strip=None, dataloader_mixed=None,
                             dataloader_10percent=None, dataloader_20percent=None,
                             dataloader_30percent=None, dataloader_40percent=None,
                             dataloader_50percent=None, dataloader_60percent=None,
                            #  dataloader_ocean=None, dataloader_land=None,
                             dataloader_1day=None, dataloader_5days=None,
                             dataloader_10days=None, dataloader_30days=None):
    """创建实验配置并运行实验"""
    
    experiment_config = {
        # 'missing_types': {
        #     'thin_cloud': dataloader_thin_cloud,
        #     'thick_cloud': dataloader_thick_cloud,
        #     'strip': dataloader_strip,
        #     'mixed': dataloader_mixed
        # },
        'missing_ratios': {
            # 10: dataloader_10percent,
            # 20: dataloader_20percent,
            # 30: dataloader_30percent,
            # 40: dataloader_40percent,
            50: dataloader_50percent,
            # 60: dataloader_60percent
        },
        # 'terrain_types': {
        #     'ocean': dataloader_ocean,
        #     'land': dataloader_land
        # },
        # 'time_intervals': {
        #     1: dataloader_1day,
        #     5: dataloader_5days,
        #     10: dataloader_10days,
        #     30: dataloader_30days
        # }
    }
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load('fine_tuned_model.pth'))
        print("模型权重加载成功")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        print("使用随机初始化的模型进行测试")
    
    # 运行所有实验
    results = run_four_experiments(model, experiment_config, out_channels)
    print("实验数据已保存到 experiment_results/ 目录")
    
    return results