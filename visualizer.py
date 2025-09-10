import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from experiments import run_four_experiments
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保 matplotlib 在无交互环境也能保存图
plt.switch_backend('Agg')

METRICS = ["SSIM", "PSNR", "MAE", "TCC"]

def _ensure_fig_dir():
    out_dir = 'experiment_results/figures'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _extract_number_from_string(s):
    """从字符串中提取数字"""
    try:
        # 尝试直接转换为数字
        return float(s)
    except ValueError:
        # 从字符串中提取数字部分
        numbers = re.findall(r'\d+', s)
        if numbers:
            return float(numbers[0])
        return None

def load_all_results_from_directories(base_dir="experiment_results"):
    """
    从目录结构加载实验结果，处理各种命名格式
    """
    exp_mapping = {
        'exp1_missing_types': 'missing_types',
        'exp2_missing_ratios': 'missing_ratios', 
        'exp3_terrain_types': 'terrain_types',
        'exp4_time_intervals': 'time_intervals'
    }
    
    all_results = {}
    
    for exp_dir, exp_key in exp_mapping.items():
        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.exists(exp_path):
            print(f"警告: 实验目录 {exp_path} 不存在")
            continue
            
        all_results[exp_key] = {}
        
        # 查找所有子目录
        for condition_dir in os.listdir(exp_path):
            condition_path = os.path.join(exp_path, condition_dir)
            if not os.path.isdir(condition_path):
                continue
                
            # 查找 metrics.json 文件
            metrics_file = os.path.join(condition_path, 'metrics.json')
            if not os.path.exists(metrics_file):
                print(f"警告: {metrics_file} 不存在")
                continue
                
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                # 转换目录结构数据为 run_four_experiments 格式
                condition_data = {
                    'metrics': {
                        'avg_metrics': metrics_data.get('avg_metrics', {}),
                        'frame_metrics': metrics_data.get('frame_metrics', {})
                    }
                }
                
                all_results[exp_key][condition_dir] = condition_data
                
            except Exception as e:
                print(f"读取 {metrics_file} 时出错: {e}")
    
    return all_results

def get_all_experiment_results(run_experiment=True, model=None, experiment_config=None, out_channels=3):
    """
    获取实验结果的统一入口函数
    """
    if run_experiment:
        if model is None or experiment_config is None:
            raise ValueError("运行实验时需要提供 model 和 experiment_config")
        print("正在运行实验...")
        return run_four_experiments(model, experiment_config, out_channels)
    else:
        print("从目录加载实验结果...")
        return load_all_results_from_directories()

def _safe_get_avg_metrics(all_results, exp_key):
    """
    处理两种数据来源的统一函数
    """
    data = {}
    if exp_key not in all_results:
        return data
    
    for cond, res in all_results[exp_key].items():
        try:
            # 处理 run_four_experiments 返回的结构
            if 'metrics' in res and 'avg_metrics' in res['metrics']:
                avg = res['metrics']['avg_metrics']
            # 处理目录加载的结构
            elif 'avg_metrics' in res:
                avg = res['avg_metrics']
            else:
                avg = {}
        except (KeyError, TypeError):
            avg = {}
        
        # 规范化：确保所有方法都有 METRICS 字段
        normalized = {}
        for method, mvals in avg.items():
            if isinstance(mvals, dict):
                normalized[method] = {k: float(mvals.get(k, 0.0)) for k in METRICS}
            else:
                normalized[method] = {k: 0.0 for k in METRICS}
        data[cond] = normalized
    
    return data

def _get_sorted_conditions(data):
    """获取排序后的条件列表，处理各种命名格式"""
    conditions = list(data.keys())
    
    # 尝试提取数字进行排序
    numeric_conditions = []
    for cond in conditions:
        num = _extract_number_from_string(cond)
        if num is not None:
            numeric_conditions.append((num, cond))
        else:
            numeric_conditions.append((float('inf'), cond))  # 非数字条件放在最后
    
    # 按数字排序，非数字条件保持原顺序
    numeric_conditions.sort(key=lambda x: x[0])
    return [cond for _, cond in numeric_conditions]

def plot_missing_types_bar(all_results, show_metrics=["SSIM", "PSNR"]):
    """
    Exp1: 不同缺失类型对比（分组柱状图）
    """
    out_dir = _ensure_fig_dir()
    data = _safe_get_avg_metrics(all_results, 'missing_types')
    if not data:
        print("missing_types 数据缺失，跳过绘图")
        return []

    conditions = _get_sorted_conditions(data)
    methods = sorted({m for cond in data.values() for m in cond.keys()})
    fig_paths = []

    for metric in show_metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(conditions)*1.5), 6))
        x = np.arange(len(conditions))
        width = 0.8 / max(1, len(methods))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            vals = [data[cond].get(method, {}).get(metric, 0.0) for cond in conditions]
            ax.bar(x + (i - (len(methods)-1)/2)*width, vals, width=width, 
                  label=method, color=colors[i], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in conditions], rotation=25, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'不同缺失类型下各方法的 {metric} 对比')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(out_dir, f'exp1_missing_types_{metric}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        fig_paths.append(path)

    return fig_paths

def plot_missing_ratios_line(all_results, metric="SSIM"):
    """
    Exp2: 不同缺失比例对比（折线图）
    """
    out_dir = _ensure_fig_dir()
    data = _safe_get_avg_metrics(all_results, 'missing_ratios')
    if not data:
        print("missing_ratios 数据缺失，跳过绘图")
        return None

    # 获取排序后的条件
    conditions = _get_sorted_conditions(data)
    
    # 提取数字用于x轴
    x_values = []
    for cond in conditions:
        num = _extract_number_from_string(cond)
        if num is not None:
            x_values.append(num)
        else:
            x_values.append(len(x_values) + 1)  # 如果没有数字，使用顺序编号
    
    methods = sorted({m for cond in data.values() for m in cond.keys()})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        y_vals = [data[cond].get(method, {}).get(metric, 0.0) for cond in conditions]
        ax.plot(x_values, y_vals, marker='o', linewidth=2, markersize=8,
               label=method, linestyle=line_styles[i % len(line_styles)], 
               color=colors[i])
    
    ax.set_xlabel('缺失比例 (%)')
    ax.set_ylabel(metric)
    ax.set_title(f'不同缺失比例下各方法的 {metric} 曲线')
    ax.set_xticks(x_values)
    ax.set_xticklabels(conditions, rotation=25, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    path = os.path.join(out_dir, f'exp2_missing_ratios_{metric}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_terrain_grouped_bar(all_results, metric="SSIM"):
    """
    Exp3: 不同地形类型对比（分组柱状图）
    """
    out_dir = _ensure_fig_dir()
    data = _safe_get_avg_metrics(all_results, 'terrain_types')
    if not data:
        print("terrain_types 数据缺失，跳过绘图")
        return None

    conditions = _get_sorted_conditions(data)
    methods = sorted({m for cond in data.values() for m in cond.keys()})
    
    fig, ax = plt.subplots(figsize=(max(10, len(methods)*1.5), 6))
    
    x = np.arange(len(methods))
    width = 0.8 / len(conditions)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(conditions)))
    
    for i, condition in enumerate(conditions):
        vals = [data[condition].get(m, {}).get(metric, 0.0) for m in methods]
        ax.bar(x + (i - (len(conditions)-1)/2)*width, vals, width, 
              label=condition, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'不同地形类型下各方法的 {metric} 对比')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(out_dir, f'exp3_terrain_{metric}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_time_intervals_log_line(all_results, metric="TCC"):
    """
    Exp4: 不同时间间隔对比（对数坐标折线图）
    """
    out_dir = _ensure_fig_dir()
    data = _safe_get_avg_metrics(all_results, 'time_intervals')
    if not data:
        print("time_intervals 数据缺失，跳过绘图")
        return None

    # 获取排序后的条件
    conditions = _get_sorted_conditions(data)
    
    # 提取数字用于x轴
    x_values = []
    for cond in conditions:
        num = _extract_number_from_string(cond)
        if num is not None:
            x_values.append(num)
        else:
            x_values.append(len(x_values) + 1)
    
    methods = sorted({m for cond in data.values() for m in cond.keys()})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        y_vals = [data[cond].get(method, {}).get(metric, 0.0) for cond in conditions]
        ax.plot(x_values, y_vals, marker='s', linewidth=2, markersize=8,
               label=method, linestyle=line_styles[i % len(line_styles)], 
               color=colors[i])
    
    ax.set_xscale('log')
    ax.set_xlabel('时间间隔（天，对数坐标）')
    ax.set_ylabel(metric)
    ax.set_title(f'不同时间间隔下各方法的 {metric}（对数坐标）')
    ax.set_xticks(x_values)
    ax.set_xticklabels(conditions, rotation=25, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    path = os.path.join(out_dir, f'exp4_time_intervals_{metric}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path

def generate_all_visualizations(all_results, metrics_to_plot=None):
    """
    生成所有实验的可视化图表
    """
    if metrics_to_plot is None:
        metrics_to_plot = {
            'exp1': ["SSIM", "PSNR"],
            'exp2': "SSIM",
            'exp3': "SSIM",
            'exp4': "TCC"
        }

    results = {}
    
    print("开始生成可视化图表...")
    
    # Exp1: 缺失类型对比
    if 'missing_types' in all_results:
        print("生成实验1图表...")
        results['exp1'] = plot_missing_types_bar(all_results, show_metrics=metrics_to_plot.get('exp1', ["SSIM"]))
    
    # Exp2: 缺失比例对比
    if 'missing_ratios' in all_results:
        print("生成实验2图表...")
        results['exp2'] = plot_missing_ratios_line(all_results, metric=metrics_to_plot.get('exp2', "SSIM"))
    
    # Exp3: 地形类型对比
    if 'terrain_types' in all_results:
        print("生成实验3图表...")
        results['exp3'] = plot_terrain_grouped_bar(all_results, metric=metrics_to_plot.get('exp3', "SSIM"))
    
    # Exp4: 时间间隔对比
    if 'time_intervals' in all_results:
        print("生成实验4图表...")
        results['exp4'] = plot_time_intervals_log_line(all_results, metric=metrics_to_plot.get('exp4', "TCC"))
    
    print("所有可视化图表已生成并保存到 experiment_results/figures/")
    return results

def generate_comprehensive_report(all_results):
    """
    生成综合分析报告，处理各种命名格式
    """
    report = "遥感数据插补方法综合评估报告\n"
    report += "="*60 + "\n\n"
    
    # 实验1分析
    if 'missing_types' in all_results:
        report += "实验1: 不同缺失类型对比分析\n"
        report += "-"*40 + "\n"
        report += "科学问题：模型是否对缺失模式具有敏感性？\n"
        report += "实际意义：真实遥感数据中缺失类型多样，需要模型具备通用性\n\n"
        
        data = _safe_get_avg_metrics(all_results, 'missing_types')
        conditions = _get_sorted_conditions(data)
        for condition in conditions:
            report += f"{condition}:\n"
            for method, metrics in data[condition].items():
                report += f"  {method}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.2f}, TCC={metrics['TCC']:.4f}\n"
            report += "\n"
    
    # 实验2分析
    if 'missing_ratios' in all_results:
        report += "实验2: 不同缺失比例对比分析\n"
        report += "-"*40 + "\n"
        report += "科学问题：模型性能如何随缺失严重程度变化？\n"
        report += "实际意义：现实应用中缺失比例各异，需要模型具有可预测的性能衰减\n\n"
        
        data = _safe_get_avg_metrics(all_results, 'missing_ratios')
        conditions = _get_sorted_conditions(data)
        for condition in conditions:
            # 提取数字部分用于显示
            num = _extract_number_from_string(condition)
            if num is not None:
                display_name = f"{num}%缺失"
            else:
                display_name = condition
                
            report += f"{display_name}:\n"
            for method, metrics in data[condition].items():
                report += f"  {method}: SSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.2f}, MAE={metrics['MAE']:.2f}, TCC={metrics['TCC']:.4f}\n"
            report += "\n"
    
    # 保存报告
    report_path = 'experiment_results/comprehensive_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path
