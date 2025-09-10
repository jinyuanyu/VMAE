import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from experiments import *
from MAE_LaMa import *

def main():
    # 数据路径配置
    data_dir = 'E:/lama/jet_S2_Daily_Mosaic/'
    ocean_mask_path = 'E:/lama/S2_Daily_Mosaic_Masked/mask.png'
    input_seq_len = 8
    out_channels = 4
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 模型初始化
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
    print("开始运行实验...")
    all_results = run_four_experiments(model, experiment_config, out_channels, input_seq_len)
    
    # 保存实验结果
    print("实验完成，保存结果...")
    save_comprehensive_results(all_results)
    generate_analysis_report(all_results)
    
    print("所有实验已完成！结果保存在 experiment_results/ 目录")

if __name__ == "__main__":
    main()
