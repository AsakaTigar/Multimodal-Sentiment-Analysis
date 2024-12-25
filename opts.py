'''
* @name: opts.py
* @description: Hyperparameter configuration. Note: For hyperparameter settings, please refer to the appendix of the paper.
* @description: 超参数配置。注意：关于超参数的设置，请参考论文的附录。
'''


import argparse  # 导入 argparse 库，用于处理命令行参数

def parse_opts():
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 定义超参数配置字典
    arguments = {
        'dataset': [  # 数据集相关配置
            dict(name='--datasetName',  # 数据集名称
                 type=str,  # 参数类型为字符串
                 default='mosi',  # 默认值为 'mosi'
                 help='mosi, mosei or sims'),  # 说明参数选择 'mosi', 'mosei' 或 'sims'
            dict(name='--dataPath',  # 数据路径
                 default="/home/lab/LAD/emotional/ATML/My_project/MOSI/Processed/unaligned_50.pkl",  # 默认数据路径
                 type=str,  # 字符串类型
                 help='./MOSEI/Processed/unaligned_50.pkl 或者 ./SIMS/Processed/unaligned_39.pkl'),  # 数据路径帮助说明
            dict(name='--seq_lens',     # 序列长度配置  
                 default=[50, 50, 50],  # 默认序列长度为 50
                 type=list,  # 列表类型
                 help=' '),  # 序列长度帮助说明
            dict(name='--num_workers',  # 加载数据时使用的工作线程数
                 default=8,  # 默认值为 8
                 type=int,  # 整型
                 help=' '),  # 工作线程数帮助说明
           dict(name='--train_mode',  # 训练模式（如回归任务）
                 default="regression",  # 默认训练模式为 'regression'（回归）
                 type=str,  # 字符串类型
                 help=' '),  # 训练模式帮助说明
            dict(name='--test_checkpoint',  # 测试时的模型检查点路径
                 default="./checkpoint/test/SIMS_Acc7_Best.pth",  # 默认检查点路径
                 type=str,  # 字符串类型
                 help=' '), # 检查点路径帮助说明
        ],
        'network': [  # 网络结构相关配置
            dict(name='--CUDA_VISIBLE_DEVICES',# 使用的 GPU 设备编号   
                 default='0',  # 默认使用 GPU 0
                 type=str), # 字符串类型
            dict(name='--fusion_layer_depth',  # 融合层的深度（层数）
                 default=2,  # 默认值为 2
                 type=int)  # 整型
        ],

        'common': [  # 常用配置项
            dict(name='--project_name',# 项目名称
                 default='ALMT_Demo',  # 默认项目名称为 'ALMT_Demo'
                 type=str  # 字符串类型
                 ),
           dict(name='--is_test',      # 是否进行测试（1 表示是，0 表示否）
                 default=1,  # 默认值为 1（表示进行测试）
                 type=int  # 整型
                 ),
            dict(name='--seed',  # 随机种子（用于确保结果可复现）
                 default=18,  # 默认种子值为 18
                 type=int  # 整型
                 ),
            dict(name='--models_save_root',  # 模型保存根路径
                 default='./checkpoint',  # 默认保存路径为 './checkpoint'
                 type=str  # 字符串类型
                 ),
            dict(name='--batch_size',  # 批次大小（每次训练时使用的数据量）
                 default=64,  # 默认批次大小为 64
                 type=int,  # 整型
                 help=' '),  # 批次大小帮助说明
            dict(
                name='--n_threads',  # 多线程数据加载的线程数
                default=3,  # 默认使用 3 个线程
                type=int,  # 整型
                help='Number of threads for multi-thread loading',  # 帮助说明
            ),
            dict(name='--lr', # 学习率
                 type=float,  # 浮动类型
                 default=1e-4),  # 默认学习率为 1e-4
            dict(name='--weight_decay',  # 权重衰减（L2 正则化系数）
                 type=float,  # 浮动类型
                 default=1e-4),  # 默认值为 1e-4
            dict(
                name='--n_epochs',  # 总训练轮数
                default=200,  # 默认训练 200 个轮次
                type=int,  # 整型
                help='训练的总轮数', # 训练轮数帮助说明
            )
        ]
    }

     # 将参数字典中的每一项添加到 ArgumentParser 中
    for group in arguments.values(): # 遍历每个参数组（如 'dataset', 'network', 'common'）
        for argument in group:  # 遍历参数组中的每个参数
               name = argument['name']  # 获取参数名称
               del argument['name']  # 删除 name 键，避免重复
               # 将每个参数添加到 ArgumentParser 中
               parser.add_argument(name, **argument)

     # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args