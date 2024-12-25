from torch import nn
import torch
from bert import BertTextEncoder
from almt_layer_copy import EfficientTransformer, CrossTransformer, HhyperLearningEncoder  # 更新后的类名
from einops import repeat  # 用于扩展张量维度

import torch
import torch.nn as nn
from transformers import BertModel

# 假设 EfficientTransformer, CrossTransformer, HhyperLearningEncoder 等类已经定义

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=True, transformers='bert', pretrained='bert-base-uncased'):
        super(BertTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.use_finetune = use_finetune

    def forward(self, x):
        # 对文本进行编码，返回 [batch_size, seq_len, hidden_size]
        outputs = self.bert(x)
        return outputs.last_hidden_state  # 返回最后一层隐藏状态

class ALMT(nn.Module):
    def __init__(self, dataset, AHL_depth=3, fusion_layer_depth=2, bert_pretrained='bert-base-uncased'):
        super(ALMT, self).__init__()

        # 初始化超参数嵌入，形状为 (1, 8, 128)
        self.h_hyper = nn.Parameter(torch.ones(1, 8, 128))

        # 加载预训练的BERT文本编码器
        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        # 根据不同数据集定义线性投影层，将输入特征映射到固定维度
        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(20, 128)  # 文本特征从 20 映射到 128 (根据你的输入调整)
            self.proj_a0 = nn.Linear(5, 128)   # 音频特征从 5 映射到 128
            self.proj_v0 = nn.Linear(50, 128)  # 视频特征从 50 映射到 128 (根据你的输入调整)
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(20, 128)  # 文本特征从 20 映射到 128 (根据你的输入调整)
            self.proj_a0 = nn.Linear(74, 128)  # 音频特征从 74 映射到 128
            self.proj_v0 = nn.Linear(35, 128)  # 视频特征从 35 映射到 128
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(20, 128)  # 文本特征从 20 映射到 128 (根据你的输入调整)
            self.proj_a0 = nn.Linear(33, 128)  # 音频特征从 33 映射到 128
            self.proj_v0 = nn.Linear(50, 128)  # 视频特征从 50 映射到 128
        else:
            assert False, "数据集名称不匹配"

        # 定义文本、音频和视频特征的 Transformer 编码器（使用新的 EfficientTransformer）
        self.proj_l = EfficientTransformer(dim=128, num_heads=8, group_split=[4, 4], kernel_sizes=[3, 5], depth=1)
        self.proj_a = EfficientTransformer(dim=128, num_heads=8, group_split=[4, 4], kernel_sizes=[3, 5], depth=1)
        self.proj_v = EfficientTransformer(dim=128, num_heads=8, group_split=[4, 4], kernel_sizes=[3, 5], depth=1)

        # 定义文本编码器，用于处理文本嵌入（使用新的 EfficientTransformer）
        self.text_encoder = EfficientTransformer(dim=128, num_heads=8, group_split=[4, 4], kernel_sizes=[3, 5], depth=AHL_depth-1)

        # 高阶超学习编码器 (HhyperLearningEncoder)，用于融合多模态信息
        self.h_hyper_layer = HhyperLearningEncoder(dim=128, depth=AHL_depth, heads=8, dim_head=16, dropout=0.)

        # 定义跨模态融合层（使用新的 CrossEfficientTransformer）
        self.fusion_layer = CrossTransformer(dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)

        # 分类头，用于生成最终输出
        self.cls_head = nn.Sequential(
            nn.Linear(128, 1)  # 将 128 维特征映射到 1 维输出（如回归任务）
        )

    def forward(self, text_input, audio_input, video_input):
        """
        前向传播方法

        参数:
            - text_input: 文本输入，形状为 [batch_size, seq_len, feature_dim]
            - audio_input: 音频输入，形状为 [batch_size, seq_len, feature_dim]
            - video_input: 视频输入，形状为 [batch_size, seq_len, feature_dim]

        返回:
            - 输出预测结果
        """
        print(f"text_input shape: {text_input.shape}")
        print(f"audio_input shape: {audio_input.shape}")
        print(f"video_input shape: {video_input.shape}")
        # 动态计算源序列和目标序列的帧数
        source_num_frames = text_input.size(1)  # 假设源序列为文本的 seq_len
        tgt_num_frames = audio_input.size(1)   # 假设目标序列为音频的 seq_len

        # 根据数据的长度来调整 CrossTransformer 的帧数
        self.fusion_layer.source_num_frames = source_num_frames
        self.fusion_layer.tgt_num_frames = tgt_num_frames

        # 投影层将输入的文本、音频、视频特征投影到相同的维度
        text_proj = self.proj_l0(text_input)  # [batch_size, seq_len, 128]
        audio_proj = self.proj_a0(audio_input)  # [batch_size, seq_len, 128]
        video_proj = self.proj_v0(video_input)  # [batch_size, seq_len, 128]
        print(f"text_proj shape: {text_proj.shape}")
        print(f"audio_proj shape: {audio_proj.shape}")
        print(f"video_proj shape: {video_proj.shape}")
        # 通过 Transformer 编码器进行处理
        text_encoded = self.text_encoder(text_proj)  # [batch_size, seq_len, 128]
        audio_encoded = self.proj_a(audio_proj)  # [batch_size, seq_len, 128]
        video_encoded = self.proj_v(video_proj)  # [batch_size, seq_len, 128]

        # 高阶超学习编码器（多模态融合）
        multimodal_features = self.h_hyper_layer(text_encoded, audio_encoded, video_encoded)  # [batch_size, seq_len, 128]

        # 使用跨模态融合层（CrossTransformer）进行融合
        fusion_output = self.fusion_layer(text_encoded, audio_encoded)

        # 后续的分类头处理
        output = self.cls_head(fusion_output)

        return output



# 构建模型的方法
def build_model(opt):
    """
    根据选项构建 ALMT 模型。

    参数:
        - opt: 包含模型配置的选项对象。

    返回:
        - model: 初始化好的 ALMT 模型实例。
    """
    if opt.datasetName == 'sims':
        l_pretrained = '/home/lab/LAD/emotional/ATML/My_project/bert-base-chinese'  # 对于 'sims' 数据集，使用中文 BERT 模型
    else:
        l_pretrained = '/home/lab/LAD/emotional/ATML/My_project/bert-base-uncased'  # 其他数据集使用英文 BERT 模型

    # 初始化ALMT模型并返回
    model = ALMT(dataset=opt.datasetName, fusion_layer_depth=opt.fusion_layer_depth, bert_pretrained=l_pretrained)

    return model  # 返回构建的模型实例
