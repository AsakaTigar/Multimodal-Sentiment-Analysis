import torch
from torch import nn, einsum
from einops import rearrange, repeat
from typing import List
from efficientnet_pytorch.model import MemoryEfficientSwish
from timm.models.layers import DropPath
def pair(t):
    """
    将输入转换为元组，如果输入已经是元组，则原样返回

    Args：
        t：任一类型，可以是单一值或元组。

    Returns：
        tuple：如果输入时单一值，则返回(t,t)的元组，否则直接返回原元组
    """
    return t if isinstance(t, tuple) else (t, t)
###############################################################################################
class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, 
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 group_split: List[int] = [2, 2, 4], kernel_sizes: List[int] = [3, 5, 7], window_size=7):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        # 使用修改后的 TransformerEncoder
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout, 
                                         group_split, kernel_sizes, window_size)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n + self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x
###############################################################################################
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., 
                 group_split: List[int] = [2, 2, 4], kernel_sizes: List[int] = [3, 5, 7], window_size=7):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, EfficientAttention(dim, heads, group_split, kernel_sizes, window_size)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x, save_hidden=False):
        if save_hidden:
            hidden_list = [x]
            for attn, ff in self.layers:
                x = attn(x, x, x)
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x)
                x = ff(x) + x
            return x
###############################################################################################
class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v)
    
###############################################################################################
class Attention(nn.Module):
    """
    多头注意力机制模块。
    """

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """
            -dim：输入特征的维度
            -heads：多头注意力机制的头数
            -dim_head：每个头的维度
            -dropout：Dropout的概率，用于防止过拟合
        """
        super().__init__()
        inner_dim = dim_head *  heads   # 内部维度 = 注意力头数 * 每头的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads              # 注意力头数
        self.scale = dim_head ** -0.5   # 缩放因子, 用于稳定梯度

        self.attend = nn.Softmax(dim = -1)  # 注意力权重归一化

        # 定义线性变换，用于生成Query、Key、Value
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        # 输出层，是否投影决定于头数与维度
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        """
            - q：Query输入，形状[batch_size, seq_len, dim]
            - k：Key输入，形状[batch_size, seq_len, dim]
            - v：Value输入，形状[batch_size, seq_len, dim]
        """
        b, n, _, h = *q.shape, self.heads   # 提取批量大小，序列长度和注意力头数
        # 生成q, k, v
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # 重排维度以适配多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        # 计算注意力得分（内积并乘以缩放因子）
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 对注意力得分进行归一化，得到注意力权重
        attn = self.attend(dots)

        # 使用注意力权重对Value继续加权求和
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 恢复原始维度
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    
###############################################################################################
class PreNormForward(nn.Module):
    """
    在前向传播中先进行LayerNorm，在传入指定的函数进行处理的模块
    用于处理输入的张量，先归一化在进行后续操作
    """
    def __init__(self, dim, fn):
        """
        初始化方法。

        Args：
            dim(int)：输入张量的维度
            fn(nn.Module)：需要应用的函数（如全连接层，激活函数等）。
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)   # LayerNorm层进行归一化
        self.fn = fn    # 后续的操作函数
    def forward(self, x, **kwargs):
        """
        前向传播过程，先进行归一化，再调用指定的函数处理输入。

        Args：
            x(Tensor)：输入张量。
            kwargs(dict)：其他可能传入的参数。

        Returns：
            Tensor：经过函数处理后的结果。
        """
        return self.fn(self.norm(x), **kwargs)
    
###############################################################################################

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# === 新引入的模块 ===

# Attention Map Module
class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
            # nn.Identity()
        )
    def forward(self, x):
        return self.act_block(x)

# Efficient Attention Module
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7, 
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size, 
                                   1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2d(dim, 3*group_head*self.dim_head, 1, 1, 0, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size !=1 else nn.Identity()
        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        q, k, v = qkv
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)
        return res

    def low_fre_attention(self, x : torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous()
        kv = avgpool(x)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h*w)//(self.window_size**2)).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv
        attn = self.scalor * q @ k.transpose(-1, -2)
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding=kernel_size//2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.act2 = nn.GELU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm(x)
        return x






# Efficient Attention with Convolutional FeedForward Network
class EfficientAttnFFN(nn.Module):
    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        self.eff_attn = EfficientAttention(dim, num_heads, group_split, kernel_sizes, window_size, 
                                          attn_drop, proj_drop, qkv_bias)
        self.conv_ffn = ConvFFN(dim, dim * 4, kernel_size=3, stride=1, out_channels=dim)

    def forward(self, x):
        attn_out = self.eff_attn(x)
        ffn_out = self.conv_ffn(attn_out)
        return ffn_out

# Efficient Transformer
class EfficientTransformer:
    def __init__(self, dim, num_heads, group_split, kernel_sizes, depth=1):
        # 保存参数
        self.dim = dim
        self.num_heads = num_heads
        self.group_split = group_split
        self.kernel_sizes = kernel_sizes
        self.depth = depth
        
        # 根据 `depth` 创建 Transformer 的多层结构
        self.layers = self._create_layers()
    
    def _create_layers(self):
        # 根据 depth 创建层，这里可以是一个示例实现
        layers = []
        for i in range(self.depth):
            # 创建每一层，可以是 Efficient Transformer 的一个子模块
            layers.append(self._create_single_layer(i))
        return layers

    def _create_single_layer(self, layer_id):
        # 创建一个单独的 transformer 层（你可以根据需要替换为适当的实现）
        return f"Layer {layer_id + 1}"

    def forward(self, x):
        # 在这里实现前向传播逻辑
        for layer in self.layers:
            # 执行每一层的前向传播
            print(f"Processing with {layer}")
        return x




###############################################################################################
class HhyperLearningEncoder(nn.Module):
    """
    超参数学习编码器：
    该模块由多个HhyperLearningLayer 层堆叠而成，用于对多模态输入特征进行逐层处理和融合。
    参数：
        - dim：输入特征的维度。
        - depth：编码器的深度，即堆叠的层数。
        - heads：注意力头的数量。
        - dim_head：每个头的维度。
        - dropout：Dropout的概率。
    """
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([]) # 使用列表存储每层
        for _ in range(depth):
            # 添加每一层HhyperLearningLayer
            self.layers.append(nn.ModuleList([
                PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_t_list, h_a, h_v, h_hyper):
        """
        参数：
            - h_t_list：不同时间步的时间特征列表
            - h_a：空间序列特征。
            - h_v：视觉信息特征。
            - h_hyper：超参数特征。
        """
        # 依次通过每一层，逐层处理特征。
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_t_list[i], h_a, h_v, h_hyper)
        return h_hyper
    
###############################################################################################
class PreNormAHL(nn.Module):
    """
    针对多模态输入的前向归一化处理模块。
    每种输入(h_t、h_a、h_v、h_hyper)单独进行归一化，保持各模态的特征分布稳定。
    参数：
        -dim：输入特征的维度
        -fn：归一化后应用的函数，用于处理多模态数据
    """
    def __init__(self, dim, fn):
        super().__init__()
        # 定义灭个输入模态的LayerNorm层
        self.norm1 = nn.LayerNorm(dim) # 时间序列特征的归一化
        self.norm2 = nn.LayerNorm(dim) # 空间序列特征的归一化
        self.norm3 = nn.LayerNorm(dim) # 视觉特征的归一化
        self.norm4 = nn.LayerNorm(dim) # 超参数的归一化
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        # 对输入模态分别进行归一化
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        # 将归一化后的特征传递到下一个函数处理
        return self.fn(h_t, h_a, h_v, h_hyper)

###############################################################################################
class HhyperLearningLayer(nn.Module):
    """
    超参数学习层：
    该层使用多头注意力机制，将多模态输入进行融合，并调整超参数特征。
    参数：
        - dim：输入特征的维度
        - heads：注意力头的数量
        - dim_head：每个头的特征维度
        - dropout：Dropout的概率
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads   # 内部维度为注意力头数乘以每个头的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # 注意力头数量
        self.scale = dim_head ** -0.5   # 缩放因子，用于稳定梯度

        self.attend = nn.Softmax(dim = -1)  # 注意力权重归一化

        # 定义线性层，分别生成Query 和多个 Key、Value
        self.to_q = nn.Linear(dim, inner_dim, bias=False)       # 时间特征Query
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)    # 空间特征Key
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)    # 视觉特征Key
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)    # 空间特征Value
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)    # 视觉特征Value
    
        # 输出层，将多头注意力输出投影会输入维度
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_t, h_a, h_v, h_hyper):
        """
        参数：
            - h_t：时间序列特征。
            - h_a：空间序列特征。
            - h_v：视觉信息特征。
            - h_hyper：超参数特征。
        """
        b, n, _, h = *h_t.shape, self.heads # 提取批量大小、序列长度和头数

        # 生成Query 和 Key、Value
        q = self.to_q(h_t)
        k_ta = self.to_k_ta(h_a)
        k_tv = self.to_k_tv(h_v)
        v_ta = self.to_v_ta(h_a)
        v_tv = self.to_v_tv(h_v)

        # 调整维度适配多头注意力机制
        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, k_tv, v_ta, v_tv))

        # 计算时间特征与空间特征之间的注意力得分和输出
        dots_ta = einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale
        attn_ta = self.attend(dots_ta)
        out_ta = einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta)
        out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')

        # 计算时间特征与视觉特征之间的注意力得分和输出
        dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
        attn_tv = self.attend(dots_tv)
        out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv)
        out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')


        # 合并空间和视觉特征输出，调整超参数特征
        h_hyper_shift = self.to_out(out_ta + out_tv)
        h_hyper += h_hyper_shift    # 更新超参数特征

        return h_hyper
    














###############################################################################################
class CrossTransformer(nn.Module):
    """
    交叉Transformer模型。
    
    参数：
        - source_num_frames: 源序列的帧数。
        - tgt_num_frames: 目标序列的帧数。
        - dim: 输入特征维度。
        - depth: 编码器深度。
        - heads: 注意力头的数量。
        - mlp_dim: 前馈网络的隐藏层维度。
        - pool: 池化方式。
        - dim_head: 每个头的维度。
        - dropout: Dropout 的概率。
        - emb_dropout: 嵌入层 Dropout 的概率。
    """
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # 定义位置编码参数：源序列和目标序列的长度
        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))  # 额外的标记嵌入

        # Dropout 层
        self.dropout = nn.Dropout(emb_dropout)

        # 定义交叉Transformer编码器
        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 池化策略
        self.pool = pool

    def forward(self, source_x, target_x):
        """
        前向传播方法。

        参数：
            - source_x: 源序列特征。
            - target_x: 目标序列特征。

        返回：
            - 经过交叉注意力处理后的目标序列特征。
        """
        b, n_s, _ = source_x.shape  # 获取源序列的批次大小和长度
        b, n_t, _ = target_x.shape  # 获取目标序列的批次大小和长度
        
        # 添加额外的标记到源序列和目标序列中
        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b=b)

        source_x = torch.cat((extra_token, source_x), dim=1)  # 将额外标记添加到源序列
        source_x = source_x + self.pos_embedding_s[:, :n_s+1]  # 添加源序列的位置编码

        target_x = torch.cat((extra_token, target_x), dim=1)  # 将额外标记添加到目标序列
        target_x = target_x + self.pos_embedding_t[:, :n_t+1]  # 添加目标序列的位置编码

        # 应用 Dropout
        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        # 通过交叉Transformer编码器
        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t  # 返回处理后的目标序列特征
    
###############################################################################################
class CrossTransformerEncoder(nn.Module):
    """
    交叉Transformer 编码器：
    该模块通过注意力机制，将一个序列的信息传递到另一个序列中。

    参数：
        - dim: 输入特征的维度。
        - depth: 编码器的深度（Transformer 层数）。
        - heads: 注意力头的数量。
        - dim_head: 每个头的维度。
        - mlp_dim: 前馈神经网络的隐藏层维度。
        - dropout: Dropout 的概率。
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        """
        前向传播方法

        参数：
            - source_x: 源序列特征。
            - target_x: 目标序列特征。

        返回：
            - target_x：经过源序列影响后的目标序列特征。
        """
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)   # 交叉注意力机制
            target_x = target_x_tmp + target_x                  # 残差连接
            target_x = ff(target_x) + target_x                  # 前馈网络和残差连接
        return target_x                                         # 返回更新后的目标序列特征