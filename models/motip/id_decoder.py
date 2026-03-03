# Copyright (c) Ruopeng Gao. All Rights Reserved.
# 版权所有 (c) Ruopeng Gao。保留所有权利。

import torch
import einops
import torch.nn as nn
from typing import Tuple
from torch.utils.checkpoint import checkpoint

from models.misc import _get_clones, label_to_one_hot
from models.ffn import FFN


class IDDecoder(nn.Module):
    """
    ID Decoder for MOTIP (Multi-Object Tracking In-context Prediction).
    MOTIP 的 ID 解码器，用于上下文内预测的多目标跟踪。
    """

    def __init__(
        self,
        feature_dim: int,  # 特征维度
        id_dim: int,  # ID 嵌入维度
        ffn_dim_ratio: int,  # FFN 维度比例
        num_layers: int,  # 解码器层数
        head_dim: int,  # 注意力头维度
        num_id_vocabulary: int,  # ID 词汇表大小（最大支持的 ID 数量）
        rel_pe_length: int,  # 相对时间位置编码的最大长度
        use_aux_loss: bool,  # 是否使用辅助损失
        use_shared_aux_head: bool,  # 是否共享辅助分类头
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim
        # 总维度是特征维度和 ID 维度之和
        self.n_heads = (self.feature_dim + self.id_dim) // self.head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length

        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head

        # ID 词（One-hot）到嵌入向量的投影
        self.word_to_embed = nn.Linear(self.num_id_vocabulary + 1, self.id_dim, bias=False)
        # 嵌入向量到 ID 词（Logits）的投影
        embed_to_word = nn.Linear(self.id_dim, self.num_id_vocabulary + 1, bias=False)

        if self.use_aux_loss and not self.use_shared_aux_head:
            # 如果不共享头，则克隆多层分类头
            self.embed_to_word_layers = _get_clones(embed_to_word, self.num_layers)
        else:
            # 如果共享，则所有层使用同一个 ModuleList 中的同一个实例
            self.embed_to_word_layers = nn.ModuleList([embed_to_word for _ in range(self.num_layers)])
        pass

        # Related Position Embeddings:
        # 相对位置编码参数，每一层、每个相对位移、每个头都有一个独立的标量偏置
        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.rel_pe_length, self.n_heads), dtype=torch.float32)
        )
        # Prepare others for rel pe:
        # 预先生成相对位置映射索引
        t_idxs = torch.arange(self.rel_pe_length, dtype=torch.int64)
        curr_t_idxs, traj_t_idxs = torch.meshgrid([t_idxs, t_idxs])
        self.rel_pos_map = (
            curr_t_idxs - traj_t_idxs
        )  # [curr_t_idx, traj_t_idx] -> rel_pos, 例如 [1, 0] = 1
        pass

        # 自注意力：用于处理同一帧内不同未知目标之间的关系
        self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        self_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        # 交叉注意力：用于将未知目标与历史轨迹进行匹配
        cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        # 前馈网络 (FFN)
        ffn = FFN(
            d_model=self.feature_dim + self.id_dim,
            d_ffn=(self.feature_dim + self.id_dim) * self.ffn_dim_ratio,
            activation=nn.GELU(),
        )
        ffn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)

        # 构建堆叠的解码器层
        self.self_attn_layers = _get_clones(self_attn, self.num_layers - 1)
        self.self_attn_norm_layers = _get_clones(self_attn_norm, self.num_layers - 1)
        self.cross_attn_layers = _get_clones(cross_attn, self.num_layers)
        self.cross_attn_norm_layers = _get_clones(cross_attn_norm, self.num_layers)
        self.ffn_layers = _get_clones(ffn, self.num_layers)
        self.ffn_norm_layers = _get_clones(ffn_norm, self.num_layers)

        # Init parameters:
        # 参数初始化
        for n, p in self.named_parameters():
            if p.dim() > 1 and "rel_pos_embeds" not in n:
                nn.init.xavier_uniform_(p)

        pass

    def forward(self, seq_info, use_decoder_checkpoint):
        """
        前向传播过程。
        Args:
            seq_info: 序列信息字典，包含轨迹特征、未知特征、标签、时间戳和掩码等。
            use_decoder_checkpoint: 是否使用 PyTorch checkpoint 来节省显存。
        """
        trajectory_features = seq_info[
            "trajectory_features"
        ]  # 历史轨迹特征: (B, G, T, N, C)
        unknown_features = seq_info[
            "unknown_features"
        ]  # 当前待识别目标特征: (B, G, T_curr, N, C)
        trajectory_id_labels = seq_info["trajectory_id_labels"]  # 历史轨迹 ID 标签
        unknown_id_labels = seq_info["unknown_id_labels"] if "unknown_id_labels" in seq_info else None
        trajectory_times = seq_info["trajectory_times"]  # 历史轨迹对应的时间步
        unknown_times = seq_info["unknown_times"]  # 待识别目标对应的时间步
        trajectory_masks = seq_info["trajectory_masks"]  # 历史轨迹掩码
        unknown_masks = seq_info["unknown_masks"]  # 待识别目标掩码
        _B, _G, _T, _N, _ = trajectory_features.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_features.shape

        # 将历史轨迹的 ID 标签转换为 ID 嵌入
        trajectory_id_embeds = self.id_label_to_embed(id_labels=trajectory_id_labels)
        # 为待识别目标生成空（占位）ID 嵌入
        unknown_id_embeds = self.generate_empty_id_embed(unknown_features=unknown_features)

        # 拼接视觉特征和 ID 嵌入
        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)

        # Prepare some common variables:
        # 准备注意力掩码：
        # 自注意力 key 掩码：处理同一帧内的填充
        self_attn_key_padding_mask = einops.rearrange(unknown_masks, "b g t n -> (b g t) n").contiguous()
        # 交叉注意力 key 掩码：处理历史轨迹中的填充
        cross_attn_key_padding_mask = einops.rearrange(trajectory_masks, "b g t n -> (b g) (t n)").contiguous()

        # 展平时间戳以便计算因果掩码和相对位置
        _trajectory_times_flatten = einops.rearrange(trajectory_times, "b g t n -> (b g) (t n)")
        _unknown_times_flatten = einops.rearrange(unknown_times, "b g t n -> (b g) (t n)")

        # 交叉注意力因果掩码：确保只关注当前或过去的目标
        cross_attn_mask = _trajectory_times_flatten[:, None, :] >= _unknown_times_flatten[:, :, None]
        # 重复掩码以匹配多头注意力的头数
        cross_attn_mask = einops.repeat(cross_attn_mask, "bg tn1 tn2 -> (bg n_heads) tn1 tn2", n_heads=self.n_heads).contiguous()

        # Prepare for rel PE: 准备相对位置编码
        self.rel_pos_map = self.rel_pos_map.to(trajectory_features.device)
        # 计算每一对 (未知目标, 历史目标) 的相对时间索引索引对
        rel_pe_idx_pairs = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        [_unknown_times_flatten[_], _trajectory_times_flatten[_]]
                    ),
                    dim=-1,
                )
                for _ in range(len(_trajectory_times_flatten))
            ],
            dim=0,
        )  # (B*G, T_curr*N, T_traj*N, 2)
        rel_pe_idx_pairs = rel_pe_idx_pairs.to(trajectory_features.device)
        # 根据时间偏移查找预定义的相对位置索引
        rel_pe_idxs = self.rel_pos_map[
            rel_pe_idx_pairs[..., 0], rel_pe_idx_pairs[..., 1]
        ]  # (B*G, TN_curr, TN_traj)
        pass

        # 将逻辑掩码转换为浮点掩码（用于加算至注意力矩阵）
        cross_attn_key_padding_mask = torch.masked_fill(
            cross_attn_key_padding_mask.float(),
            mask=cross_attn_key_padding_mask,
            value=float("-inf"),
        ).to(self.dtype)
        cross_attn_mask = torch.masked_fill(
            cross_attn_mask.float(),
            mask=cross_attn_mask,
            value=float("-inf"),
        ).to(self.dtype)
        pass

        all_unknown_id_logits = None
        all_unknown_id_labels = None
        all_unknown_id_masks = None

        # 逐层进行解码
        for layer in range(self.num_layers):
            if use_decoder_checkpoint:
                # 使用激活值检查点以节省显存
                unknown_embeds = checkpoint(
                    self._forward_a_layer,
                    layer,
                    unknown_embeds, trajectory_embeds,
                    self_attn_key_padding_mask, cross_attn_key_padding_mask,
                    cross_attn_mask, rel_pe_idxs,
                    use_reentrant=False,
                )
            else:
                unknown_embeds = self._forward_a_layer(
                    layer=layer,
                    unknown_embeds=unknown_embeds,
                    trajectory_embeds=trajectory_embeds,
                    self_attn_key_padding_mask=self_attn_key_padding_mask,
                    cross_attn_key_padding_mask=cross_attn_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pe_idx=rel_pe_idxs,
                )

            # 通过当前层的分类头预测 ID logits
            _unknown_id_logits = self.embed_to_word_layers[layer](unknown_embeds[..., -self.id_dim:])
            _unknown_id_masks = unknown_masks.clone()
            _unknown_id_labels = None if not self.training else unknown_id_labels

            # 收集所有层的输出以计算辅助损失
            if all_unknown_id_logits is None:
                all_unknown_id_logits = _unknown_id_logits
                all_unknown_id_labels = _unknown_id_labels
                all_unknown_id_masks = _unknown_id_masks
            else:
                all_unknown_id_logits = torch.cat([all_unknown_id_logits, _unknown_id_logits], dim=0)
                all_unknown_id_labels = torch.cat([all_unknown_id_labels, _unknown_id_labels], dim=0) if _unknown_id_labels is not None else None
                all_unknown_id_masks = torch.cat([all_unknown_id_masks, _unknown_id_masks], dim=0)

        # 训练模式且启用辅助损失时返回堆叠的结果，否则只返回最后一层结果
        if self.training and self.use_aux_loss:
            return all_unknown_id_logits, all_unknown_id_labels, all_unknown_id_masks
        else:
            return _unknown_id_logits, _unknown_id_labels, _unknown_id_masks

    def _forward_a_layer(
            self,
            layer: int,
            unknown_embeds: torch.Tensor,
            trajectory_embeds: torch.Tensor,
            self_attn_key_padding_mask: torch.Tensor,
            cross_attn_key_padding_mask: torch.Tensor,
            cross_attn_mask: torch.Tensor,
            rel_pe_idx: torch.Tensor,
    ):
        """
        单层解码器的前向传播。
        包含：同一帧内的自注意力、跨时间的交叉注意力、前馈网络。
        """
        _B, _G, _T, _N, _ = trajectory_embeds.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_embeds.shape

        # 1. 自注意力 (Self-Attention): 在同一时间步的目标之间传递信息
        if layer > 0:  # 第一层跳过自注意力，因为 ID 信息尚未合并
            self_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g t) n c").contiguous()
            self_out, _ = self.self_attn_layers[layer - 1](
                query=self_unknown_embeds, key=self_unknown_embeds, value=self_unknown_embeds,
                key_padding_mask=self_attn_key_padding_mask,
            )
            self_out = self_unknown_embeds + self_out
            self_out = self.self_attn_norm_layers[layer - 1](self_out)
            unknown_embeds = einops.rearrange(self_out, "(b g t) n c -> b g t n c", b=_B, g=_G, t=_curr_T)

        # 2. 交叉注意力 (Cross-Attention): 处理视觉特征和 ID 信息的关联
        cross_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        cross_trajectory_embeds = einops.rearrange(trajectory_embeds, "b g t n c -> (b g) (t n) c").contiguous()

        # 将相对位置编码映射到对应的注意力偏移掩码中
        rel_pe_mask = self.rel_pos_embeds[layer][
            rel_pe_idx
        ]  # (B*G, TN_curr, TN_traj, n_heads)
        # 将偏置加到因果掩码和填充掩码中
        cross_attn_mask_with_rel_pe = cross_attn_mask + einops.rearrange(rel_pe_mask, "bg l1 l2 n -> (bg n) l1 l2")

        # 执行交叉注意力
        cross_out, _ = self.cross_attn_layers[layer](
            query=cross_unknown_embeds, key=cross_trajectory_embeds, value=cross_trajectory_embeds,
            key_padding_mask=cross_attn_key_padding_mask,
            attn_mask=cross_attn_mask_with_rel_pe,
        )
        cross_out = cross_unknown_embeds + cross_out
        cross_out = self.cross_attn_norm_layers[layer](cross_out)

        # 3. 前馈网络 (FFN):
        cross_out = cross_out + self.ffn_layers[layer](cross_out)
        cross_out = self.ffn_norm_layers[layer](cross_out)

        # 将展平的特征重新变回原始形状
        unknown_embeds = einops.rearrange(cross_out, "(b g) (t n) c -> b g t n c", b=_B, g=_G, t=_curr_T)

        return unknown_embeds

    def id_label_to_embed(self, id_labels):
        """
        将整数形式的 ID 标签转换为对应的 ID 嵌入（Embedding）。
        """
        # 转换为 one-hot 编码
        id_words = label_to_one_hot(id_labels, self.num_id_vocabulary + 1, dtype=self.dtype)
        # 通过线性层投影到嵌入空间
        id_embeds = self.word_to_embed(id_words)
        return id_embeds

    def generate_empty_id_embed(self, unknown_features):
        """
        为未知目标生成空的（Null）ID 嵌入。通常使用词汇表中最后一个 ID 作为背景或未知类别。
        """
        _shape = unknown_features.shape[:-1]
        # 用 num_id_vocabulary (最后一个索引) 填充作为空标签
        empty_id_labels = self.num_id_vocabulary * torch.ones(_shape, dtype=torch.int64, device=unknown_features.device)
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds

    def shuffle(self):
        """
        对 ID 词汇表进行随机打乱，可能用于鲁棒性训练或某些增强策略。
        """
        shuffle_index = torch.randperm(self.num_id_vocabulary, device=self.word_to_embed.weight.device)
        # 保持最后一个 (Null ID) 在最后
        shuffle_index = torch.cat([shuffle_index, torch.tensor([self.num_id_vocabulary], device=self.word_to_embed.weight.device)])
        # 更新投影矩阵的权重数据
        self.word_to_embed.weight.data = self.word_to_embed.weight.data[:, shuffle_index]
        self.embed_to_word.weight.data = self.embed_to_word.weight.data[shuffle_index, :]
        pass

    @property
    def dtype(self):
        """
        返回模型当前权重的数据类型。
        """
        return self.word_to_embed.weight.dtype
