import math

import torch
from torch import nn
from torch.nn import functional as F


class UnitA(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [UnitB(hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, padding_mask=None):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return self.norm(x)


class UnitB(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class UnitC(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, attn = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(query + self.drop(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x, attn


class UnitD(nn.Module):

    def __init__(
        self,
        feature_dim=128,
        explanation_dim=768,
        hidden_dim=128,
        heads=4,
        dropout=0.1,
        num_layers=1,
    ):
        super().__init__()
        self.query_proj = nn.Linear(explanation_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.cls_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.layers = nn.ModuleList(
            [
                UnitC(hidden_dim, heads=heads, dropout=dropout)
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, feature_tokens, explanation_tokens, feature_padding_mask=None):
        batch_size = feature_tokens.size(0)
        x = self.query_proj(explanation_tokens)
        cls_query = self.cls_query.expand(batch_size, -1, -1)
        x = torch.cat([cls_query, x], dim=1)
        key = self.key_proj(feature_tokens)
        value = self.value_proj(feature_tokens)
        attn = None
        for layer in self.layers:
            x, attn = layer(x, key, value, key_padding_mask=feature_padding_mask)
        return x, attn


class UnitE(nn.Module):

    def __init__(self, token_dim=128, description_dim=768, heads=4, scale_a=0.15, dropout=0.1, detach_x=False):
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError("token_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = token_dim // heads
        self.scale_a = scale_a
        self.detach_x = detach_x
        self.q_proj = nn.Linear(description_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, description_tokens, token_mask=None, description_mask=None):
        bsz, token_len, dim = tokens.shape
        q = self.q_proj(description_tokens).view(bsz, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(bsz, token_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if token_mask is not None:
            scores = scores.masked_fill(token_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        if description_mask is not None:
            valid_desc = description_mask[:, None, :, None].to(dtype=attn.dtype)
            denom_desc = (valid_desc.sum(dim=(1, 2)) * self.heads).clamp_min(1.0)
            importance = (attn * valid_desc).sum(dim=(1, 2)) / denom_desc
        else:
            importance = attn.mean(dim=(1, 2))

        if token_mask is None:
            valid = torch.ones_like(importance)
        else:
            valid = (~token_mask.bool()).to(dtype=importance.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_importance = (importance * valid).sum(dim=1, keepdim=True) / denom
        relative = importance / mean_importance.clamp_min(1e-6)
        gate = 1.0 + self.scale_a * (relative - 1.0)
        gate = torch.clamp(gate, 1.0 - self.scale_a, 1.0 + self.scale_a)
        gate = gate * valid + (1.0 - valid)
        gate_mean = (gate * valid).sum(dim=1, keepdim=True) / denom
        gate = gate / gate_mean.clamp_min(1e-6)
        if self.detach_x:
            gate = gate.detach()
        return tokens * gate.unsqueeze(-1), gate, importance


class UnitF(nn.Module):

    def __init__(
        self,
        dim=128,
        dim_x=768,
        scale_x=0.5,
        dropout=0.1,
        detach_x=False,
    ):
        super().__init__()
        self.scale_x = float(scale_x)
        self.detach_x = detach_x
        self.norm_x = nn.LayerNorm(dim_x)
        self.proj_x = nn.Sequential(
            nn.Linear(dim_x, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm_y = nn.LayerNorm(dim)
        self.proj_y = nn.Linear(dim, dim, bias=False)
        self.proj_z = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def _pool_x(self, description_tokens, description_mask=None):
        if description_mask is None:
            mean_pooled = description_tokens.mean(dim=1)
        else:
            valid = description_mask.to(device=description_tokens.device, dtype=description_tokens.dtype).unsqueeze(-1)
            mean_pooled = (description_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return 0.5 * description_tokens[:, 0] + 0.5 * mean_pooled

    def forward(self, modal_reprs, description_tokens, description_mask=None):
        if self.scale_x == 0.0 or description_tokens is None:
            return torch.zeros(modal_reprs.shape[:2], device=modal_reprs.device, dtype=modal_reprs.dtype)
        modal = modal_reprs.detach() if self.detach_x else modal_reprs
        desc = self._pool_x(description_tokens.to(dtype=modal.dtype), description_mask)
        desc = self.proj_x(self.norm_x(desc))
        desc_query = F.normalize(desc, dim=-1)
        modal_key = F.normalize(self.proj_y(self.norm_y(modal)), dim=-1)
        compatibility = torch.sum(modal_key * desc_query[:, None, :], dim=-1)
        desc_expanded = desc[:, None, :].expand(-1, modal.size(1), -1)
        pair_score = self.proj_z(torch.cat([modal, desc_expanded], dim=-1)).squeeze(-1)
        scores = compatibility + pair_score
        scores = scores - scores.mean(dim=-1, keepdim=True)
        return self.scale_x * torch.tanh(scores)


def op_a(logits, limit_x=0.1):
    count = logits.size(-1)
    if count * limit_x >= 1.0:
        raise ValueError("invalid limit_x")
    probs = torch.softmax(logits, dim=-1)
    return limit_x + (1.0 - count * limit_x) * probs


def op_b(values, limit_x=0.1, eps=1e-6):
    count = values.size(-1)
    residual_mass = 1.0 - count * limit_x
    residual = (values - limit_x) / residual_mass
    residual = residual.clamp_min(eps)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.log(residual)


class UnitG(nn.Module):

    def __init__(self, dim=128, limit_x=0.1, scale_x=1.0, dropout=0.1):
        super().__init__()
        self.limit_x = limit_x
        self.scale_x = scale_x
        self.path_x = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        self.path_y = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                )
                for _ in range(3)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, modal_reprs, values_x=None, input_x=None, offset_x=None):
        if modal_reprs.dim() != 3 or modal_reprs.size(1) != 3:
            raise ValueError("modal_reprs must have shape (batch, 3, dim)")
        if input_x is None:
            input_x = modal_reprs[:, 0]
        logits = self.path_x(input_x)
        if offset_x is not None:
            delta = offset_x.to(dtype=logits.dtype, device=logits.device)
            if delta.shape != logits.shape:
                raise ValueError("offset_x must have shape (batch, 3)")
            logits = logits + delta
        if values_x is not None:
            values = values_x.to(dtype=logits.dtype, device=logits.device)
            values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = logits + self.scale_x * op_b(values, limit_x=self.limit_x)
        coeff = op_a(logits, limit_x=self.limit_x)
        branch_values = torch.stack([part(modal_reprs[:, i]) for i, part in enumerate(self.path_y)], dim=1)
        mixed = torch.sum(coeff.unsqueeze(-1) * branch_values, dim=1)
        return self.out_norm(mixed), coeff


class UnitH(nn.Module):

    def __init__(self, dim=128, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.audio_norm = nn.LayerNorm(dim)
        self.vision_norm = nn.LayerNorm(dim)
        self.audio_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.vision_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.audio_out = nn.Linear(dim, dim)
        self.vision_out = nn.Linear(dim, dim)
        self.audio_gate = nn.Linear(dim, dim)
        self.vision_gate = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def _conv_x(self, x, conv):
        return conv(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, audio_tokens, vision_tokens):
        audio_norm = self.audio_norm(audio_tokens)
        vision_norm = self.vision_norm(vision_tokens)
        audio_update = self.audio_out(self._conv_x(audio_norm, self.audio_conv))
        vision_update = self.vision_out(self._conv_x(vision_norm, self.vision_conv))
        audio_gate = torch.sigmoid(self.vision_gate(vision_norm))
        vision_gate = torch.sigmoid(self.audio_gate(audio_norm))
        audio_aligned = audio_tokens + self.drop(audio_update * audio_gate)
        vision_aligned = vision_tokens + self.drop(vision_update * vision_gate)
        return self.out_norm(audio_aligned), self.out_norm(vision_aligned)


class UnitI(nn.Module):

    def __init__(
        self,
        dim=128,
        dim_x=768,
        dropout=0.1,
        scale_x=0.12,
        offset_x=0.35,
        scale_y=0.20,
        flag_x=False,
        bound_x=0.35,
        sharp_x=8.0,
        amp_x=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.scale_x = scale_x
        self.offset_x = offset_x
        self.scale_y = scale_y
        self.flag_x = flag_x
        self.bound_x = bound_x
        self.sharp_x = sharp_x
        self.amp_x = amp_x

        self.audio_norm = nn.LayerNorm(dim)
        self.vision_norm = nn.LayerNorm(dim)
        self.audio_q = nn.Linear(dim, dim, bias=False)
        self.audio_k = nn.Linear(dim, dim, bias=False)
        self.audio_v = nn.Linear(dim, dim, bias=False)
        self.vision_q = nn.Linear(dim, dim, bias=False)
        self.vision_k = nn.Linear(dim, dim, bias=False)
        self.vision_v = nn.Linear(dim, dim, bias=False)

        self.desc_q = nn.Linear(dim_x, dim, bias=False)
        self.desc_k = nn.Linear(dim, dim, bias=False)

        pair_dim = dim * 4 + 1
        self.audio_update = nn.Sequential(
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.vision_update = nn.Sequential(
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.audio_gate = nn.Linear(pair_dim, dim)
        self.vision_gate = nn.Linear(pair_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.audio_out_norm = nn.LayerNorm(dim)
        self.vision_out_norm = nn.LayerNorm(dim)

    def _mask(self, clue_mask, tokens):
        batch_size, token_len, _ = tokens.shape
        if clue_mask is None:
            return torch.ones(batch_size, token_len, device=tokens.device, dtype=torch.bool)
        valid = clue_mask.to(device=tokens.device).bool()
        if valid.size(1) == token_len - 1:
            cls_valid = torch.ones(batch_size, 1, device=tokens.device, dtype=torch.bool)
            valid = torch.cat([cls_valid, valid], dim=1)
        elif valid.size(1) > token_len:
            valid = valid[:, :token_len]
        elif valid.size(1) < token_len:
            pad = torch.zeros(batch_size, token_len - valid.size(1), device=tokens.device, dtype=torch.bool)
            valid = torch.cat([valid, pad], dim=1)
        return valid

    def _coords(self, align_attn, tokens):
        batch_size, token_len, _ = tokens.shape
        if align_attn is None:
            positions = torch.linspace(0.0, 1.0, token_len, device=tokens.device, dtype=tokens.dtype)
            anchors = positions.unsqueeze(0).expand(batch_size, -1)
            reliability = torch.ones_like(anchors)
            return anchors, reliability
        attn = align_attn
        if attn.dim() == 3:
            attn = attn.unsqueeze(1)
        attn = attn.mean(dim=1).to(dtype=tokens.dtype)
        attn = attn.clamp_min(0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        source_len = attn.size(-1)
        positions = torch.linspace(0.0, 1.0, source_len, device=tokens.device, dtype=tokens.dtype)
        anchors = torch.sum(attn * positions.view(1, 1, -1), dim=-1)
        probs = attn.clamp_min(1e-6)
        entropy = -(probs * probs.log()).sum(dim=-1)
        max_entropy = math.log(max(source_len, 2))
        reliability = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        return anchors, reliability

    def _offsets(self, mixed_tokens, final_hidden, final_mask, token_valid):
        if final_hidden is None:
            return torch.zeros(mixed_tokens.shape[:2], device=mixed_tokens.device, dtype=mixed_tokens.dtype)
        query = self.desc_q(final_hidden.to(dtype=mixed_tokens.dtype))
        key = self.desc_k(mixed_tokens)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.dim)
        scores = scores.masked_fill(~token_valid[:, None, :], -1e4)
        attn = torch.softmax(scores, dim=-1)
        if final_mask is not None:
            desc_valid = final_mask.to(device=mixed_tokens.device).bool()
            attn = attn * desc_valid[:, :, None].to(dtype=attn.dtype)
            denom = desc_valid.sum(dim=1, keepdim=True).to(dtype=attn.dtype).clamp_min(1.0)
            importance = attn.sum(dim=1) / denom
        else:
            importance = attn.mean(dim=1)
        valid = token_valid.to(dtype=importance.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_importance = (importance * valid).sum(dim=1, keepdim=True) / denom
        relative = (importance / mean_importance.clamp_min(1e-6)).clamp(0.25, 4.0)
        return self.offset_x * torch.log(relative.clamp_min(1e-6)) * valid

    def _directed_context(
        self,
        source_tokens,
        target_tokens,
        source_anchors,
        target_anchors,
        source_reliability,
        target_reliability,
        source_valid,
        target_valid,
        target_offsets,
        query_proj,
        key_proj,
        value_proj,
    ):
        query = query_proj(source_tokens)
        key = key_proj(target_tokens)
        value = value_proj(target_tokens)
        semantic_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.dim)
        distance = (source_anchors[:, :, None] - target_anchors[:, None, :]).pow(2)
        sigma = max(float(self.scale_x), 1e-4)
        time_logits = -distance / (2.0 * sigma * sigma)
        pair_confidence = (source_reliability[:, :, None] * target_reliability[:, None, :]).clamp_min(0.0).sqrt()
        confidence_logits = self.scale_y * torch.log(pair_confidence.clamp_min(1e-4))
        logits = semantic_logits + time_logits + confidence_logits + target_offsets[:, None, :]
        logits = logits.masked_fill(~target_valid[:, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, value)
        confidence = torch.sum(weights * pair_confidence, dim=-1, keepdim=True)
        confidence = confidence * source_valid[:, :, None].to(dtype=confidence.dtype)
        return context, confidence

    def forward(
        self,
        audio_tokens,
        vision_tokens,
        audio_align_attn=None,
        vision_align_attn=None,
        audio_clue_mask=None,
        vision_clue_mask=None,
        final_hidden=None,
        final_mask=None,
    ):
        audio_norm = self.audio_norm(audio_tokens)
        vision_norm = self.vision_norm(vision_tokens)
        audio_valid = self._mask(audio_clue_mask, audio_tokens)
        vision_valid = self._mask(vision_clue_mask, vision_tokens)
        audio_anchors, audio_reliability = self._coords(audio_align_attn, audio_tokens)
        vision_anchors, vision_reliability = self._coords(vision_align_attn, vision_tokens)
        audio_offsets = self._offsets(audio_norm, final_hidden, final_mask, audio_valid)
        vision_offsets = self._offsets(vision_norm, final_hidden, final_mask, vision_valid)

        vision_context, audio_confidence = self._directed_context(
            audio_norm,
            vision_norm,
            audio_anchors,
            vision_anchors,
            audio_reliability,
            vision_reliability,
            audio_valid,
            vision_valid,
            vision_offsets,
            self.audio_q,
            self.vision_k,
            self.vision_v,
        )
        audio_context, vision_confidence = self._directed_context(
            vision_norm,
            audio_norm,
            vision_anchors,
            audio_anchors,
            vision_reliability,
            audio_reliability,
            vision_valid,
            audio_valid,
            audio_offsets,
            self.vision_q,
            self.audio_k,
            self.audio_v,
        )

        audio_pair = torch.cat(
            [audio_norm, vision_context, vision_context - audio_norm, audio_norm * vision_context, audio_confidence],
            dim=-1,
        )
        vision_pair = torch.cat(
            [vision_norm, audio_context, audio_context - vision_norm, vision_norm * audio_context, vision_confidence],
            dim=-1,
        )
        audio_gate = torch.sigmoid(self.audio_gate(audio_pair)) * audio_valid[:, :, None].to(dtype=audio_tokens.dtype)
        vision_gate = torch.sigmoid(self.vision_gate(vision_pair)) * vision_valid[:, :, None].to(dtype=vision_tokens.dtype)
        if self.flag_x:
            audio_need_vision = torch.sigmoid(
                float(self.sharp_x) * (float(self.bound_x) - audio_reliability)
            )
            vision_need_audio = torch.sigmoid(
                float(self.sharp_x) * (float(self.bound_x) - vision_reliability)
            )
            audio_gate = audio_gate * (float(self.amp_x) * audio_need_vision[:, :, None]).clamp(0.0, 1.0)
            vision_gate = vision_gate * (float(self.amp_x) * vision_need_audio[:, :, None]).clamp(0.0, 1.0)
        audio_update = self.audio_update(audio_pair)
        vision_update = self.vision_update(vision_pair)
        audio_aligned = audio_tokens + self.drop(audio_gate * audio_update)
        vision_aligned = vision_tokens + self.drop(vision_gate * vision_update)
        return self.audio_out_norm(audio_aligned), self.vision_out_norm(vision_aligned)
