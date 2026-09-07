from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .units import (
    UnitA,
    UnitD,
    UnitE,
    UnitF,
    UnitG,
    UnitH,
    UnitI,
)


class UnitJ(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = torch.matmul(x, self.fc1.weight.t()) + self.fc1.bias
        x = F.gelu(x)
        x = self.drop(x)
        x = (x * self.fc2.weight.squeeze(0)).sum(dim=-1, keepdim=True) + self.fc2.bias
        return x


class ModelA(nn.Module):
    def __init__(
        self,
        bert_dir,
        text_dim=768,
        audio_dim=74,
        vision_dim=35,
        hidden_dim=128,
        heads=4,
        feature_layers=1,
        align_layers=1,
        dropout=0.1,
        scale_a=0.15,
        limit_x=0.1,
        scale_x=1.0,
        mode_e="b",
        scale_y=0.0,
        detach_y=False,
        flag_i=True,
        mode_i="i",
        width_i=3,
        sigma_i=0.12,
        offset_i=0.35,
        scale_i=0.20,
        flag_h=False,
        bound_h=0.35,
        sharp_h=8.0,
        amp_h=1.0,
        flag_j=False,
        flag_k=False,
        mix_k=0.0,
        shrink_k=0.0,
        flag_b=False,
    ):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(bert_dir, local_files_only=True)
        self.flag_b = flag_b
        self.mode_e = mode_e
        self.scale_y = float(scale_y)
        self.flag_i = flag_i
        self.mode_i = mode_i
        self.flag_k = flag_k
        self.mix_k = float(mix_k)
        self.shrink_k = float(shrink_k)
        if mode_e not in {"a", "b", "none"}:
            raise ValueError("mode_e must be one of: a, b, none")
        if mode_i not in {"i", "h", "none"}:
            raise ValueError("mode_i must be one of: i, h, none")
        if not flag_b:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

        self.unit_0 = UnitA(text_dim, hidden_dim, num_layers=feature_layers, heads=heads, dropout=dropout)
        self.unit_1 = UnitA(audio_dim, hidden_dim, num_layers=feature_layers, heads=heads, dropout=dropout)
        self.unit_2 = UnitA(vision_dim, hidden_dim, num_layers=feature_layers, heads=heads, dropout=dropout)

        self.unit_3 = UnitD(
            hidden_dim, 768, hidden_dim, heads=heads, dropout=dropout, num_layers=align_layers
        )
        self.unit_4 = UnitD(
            hidden_dim, 768, hidden_dim, heads=heads, dropout=dropout, num_layers=align_layers
        )
        self.unit_5 = UnitD(
            hidden_dim, 768, hidden_dim, heads=heads, dropout=dropout, num_layers=align_layers
        )

        self.unit_6 = UnitE(hidden_dim, 768, heads=heads, scale_a=scale_a, dropout=dropout)
        self.unit_7 = UnitE(hidden_dim, 768, heads=heads, scale_a=scale_a, dropout=dropout)
        self.unit_8 = UnitE(hidden_dim, 768, heads=heads, scale_a=scale_a, dropout=dropout)

        if mode_i == "i":
            self.unit_i = UnitI(
                hidden_dim,
                dim_x=768,
                dropout=dropout,
                scale_x=sigma_i,
                offset_x=offset_i,
                scale_y=scale_i,
                flag_x=flag_h,
                bound_x=bound_h,
                sharp_x=sharp_h,
                amp_x=amp_h,
            )
        elif mode_i == "h":
            self.unit_i = UnitH(hidden_dim, kernel_size=width_i, dropout=dropout)
        else:
            self.unit_i = None
        self.unit_g = UnitG(hidden_dim, limit_x=limit_x, scale_x=scale_x, dropout=dropout)
        self.unit_f = UnitF(
            hidden_dim,
            768,
            scale_x=scale_y,
            dropout=dropout,
            detach_x=detach_y,
        )
        self.unit_j = UnitJ(hidden_dim, dropout=dropout)
        if flag_k:
            self.path_k = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.head_k = nn.Sequential(
                nn.LayerNorm(hidden_dim + 4),
                nn.Linear(hidden_dim + 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.flag_j = flag_j
        if flag_j:
            self.unit_j_0 = UnitJ(hidden_dim, dropout=dropout)
            self.unit_j_1 = UnitJ(hidden_dim, dropout=dropout)
            self.unit_j_2 = UnitJ(hidden_dim, dropout=dropout)

    def train(self, mode=True):
        super().train(mode)
        if not self.flag_b:
            self.bert.eval()
        return self

    def _bert_encode(self, bert_input):
        if isinstance(bert_input, dict):
            hidden = bert_input["hidden"].float()
            attention_mask = bert_input.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
            else:
                attention_mask = attention_mask.long()
            return hidden, attention_mask

        input_ids = bert_input[:, 0, :].long()
        attention_mask = bert_input[:, 1, :].long()
        token_type_ids = bert_input[:, 2, :].long()
        ctx = nullcontext() if self.flag_b else torch.no_grad()
        with ctx:
            hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state
        return hidden, attention_mask

    def _pool_explanation_queries(self, aligned_tokens, clue_mask):
        query_tokens = aligned_tokens[:, 1:]
        valid = clue_mask.to(dtype=query_tokens.dtype, device=query_tokens.device).unsqueeze(-1)
        pooled = (query_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return 0.5 * query_tokens[:, 0] + 0.5 * pooled

    def _aligned_padding_mask(self, aligned_tokens, clue_mask):
        batch_size, token_len, _ = aligned_tokens.shape
        if clue_mask is None:
            return None
        clue_valid = clue_mask.to(device=aligned_tokens.device).bool()
        if clue_valid.size(1) == token_len - 1:
            cls_valid = torch.ones(batch_size, 1, device=aligned_tokens.device, dtype=torch.bool)
            valid = torch.cat([cls_valid, clue_valid], dim=1)
        elif clue_valid.size(1) >= token_len:
            valid = clue_valid[:, :token_len]
        else:
            pad = torch.zeros(batch_size, token_len - clue_valid.size(1), device=aligned_tokens.device, dtype=torch.bool)
            valid = torch.cat([clue_valid, pad], dim=1)
        return ~valid

    def _encode_one(
        self,
        feature_encoder,
        aligner,
        desc_gate,
        features,
        clue_hidden,
        clue_mask,
        final_hidden,
        final_mask,
        apply_desc_reweight=True,
    ):
        feature_tokens = feature_encoder(features)
        if apply_desc_reweight and self.mode_e == "a":
            feature_tokens, desc_gate_values, desc_importance = desc_gate(
                feature_tokens,
                final_hidden,
                description_mask=final_mask,
            )
        else:
            desc_gate_values = None
            desc_importance = None
        aligned_tokens, align_attn = aligner(feature_tokens, clue_hidden)
        if apply_desc_reweight and self.mode_e == "b":
            aligned_tokens, desc_gate_values, desc_importance = desc_gate(
                aligned_tokens,
                final_hidden,
                token_mask=self._aligned_padding_mask(aligned_tokens, clue_mask),
                description_mask=final_mask,
            )
        modality_repr = self._pool_explanation_queries(aligned_tokens, clue_mask)
        return modality_repr, {
            "tokens": aligned_tokens,
            "repr": modality_repr,
            "align_attn": align_attn,
            "desc_gate": desc_gate_values,
            "desc_importance": desc_importance,
        }

    def _pool_aligned_tokens(self, aligned_tokens, clue_mask):
        return self._pool_explanation_queries(aligned_tokens, clue_mask)

    def forward(self, batch):
        text_clue_hidden, text_clue_mask = self._bert_encode(batch["text_clue_bert"])
        audio_clue_hidden, audio_clue_mask = self._bert_encode(batch["audio_clue_bert"])
        visual_clue_hidden, visual_clue_mask = self._bert_encode(batch["visual_clue_bert"])
        final_hidden, final_mask = self._bert_encode(batch["final_description_bert"])

        text_repr, text_aux = self._encode_one(
            self.unit_0,
            self.unit_3,
            self.unit_6,
            batch["text"].float(),
            text_clue_hidden,
            text_clue_mask,
            final_hidden,
            final_mask,
            apply_desc_reweight=False,
        )
        audio_repr, audio_aux = self._encode_one(
            self.unit_1,
            self.unit_4,
            self.unit_7,
            batch["audio"].float(),
            audio_clue_hidden,
            audio_clue_mask,
            final_hidden,
            final_mask,
        )
        vision_repr, vision_aux = self._encode_one(
            self.unit_2,
            self.unit_5,
            self.unit_8,
            batch["vision"].float(),
            visual_clue_hidden,
            visual_clue_mask,
            final_hidden,
            final_mask,
        )
        if self.flag_i and self.unit_i is not None:
            if self.mode_i == "i":
                audio_tokens, vision_tokens = self.unit_i(
                    audio_aux["tokens"],
                    vision_aux["tokens"],
                    audio_align_attn=audio_aux["align_attn"],
                    vision_align_attn=vision_aux["align_attn"],
                    audio_clue_mask=audio_clue_mask,
                    vision_clue_mask=visual_clue_mask,
                    final_hidden=final_hidden,
                    final_mask=final_mask,
                )
            else:
                audio_tokens, vision_tokens = self.unit_i(audio_aux["tokens"], vision_aux["tokens"])
            audio_aux["tokens_i"] = audio_tokens
            vision_aux["tokens_i"] = vision_tokens
            audio_repr = self._pool_aligned_tokens(audio_tokens, audio_clue_mask)
            vision_repr = self._pool_aligned_tokens(vision_tokens, visual_clue_mask)
            audio_aux["repr_i"] = audio_repr
            vision_aux["repr_i"] = vision_repr

        modal_reprs = torch.stack([text_repr, audio_repr, vision_repr], dim=1)
        aux_k = None
        input_x = text_repr
        if self.flag_k and "u_k" in batch:
            aux_k = batch["u_k"].float()
            delta_k = self.path_k(aux_k)
            input_x = text_repr + self.mix_k * delta_k
        values_x = batch.get("u_g")
        offset_x = None
        if self.scale_y > 0:
            offset_x = self.unit_f(modal_reprs, final_hidden, final_mask)
        fused, coeff_x = self.unit_g(
            modal_reprs,
            values_x=values_x,
            input_x=input_x,
            offset_x=offset_x,
        )
        raw_prediction = self.unit_j(fused).squeeze(-1)
        prediction = raw_prediction
        output = {
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "u_h": coeff_x,
            "modal_reprs": modal_reprs,
            "text": text_aux,
            "audio": audio_aux,
            "vision": vision_aux,
        }
        if offset_x is not None:
            output["offset_x"] = offset_x
        if self.flag_k and aux_k is not None:
            input_k = torch.cat([fused, aux_k], dim=-1).contiguous()
            logit_k = self.head_k(input_k).squeeze(-1)
            prob_k = torch.sigmoid(logit_k)
            if self.shrink_k > 0:
                strength_k = aux_k[:, 0].clamp(0.0, 1.0)
                shrink = (self.shrink_k * strength_k * prob_k).clamp(0.0, 0.95)
                prediction = raw_prediction * (1.0 - shrink)
                output["prediction"] = prediction
            output["out_k"] = logit_k
            output["prob_k"] = prob_k
        if self.flag_j:
            output["u_j"] = {
                "text": self.unit_j_0(text_repr).squeeze(-1),
                "audio": self.unit_j_1(audio_repr).squeeze(-1),
                "vision": self.unit_j_2(vision_repr).squeeze(-1),
            }
        return output
