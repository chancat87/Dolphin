# encoding: utf8
"""
Hotword encoder module for deep-biasing.

This module provides hotword encoding using LSTM + Attention fusion
to boost recognition of specific words/phrases during ASR decoding.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class HotwordEncoder(nn.Module):
    """Hotword encoder using deep-biasing approach.

    This module encodes hotword token sequences using a bidirectional LSTM,
    then fuses the embeddings with encoder outputs using multi-headed attention.

    Args:
        vocab_size (int): Vocabulary size for embedding layer.
        embedding_size (int): Embedding dimension (must match encoder d_model).
        encoder_layers (int): Number of LSTM layers for context encoding.
        attention_heads (int): Number of attention heads for fusion.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.vocab_size = vocab_size
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        self.context_extractor = _BLSTM(
            vocab_size, embedding_size, encoder_layers, dropout_rate
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_size * 4, embedding_size),
            nn.LayerNorm(embedding_size)
        )

        from dolphin.model import MultiHeadedAttention
        self.biasing_layer = MultiHeadedAttention(
            n_head=attention_heads,
            n_feat=embedding_size,
            dropout_rate=dropout_rate
        )

        self.combiner = nn.Linear(embedding_size, embedding_size)
        self.norm_aft_combiner = nn.LayerNorm(embedding_size)

    def forward_context_emb(
        self,
        context_list: torch.Tensor,
        context_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Extract context embeddings from hotword token sequences.

        Args:
            context_list (torch.Tensor): (batch, max_context_len) token IDs.
            context_lengths (torch.Tensor): (batch,) actual lengths.

        Returns:
            torch.Tensor: (1, num_hotwords, embedding_size) context embeddings.
        """
        context_emb = self.context_extractor(context_list, context_lengths)
        context_emb = self.context_encoder(context_emb.unsqueeze(0))
        return context_emb

    def forward(
        self,
        context_emb: torch.Tensor,
        encoder_out: torch.Tensor,
        biasing_score: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse context embeddings with encoder outputs using attention.

        Args:
            context_emb (torch.Tensor): (1, num_hotwords, embedding_size) context embeddings.
            encoder_out (torch.Tensor): (batch, T, d_model) encoder outputs.
            biasing_score (float): Degree of context biasing.
            recognize (bool): If True, skip context decoder computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - encoder_bias_out: (batch, T, d_model) fused encoder outputs.
                - bias_out: dummy tensor (for compatibility).
        """
        context_emb = context_emb.expand(encoder_out.shape[0], -1, -1)
        context_emb, _ = self.biasing_layer(
            encoder_out, context_emb, context_emb
        )
        encoder_bias_out = self.norm_aft_combiner(
            encoder_out + self.combiner(context_emb) * biasing_score
        )
        return encoder_bias_out, torch.tensor(0.0)


class _BLSTM(nn.Module):
    """Bidirectional LSTM for encoding hotword token sequences.

    Encodes variable-length hotword sequences into fixed-size embeddings
    by concatenating the last hidden states from both directions.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        self.sen_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

    def forward(
        self,
        sen_batch: torch.Tensor,
        sen_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode hotword sequences.

        Args:
            sen_batch (torch.Tensor): (batch, max_len) token IDs.
            sen_lengths (torch.Tensor): (batch,) actual lengths.

        Returns:
            torch.Tensor: (batch, embedding_size * 4) concatenated last hidden states.
        """
        sen_batch = torch.clamp(sen_batch, 0)
        sen_batch = self.word_embedding(sen_batch)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            sen_batch,
            sen_lengths.to('cpu').type(torch.int32),
            batch_first=True,
            enforce_sorted=False
        )
        _, last_state = self.sen_rnn(packed_seq)
        last_h = last_state[0]
        last_c = last_state[1]

        state = torch.cat([
            last_h[-1, :, :],
            last_h[-2, :, :],
            last_c[-1, :, :],
            last_c[-2, :, :]
        ], dim=-1)
        return state


def two_stage_filtering(
    context_list: List[List[int]],
    ctc_posterior: torch.Tensor,
    filter_threshold: float = -4.0,
    filter_window_size: int = 64
) -> List[List[int]]:
    """Filter hotwords using CTC posterior probabilities.

    Uses PSC (Peak-to-Side-Cluster) and SOC (Sum-to-Side-Cluster) scores
    to select relevant hotwords based on CTC posteriors.

    Args:
        context_list (List[List[int]]): List of hotword token ID sequences.
        ctc_posterior (torch.Tensor): (T, vocab_size) CTC log probabilities.
        filter_threshold (float): Minimum score threshold.
        filter_window_size (int): Window size for PSC/SOC calculation.

    Returns:
        List[List[int]]: Filtered hotword list.
    """
    if len(context_list) == 0:
        return context_list

    device = ctc_posterior.device
    vocab_size = ctc_posterior.shape[-1]
    SOC_score = {}

    for t in range(1, ctc_posterior.shape[0]):
        if t % (filter_window_size // 2) != 0 and t != ctc_posterior.shape[0] - 1:
            continue

        PSC_score = {}
        max_posterior, _ = torch.max(
            ctc_posterior[max(0, t - filter_window_size):t, :],
            dim=0,
            keepdim=False
        )
        max_posterior = max_posterior.tolist()

        for i in range(len(context_list)):
            # Filter out token IDs that are out of vocabulary range
            valid_tokens = [j for j in context_list[i] if 0 <= j < vocab_size]
            if not valid_tokens:
                continue
            score = sum(max_posterior[j] for j in valid_tokens) / len(valid_tokens)
            PSC_score[i] = max(SOC_score.get(i, -float('inf')), score)

        PSC_filtered_index = []
        for i in PSC_score:
            if PSC_score[i] > filter_threshold:
                PSC_filtered_index.append(i)

        if len(PSC_filtered_index) == 0:
            continue

        filtered_context_list = []
        for i in PSC_filtered_index:
            filtered_context_list.append(context_list[i])

        win_posterior = ctc_posterior[max(0, t - filter_window_size):t, :]
        win_posterior = win_posterior.unsqueeze(0).expand(
            len(filtered_context_list), -1, -1
        )

        select_win_posterior = []
        for i in range(len(filtered_context_list)):
            # Filter valid tokens for index_select
            valid_tokens = [j for j in filtered_context_list[i] if 0 <= j < vocab_size]
            if not valid_tokens:
                # Use dummy tensor if no valid tokens
                select_win_posterior.append(torch.zeros(1, 1, device=device))
                continue
            select_win_posterior.append(torch.index_select(
                win_posterior[0], 1,
                torch.tensor(valid_tokens, device=device)
            ).transpose(0, 1))

        select_win_posterior = nn.utils.rnn.pad_sequence(
            select_win_posterior, batch_first=True
        ).transpose(1, 2).contiguous()

        dp = torch.full(
            (select_win_posterior.shape[0], select_win_posterior.shape[2]),
            -10000.0,
            dtype=torch.float32,
            device=select_win_posterior.device
        )
        dp[:, 0] = select_win_posterior[:, 0, 0]

        for win_t in range(1, select_win_posterior.shape[1]):
            temp = dp[:, :-1] + select_win_posterior[:, win_t, 1:]
            idx = torch.where(temp > dp[:, 1:])
            idx_ = (idx[0], idx[1] + 1)
            dp[idx_] = temp[idx]
            dp[:, 0] = torch.where(
                select_win_posterior[:, win_t, 0] > dp[:, 0],
                select_win_posterior[:, win_t, 0],
                dp[:, 0]
            )

        for i in range(len(filtered_context_list)):
            SOC_score[PSC_filtered_index[i]] = max(
                SOC_score.get(PSC_filtered_index[i], -float('inf')),
                dp[i][len(filtered_context_list[i]) - 1] / len(filtered_context_list[i])
            )

    filtered_context_list = []
    for i in range(len(context_list)):
        if SOC_score.get(i, -float('inf')) > filter_threshold:
            filtered_context_list.append(context_list[i])

    return filtered_context_list


def prepare_hotword_tensor(
    context_list: List[List[int]],
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare hotword token tensor from token ID sequences.

    Adds a no-bias token (ID=0) at the beginning and pads sequences.

    Args:
        context_list (List[List[int]]): List of hotword token ID sequences.
        device: Target device for tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - context_list_tensor: (1, max_len) padded tensor.
            - context_list_lengths: (1,) actual lengths.
    """
    context_list_tensor = [torch.tensor([0], dtype=torch.int32, device=device)]
    for context_token in context_list:
        context_list_tensor.append(torch.tensor(context_token, dtype=torch.int32, device=device))

    context_list_lengths = torch.tensor(
        [x.size(0) for x in context_list_tensor],
        dtype=torch.int32,
        device=device
    )
    context_list_tensor = nn.utils.rnn.pad_sequence(
        context_list_tensor,
        batch_first=True,
        padding_value=-1
    )
    return context_list_tensor, context_list_lengths


def apply_deep_biasing(
    encoder_out: torch.Tensor,
    ctc_logprobs: torch.Tensor,
    hotword_encoder: HotwordEncoder,
    context_list: List[List[int]],
    use_two_stage_filter: bool = False,
    filter_threshold: float = -4.0,
    deep_biasing_score: float = 1.0
) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
    """Apply deep-biasing to encoder outputs using hotwords.

    Args:
        encoder_out (torch.Tensor): (batch, T, d_model) encoder outputs.
        ctc_logprobs (torch.Tensor): (batch, T, vocab_size) CTC log probabilities.
        hotword_encoder (HotwordEncoder): Hotword encoder module.
        context_list (List[List[int]]): List of hotword token ID sequences.
        use_two_stage_filter (bool): Whether to use two-stage filtering.
        filter_threshold (float): Threshold for two-stage filtering.
        deep_biasing_score (float): Biasing score multiplier.

    Returns:
        Tuple[torch.Tensor, Optional[List[List[int]]]]]:
            - Fused encoder outputs.
            - Filtered context list if two-stage filtering was used.
    """
    device = encoder_out.device
    batch_size = encoder_out.shape[0]

    if use_two_stage_filter:
        # Per-sample filtering: each sample in batch gets its own filtered hotwords
        filtered_context_list = []

        for b in range(batch_size):
            # Extract CTC posterior for this sample: (T, vocab_size)
            ctc_probs = ctc_logprobs[b]
            if ctc_probs.dim() == 3:
                ctc_probs = ctc_probs.squeeze(0)

            # Filter hotwords based on this sample's CTC posterior
            sample_filtered = two_stage_filtering(
                context_list, ctc_probs, filter_threshold
            )
            filtered_context_list.append(sample_filtered)

        # Merge all filtered hotwords from all samples for batch use
        # Each sample may have different relevant hotwords, merge them all
        context_list = []
        for b_context_list in filtered_context_list:
            if not b_context_list:
                continue
            for each_in_b in b_context_list:
                if each_in_b not in context_list:
                    context_list.append(each_in_b)
        print(context_list)
    # Prepare hotword tensor
    context_list_tensor, context_list_lengths = prepare_hotword_tensor(
        context_list, device
    )

    # Encode hotwords to embeddings
    context_emb = hotword_encoder.forward_context_emb(
        context_list_tensor, context_list_lengths
    )

    # Apply attention fusion
    encoder_out, _ = hotword_encoder(
        context_emb, encoder_out, deep_biasing_score
    )

    return encoder_out, context_list if use_two_stage_filter else None
