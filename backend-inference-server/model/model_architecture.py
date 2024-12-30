""" 
Architecture for the model to load from pytorch into torchscript
"""

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Implements the self-attention mechanism used in the Transformer model.

    Attributes:
        embed_size (int): The size of the embedding vector.
        heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        values (nn.Linear): Linear layer to compute value vectors.
        keys (nn.Linear): Linear layer to compute key vectors.
        queries (nn.Linear): Linear layer to compute query vectors.
        fc_out (nn.Linear): Linear layer to project concatenated attention outputs back to embedding size.
    """

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head_dim = embed_size // heads
        self.heads = heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Performs the forward pass for self-attention.

        Args:
            values (torch.Tensor): Value vectors of shape (N, value_len, embed_size).
            keys (torch.Tensor): Key vectors of shape (N, key_len, embed_size).
            query (torch.Tensor): Query vectors of shape (N, query_len, embed_size).
            mask (torch.Tensor): Mask tensor to prevent attention on certain positions.

        Returns:
            torch.Tensor: Output tensor after self-attention of shape (N, query_len, embed_size).
        """
        # Number of training examples
        N = query.shape[0]

        # Length of values, keys, and queries
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Allows us to multiply without flattening / batch multiply
        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # Queries shape: (N, query_len, heads, heads_dim)
        # Keys shape: (N, key_len, heads, heads_dim)
        # Energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # Attention shape: (N, heads, query_len, key_len)
        # Values shape: (N, value_len, heads, heads_dim)
        # Result" (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    Implements a single Transformer block combining self-attention and feedforward layers.

    Attributes:
        attention (SelfAttention): Self-attention layer.
        norm1 (nn.LayerNorm): Layer normalization applied after self-attention.
        norm2 (nn.LayerNorm): Layer normalization applied after feedforward network.
        feed_forward (nn.Sequential): Feedforward neural network.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        Performs the forward pass for the Transformer block.

        Args:
            value (torch.Tensor): Value vectors.
            key (torch.Tensor): Key vectors.
            query (torch.Tensor): Query vectors.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape (N, query_len, embed_size).
        """
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    """
    Encoder module for the Transformer model, consisting of embedding, positional encoding,
    and multiple Transformer blocks.

    Attributes:
        word_embedding (nn.Embedding): Embedding layer for input tokens.
        position_embedding (nn.Embedding): Embedding layer for positional encoding.
        layers (nn.ModuleList): List of Transformer blocks.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Encodes the input sequence using the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, seq_length).
            mask (torch.Tensor): Mask tensor for input padding.

        Returns:
            torch.Tensor: Encoded representation of shape (N, seq_length, embed_size).
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    """
    Implements a single Decoder block with self-attention, cross-attention,
    and feedforward layers.

    Attributes:
        attention (SelfAttention): Self-attention layer.
        norm (nn.LayerNorm): Layer normalization applied after self-attention.
        transformer_block (TransformerBlock): Cross-attention and feedforward layers.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        """
        Performs the forward pass for the Decoder block.

        Args:
            x (torch.Tensor): Target input tensor.
            value (torch.Tensor): Encoded source sequence.
            key (torch.Tensor): Encoded source sequence.
            src_mask (torch.Tensor): Mask for source sequence.
            tgt_mask (torch.Tensor): Mask for target sequence.

        Returns:
            torch.Tensor: Output tensor of shape (N, seq_length, embed_size).
        """
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    """
    Decoder module for the Transformer model, consisting of embedding, positional encoding,
    and multiple Decoder blocks.

    Attributes:
        word_embedding (nn.Embedding): Embedding layer for target tokens.
        position_embedding (nn.Embedding): Embedding layer for positional encoding.
        layers (nn.ModuleList): List of Decoder blocks.
        fc_out (nn.Linear): Fully connected layer for generating output logits.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self,
        tgt_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        Decodes the target sequence using the Transformer decoder.

        Args:
            x (torch.Tensor): Target input tensor of shape (N, tgt_seq_length).
            enc_out (torch.Tensor): Encoded source sequence.
            src_mask (torch.Tensor): Mask for source sequence.
            tgt_mask (torch.Tensor): Mask for target sequence.

        Returns:
            torch.Tensor: Output logits of shape (N, tgt_seq_length, tgt_vocab_size).
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    """
    Transformer model combining an Encoder and a Decoder.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        src_pad_idx (int): Padding index for source tokens.
        tgt_pad_idx (int): Padding index for target tokens.
        device (str): Device on which the model runs.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to ignore padding tokens.

        Args:
            src (torch.Tensor): Source input tensor of shape (N, src_seq_length).

        Returns:
            torch.Tensor: Source mask of shape (N, 1, 1, src_seq_length).
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        """
        Creates a causal mask for the target sequence to prevent attending to future tokens.

        Args:
            tgt (torch.Tensor): Target input tensor of shape (N, tgt_seq_length).

        Returns:
            torch.Tensor: Target mask of shape (N, 1, tgt_seq_length, tgt_seq_length).
        """
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            N, 1, tgt_len, tgt_len
        )
        return tgt_mask.to(self.device)

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (N, src_seq_length).
            tgt (torch.Tensor): Target input tensor of shape (N, tgt_seq_length).

        Returns:
            torch.Tensor: Output logits of shape (N, tgt_seq_length, tgt_vocab_size).
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out


class TransformerForClassification(nn.Module):
    """
    Transformer model for text classification, combining an Encoder and a classification head.

    Attributes:
        encoder (Encoder): Encoder module to process input sequences.
        fc_out (nn.Sequential): Fully connected layer for classification output.
        src_pad_idx (int): Padding index for source tokens.
        device (str): Device on which the model runs.
    """

    def __init__(
        self,
        src_vocab_size,
        src_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        max_length=1200,  # UPDATED to match data preprocessing
        num_classes=5,
        device="cuda",
    ):
        super(TransformerForClassification, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, num_classes),
        )
        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Creates a mask for the source sequence to ignore padding tokens.

        Args:
            src (torch.Tensor): Source input tensor of shape (N, src_seq_length).

        Returns:
            torch.Tensor: Source mask of shape (N, 1, 1, src_seq_length).
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src):
        """
        Forward pass for the Transformer model for classification.

        Args:
            src (torch.Tensor): Input tensor of shape (N, src_seq_length).

        Returns:
            torch.Tensor: Output logits of shape (N, num_classes).
        """
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        
        # Global average pooling over sequence length
        enc_src = enc_src.mean(dim=1)
        out = self.fc_out(enc_src)
        return out