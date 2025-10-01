import torch
import torch.nn as nn

__all__ = [
    "MultiScale",
]

class MultiScale(nn.Module):
    """
    MultiScale layer for time series decomposition at multiple scales.
    """
    def __init__(self, input_shape, k=2, c=2, dropout=0.2, layernorm=True):
        """
        Initialize the MultiScale object.

        Args:
            input_shape: Tuple indicating the input tensor shape (batch_size, seq_len, n_channels).
            k: The number of scales for decomposition.
            c: Scaling factor for the number of channels.
            layernorm: If True, apply LayerNorm after the pooling operation.
        """
        super(MultiScale, self).__init__()
        
        # Store input sequence length and feature size
        self.seq_len = input_shape[0]
        self.n_channels = input_shape[1]
        self.k = k
        self.k_list = [c ** i for i in range(k, 0, -1)]  # Scaling factor list
        # 设置隐层尺度
        self.hidden_size = [256 if k>=750 else 128 for k in self.k_list]
        assert len(self.k_list)
        self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])  # Average Pooling layers
        self.linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.seq_len // self.k_list[i], self.hidden_size[i]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_size[i], self.seq_len * c // self.k_list[i]),
                )
                for i in range(len(self.k_list))
            ]
        )
        
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])  # Batch normalization after flattening

    def forward(self, emb):
        """
        Apply multi-scale decomposition to the input tensor.

        Args:
            emb: (batch_size, seq_len, d_channel) - The input tensor.

        Returns:
            multi_scales: (batch_size, seq_len, d_channel) - The multi-scale decomposed tensor.
        """
        # Apply layer normalization if needed
        if self.layernorm:
            if emb.shape[0] == 1:
                emb = emb
            else:
                emb = self.norm(torch.flatten(emb, 1, -1)).reshape(emb.shape)

        # 转换为[batch_size, d_channel, seq_len]的形式
        emb = emb.permute(0, 2, 1)
        
        # Decompose input into multiple scales
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](emb))
        sample_x.append(emb)  # Add original input as the finest scale
        
        n = len(sample_x)
        
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)

         # 归一化：除以层数（n），可以帮助平衡不同尺度的输出
        for i in range(n):
            sample_x[i] = sample_x[i] / n

        # 转换回[batch_size, seq_len, d_channel]的形式
        sample_x = [x.permute(0, 2, 1) for x in sample_x]
        
        # Return the final scale
        return sample_x[n - 1]


if __name__ == "__main__":
    # Example parameters for testing
    batch_size = 32
    emb_len = 20  # Length of the sequence
    d_model = 128  # Dimension of the model
    num_scales = 3  # Number of scales to decompose into
    
    # Initialize the MultiScale module
    multi_scale_layer = MultiScale(input_shape=(emb_len, d_model), k=num_scales)

    # Initialize input `emb` (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)

    # Forward the input through the multi-scale decomposition layer
    multi_scales = multi_scale_layer(emb)

    # Print the shape of the output (batch_size, num_scales, seq_len, d_neural)
    print(multi_scales.shape)
