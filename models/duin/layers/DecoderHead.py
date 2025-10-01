#!/usr/bin/env python3
"""
Created on 21:35, Jan. 21st, 2024

@author: Norbert Zheng
"""
import torch
import copy as cp
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
else:
    from models.duin.layers.LambdaLayer import LambdaLayer

__all__ = [
    # Regression Heads.
    "TimeRGSHead",
    "PatchRGSHead",
    "LLMRGSHead",
    # Classification Heads.
    "LabelCLSHead",
    "TokenCLSHead",
    "LLMCLSHead",
]

"""
regression heads
"""
# def TimeRGSHead class
class TimeRGSHead(nn.Module):
    """
    Time series regression head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `TimeRGSHead` object.

        Args:
            params: DotDict - The parameters of `TimeRGSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TimeRGSHead, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize regression head.
        self.rgs_head = nn.Sequential()
        # Add `ConvTNDBlock` layers.
        # TODO: Add `ConvNDBlock` layers to improve model ability.
        seq_len = self.params.emb_len
        for deconv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for deconvolution block.
            n_channels = self.params.n_filters[deconv_idx-1] if deconv_idx > 0 else self.params.d_model
            seq_len = seq_len * self.params.n_strides[deconv_idx]
            n_filters = self.params.n_filters[deconv_idx]; kernel_size = self.params.kernel_sizes[deconv_idx]
            n_strides = self.params.n_strides[deconv_idx]; dilation_rate = self.params.dilation_rates[deconv_idx]
            # Add the deconvolution layer.
            self.rgs_head.append(TimeRGSHead._make_deconv_block(
                # Modified `_make_deconv_block` parameters.
                n_channels=n_channels, seq_len=seq_len, n_filters=n_filters,
                kernel_size=kernel_size, n_strides=n_strides, dilation_rate=dilation_rate
            ))
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.rgs_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.n_filters[-1]),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.GELU(approximate="none"),
                nn.LayerNorm(
                    # Modified `LayerNorm` layer parameters.
                    normalized_shape=(self.params.d_hidden[hidden_idx],),
                    # Default `LayerNorm` layer parameters.
                    eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
                ),
            ))
        # Add the final regression `Linear` layer.
        # 'rgs': {'emb_len': 16, 'seg_len': 8, 'd_model': 16, 'n_filters': [128, 128, 128, 128, 16], 'kernel_sizes': [3, 3, 10, 9, 19], 'n_strides': [1, 1, 10, 1, 10], 'dilation_rates': [1, 1, 1, 1, 1], 'd_hidden': [], 'd_neural': 16},
        print("in_features:",self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.n_filters[-1])
        print("out_features:",self.params.d_neural)
        self.rgs_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.n_filters[-1],
                out_features=self.params.d_neural,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        ))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    # def _make_deconv_block func
    @staticmethod
    def _make_deconv_block(n_channels, seq_len, n_filters, kernel_size, n_strides, dilation_rate, **kwargs):
        """
        Make one deconvolution block, which contains [ConvTranspose1d,Conv1d,Conv1d].

        Args:
            n_channels: int - The number of input channels.
            seq_len: int - The length of embedding sequence.
            n_filters: int - The number of deconvolution filters.
            kernel_size: int - The dimensions of deconvolution kernel.
            n_strides: int - The number of deconvolution strides.
            dilation_rate: int - The dilation rate of deconvolution.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            deconv_block: nn.Module - The deconvolution block, which contains [ConvTranspose1d,Conv1d,Conv1d].
        """
        # Initialize the deconvolution block.
        deconv_block = nn.Sequential(**kwargs)
        # Add `ConvTranspose1d` layer.
        padding, output_padding = _cal_conv_padding(input_len=(seq_len * n_strides), output_len=seq_len,
            kernel_size=kernel_size, n_strides=n_strides, dilation_rate=dilation_rate)
        deconv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.ConvTranspose1d(
                # Modified `ConvTranspose1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                stride=n_strides, padding=padding, output_padding=output_padding, dilation=dilation_rate,
                # Default `ConvTranspose1d` layer parameters.
                groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        # Return the final `deconv_block`.
        return deconv_block

    """
    network funcs
    """
    # def forward func
    def forward(self, E):
        """
        Forward layers in `TimeRGSHead` to get the regression time series.

        Args:
            E: (batch_size, emb_len, d_model) - The encoder-transformed embeddings.

        Returns:
            X: (batch_size, seq_len, d_neural) - The regression time series.
        """
        return self.rgs_head(E)

# def PatchRGSHead class
class PatchRGSHead(nn.Module):
    """
    Time series regression head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `PatchRGSHead` object.

        Args:
            params: DotDict - The parameters of `PatchRGSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(PatchRGSHead, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize regression head.
        self.rgs_head = nn.Sequential()
        # Add the reshape layer to unwarp the sequence.
        d_model = (self.params.d_model // (self.params.seg_len // np.prod(self.params.n_strides)))
        self.rgs_head.append(LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], -1, d_model)))))
        # Add `ConvTNDBlock` layers.
        # TODO: Add `ConvNDBlock` layers to improve model ability.
        seq_len = (self.params.emb_len * (self.params.seg_len // np.prod(self.params.n_strides)))
        for deconv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for deconvolution block.
            n_channels = self.params.n_filters[deconv_idx-1] if deconv_idx > 0 else d_model
            seq_len = seq_len * self.params.n_strides[deconv_idx]
            n_filters = self.params.n_filters[deconv_idx]; kernel_size = self.params.kernel_sizes[deconv_idx]
            n_strides = self.params.n_strides[deconv_idx]; dilation_rate = self.params.dilation_rates[deconv_idx]
            # Add the deconvolution layer.
            self.rgs_head.append(PatchRGSHead._make_deconv_block(
                # Modified `_make_deconv_block` parameters.
                n_channels=n_channels, seq_len=seq_len, n_filters=n_filters,
                kernel_size=kernel_size, n_strides=n_strides, dilation_rate=dilation_rate
            ))
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.rgs_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.n_filters[-1]),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.GELU(approximate="none"),
                nn.LayerNorm(
                    # Modified `LayerNorm` layer parameters.
                    normalized_shape=(self.params.d_hidden[hidden_idx],),
                    # Default `LayerNorm` layer parameters.
                    eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
                ),
            ))
        # Add the final regression `Linear` layer.
        self.rgs_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.n_filters[-1],
                out_features=self.params.d_neural,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        ))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    # def _make_deconv_block func
    @staticmethod
    def _make_deconv_block(n_channels, seq_len, n_filters, kernel_size, n_strides, dilation_rate, **kwargs):
        """
        Make one deconvolution block, which contains [ConvTranspose1d,Conv1d,Conv1d].

        Args:
            n_channels: int - The number of input channels.
            seq_len: int - The length of embedding sequence.
            n_filters: int - The number of deconvolution filters.
            kernel_size: int - The dimensions of deconvolution kernel.
            n_strides: int - The number of deconvolution strides.
            dilation_rate: int - The dilation rate of deconvolution.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            deconv_block: nn.Module - The deconvolution block, which contains [ConvTranspose1d,Conv1d,Conv1d].
        """
        # Initialize the deconvolution block.
        deconv_block = nn.Sequential(**kwargs)
        # Add `ConvTranspose1d` layer.
        padding, output_padding = _cal_conv_padding(input_len=(seq_len * n_strides), output_len=seq_len,
            kernel_size=kernel_size, n_strides=n_strides, dilation_rate=dilation_rate)
        deconv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.ConvTranspose1d(
                # Modified `ConvTranspose1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                stride=n_strides, padding=padding, output_padding=output_padding, dilation=dilation_rate,
                # Default `ConvTranspose1d` layer parameters.
                groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        # Return the final `deconv_block`.
        return deconv_block

    """
    network funcs
    """
    # def forward func
    def forward(self, E):
        """
        Forward layers in `PatchRGSHead` to get the regression time series.

        Args:
            E: (batch_size, emb_len, d_model) - The encoder-transformed embeddings.

        Returns:
            X: (batch_size, seq_len, d_neural) - The regression time series.
        """
        return self.rgs_head(E)

# def LLMRGSHead class
class LLMRGSHead(nn.Module):
    """
    LLM regression head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `LLMRGSHead` object.

        Args:
            params: DotDict - The parameters of `LLMRGSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LLMRGSHead, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize regression head.
        self.rgs_head = nn.Sequential()
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.rgs_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_model),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.ReLU(inplace=False),
            ))
        # Add `Dropout` after hidden `Linear` layers.
        if self.params.dropout > 0.:
            self.rgs_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final regression `Linear` layer.
        self.rgs_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.d_model),
                out_features=self.params.d_llm,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        ))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, E):
        """
        Forward layers in `LLMRGSHead` to get the regression LLM embeddings.

        Args:
            E: (batch_size, emb_len, d_model) - The encoder-transformed embeddings.

        Returns:
            L: (batch_size, emb_len, d_llm) - The regression LLM embeddings.
        """
        return self.rgs_head(E)

"""
classification heads
"""
# def LabelCLSHead class
class LabelCLSHead(nn.Module):
    """
    Label classification head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `LabelCLSHead` object.

        Args:
            params: DotDict - The parameters of `LabelCLSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LabelCLSHead, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize classification head.
        self.cls_head = nn.Sequential()
        # Add `Flatten` layer to flatten embedding sequence.
        # TODO: Support multiple time-integration methods, e.g., average.
        self.cls_head.append(nn.Flatten(start_dim=1, end_dim=-1))
        
        # BatchNorm after Flatten
        self.cls_head.append(nn.BatchNorm1d(self.params.d_feature))  # <------ 添加这一行
        
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.cls_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_feature),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.BatchNorm1d(self.params.d_hidden[hidden_idx]),     # <--- 每个Linear后也可加

                nn.ReLU(inplace=False),
            ))
        # Add `Dropout` after hidden `Linear` layers.
        if self.params.dropout > 0.:
            self.cls_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final classification `Linear` layer.
        # Note: Add `nn.Sigmoid()` to avoid gradient explotion.
        self.cls_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.d_feature),
                out_features=self.params.n_labels,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
            # nn.Sigmoid(),
        ))
        self.sigmoid = nn.Sigmoid()

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    def get_flatten_feature(self, E):
        """
        Get the flattened embedding of the input feature.

        Args:
            E: (batch_size, emb_len, d_model)

        Returns:
            flat_E: (batch_size, feature_dim)
        """
        features_vec = E.mean(dim=1)
        return features_vec
        # return nn.Flatten(start_dim=1, end_dim=-1)(E)


    """
    network funcs
    """
    # def forward func
    def forward(self, E):
        """
        Forward layers in `LabelCLSHead` to get the predicted labels.

        Args:
            E: (batch_size, emb_len, d_model) - The encoder-transformed embeddings.

        Returns:
            y_pred: (batch_size, n_labels) - The predicted labels.
        """
        # return self.cls_head(E)
        logits = self.cls_head(E)
        probs = self.sigmoid(logits)
        return probs, logits

# def TokenCLSHead class
class TokenCLSHead(nn.Module):
    """
    Token classification head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `TokenCLSHead` object.

        Args:
            params: DotDict - The parameters of `TokenCLSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TokenCLSHead, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize classification head.
        self.cls_head = nn.Sequential()
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.cls_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_model),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.ReLU(inplace=False),
            ))
        # Add `Dropout` after hidden `Linear` layers.
        if self.params.dropout > 0.:
            self.cls_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final classification `Linear` layer.
        # Note: Add `nn.Sigmoid()` to avoid gradient explotion.
        self.cls_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.d_model),
                out_features=self.params.n_tokens,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        ))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `TokenCLSHead` to get the predicted tokens.

        Args:
            emb: (batch_size, emb_len, d_model) - The encoder-transformed embedding sequence.

        Returns:
            c_pred: (batch_size, emb_len, n_tokens) - The predicted tokens.
        """
        return self.cls_head(emb)

# def LLMCLSHead class
class LLMCLSHead(nn.Module):
    """
    LLM classification head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `LLMCLSHead` object.

        Args:
            params: DotDict - The parameters of `LLMCLSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LLMCLSHead, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize classification head.
        self.cls_head = nn.Sequential()
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.cls_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_model),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.ReLU(inplace=False),
            ))
        # Add `Dropout` after hidden `Linear` layers.
        if self.params.dropout > 0.:
            self.cls_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final classification `Linear` layer.
        # Note: Add `nn.Sigmoid()` to avoid gradient explotion.
        self.cls_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.d_model),
                out_features=self.params.n_words,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        ))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `LLMCLSHead` to get the predicted words.

        Args:
            emb: (batch_size, emb_len, d_model) - The encoder-transformed embedding sequence.

        Returns:
            c_pred: (batch_size, emb_len, n_words) - The predicted words.
        """
        return self.cls_head(emb)

"""
tool funcs
"""
# def _cal_conv_padding func
def _cal_conv_padding(input_len, output_len, kernel_size, n_strides, dilation_rate):
    """
    Calculate the padding of convolution.

    Args:
        input_len: int - The length of input embedding sequence.
        output_len: int - The length of output embedding sequence.
        kernel_size: int - The size of convolution kernel.
        n_strides: int - The number of convolution strides.
        dilation_rate: int - The dilation rate of convolution.

    Returns:
        padding: int - The padding will be added to both sides of the input.
        output_padding: int - The additional size added to one side of the output shape.
    """
    # Calculate the padding of convolution.
    padding_diff = (output_len - 1) * n_strides + (dilation_rate * (kernel_size - 1) + 1) - input_len
    padding = int(np.ceil(padding_diff / 2))
    output_padding = 2 * padding - padding_diff
    # Return the final `padding`.
    return padding, output_padding

if __name__ == "__main__":
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128; d_neural = 256; n_labels = 10; n_tokens = 2048; d_llm = 1024

    ## Forward regression heads.
    # Initialize params for TimeRGSHead.
    rgsh_time_params_inst = DotDict({
        # The dimensions of model embedding.
        "d_model": d_model,
        # The length of embedding sequence.
        "emb_len": emb_len,
        # The number of filters of each deconvolution block.
        "n_filters": [256, 128],
        # The size of kernel of each deconvolution block.
        "kernel_sizes": [11, 9],
        # The number of strides of each deconvolution block.
        "n_strides": [3, 4],
        # The dilation rate of each deconvolution block.
        "dilation_rates": [1, 1],
        # The dimensions of the hidden layers after deconvolution.
        "d_hidden": [128,],
        # The dimensions of common hidden neural space.
        "d_neural": d_neural,
    })
    # Initialize input `E`.
    # E - (batch_size, emb_len, d_model)
    E = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate TimeRGSHead.
    rgsh_time_inst = TimeRGSHead(params=rgsh_time_params_inst)
    # Forward layers in `rgsh_time_inst`.
    # X - (batch_size, seq_len, d_neural)
    X = rgsh_time_inst(E)
    # Initialize params for PatchRGSHead.
    rgsh_patch_params_inst = DotDict({
        # The length of embedding sequence.
        "emb_len": emb_len,
        # The length of element segment.
        "seg_len": 20,
        # The dimensions of model embedding.
        "d_model": d_model,
        # The number of filters of each deconvolution block.
        "n_filters": [256, 128],
        # The size of kernel of each deconvolution block.
        "kernel_sizes": [3, 9],
        # The number of strides of each deconvolution block.
        "n_strides": [2, 5],
        # The dilation rate of each deconvolution block.
        "dilation_rates": [1, 1],
        # The dimensions of the hidden layers after deconvolution.
        "d_hidden": [128,],
        # The dimensions of common hidden neural space.
        "d_neural": d_neural,
    })
    # Initialize input `E`.
    # E - (batch_size, emb_len, d_model)
    E = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate PatchRGSHead.
    rgsh_patch_inst = PatchRGSHead(params=rgsh_patch_params_inst)
    # Forward layers in `rgsh_patch_inst`.
    # X - (batch_size, seq_len, d_neural)
    X = rgsh_patch_inst(E)
    # Initialize params for LLMRGSHead.
    clsh_label_params_inst = DotDict({
        # The dimensions of model embedding.
        "d_model": d_model,
        # The dimensions of the hidden layer in classification block.
        "d_hidden": [128,],
        # The dropout probability of the hidden layer in classification block.
        "dropout": 0.,
        # The dimension of LLM embedding.
        "d_llm": 1024,
    })
    # Initialize input `E`.
    # E - (batch_size, emb_len, d_model)
    E = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate LLMRGSHead.
    rgsh_llm_inst = LLMRGSHead(params=clsh_label_params_inst)
    # Forward layers in `rgsh_llm_inst`.
    # L - (batch_size, emb_len, d_llm)
    L = rgsh_llm_inst(E)
    ## Forward classification heads.
    # Initialize params for LabelCLSHead.
    clsh_label_params_inst = DotDict({
        # The dimensions of feature embedding.
        "d_feature": (emb_len * d_model),
        # The dimensions of the hidden layer in classification block.
        "d_hidden": [128,],
        # The dropout probability of the hidden layer in classification block.
        "dropout": 0.,
        # The number of output classes.
        "n_labels": n_labels,
    })
    # Initialize input `E`.
    # E - (batch_size, emb_len, d_model)
    E = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate LabelCLSHead.
    clsh_label_inst = LabelCLSHead(params=clsh_label_params_inst)
    # Forward layers in `clsh_label_inst`.
    # y_pred - (batch_size, n_labels)
    y_pred = clsh_label_inst(E)
    # Initialize params for TokenCLSHead.
    clsh_token_params_inst = DotDict({
        # The dimensions of model embedding.
        "d_model": d_model,
        # The dimensions of the hidden layer in classification block.
        "d_hidden": [128,],
        # The dropout probability of the hidden layer in classification block.
        "dropout": 0.,
        # The number of output tokens.
        "n_tokens": n_tokens,
    })
    # Initialize input `E`.
    # E - (batch_size, emb_len, d_model)
    E = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate TokenCLSHead.
    clsh_token_inst = TokenCLSHead(params=clsh_token_params_inst)
    # Forward layers in `clsh_token_inst`.
    # c_pred - (batch_size, emb_len, n_tokens)
    c_pred = clsh_token_inst(E)

