#!/usr/bin/env python3
"""
Created on 21:51, Jan. 19th, 2024

@author: Norbert Zheng
"""
import re, torch
import copy as cp
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
from utils import DotDict

__all__ = [
    "duin_vqvae",
    "duin_mae",
    "duin_cls",
    "duin_llm",
]

# def duin_vqvae class
class duin_vqvae(nn.Module):
    """
    DuIN model for neural signal prediction.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_vqvae` object.

        Args:
            params: DotDict - Model parameters initialized by duin_vqvae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_vqvae, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        self.isMDM = self.params.isMDM

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
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize Multi-Scale block.
        # multi_scale - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.multi_scale = MultiScale((self.params.seq_len, self.params.subj.d_output))
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None) and (self.params.decoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize decoder block.
        # decoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.decoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.decoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize regression block.
        # rgs_block - (batch_size, token_len, d_model) -> (batch_size, seq_len, d_neural)
        self.rgs_block = TimeRGSHead(params=self.params.rgs)
        # Initialize de-subject block.
        # desubj_block - (batch_size, seq_len, d_neural) -> (batch_size, seq_len, n_channels)
        self.desubj_block = SubjectBlock(params=self.params.desubj)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*multi_scale": "multi_scale",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
            "([^.]*\.)*vq_block": "vq_block",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_vqvae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_vqvae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,subj_id,channel_mask].

        Returns:
            X_reconstr: (batch_size, seq_len, n_channels) - The reconstructed signals.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects); channel_mask - (batch_size, n_channels)
        X = inputs[0]; subj_id = inputs[1]; channel_mask = inputs[2]
        # 处理X，有可能在数据中存在Nan，用0填充
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # 将X_h变成multi-scale的输入，然后分别进行tokenization，然后进行encoder，然后进行vq，然后进行contrastive，然后进行classification
        # X_h - batch_size, seq_len, d_neural
        if self.isMDM:
            X_h = self.multi_scale(X_h)
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E, b2_emb, b3_emb, b4_emb, b5_emb = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward decoder & regression block to get the corresponding reconstructon.
        # TODO: Support subject-layer in `rgs_block`, we do not reconstruct the intermediate `X_h_reconstr`.
        # T_reconstr - (batch_size, token_len, d_model)
        E_de, b2_emb_de, b3_emb_de, b4_emb_de, b5_emb_de = self.decoder(E_vq)
        T_reconstr = torch.reshape(E_de, shape=token_shape)
        # X_reconstr - (batch_size, seq_len, n_channels)
        X_reconstr = self.desubj_block((self.rgs_block(T_reconstr), subj_id))
        # Calculate the regression loss.
        # loss_rgs - torch.float32
        loss_rgs = self._loss_rgs(X_reconstr, X, weight=channel_mask.to(dtype=X.dtype))
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.rgs_loss_scale * loss_rgs +\
            self.params.vq_loss_scale * loss_vq
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "vq": loss_vq,
            "rgs": loss_rgs,
        })
        # Return the final `X_reconstr` & `loss`.
        return X_reconstr, loss

    # def quantize func
    def quantize(self, inputs):
        """
        Forward `duin_vqvae` to get the quantized embeddings.

        Args:
            inputs: tuple - The input data, including [X,subj_id].

        Returns:
            E_vq: (batch_size, emb_len, d_model) - The quantized embeddings.
            loss_vq: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, emb_len, codex_size) - The one-hot probabilities of the embeddings.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects)
        X = inputs[0]; subj_id = inputs[1]
        # 处理X，有可能在数据中存在Nan，用0填充
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
        E_vq, loss_vq, codex_probs = self.vq_block(E)
        # Return the final `E_vq` & `loss_vq` & `codex_probs`.
        return E_vq, loss_vq, codex_probs

    """
    loss funcs
    """
    # def _loss_rgs func
    def _loss_rgs(self, value, target, weight=None):
        """
        Calculate regresion error between (list of) tensors value and target. Include a factor
        0.5 to squared error by convention. Set `keepdims` to false, then get sum over last dimension to keep
        losses of different batches separate.

        Args:
            value: (batch_size, seq_len, n_channels) - Value of the object.
            target: (batch_size, seq_len, n_channels) - Traget of the object.
            weight: (batch_size, n_channels) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, seq_len, n_channels)
        loss = torch.square(target - value)
        # Average over all locations.
        # loss - (batch_size, n_channels)
        loss = torch.mean(torch.flatten(torch.permute(loss, dims=[0,-1,*range(1, len(loss.shape)-1)]), start_dim=2, end_dim=-1), dim=-1)
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

# def duin_mae class
class duin_mae(nn.Module):
    """
    DuIN model for neural token prediction.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_mae` object.

        Args:
            params: DotDict - Model parameters initialized by duin_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_mae, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        self.isMDM = self.params.isMDM

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
        # Initialize mask embedding.
        # mask_emb - (d_model,)
        mask_emb = torch.ones((self.params.encoder.d_model,), dtype=torch.float32)
        self.mask_emb = nn.Parameter(mask_emb, requires_grad=True)
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize Multi-Scale block.
        # multi_scale - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.multi_scale = MultiScale((self.params.seq_len, self.params.subj.d_output))
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_tokens)
        self.cls_block = TokenCLSHead(params=self.params.cls)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `mask_emb`.
        nn.init.trunc_normal_(self.mask_emb, mean=0., std=0.02)

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*multi_scale": "multi_scale",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_mae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_mae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,c_true,subj_id].

        Returns:
            c_pred: (batch_size, emb_len, codex_size) - The predicted codex.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); c_true - (batch_size, emb_len, codex_size); subj_id - (batch_size, n_subjects)
        X = inputs[0]; c_true = inputs[1]; subj_id = inputs[2]
        # 处理X，有可能在数据中存在Nan，用0填充
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # 将X_h变成multi-scale的输入，然后分别进行tokenization，然后进行encoder，然后进行vq，然后进行contrastive，然后进行classification
        # X_h - batch_size, seq_len, d_neural
        if self.isMDM:
            X_h = self.multi_scale(X_h)
            # print("with MDM")
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E_init - (batch_size, emb_len, d_model)
        E_init = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Generate mask according to the init embedding `E`.
        # mask - (batch_size, emb_len)
        mask = self.gen_mask(E_init, mask_ratio=self.params.mask_ratio)
        # Get the masked embedding `E_masked` according to `mask`.
        # mask_emb - (batch_size, emb_len, d_model)
        mask_emb = self.mask_emb[None,None,...].expand(*mask.shape, -1)
        # E_masked - (2[list], batch_size, emb_len, d_model)
        E_masked = [
            (E_init * (1. - mask[...,None].to(dtype=E_init.dtype)) + mask_emb * mask[...,None].to(dtype=E_init.dtype)),
            (E_init * mask[...,None].to(dtype=E_init.dtype) + mask_emb * (1. - mask[...,None].to(dtype=E_init.dtype))),
        ]
        # Forward encoder block to get time-aligned token sequence.
        # E - (2[list], batch_size, emb_len, d_model)
        E = [self.encoder(E_i) for E_i in E_masked]
        # Forward classification block to get the corresponding prediction.
        # c_pred - (batch_size, emb_len, codex_size)
        c_pred = [self.cls_block(E_i) for E_i in E]
        c_pred = (
            (c_pred[0] * mask[...,None].to(dtype=c_pred[0].dtype)) +\
            (c_pred[1] * (1. - mask[...,None].to(dtype=c_pred[1].dtype)))
        )
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(
            torch.reshape(c_pred, shape=(-1, c_pred.shape[-1])),
            torch.reshape(c_true, shape=(-1, c_true.shape[-1])),
        )
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `c_pred` & `loss`.
        return c_pred, loss

    # def gen_mask func
    def gen_mask(self, E, mask_ratio=0.5):
        """
        Generate mask for embedding sequence.

        Args:
            E: (batch_size, emb_len, d_model) - The embedding sequence.
            mask_ratio: float - The mask ratio of each embedding item.

        Returns:
            mask: (batch_size, emb_len) - The generated mask.
        """
        # Initialize `batch_size` & `emb_len` & `d_model` from `E`.
        batch_size, emb_len, d_model = E.shape
        # Initialize the length of keep embedding items.
        keep_len = int(emb_len * (1. - mask_ratio))
        # Initialize the noise for further argsort.
        # noise - (batch_size, emb_len)
        noise = torch.rand((batch_size, emb_len), dtype=E.dtype).to(device=E.device)
        # Get the corresponding `shuffle_idxs` & `restore_idxs`.
        # Note: `torch.argsort` is reversible, we have `shuffle_idxs = torch.argsort(restore_idxs)`.
        shuffle_idxs = torch.argsort(noise, dim=-1); restore_idxs = torch.argsort(shuffle_idxs, dim=-1)
        # Generate the bool mask: `False` is keep, `True` is remove.
        # mask - (batch_size, emb_len)
        mask = torch.ones((batch_size, emb_len), dtype=torch.bool).to(device=E.device); mask[:,:keep_len] = False
        # Un-shuffle to get the bool mask.
        mask = torch.gather(mask, dim=-1, index=restore_idxs)
        # Return the final `mask`.
        return mask

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

# def duin_cls class
class duin_cls(nn.Module):
    """
    DuIN model for classification task.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_cls` object.

        Args:
            params: DotDict - Model parameters initialized by duin_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_cls, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        self.isMDM = self.params.isMDM
        self.isDistill = self.params.isDistill

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
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize Multi-Scale block.
        # multi_scale - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.multi_scale = MultiScale((self.params.seq_len, self.params.subj.d_output))
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), 
            LambdaLayer(func=(lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4]))),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize contrastive block.
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, n_labels)
        self.cls_block = LabelCLSHead(params=self.params.cls)

        # self distillation
        # 辅助分类头（和cls_block类似但独立的）
        self.aux_cls_block = LabelCLSHead(params=self.params.cls)
        # self.aux_cls_block_2 = LabelCLSHead(params=self.params.cls)
        # self.aux_cls_block_3 = LabelCLSHead(params=self.params.cls)
        # self.aux_cls_block_4 = LabelCLSHead(params=self.params.cls)
        # self.aux_cls_block_5 = LabelCLSHead(params=self.params.cls)

        # 自蒸馏头的loss加权可学习参数，不同的block有不同的权重
        self.distill_loss_scale = nn.Parameter(torch.ones(4), requires_grad=True)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*multi_scale": "multi_scale",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
            "([^.]*\.)*cls_block.cls_head.1": "cls_block.cls_head.1",
            "([^.]*\.)*cls_block.cls_head.2": "cls_block.cls_head.2",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,y_true,subj_id].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels); subj_id - (batch_size, n_subjects)
        X = inputs[0]; y_true = inputs[1]; subj_id = inputs[2]
        # 处理X，有可能在数据中存在Nan，用0填充
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # 将X_h变成multi-scale的输入，然后分别进行tokenization，然后进行encoder，然后进行vq，然后进行contrastive，然后进行classification
        # X_h - batch_size, seq_len, d_neural
        if self.isMDM:
            X_h = self.multi_scale(X_h)
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))   # [32, 30, 160]
        # Forward encoder block to get time-aligned token sequence.
        # E = self.encoder(E)
        E, b2_emb, b3_emb, b4_emb, b5_emb = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Calculate the contrastive loss.
        # loss_contra - torch.float32
        loss_contra, _ = self.contra_block(((E, E), (y_true, y_true)))
        # Forward classification block to get the corresponding prediction.
        # y_pred - (batch_size, n_labels)
        flat_feat = self.cls_block.get_flatten_feature(E)
        y_pred, logits = self.cls_block(E)
        # 辅助分类头 logits（自蒸馏用）
        y_pred_b2, logits_b2 = self.aux_cls_block(b2_emb)
        y_pred_b3, logits_b3 = self.aux_cls_block(b3_emb)
        y_pred_b4, logits_b4 = self.aux_cls_block(b4_emb)
        y_pred_b5, logits_b5 = self.aux_cls_block(b5_emb)
        # y_pred_b2 = self.aux_cls_block_2(b2_emb)
        # y_pred_b3 = self.aux_cls_block_3(b3_emb)
        # y_pred_b4 = self.aux_cls_block_4(b4_emb)
        # y_pred_b5 = self.aux_cls_block_5(b5_emb)
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(y_pred, y_true)
        # 蒸馏损失（KL散度）
        # 目前的对齐使用的是probs的对齐，而自蒸馏更好的使用logits进行对齐的，换试试
        T = 2.0
        # 使用logits自蒸馏
        # distill_loss_b2 = F.kl_div(
        #     F.log_softmax(logits_b2 / T, dim=-1),
        #     F.softmax(logits.detach() / T, dim=-1),
        #     reduction='batchmean'
        # ) * (T * T)
        # distill_loss_b3 = F.kl_div(
        #     F.log_softmax(logits_b3 / T, dim=-1),
        #     F.softmax(logits.detach() / T, dim=-1),
        #     reduction='batchmean'
        # ) * (T * T)
        # distill_loss_b4 = F.kl_div(
        #     F.log_softmax(logits_b4 / T, dim=-1),
        #     F.softmax(logits.detach() / T, dim=-1),
        #     reduction='batchmean'
        # ) * (T * T)
        # distill_loss_b5 = F.kl_div(
        #     F.log_softmax(logits_b5 / T, dim=-1),
        #     F.softmax(logits.detach() / T, dim=-1),
        #     reduction='batchmean'
        # ) * (T * T)
        # 使用probs自蒸馏
        distill_loss_b2 = F.kl_div(
            F.log_softmax(y_pred_b2 / T, dim=-1),
            F.softmax(y_pred.detach() / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        distill_loss_b3 = F.kl_div(
            F.log_softmax(y_pred_b3 / T, dim=-1),
            F.softmax(y_pred.detach() / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        distill_loss_b4 = F.kl_div(
            F.log_softmax(y_pred_b4 / T, dim=-1),
            F.softmax(y_pred.detach() / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        distill_loss_b5 = F.kl_div(
            F.log_softmax(y_pred_b5 / T, dim=-1),
            F.softmax(y_pred.detach() / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        # Calculate the total loss.
        # loss_total - torch.float32
        # if self.isDistill:
        #     loss_total = (
        #         self.params.cls_loss_scale * loss_cls +\
        #         self.params.contra_loss_scale * loss_contra +\
        #         0.2 * (distill_loss_b2 + distill_loss_b3 + distill_loss_b4 + distill_loss_b5)
        #     )
        if self.isDistill:
            loss_total = (
                self.params.cls_loss_scale * loss_cls +\
                self.params.contra_loss_scale * loss_contra +\
                0.1 * distill_loss_b2 +\
                0.2 * distill_loss_b3 +\
                0.3 * distill_loss_b4 +\
                0.4 * distill_loss_b5
            )
        else:
            loss_total = (
                self.params.cls_loss_scale * loss_cls +\
                self.params.contra_loss_scale * loss_contra
            )
        # 可学习参数
        # loss_total = (
        #     self.params.cls_loss_scale * loss_cls +\
        #     self.params.contra_loss_scale * loss_contra +\
        #     self.distill_loss_scale[0] * distill_loss_b2 +\
        #     self.distill_loss_scale[1] * distill_loss_b3 +\
        #     self.distill_loss_scale[2] * distill_loss_b4 +\
        #     self.distill_loss_scale[3] * distill_loss_b5
        # )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
            "contra": loss_contra,
            "distill_b2": distill_loss_b2,
            "distill_b3": distill_loss_b3,
            "distill_b4": distill_loss_b4,
            "distill_b5": distill_loss_b5,
        })
        # Return the final `y_pred` & `loss`.
        return y_pred, loss, flat_feat

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

# def duin_llm class
class duin_llm(nn.Module):
    """
    DuIN model for open-set language decoding task.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_llm` object.

        Args:
            params: DotDict - Model parameters initialized by duin_llm_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_llm, self).__init__(**kwargs)

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
        # Initialize subject block.
        # subj_block - (batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.subj_block = SubjectBlock(params=self.params.subj)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize classification blocks.
        # cls_blocks - (batch_size, token_len, d_model) -> (batch_size, token_len, n_phonemes)
        cls_initials_params = cp.deepcopy(self.params.cls); cls_initials_params.n_tokens = cls_initials_params.n_initials
        cls_finals_params = cp.deepcopy(self.params.cls); cls_finals_params.n_tokens = cls_finals_params.n_finals
        self.cls_blocks = nn.ModuleList(modules=[
            TokenCLSHead(params=cls_initials_params),
            TokenCLSHead(params=cls_finals_params),
        ])

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*subj_block": "subj_block",
            "([^.]*\.)*multi_scale": "multi_scale",
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.duin.duin_llm."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_llm` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,p_true,subj_id,token_mask].

        Returns:
            p_pred: (2[list], batch_size, token_len, n_phonemes) - The predicted phonemes.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); p_true - (2[list], batch_size, token_len, n_phonemes)
        # subj_id - (batch_size, n_subjects); token_mask - (batch_size, token_len)
        X = inputs[0]; p_true = inputs[1]; subj_id = inputs[2]; token_mask = inputs[3]
        # 处理X，有可能在数据中存在Nan，用0填充
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
        # Forward subject block to get the subject-transformed signals (which share the same space).
        # `X_h` is the projection of the original data in the common hidden space (shared by all subjects).
        # This process will not reduce the resolution, e.g., for 1D-signal, `data_len` holds for `X_h`.
        # X_h - (batch_size, seq_len, d_neural)
        X_h = self.subj_block((X, subj_id))
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, d_neural), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X_h); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward classification block to get the prediction phonemes.
        # p_pred - (2[list], batch_size, token_len, n_phonemes)
        p_pred = [self.cls_blocks[phoneme_idx](E) for phoneme_idx in range(len(self.cls_blocks))]
        # Calculate the classification loss.
        # loss_cls - torch.float32
        weight = token_mask.to(dtype=p_pred[0].dtype)
        #loss_cls = torch.mean(torch.stack([self._loss_cls(p_pred_i, p_true_i, weight=weight)\
        #    for p_pred_i, p_true_i in zip(p_pred, p_true)], dim=0))
        loss_cls = [self._loss_cls(p_pred_i, p_true_i, weight=weight)\
            for p_pred_i, p_true_i in zip(p_pred, p_true)][1]
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `p_pred` & `loss`.
        return p_pred, loss

    """
    loss funcs
    """
    # def _loss_rgs func
    def _loss_rgs(self, value, target, weight=None):
        """
        Calculate regresion error between (list of) tensors value and target. Include a factor
        0.5 to squared error by convention. Set `keepdims` to false, then get sum over last dimension to keep
        losses of different batches separate.

        Args:
            value: (batch_size, emb_len, d_llm) - Value of the object.
            target: (batch_size, emb_len, d_llm) - Traget of the object.
            weight: (batch_size, emb_len, d_llm) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, emb_len, d_llm)
        loss = torch.square(target - value)
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

    # def _loss_cls func
    def _loss_cls(self, value, target, weight=None):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, emb_len, n_words) - Value of the object.
            target: (batch_size, emb_len, n_words) - Target of the object.
            weight: (batch_size, d_llm) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Initialize `batch_size` & `emb_len` & `n_words` from `value`.
        batch_size, emb_len, n_words = value.shape
        # Calculate the cross-entropy loss.
        # loss - (batch_size, emb_len)
        loss = torch.reshape(F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=torch.reshape(value, shape=(-1, n_words)), target=torch.reshape(target, shape=(-1, n_words)),
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="none", label_smoothing=0.
        ), shape=(batch_size, emb_len))
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            ch_weights: (n_subjects, n_channels) - The contribution weights of each input channel.
        """
        return self.subj_block.get_weight_i()

if __name__ == "__main__":
    import numpy as np
    # local dep
    import utils.model.torch
    from params.duin_params import duin_vqvae_params, duin_mae_params, duin_cls_params, duin_llm_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 3000; n_channels = 16; n_labels = 61; n_subjects = 10; d_llm = 1024

    # Initialize training process.
    utils.model.torch.set_seeds(42)

    ## Forward duin_vqvae.
    # Instantiate params.
    duin_vqvae_params_inst = duin_vqvae_params(dataset=dataset)
    duin_vqvae_params_inst.model.n_subjects = n_subjects
    duin_vqvae_params_inst.model.desubj.n_subjects = duin_vqvae_params_inst.model.subj.n_subjects = n_subjects
    duin_vqvae_params_inst.model.n_channels = n_channels
    duin_vqvae_params_inst.model.desubj.d_output = duin_vqvae_params_inst.model.subj.d_input = n_channels
    assert seq_len % duin_vqvae_params_inst.model.seg_len == 0; duin_vqvae_params_inst.model.seq_len = seq_len
    token_len = duin_vqvae_params_inst.model.seq_len // duin_vqvae_params_inst.model.tokenizer.seg_len
    duin_vqvae_params_inst.model.tokenizer.token_len = token_len
    duin_vqvae_params_inst.model.decoder.emb_len = duin_vqvae_params_inst.model.encoder.emb_len = token_len
    # Initialize input `X` & `subj_id` & `channel_mask`.
    # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects); channel_mask - (batch_size, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    channel_mask = torch.ones((batch_size, n_channels), dtype=torch.bool)
    # Instantiate duin_vqvae.
    duin_vqvae_inst = duin_vqvae(duin_vqvae_params_inst.model); print(duin_vqvae_inst)
    # Forward layers in `duin_vqvae_inst`.
    # X_reconstr - (batch_size, seq_len, n_channels); loss - torch.float32
    X_reconstr, loss = duin_vqvae_inst((X, subj_id, channel_mask))
    # Forward layers before vector-quantizer in `duin_vqvae_inst`.
    # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
    E_vq, loss_vq, codex_probs = duin_vqvae_inst.quantize((X, subj_id))
    ## Forward duin_mae.
    # Instantiate params.
    duin_mae_params_inst = duin_mae_params(dataset=dataset)
    duin_mae_params_inst.model.subj.n_subjects = duin_mae_params_inst.model.n_subjects = n_subjects
    duin_mae_params_inst.model.subj.d_input = duin_mae_params_inst.model.n_channels = n_channels
    assert seq_len % duin_mae_params_inst.model.seg_len == 0; duin_mae_params_inst.model.seq_len = seq_len
    token_len = duin_mae_params_inst.model.seq_len // duin_mae_params_inst.model.tokenizer.seg_len
    duin_mae_params_inst.model.encoder.emb_len = duin_mae_params_inst.model.tokenizer.token_len = token_len
    # Initialize input `X` & `c_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); c_true - (batch_size, emb_len, codex_size); subj_id - (batch_size, n_subjects)
    emb_len = token_len; codex_size = duin_mae_params_inst.model.vq.codex_size
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    c_true = torch.tensor(np.eye(codex_size)[np.random.randint(0, codex_size, size=(batch_size, emb_len))], dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_mae.
    duin_mae_inst = duin_mae(duin_mae_params_inst.model); print(duin_mae_inst)
    # Forward layers in `duin_mae_inst`.
    # c_pred - (batch_size, emb_len, codex_size); loss - torch.float32
    c_pred, loss = duin_mae_inst((X, c_true, subj_id))
    ## Forward duin_cls.
    # Instantiate params.
    duin_cls_params_inst = duin_cls_params(dataset=dataset)
    duin_cls_params_inst.model.subj.n_subjects = duin_cls_params_inst.model.n_subjects = n_subjects
    duin_cls_params_inst.model.subj.d_input = duin_cls_params_inst.model.n_channels = n_channels
    assert seq_len % duin_cls_params_inst.model.seg_len == 0; duin_cls_params_inst.model.seq_len = seq_len
    token_len = duin_cls_params_inst.model.seq_len // duin_cls_params_inst.model.tokenizer.seg_len
    duin_cls_params_inst.model.tokenizer.token_len = token_len
    duin_cls_params_inst.model.encoder.emb_len = token_len
    duin_cls_params_inst.model.cls.d_feature = (
        duin_cls_params_inst.model.encoder.d_model * duin_cls_params_inst.model.encoder.emb_len
    )
    duin_cls_params_inst.model.cls.n_labels = n_labels
    # Initialize input `X` & `y_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels); subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_cls.
    duin_cls_inst = duin_cls(duin_cls_params_inst.model); print(duin_cls_inst)
    # Forward layers in `duin_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - torch.float32
    y_pred, loss = duin_cls_inst((X, y_true, subj_id))
    ## Forward duin_llm.
    # Instantiate params.
    duin_llm_params_inst = duin_llm_params(dataset=dataset)
    duin_llm_params_inst.model.subj.n_subjects = duin_llm_params_inst.model.n_subjects = n_subjects
    duin_llm_params_inst.model.subj.d_input = duin_llm_params_inst.model.n_channels = n_channels
    assert seq_len % duin_llm_params_inst.model.seg_len == 0; duin_llm_params_inst.model.seq_len = seq_len
    token_len = duin_llm_params_inst.model.seq_len // duin_llm_params_inst.model.tokenizer.seg_len
    duin_llm_params_inst.model.tokenizer.token_len = token_len
    duin_llm_params_inst.model.encoder.emb_len = token_len
    duin_llm_params_inst.model.rgs.d_model = duin_llm_params_inst.model.encoder.d_model
    duin_llm_params_inst.model.rgs.d_llm = d_llm
    # Initialize input `X` & `y_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); L - (batch_size, emb_len, d_llm); subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    L = torch.rand((batch_size, token_len, d_llm), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_llm.
    duin_llm_inst = duin_llm(duin_llm_params_inst.model); print(duin_llm_inst)
    # Forward layers in `duin_llm_inst`.
    # L_rgs - (batch_size, emb_len, d_llm); loss - torch.float32
    L_rgs, loss = duin_llm_inst((X, L, subj_id))

