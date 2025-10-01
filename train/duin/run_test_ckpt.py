#!/usr/bin/env python3
"""
Test DuIN CLS model from a specified checkpoint.
"""
import torch
import os, argparse, copy as cp
import numpy as np
# KL散度修复相关导入
from enhanced_distribution_analysis import (
    apply_transform_to_numpy_data,
    analyze_dataset_stats_enhanced,
    compute_kl_divergences,
    create_distribution_plots
)
import scipy as sp
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy

# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import save_pickle, load_pickle
from models.duin import duin_cls as duin_model

def get_args_parser():
    parser = argparse.ArgumentParser("DuIN CLS Test from checkpoint", add_help=False)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,])
    parser.add_argument("--subjs", type=str, nargs="+", default=["011",])
    parser.add_argument("--subj_idxs", type=int, nargs="+", default=[0,])
    parser.add_argument("--pt_ckpt", type=str, default=None)
    parser.add_argument("--foldName", type=str, default=None)
    parser.add_argument("--cls_part", type=str, default='_vocal', choices=['_vocal', '_mimed', '_imagined'], help="choose subj13's cls part for downstream")
    parser.add_argument("--use_senet", type=int, default=None, help="senet")
    parser.add_argument("--all_channels", action='store_true', help="use all channels")
    parser.add_argument("--ttt_compare", action='store_true', help="Compare different TTT strategies")
    
    # Domain shift / Transform arguments
    parser.add_argument("--test_transform", type=str, default="none", 
                       choices=["none", "gaussian_noise", "channel_noise", "scaling"],
                       help="Type of domain shift transform to apply")
    parser.add_argument("--transform_strength", type=str, default="medium",
                       choices=["light", "medium", "heavy"],
                       help="Strength of the transform")
    parser.add_argument("--noise_sigma", type=float, default=0.2,
                       help="Standard deviation for Gaussian noise")
    parser.add_argument("--channel_noise_sigma", type=float, default=0.1,
                       help="Standard deviation for per-channel noise")
    parser.add_argument("--scale_min", type=float, default=0.8,
                       help="Minimum scaling factor")
    parser.add_argument("--scale_max", type=float, default=1.2,
                       help="Maximum scaling factor")
    
    # TTT hyperparameters
    parser.add_argument("--ttt_lr", type=float, default=1e-3, help="TTT learning rate")
    parser.add_argument("--ttt_momentum", type=float, default=0.9, help="TTT momentum for SGD")
    parser.add_argument("--ttt_optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="TTT optimizer type")
    parser.add_argument("--ttt_steps", type=int, default=1, help="Number of gradient steps per batch")
    parser.add_argument("--ttt_warmup", type=int, default=0, help="Number of warmup batches before adaptation")
    # 注意：TENT方法不应该重置BN统计量，保留此选项仅为兼容性，但会被忽略
    parser.add_argument("--ttt_reset_bn_stats", action='store_true', help="[DEPRECATED] Reset BN running statistics (ignored in TENT)")
    parser.add_argument("--ttt_entropy_weight", type=float, default=1.0, help="Weight for entropy loss")
    parser.add_argument("--ttt_temperature", type=float, default=1.0, help="Temperature for softmax in entropy calculation")
    
    # Advanced TTT options for BN parameter scaling issues
    parser.add_argument("--ttt_normalize_params", action='store_true', help="Normalize BN parameters before optimization")
    parser.add_argument("--ttt_adaptive_lr", action='store_true', help="Use adaptive learning rate per parameter")
    parser.add_argument("--ttt_loss_type", type=str, default="entropy", choices=["entropy", "confidence", "diversity", "combined"], help="Type of adaptation loss")
    parser.add_argument("--ttt_grad_clip", type=float, default=None, help="Gradient clipping value")
    parser.add_argument("--ttt_weight_decay", type=float, default=0.0, help="Weight decay for BN parameters")
    # 注意：TENT只更新仿射参数，此选项保留为兼容性
    parser.add_argument("--ttt_bn_affine_only", action='store_true', help="[DEFAULT] Only adapt affine parameters (weight/bias), not running stats")
    parser.add_argument("--ttt_separate_lr", action='store_true', help="Use separate learning rates for weight and bias")
    parser.add_argument("--ttt_lr_weight", type=float, default=None, help="Learning rate for BN weights")
    parser.add_argument("--ttt_lr_bias", type=float, default=None, help="Learning rate for BN biases")
    
    return parser

def init(params_, foldName=None):
    global params, paths
    params = cp.deepcopy(params_)
    
    # 确保必要的参数存在，如果不存在则设置默认值
    if not hasattr(params, '_precision'):
        params._precision = 'float32'
    
    newFoldName = foldName
    paths = utils.Paths(base=params.train.base, params=params, foldName=newFoldName)
    paths.run.logger.tensorboard = SummaryWriter(paths.run.train)
    torch.set_default_dtype(getattr(torch, params._precision))
    torch.set_float32_matmul_precision("high")

def load_data(load_params):
    global params
    try:
        func = getattr(sys.modules[__name__], "_".join(["_load_data", params.train.dataset]))
        dataset_train, dataset_validation, dataset_test = func(load_params)
    except Exception:
        raise ValueError(("ERROR: Unknown dataset type {} in test script.").format(params.train.dataset))
    return dataset_train, dataset_validation, dataset_test

def _load_data_seeg_he2023xuanwu(load_params):
    global params, paths
    subjs_cfg = load_params.subjs_cfg
    n_channels = load_params.n_channels if load_params.n_channels is not None else None
    n_subjects = load_params.n_subjects if load_params.n_subjects is not None else len(subjs_cfg)
    subj_idxs = load_params.subj_idxs if load_params.subj_idxs is not None else np.arange(n_subjects)
    seq_len = None; n_labels = None
    Xs_train = []; ys_train = []; subj_ids_train = []
    Xs_validation = []; ys_validation = []; subj_ids_validation = []
    Xs_test = []; ys_test = []; subj_ids_test = []
    for subj_idx, subj_cfg_i in zip(subj_idxs, subjs_cfg):
        print('subj_cfg_i', subj_cfg_i)
        if subj_cfg_i.name != "013":
            func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))
        else:
            task, _ = load_params.task.rsplit('_', 1)
            func = getattr(getattr(utils.data.seeg.he2023xuanwu, task), "load_subj_{}_013".format(load_params.type))
        dataset = func(subj_cfg_i.path, ch_names=subj_cfg_i.ch_names, use_align=load_params.use_align, task=load_params.cls_part)
        X = dataset.X_s.astype(np.float32); y = dataset.y.astype(np.int64)
        if load_params.type.startswith("bipolar"):
            X = X
            sample_rate = 1000; X = sp.signal.resample(X, int(np.round(X.shape[1] / (sample_rate / load_params.resample_rate))), axis=1)
            X = X[:,int(np.round((-0.5 - (-0.5)) * load_params.resample_rate)):
                int(np.round((2.5 - (-0.5)) * load_params.resample_rate)),:]
            X = (X - np.mean(X, axis=(0,1), keepdims=True)) / np.std(X, axis=(0,1), keepdims=True)
        else:
            raise ValueError("ERROR: Unknown type {} of dataset.".format(load_params.type))
        train_ratio = params.train.train_ratio; train_idxs = []; test_idxs = []
        for label_i in sorted(set(y)):
            label_idxs = np.where(y == label_i)[0].tolist()
            train_idxs.extend(label_idxs[:int(train_ratio * len(label_idxs))])
            test_idxs.extend(label_idxs[int(train_ratio * len(label_idxs)):])
        for train_idx in train_idxs: assert train_idx not in test_idxs
        train_idxs = np.array(train_idxs, dtype=np.int64); test_idxs = np.array(test_idxs, dtype=np.int64)
        X_train = X[train_idxs,:,:]; y_train = y[train_idxs]; X_test = X[test_idxs,:,:]; y_test = y[test_idxs]
        if len(X_train) == 0 or len(X_test) == 0: return ([], []), ([], [])
        # 与训练脚本完全一致的检查
        samples_same = None; n_samples = 10; assert X_train.shape[1] == X_test.shape[1]
        for _ in range(n_samples):
            sample_idx = np.random.randint(X_train.shape[1])
            sample_same_i = np.intersect1d(X_train[:,sample_idx,0], X_test[:,sample_idx,0], return_indices=True)[-1].tolist()
            samples_same = set(sample_same_i) if samples_same is None else set(sample_same_i) & samples_same
        assert len(samples_same) == 0
        assert len(set(y_train)) == len(set(y_test)); labels = sorted(set(y_train))
        y_train = np.array([labels.index(y_i) for y_i in y_train], dtype=np.int64); y_train = np.eye(len(labels))[y_train]
        y_test = np.array([labels.index(y_i) for y_i in y_test], dtype=np.int64); y_test = np.eye(len(labels))[y_test]
        # Execute sample permutation. We only shuffle along the axis.
        if load_params.permutation: np.random.shuffle(y_train)
        # valid/test split
        validation_idxs = np.random.choice(np.arange(X_test.shape[0]), size=int(X_test.shape[0]/2), replace=False)
        validation_mask = np.zeros((X_test.shape[0],), dtype=np.bool_); validation_mask[validation_idxs] = True
        X_validation = X_test[validation_mask,:,:]; y_validation = y_test[validation_mask,:]
        X_test = X_test[~validation_mask,:,:]; y_test = y_test[~validation_mask,:]
        subj_id_train = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_train.shape[0])])
        subj_id_validation = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_validation.shape[0])])
        subj_id_test = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_test.shape[0])])
        msg = (
            "INFO: Data preparation for subject ({}) complete, with train-set ({}) & validation-set ({}) & test-set ({})."
        ).format(subj_cfg_i.name, X_train.shape, X_validation.shape, X_test.shape)
        print(msg); paths.run.logger.summaries.info(msg)
        Xs_train.append(X_train); ys_train.append(y_train); subj_ids_train.append(subj_id_train)
        Xs_validation.append(X_validation); ys_validation.append(y_validation); subj_ids_validation.append(subj_id_validation)
        Xs_test.append(X_test); ys_test.append(y_test); subj_ids_test.append(subj_id_test)
        n_channels = max(X.shape[-1], n_channels) if n_channels is not None else X.shape[-1]
        seq_len = X.shape[-2] if seq_len is None else seq_len; assert seq_len == X.shape[-2]
        n_labels = len(labels) if n_labels is None else n_labels; assert n_labels == len(labels)
    if load_params.n_channels is not None: assert n_channels == load_params.n_channels
    # padding
    Xs_train = [np.concatenate([X_train_i,
        np.zeros((*X_train_i.shape[:-1], (n_channels - X_train_i.shape[-1])), dtype=X_train_i.dtype)
    ], axis=-1) for X_train_i in Xs_train]
    Xs_validation = [np.concatenate([X_validation_i,
        np.zeros((*X_validation_i.shape[:-1], (n_channels - X_validation_i.shape[-1])), dtype=X_validation_i.dtype)
    ], axis=-1) for X_validation_i in Xs_validation]
    Xs_test = [np.concatenate([X_test_i,
        np.zeros((*X_test_i.shape[:-1], (n_channels - X_test_i.shape[-1])), dtype=X_test_i.dtype)
    ], axis=-1) for X_test_i in Xs_test]
    Xs_train = np.concatenate(Xs_train, axis=0); ys_train = np.concatenate(ys_train, axis=0)
    subj_ids_train = np.concatenate(subj_ids_train, axis=0)
    Xs_validation = np.concatenate(Xs_validation, axis=0); ys_validation = np.concatenate(ys_validation, axis=0)
    subj_ids_validation = np.concatenate(subj_ids_validation, axis=0)
    Xs_test = np.concatenate(Xs_test, axis=0); ys_test = np.concatenate(ys_test, axis=0)
    subj_ids_test = np.concatenate(subj_ids_test, axis=0)
    # shuffle
    train_idxs = np.arange(Xs_train.shape[0]); np.random.shuffle(train_idxs)
    validation_idxs = np.arange(Xs_validation.shape[0]); np.random.shuffle(validation_idxs)
    test_idxs = np.arange(Xs_test.shape[0]); np.random.shuffle(test_idxs)
    Xs_train = Xs_train[train_idxs,...]; ys_train = ys_train[train_idxs,...]; subj_ids_train = subj_ids_train[train_idxs,...]
    Xs_validation = Xs_validation[validation_idxs,...]; ys_validation = ys_validation[validation_idxs,...]; subj_ids_validation = subj_ids_validation[validation_idxs,...]
    Xs_test = Xs_test[test_idxs,...]; ys_test = ys_test[test_idxs,...]; subj_ids_test = subj_ids_test[test_idxs,...]
    msg = (
        "INFO: Data preparation complete, with train-set ({}) & validation-set ({}) & test-set ({})."
    ).format(Xs_train.shape, Xs_validation.shape, Xs_test.shape)
    print(msg); paths.run.logger.summaries.info(msg)
    dataset_train = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_train, ys_train, subj_ids_train)], use_aug=True)
    dataset_validation = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_validation, ys_validation, subj_ids_validation)], use_aug=False)
    dataset_test = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_test, ys_test, subj_ids_test)], use_aug=False)
    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=params.train.batch_size, shuffle=False, drop_last=False)
    dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False, drop_last=False)
    params.model.subj.n_subjects = params.model.n_subjects = n_subjects
    params.model.subj.d_input = params.model.n_channels = n_channels
    params.model.subj.use_senet = load_params.use_senet
    assert seq_len % params.model.seg_len == 0; params.model.seq_len = seq_len
    token_len = params.model.seq_len // params.model.tokenizer.seg_len
    params.model.tokenizer.token_len = token_len
    params.model.encoder.emb_len = token_len
    params.model.cls.d_feature = (
        params.model.encoder.d_model * params.model.encoder.emb_len
    )
    params.model.cls.n_labels = n_labels
    
    # 与训练脚本对齐的数据分布分析
    msg = "\nAnalyzing dataset distributions..."
    print(msg); paths.run.logger.summaries.info(msg)
    
    # === 修复后的KL散度分析（在transform之后计算） ===
    # 注意：这里暂时保留原始分析作为参考，实际的修复在TTT测试之后
    train_stats, train_dist = analyze_dataset_stats(Xs_train, ys_train, "Training", paths.run.logger.summaries)
    val_stats, val_dist = analyze_dataset_stats(Xs_validation, ys_validation, "Validation", paths.run.logger.summaries)
    test_stats, test_dist = analyze_dataset_stats(Xs_test, ys_test, "Test", paths.run.logger.summaries)

    # 防止出现0
    epsilon = 1e-10
    train_dist += epsilon
    val_dist += epsilon
    test_dist += epsilon
    train_dist /= np.sum(train_dist)
    val_dist /= np.sum(val_dist)
    test_dist /= np.sum(test_dist)
    
    # Compare distributions using KL divergence
    kl_train_val = entropy(train_dist, val_dist)
    kl_train_test = entropy(train_dist, test_dist)
    kl_val_test = entropy(val_dist, test_dist)
    
    msg = "\nDistribution similarities (KL divergence):"
    msg += f"\nTrain-Val: {kl_train_val:.4f}"
    msg += f"\nTrain-Test: {kl_train_test:.4f}"
    msg += f"\nVal-Test: {kl_val_test:.4f}"
    print(msg); paths.run.logger.summaries.info(msg)
    
    # Add to tensorboard
    writer = paths.run.logger.tensorboard
    for split, stats, dist in [
        ('train', train_stats, train_dist),
        ('val', val_stats, val_dist),
        ('test', test_stats, test_dist)
    ]:
        writer.add_scalar(f'stats/{split}/mean', stats['mean'], 0)
        writer.add_scalar(f'stats/{split}/std', stats['std'], 0)
        for i, class_prob in enumerate(dist):
            writer.add_scalar(f'class_dist/{split}/class_{i}', class_prob, 0)
    
    return dataset_train, dataset_validation, dataset_test

def analyze_dataset_stats(X, y, name, logger):
    """
    Analyze basic statistics of a dataset.
    
    Args:
        X: np.array - Signal data (n_samples, seq_len, n_channels)
        y: np.array - Labels in one-hot format (n_samples, n_classes)
        name: str - Name of the dataset for logging
        logger: Logger object
    """
    # Basic signal statistics
    signal_stats = {
        'mean': np.mean(X),
        'std': np.std(X),
        'min': np.min(X),
        'max': np.max(X)
    }
    
    # Class distribution
    y_labels = np.argmax(y, axis=1)
    class_counts = np.bincount(y_labels)
    class_dist = class_counts / len(y_labels)
    
    # Log statistics
    msg = f"\n{name} Dataset Statistics:"
    msg += f"\nSamples: {len(X)}"
    msg += f"\nSignal - Mean: {signal_stats['mean']:.3f}, Std: {signal_stats['std']:.3f}"
    msg += f"\nSignal - Range: [{signal_stats['min']:.3f}, {signal_stats['max']:.3f}]"
    msg += f"\nClass distribution: " + ", ".join([f"Class {i}: {dist:.3f}" for i, dist in enumerate(class_dist)])
    logger.info(msg)
    
    return signal_stats, class_dist

class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, data_items, use_aug=False, **kwargs):
        super(FinetuneDataset, self).__init__(**kwargs)
        self.data_items = data_items; self.use_aug = use_aug
        self._init_dataset()
    def _init_dataset(self):
        self.max_steps = self.data_items[0].X.shape[1] // 10
    def __len__(self):
        return len(self.data_items)
    def __getitem__(self, index):
        data_item = self.data_items[index]
        X = data_item.X; y = data_item.y; subj_id = data_item.subj_id
        # 与训练脚本一致的数据增强
        if self.use_aug:
            X_shifted = np.zeros(X.shape, dtype=X.dtype)
            n_steps = np.random.choice((np.arange(2 * self.max_steps + 1, dtype=np.int64) - self.max_steps))
            if n_steps > 0:
                X_shifted[:,n_steps:] = X[:,:-n_steps]
            elif n_steps < 0:
                X_shifted[:,:n_steps] = X[:,-n_steps:]
            else:
                pass
            X = X_shifted
        data = {
            "X": torch.from_numpy(X.T).to(dtype=torch.float32),
            "y": torch.from_numpy(y).to(dtype=torch.float32),
            "subj_id": torch.from_numpy(subj_id).to(dtype=torch.float32),
        }
        return data

def test_model(model, dataset_test, device):
    model.eval()
    accuracy_test = []
    all_test_pred, all_test_true = [], []
    with torch.no_grad():
        for test_batch in dataset_test:
            batch_i = [
                test_batch["X"].to(device=device),
                test_batch["y"].to(device=device),
                test_batch["subj_id"].to(device=device),
            ]
            y_pred_i, _, _ = model(batch_i)
            y_pred_i = y_pred_i.detach().cpu().numpy(); y_true_i = batch_i[1].detach().cpu().numpy()
            accuracy_i = np.stack([
                (np.argmax(y_pred_i, axis=-1) == np.argmax(y_true_i, axis=-1)).astype(np.int64),
                np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
            ], axis=0).T
            accuracy_test.append(accuracy_i)
            all_test_pred.append(y_pred_i)
            all_test_true.append(y_true_i)
    accuracy_test = np.concatenate(accuracy_test, axis=0)
    # 按 subject 统计准确率
    subj_ids = sorted(set(accuracy_test[:,1]))
    accs = [accuracy_test[np.where(accuracy_test[:,1] == subj_idx),0].mean() for subj_idx in subj_ids]
    return accs, all_test_pred, all_test_true

def _ttt(dataset_test, model, device, params, ttt_args):
    """
    TENT: Test-Time Adaptation by Entropy Minimization with advanced BN parameter handling
    
    Key principles of TENT:
    1. Only update BatchNorm affine parameters (weight, bias)
    2. Do NOT update running statistics (running_mean, running_var)
    3. Keep model in eval mode but enable gradients for BN affine params
    """
    # 关键：保持模型在eval模式，这样BN层不会更新running statistics
    model.eval()
    
    # 保存初始BN参数（用于可选的重置）
    def save_bn_params(model):
        bn_state = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    bn_state[name + '.weight'] = module.weight.detach().clone()
                if module.bias is not None:
                    bn_state[name + '.bias'] = module.bias.detach().clone()
        return bn_state
    
    def load_bn_params(model, bn_state):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None and name + '.weight' in bn_state:
                    module.weight.data.copy_(bn_state[name + '.weight'])
                if module.bias is not None and name + '.bias' in bn_state:
                    module.bias.data.copy_(bn_state[name + '.bias'])
    
    def collect_bn_params_advanced(model, ttt_args):
        """Advanced BN parameter collection with scaling handling"""
        weight_params = []
        bias_params = []
        param_info = []
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                # 确保BN层在eval模式下，但仍然可以计算梯度
                module.eval()
                
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    weight_params.append(module.weight)
                    param_info.append({
                        'name': f'{name}.weight',
                        'param': module.weight,
                        'type': 'weight',
                        'initial_norm': module.weight.data.norm().item(),
                        'initial_mean': module.weight.data.mean().item(),
                        'initial_std': module.weight.data.std().item()
                    })
                
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    bias_params.append(module.bias)
                    param_info.append({
                        'name': f'{name}.bias',
                        'param': module.bias,
                        'type': 'bias',
                        'initial_norm': module.bias.data.norm().item(),
                        'initial_mean': module.bias.data.mean().item(),
                        'initial_std': module.bias.data.std().item()
                    })
            else:
                # 确保非BN层的参数不可训练
                for p in getattr(module, 'parameters', lambda:[])():
                    p.requires_grad_(False)
        
        return weight_params, bias_params, param_info
    
    def normalize_bn_params(model, param_info):
        """Normalize BN parameters to similar scales"""
        for info in param_info:
            param = info['param']
            if info['type'] == 'weight':
                # Normalize weights to have norm around 1
                current_norm = param.data.norm()
                if current_norm > 1e-6:
                    param.data = param.data / current_norm
            else:  # bias
                # Normalize biases to have std around 1
                current_std = param.data.std()
                if current_std > 1e-6:
                    param.data = param.data / current_std
    
    def get_adaptive_loss(logits, loss_type, temperature=1.0):
        """Different types of adaptation losses"""
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        
        if loss_type == "entropy":
            # Standard entropy minimization
            return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        elif loss_type == "confidence":
            # Maximize confidence (minimize negative max probability)
            max_probs, _ = torch.max(probs, dim=1)
            return -max_probs.mean()
        
        elif loss_type == "diversity":
            # Encourage diversity in predictions across batch
            mean_probs = probs.mean(dim=0)
            return -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
        
        elif loss_type == "combined":
            # Combination of entropy and confidence
            entropy_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            max_probs, _ = torch.max(probs, dim=1)
            confidence_loss = -max_probs.mean()
            return 0.7 * entropy_loss + 0.3 * confidence_loss
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    # 保存初始状态
    init_bn_state = save_bn_params(model)
    
    # TENT不应该重置BN统计量，移除这个选项
    # 如果用户设置了重置选项，给出警告
    if hasattr(ttt_args, 'ttt_reset_bn_stats') and ttt_args.ttt_reset_bn_stats:
        print("WARNING: TENT method should NOT reset BN running statistics. Ignoring ttt_reset_bn_stats=True")
    
    # 冻结所有参数，只保留BN仿射参数可训练
    for n, p in model.named_parameters():
        p.requires_grad = False
    
    # 高级BN参数收集
    weight_params, bias_params, param_info = collect_bn_params_advanced(model, ttt_args)
    all_params = weight_params + bias_params
    
    print(f"TTT (TENT): Found {len(weight_params)} weight params, {len(bias_params)} bias params")
    print(f"TTT (TENT): Model in eval mode, only updating BN affine parameters")
    print(f"TTT (TENT): Using {ttt_args.ttt_optimizer.upper()} optimizer with lr={ttt_args.ttt_lr}")
    print(f"TTT (TENT): Loss type={ttt_args.ttt_loss_type}, Steps per batch={ttt_args.ttt_steps}")
    
    # 验证BN层确实在eval模式
    bn_train_count = 0
    bn_eval_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            if module.training:
                bn_train_count += 1
            else:
                bn_eval_count += 1
    
    print(f"TTT (TENT): BN layers - {bn_eval_count} in eval mode, {bn_train_count} in train mode")
    if bn_train_count > 0:
        print("WARNING: Some BN layers are in train mode, this may update running statistics!")
    
    # 打印参数统计信息
    weight_norms = [info['initial_norm'] for info in param_info if info['type'] == 'weight']
    bias_norms = [info['initial_norm'] for info in param_info if info['type'] == 'bias']
    
    if weight_norms:
        print(f"TTT: Weight norms - min: {min(weight_norms):.4f}, max: {max(weight_norms):.4f}, mean: {sum(weight_norms)/len(weight_norms):.4f}")
    if bias_norms:
        print(f"TTT: Bias norms - min: {min(bias_norms):.4f}, max: {max(bias_norms):.4f}, mean: {sum(bias_norms)/len(bias_norms):.4f}")
    
    # 可选：参数归一化
    if hasattr(ttt_args, 'ttt_normalize_params') and ttt_args.ttt_normalize_params:
        print("TTT: Normalizing BN parameters")
        normalize_bn_params(model, param_info)
    
    # 创建优化器
    if (hasattr(ttt_args, 'ttt_separate_lr') and ttt_args.ttt_separate_lr and 
        hasattr(ttt_args, 'ttt_lr_weight') and ttt_args.ttt_lr_weight is not None and 
        hasattr(ttt_args, 'ttt_lr_bias') and ttt_args.ttt_lr_bias is not None):
        # 使用分离的学习率
        param_groups = [
            {'params': weight_params, 'lr': ttt_args.ttt_lr_weight},
            {'params': bias_params, 'lr': ttt_args.ttt_lr_bias}
        ]
        print(f"TTT: Using separate LRs - weight: {ttt_args.ttt_lr_weight}, bias: {ttt_args.ttt_lr_bias}")
    else:
        param_groups = [{'params': all_params, 'lr': ttt_args.ttt_lr}]
    
    # 获取权重衰减参数
    weight_decay = getattr(ttt_args, 'ttt_weight_decay', 0.0)
    
    if ttt_args.ttt_optimizer == "sgd":
        momentum = getattr(ttt_args, 'ttt_momentum', 0.9)
        tent_optimizer = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    elif ttt_args.ttt_optimizer == "adam":
        tent_optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif ttt_args.ttt_optimizer == "adamw":
        tent_optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # 自适应学习率调度器
    if hasattr(ttt_args, 'ttt_adaptive_lr') and ttt_args.ttt_adaptive_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(tent_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    accuracy_test = []
    total_loss = 0.0
    batch_count = 0
    loss_history = []
    
    # 记录BN参数变化
    initial_param_values = {}
    for info in param_info:
        initial_param_values[info['name']] = info['param'].data.clone()
    
    # 获取TTT参数，提供默认值
    warmup = getattr(ttt_args, 'ttt_warmup', 0)
    steps = getattr(ttt_args, 'ttt_steps', 1)
    temperature = getattr(ttt_args, 'ttt_temperature', 1.0)
    entropy_weight = getattr(ttt_args, 'ttt_entropy_weight', 1.0)
    loss_type = getattr(ttt_args, 'ttt_loss_type', 'entropy')
    grad_clip = getattr(ttt_args, 'ttt_grad_clip', None)
    
    for batch_idx, test_batch in enumerate(dataset_test):
        batch_i = [
            test_batch["X"].to(device=device),
            test_batch["y"].to(device=device),
            test_batch["subj_id"].to(device=device),
        ]
        
        # Warmup期间不进行适应
        if batch_idx < warmup:
            with torch.no_grad():
                logits, _, _ = model(batch_i)
                y_pred_i = logits.detach().cpu().numpy()
                y_true_i = batch_i[1].detach().cpu().numpy()
                
                accuracy_i = np.stack([
                    (np.argmax(y_pred_i, axis=-1) == np.argmax(y_true_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T
                accuracy_test.append(accuracy_i)
            continue
        
        # 多步优化
        batch_losses = []
        for step in range(steps):
            # Forward pass
            logits, _, _ = model(batch_i)
            
            # 计算适应损失
            loss = get_adaptive_loss(logits, loss_type, temperature) * entropy_weight
            
            if step == 0:  # 只在第一步记录损失
                total_loss += loss.item()
                batch_count += 1
                batch_losses.append(loss.item())
            
            # Backward pass and optimization
            tent_optimizer.zero_grad()
            loss.backward()
            
            # 可选：梯度裁剪
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            
            tent_optimizer.step()
        
        # 自适应学习率调整
        if hasattr(ttt_args, 'ttt_adaptive_lr') and ttt_args.ttt_adaptive_lr and batch_losses:
            scheduler.step(batch_losses[-1])
        
        loss_history.extend(batch_losses)
        
        # 获取更新后的预测
        with torch.no_grad():
            logits_updated, _, _ = model(batch_i)
            y_pred_i = logits_updated.detach().cpu().numpy()
            y_true_i = batch_i[1].detach().cpu().numpy()
            
            accuracy_i = np.stack([
                (np.argmax(y_pred_i, axis=-1) == np.argmax(y_true_i, axis=-1)).astype(np.int64),
                np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
            ], axis=0).T
            accuracy_test.append(accuracy_i)
        
        # 每10个batch打印一次进度
        if (batch_idx + 1) % 10 == 0:
            # 计算参数变化
            param_changes = {}
            total_change = 0.0
            max_change = 0.0
            
            for info in param_info:
                name = info['name']
                current_val = info['param'].data
                initial_val = initial_param_values[name]
                
                change = (current_val - initial_val).norm().item()
                relative_change = change / (initial_val.norm().item() + 1e-8)
                
                param_changes[name] = {
                    'absolute_change': change,
                    'relative_change': relative_change
                }
                
                total_change += change
                max_change = max(max_change, change)
            
            print(f"TTT Batch {batch_idx + 1}: Loss = {loss.item():.6f}")
            print(f"  Total param change: {total_change:.6f}, Max change: {max_change:.6f}")
            print(f"  Recent loss trend: {loss_history[-5:] if len(loss_history) >= 5 else loss_history}")
    
    # 计算最终准确率
    accuracy_test = np.concatenate(accuracy_test, axis=0)
    subj_ids = sorted(set(accuracy_test[:,1]))
    accs = [accuracy_test[np.where(accuracy_test[:,1] == subj_idx),0].mean() for subj_idx in subj_ids]
    
    # 打印详细的适应统计信息
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    print(f"TTT Summary:")
    print(f"  Average {loss_type} loss: {avg_loss:.6f}")
    print(f"  Loss reduction: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}" if len(loss_history) > 1 else "  Single loss value")
    
    # 详细的参数变化分析
    print(f"  Parameter changes by type:")
    weight_changes = []
    bias_changes = []
    
    for info in param_info:
        name = info['name']
        current_val = info['param'].data
        initial_val = initial_param_values[name]
        
        abs_change = (current_val - initial_val).norm().item()
        rel_change = abs_change / (initial_val.norm().item() + 1e-8)
        
        if info['type'] == 'weight':
            weight_changes.append(rel_change)
        else:
            bias_changes.append(rel_change)
    
    if weight_changes:
        print(f"    Weights - mean rel change: {np.mean(weight_changes):.6f}, max: {np.max(weight_changes):.6f}")
    if bias_changes:
        print(f"    Biases - mean rel change: {np.mean(bias_changes):.6f}, max: {np.max(bias_changes):.6f}")
    
    return accs

def _ttt_variants(dataset_test, model, device, params, ttt_args, variant="cumulative"):
    """
    TTT的不同变体实现 - 符合TENT方法
    variant: 
    - "cumulative": 累积适应（推荐，符合TENT论文）
    - "per_batch": 每个batch重置
    """
    # 关键：保持模型在eval模式
    model.eval()
    
    def save_bn_params(model):
        bn_state = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None:
                    bn_state[name + '.weight'] = module.weight.detach().clone()
                if module.bias is not None:
                    bn_state[name + '.bias'] = module.bias.detach().clone()
        return bn_state
    
    def load_bn_params(model, bn_state):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                if module.weight is not None and name + '.weight' in bn_state:
                    module.weight.data.copy_(bn_state[name + '.weight'])
                if module.bias is not None and name + '.bias' in bn_state:
                    module.bias.data.copy_(bn_state[name + '.bias'])
    
    def collect_bn_params(model):
        params = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
                # 确保BN层在eval模式
                m.eval()
                
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
            else:
                for p in getattr(m, 'parameters', lambda:[])():
                    p.requires_grad_(False)
        return params
    
    def entropy_minimize_loss(logits, temperature=1.0):
        logits = logits / temperature
        p = torch.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
    
    # 保存初始状态
    init_bn_state = save_bn_params(model)
    
    # 冻结所有参数，只保留BN仿射参数可训练
    for n, p in model.named_parameters():
        p.requires_grad = False
    bn_params = collect_bn_params(model)
    
    print(f"TTT {variant}: Found {len(bn_params)} BN parameters to optimize")
    print(f"TTT {variant}: Model in eval mode, only updating BN affine parameters")
    
    # 验证BN层确实在eval模式
    bn_train_count = 0
    bn_eval_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            if module.training:
                bn_train_count += 1
            else:
                bn_eval_count += 1
    
    print(f"TTT {variant}: BN layers - {bn_eval_count} in eval mode, {bn_train_count} in train mode")
    if bn_train_count > 0:
        print("WARNING: Some BN layers are in train mode, this may update running statistics!")
    
    accuracy_test = []
    total_loss = 0.0
    batch_count = 0
    
    # 获取TTT参数，提供默认值
    lr = getattr(ttt_args, 'ttt_lr', 1e-3)
    momentum = getattr(ttt_args, 'ttt_momentum', 0.9)
    temperature = getattr(ttt_args, 'ttt_temperature', 1.0)
    entropy_weight = getattr(ttt_args, 'ttt_entropy_weight', 1.0)
    optimizer_type = getattr(ttt_args, 'ttt_optimizer', 'sgd')
    
    for batch_idx, test_batch in enumerate(dataset_test):
        # 根据变体决定是否重置
        if variant == "per_batch" and batch_idx > 0:
            load_bn_params(model, init_bn_state)
        
        # 为每个batch创建新的优化器（如果需要重置）
        if variant == "per_batch" or batch_idx == 0:
            if optimizer_type == "sgd":
                tent_optimizer = torch.optim.SGD(bn_params, lr=lr, momentum=momentum)
            elif optimizer_type == "adam":
                tent_optimizer = torch.optim.Adam(bn_params, lr=lr)
            elif optimizer_type == "adamw":
                tent_optimizer = torch.optim.AdamW(bn_params, lr=lr)
        
        batch_i = [
            test_batch["X"].to(device=device),
            test_batch["y"].to(device=device),
            test_batch["subj_id"].to(device=device),
        ]
        
        # batch级别的处理
        logits, _, _ = model(batch_i)
        loss = entropy_minimize_loss(logits, temperature) * entropy_weight
        total_loss += loss.item()
        batch_count += 1
        
        tent_optimizer.zero_grad()
        loss.backward()
        tent_optimizer.step()
        
        with torch.no_grad():
            logits_updated, _, _ = model(batch_i)
            y_pred_i = logits_updated.detach().cpu().numpy()
            y_true_i = batch_i[1].detach().cpu().numpy()
            
            accuracy_i = np.stack([
                (np.argmax(y_pred_i, axis=-1) == np.argmax(y_true_i, axis=-1)).astype(np.int64),
                np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
            ], axis=0).T
            accuracy_test.append(accuracy_i)
    
    # 计算最终准确率
    accuracy_test = np.concatenate(accuracy_test, axis=0)
    subj_ids = sorted(set(accuracy_test[:,1]))
    accs = [accuracy_test[np.where(accuracy_test[:,1] == subj_idx),0].mean() for subj_idx in subj_ids]
    
    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"TTT {variant} - Average entropy loss: {avg_loss:.6f}")
    
    return accs

def apply_domain_shift_transform(dataset, transform_type, transform_args):
    """Apply domain shift transforms to the dataset"""
    import torch
    import numpy as np
    
    if transform_type == "none":
        return dataset
    
    print(f"Applying domain shift transform: {transform_type}")
    
    # Create a new dataset with transformed data
    transformed_items = []
    
    for batch in dataset:
        X = batch["X"]  # Shape: [batch_size, channels, time]
        y = batch["y"]
        subj_id = batch["subj_id"]
        
        # Apply transform based on type
        if transform_type == "gaussian_noise":
            noise_std = transform_args.get("noise_sigma", 0.2)
            noise = torch.randn_like(X) * noise_std
            X_transformed = X + noise
            print(f"Applied Gaussian noise with σ={noise_std}")
            
        elif transform_type == "channel_noise":
            noise_std = transform_args.get("channel_noise_sigma", 0.1)
            # Apply different noise to each channel
            noise = torch.randn_like(X) * noise_std
            X_transformed = X + noise
            print(f"Applied per-channel noise with σ={noise_std}")
            
        elif transform_type == "scaling":
            scale_min = transform_args.get("scale_min", 0.8)
            scale_max = transform_args.get("scale_max", 1.2)
            # Random scaling factor for each sample
            batch_size = X.shape[0]
            scale_factors = torch.rand(batch_size, 1, 1) * (scale_max - scale_min) + scale_min
            X_transformed = X * scale_factors
            print(f"Applied amplitude scaling [{scale_min}, {scale_max}]")
            
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Create new batch with transformed data
        transformed_batch = {
            "X": X_transformed,
            "y": y,
            "subj_id": subj_id
        }
        transformed_items.append(transformed_batch)
    
    # Create new DataLoader with transformed data
    class TransformedDataset:
        def __init__(self, items):
            self.items = items
        
        def __iter__(self):
            return iter(self.items)
        
        def __len__(self):
            return len(self.items)
    
    return TransformedDataset(transformed_items)

def test_with_seed(seed, cls_part, use_senet, all_channels, ckpt_path, ttt_compare=False, ttt_args=None):
    """
    Test the model with a specific seed - 完全按照训练脚本的逻辑
    """
    global params, paths
    
    # 设置随机种子 - 与训练脚本完全一致的时机
    utils.model.torch.set_seeds(seed)
    
    # 初始化设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params.model.device = device
    print(params.model.device); paths.run.logger.summaries.info(params.model.device)
    
    # 与训练脚本完全一致的subject配置
    if params.train.dataset == "seeg_he2023xuanwu":
        duin_path = '/data/seeg/liyangyang/duin'
        if all_channels:
            # 这里应该包含完整的all_channels配置，但为了简化，我们使用默认配置
            # 如果需要all_channels，需要从训练脚本复制完整的配置
            subjs_cfg = utils.DotDict({
                "001": utils.DotDict({
                    "name": "001", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "001"),
                    "ch_names": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
                }),
                "002": utils.DotDict({
                    "name": "002", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "002"),
                    "ch_names": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
                }),
                "003": utils.DotDict({
                    "name": "003", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "003"),
                    "ch_names": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"] ,
                }),
                "004": utils.DotDict({
                    "name": "004", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "004"),
                    "ch_names": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
                }),
                "005": utils.DotDict({
                    "name": "005", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "005"),
                    "ch_names": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
                }),
                "006": utils.DotDict({
                    "name": "006", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "006"),
                    "ch_names": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
                }),
                "007": utils.DotDict({
                    "name": "007", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "007"),
                    "ch_names": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
                }),
                "008": utils.DotDict({
                    "name": "008", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "008"),
                    "ch_names": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
                }),
                "009": utils.DotDict({
                    "name": "009", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "009"),
                    "ch_names": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
                }),
                "010": utils.DotDict({
                    "name": "010", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "010"),
                    "ch_names": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
                }),
                "011": utils.DotDict({
                    "name": "011", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "011"),
                    "ch_names": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
                }),
                "012": utils.DotDict({
                    "name": "012", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "012"),
                    "ch_names": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
                }),
            })
        else:
            subjs_cfg = utils.DotDict({
                "001": utils.DotDict({
                    "name": "001", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "001"),
                    "ch_names": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
                }),
                "002": utils.DotDict({
                    "name": "002", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "002"),
                    "ch_names": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
                }),
                "003": utils.DotDict({
                    "name": "003", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "003"),
                    "ch_names": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"] ,
                }),
                "004": utils.DotDict({
                    "name": "004", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "004"),
                    "ch_names": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
                }),
                "005": utils.DotDict({
                    "name": "005", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "005"),
                    "ch_names": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
                }),
                "006": utils.DotDict({
                    "name": "006", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "006"),
                    "ch_names": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
                }),
                "007": utils.DotDict({
                    "name": "007", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "007"),
                    "ch_names": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
                }),
                "008": utils.DotDict({
                    "name": "008", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "008"),
                    "ch_names": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
                }),
                "009": utils.DotDict({
                    "name": "009", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "009"),
                    "ch_names": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
                }),
                "010": utils.DotDict({
                    "name": "010", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "010"),
                    "ch_names": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
                }),
                "011": utils.DotDict({
                    "name": "011", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "011"),
                    "ch_names": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
                }),
                "012": utils.DotDict({
                    "name": "012", "path": os.path.join(duin_path, "data", "seeg.he2023xuanwu", "012"),
                    "ch_names": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
                }),
            })
        
        load_type = "bipolar_default"; load_task = "word_recitation"; use_align = False
        subjs_cfg = [subjs_cfg[subj_i] for subj_i in params.train.subjs]
        subj_idxs = params.train.subj_idxs; assert len(subj_idxs) == len(subjs_cfg)
        if load_type.startswith("bipolar"):
            resample_rate = 1000
        
        # 与训练脚本完全一致的load_params
        load_params = [
            utils.DotDict({
                "name": "train-task-all-speak-test-task-all-speak", "type": load_type,
                "permutation": False, "resample_rate": resample_rate, "task": load_task, "use_align": use_align,
                "n_channels": None, "n_subjects": None, "subj_idxs": subj_idxs,
                "use_senet": use_senet,
            }),
        ]
        
        # 对于013，load_task加上对应的任务
        for i in range(len(load_params)):
            if len(subjs_cfg) > 0 and subjs_cfg[0].name == '013':  # 检查是否有013
                load_params[i].task = load_params[i].task + cls_part
                # 顺便把通道名字也更新
                if cls_part == '_vocal':
                    subjs_cfg[0].ch_names = ['G11', 'H10', 'C5', 'G9', 'G7', 'C15', 'C9', 'H14', 'H13', 'H12']
                elif cls_part == '_mimed':
                    subjs_cfg[0].ch_names = ['C11', 'G11', 'H15', 'G8', 'H10', 'H9', 'C14', 'C13', 'C5', 'H6']
                elif cls_part == '_imagined':
                    subjs_cfg[0].ch_names = ['C11', 'G8', 'H10', 'C8', 'G11', 'C13', 'C12', 'H9', 'H6', 'C6']
        
        # 加载数据
        for load_params_idx in range(len(load_params)):
            load_params_i = cp.deepcopy(load_params[load_params_idx]); load_params_i.subjs_cfg = subjs_cfg
            load_params_i.cls_part = cls_part  # 添加cls_part参数
            
            # 添加TTT超参数到load_params_i
            if ttt_args:
                for key, value in vars(ttt_args).items():
                    if key.startswith('ttt_'):
                        setattr(load_params_i, key, value)
            
            msg = (
                "Testing started with experiment {} with {:d} subjects."
            ).format(load_params_i.name, len(load_params_i.subjs_cfg))
            print(msg); paths.run.logger.summaries.info(msg)
            
            # 加载数据
            dataset_train, dataset_validation, dataset_test = load_data(load_params_i)
            
            # Apply domain shift transform if specified
            if hasattr(ttt_args, 'test_transform') and ttt_args.test_transform != "none":
                transform_args = {
                    "noise_sigma": getattr(ttt_args, 'noise_sigma', 0.2),
                    "channel_noise_sigma": getattr(ttt_args, 'channel_noise_sigma', 0.1),
                    "scale_min": getattr(ttt_args, 'scale_min', 0.8),
                    "scale_max": getattr(ttt_args, 'scale_max', 1.2),
                }
                dataset_test = apply_domain_shift_transform(dataset_test, ttt_args.test_transform, transform_args)

    # === 修复后的KL散度分析 ===
    # 现在在应用transform之后重新计算KL散度
    if ttt_args and hasattr(ttt_args, 'test_transform'):
        transform_type = getattr(ttt_args, 'test_transform', 'none')
        transform_args = {
            'noise_sigma': getattr(ttt_args, 'noise_sigma', 0.2),
            'channel_noise_sigma': getattr(ttt_args, 'channel_noise_sigma', 0.1),
            'scale_min': getattr(ttt_args, 'scale_min', 0.8),
            'scale_max': getattr(ttt_args, 'scale_max', 1.2)
        }
        
        # 提取numpy数据进行分析
        def extract_numpy_data(dataset):
            X_list, y_list = [], []
            for batch in dataset:
                X_list.append(batch['X'].numpy().transpose(0, 2, 1))  # 转换为 (batch, seq_len, channels)
                y_list.append(batch['y'].numpy())
            return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
        
        try:
            # 重新提取transform后的数据
            Xs_test_transformed, ys_test_transformed = extract_numpy_data(dataset_test)
            
            # 使用增强的分析函数
            train_stats_fixed, train_dist_fixed, _ = analyze_dataset_stats_enhanced(
                Xs_train, ys_train, 'Training (Fixed)', 'none', None)
            val_stats_fixed, val_dist_fixed, _ = analyze_dataset_stats_enhanced(
                Xs_validation, ys_validation, 'Validation (Fixed)', 'none', None)
            test_stats_fixed, test_dist_fixed, _ = analyze_dataset_stats_enhanced(
                Xs_test_transformed, ys_test_transformed, f'Test (Fixed with {transform_type})', 'none', None)
            
            # 计算修复后的KL散度
            kl_results_fixed = compute_kl_divergences(
                train_stats_fixed, val_stats_fixed, test_stats_fixed,
                train_dist_fixed, val_dist_fixed, test_dist_fixed)
            
            # 打印修复后的结果
            msg_fixed = f'\n=== FIXED KL Divergence Analysis (after {transform_type} transform) ==='
            msg_fixed += f'\nClass Distribution KL Divergences:'
            msg_fixed += f'\n  Train-Val: {kl_results_fixed["class_kl"]["train_val"]:.4f}'
            msg_fixed += f'\n  Train-Test: {kl_results_fixed["class_kl"]["train_test"]:.4f}'
            msg_fixed += f'\n  Val-Test: {kl_results_fixed["class_kl"]["val_test"]:.4f}'
            msg_fixed += f'\nSignal Distribution KL Divergences:'
            msg_fixed += f'\n  Train-Val: {kl_results_fixed["signal_kl"]["train_val"]:.4f}'
            msg_fixed += f'\n  Train-Test: {kl_results_fixed["signal_kl"]["train_test"]:.4f}'
            msg_fixed += f'\n  Val-Test: {kl_results_fixed["signal_kl"]["val_test"]:.4f}'
            print(msg_fixed)
            paths.run.logger.summaries.info(msg_fixed)
            
            # 可选：生成可视化图
            try:
                viz_dir = os.path.join(paths.run.run_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                
                plot_path = create_distribution_plots(
                    Xs_train, Xs_validation, Xs_test_transformed,
                    ys_train, ys_validation, ys_test_transformed,
                    transform_type, viz_dir, cls_part, seed)
                
                print(f'Visualization saved: {plot_path}')
            except Exception as viz_e:
                print(f'Warning: Visualization failed: {viz_e}')
            
        except Exception as e:
            print(f'Warning: Fixed KL divergence analysis failed: {e}')
            print('Continuing with original analysis...')
    
        if hasattr(ttt_args, 'test_transform') and ttt_args.test_transform != "none":
            print(f"Applied domain shift transform: {ttt_args.test_transform}")
        
        # 模型初始化和加载
        model = duin_model(params.model)
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        log = model.load_state_dict(ckpt, strict=True)
        print("Load state dict result:", log)
        model = model.to(device=device)
        
        # 测试
        accs, all_test_pred, all_test_true = test_model(model, dataset_test, device)
        print("Test accuracy by subject:", accs)
        print("Mean test accuracy: {:.4f}".format(np.mean(accs)))
        # logger 写入
        paths.run.logger.summaries.info("Test accuracy by subject: {}".format(accs))
        paths.run.logger.summaries.info("Mean test accuracy: {:.4f}".format(np.mean(accs)))
        
        # TTT后处理 - 使用改进的TENT实现
        print("\n=== TTT (TENT) Results ===")
        ttt_accs = _ttt(dataset_test, model, device, params, load_params_i)
        print("TTT (Cumulative) accuracy by subject:", ttt_accs)
        print("Mean TTT (Cumulative) accuracy: {:.4f}".format(np.mean(ttt_accs)))
        
        # 可选：对比不同的TTT策略
        if ttt_compare:
            print("\n=== Comparing TTT Strategies ===")
            
            # 重新加载模型以确保公平比较
            model_copy = duin_model(params.model)
            model_copy.load_state_dict(ckpt, strict=True)
            model_copy = model_copy.to(device=device)
            
            ttt_per_batch_accs = _ttt_variants(dataset_test, model_copy, device, params, load_params_i, variant="per_batch")
            print("TTT (Per-batch reset) accuracy by subject:", ttt_per_batch_accs)
            print("Mean TTT (Per-batch reset) accuracy: {:.4f}".format(np.mean(ttt_per_batch_accs)))
            
            # 比较结果
            print("\n=== TTT Strategy Comparison ===")
            print(f"Baseline (no adaptation):     {np.mean(accs):.4f}")
            print(f"TTT Cumulative (TENT):        {np.mean(ttt_accs):.4f} (improvement: {np.mean(ttt_accs) - np.mean(accs):+.4f})")
            print(f"TTT Per-batch reset:          {np.mean(ttt_per_batch_accs):.4f} (improvement: {np.mean(ttt_per_batch_accs) - np.mean(accs):+.4f})")
            
            # logger 写入
            paths.run.logger.summaries.info("TTT (Per-batch) accuracy by subject: {}".format(ttt_per_batch_accs))
            paths.run.logger.summaries.info("Mean TTT (Per-batch) accuracy: {:.4f}".format(np.mean(ttt_per_batch_accs)))
            
            return accs, ttt_accs, ttt_per_batch_accs
        else:
            # logger 写入
            paths.run.logger.summaries.info("TTT (Cumulative) accuracy by subject: {}".format(ttt_accs))
            paths.run.logger.summaries.info("Mean TTT (Cumulative) accuracy: {:.4f}".format(np.mean(ttt_accs)))
            
            return accs, ttt_accs

if __name__ == "__main__":
    from params.duin_params import duin_cls_params as duin_params
    dataset = "seeg_he2023xuanwu"
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    
    # 首先尝试从checkpoint目录读取训练时的参数
    ckpt_dir = os.path.dirname(args.ckpt)
    params_path = os.path.join(ckpt_dir, "..", "save", "params")
    if os.path.exists(params_path):
        print(f"Loading training parameters from {params_path}")
        try:
            saved_params = load_pickle(params_path)
            duin_params_inst = saved_params
            # 更新必要的路径参数
            duin_params_inst.train.base = base
            duin_params_inst.train.subjs = args.subjs
            duin_params_inst.train.subj_idxs = args.subj_idxs
            
            # 确保关键参数存在
            if not hasattr(duin_params_inst, '_precision'):
                duin_params_inst._precision = 'float32'
            if not hasattr(duin_params_inst.model, 'isMDM'):
                duin_params_inst.model.isMDM = False
            if not hasattr(duin_params_inst.model, 'isDistill'):
                duin_params_inst.model.isDistill = False
                
            print("Successfully loaded training parameters")
        except Exception as e:
            print(f"Failed to load training parameters: {e}")
            print("Using default parameters instead")
            duin_params_inst = duin_params(dataset=dataset)
            duin_params_inst.train.base = base
            duin_params_inst.train.subjs = args.subjs
            duin_params_inst.train.subj_idxs = args.subj_idxs
            duin_params_inst.train.pt_ckpt = args.pt_ckpt
            duin_params_inst.model.isMDM = False
            duin_params_inst.model.isDistill = False
    else:
        print("Training parameters not found, using default parameters")
        duin_params_inst = duin_params(dataset=dataset)
        duin_params_inst.train.base = base
        duin_params_inst.train.subjs = args.subjs
        duin_params_inst.train.subj_idxs = args.subj_idxs
        duin_params_inst.train.pt_ckpt = args.pt_ckpt
        duin_params_inst.model.isMDM = False
        duin_params_inst.model.isDistill = False
    
    init(duin_params_inst, foldName=args.foldName)
    
    # 与训练脚本完全一致的循环结构
    for seed_i in args.seeds:
        print(f"\n=== Testing with seed {seed_i} ===")
        if args.ttt_compare:
            accs, ttt_accs, ttt_per_batch_accs = test_with_seed(seed_i, args.cls_part, args.use_senet, args.all_channels, args.ckpt, args.ttt_compare, args)
        else:
            accs, ttt_accs = test_with_seed(seed_i, args.cls_part, args.use_senet, args.all_channels, args.ckpt, args.ttt_compare, args)