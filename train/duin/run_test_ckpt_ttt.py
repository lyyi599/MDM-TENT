#!/usr/bin/env python3
"""
Test DuIN CLS model from a specified checkpoint with TTT (TENT only).
"""
import torch
import os, argparse, copy as cp
import numpy as np
# MSP-based OOD检测相关导入
import scipy as sp
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import save_pickle, load_pickle
from models.duin import duin_cls as duin_model

def get_args_parser():
    parser = argparse.ArgumentParser("DuIN CLS Test from checkpoint (TTT only)", add_help=False)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,])
    parser.add_argument("--subjs", type=str, nargs="+", default=["011",])
    parser.add_argument("--subj_idxs", type=int, nargs="+", default=[0,])
    parser.add_argument("--pt_ckpt", type=str, default=None)
    parser.add_argument("--foldName", type=str, default=None)
    parser.add_argument("--cls_part", type=str, default='_vocal', choices=['_vocal', '_mimed', '_imagined'], help="choose subj13's cls part for downstream")
    parser.add_argument("--use_senet", type=int, default=None, help="senet")
    parser.add_argument("--all_channels", action='store_true', help="use all channels")
    
    # Domain shift / Transform arguments
    parser.add_argument("--test_transform", type=str, default="none", 
                       choices=["none", "gaussian_noise", "scaling", "shift"],
                       help="Type of domain shift transform to apply")
    parser.add_argument("--transform_strength", type=str, default="medium",
                       choices=["light", "medium", "heavy"],
                       help="Strength of the transform")
    parser.add_argument("--noise_sigma", type=float, default=0.2,
                       help="Standard deviation for Gaussian noise")
    parser.add_argument("--scale_min", type=float, default=0.8,
                       help="Minimum scaling factor")
    parser.add_argument("--scale_max", type=float, default=1.2,
                       help="Maximum scaling factor")
    parser.add_argument("--shift_max", type=int, default=10, help="Max shift steps for shift transform")
    
    # TTT hyperparameters
    parser.add_argument("--ttt_lr", type=float, default=1e-3, help="TTT learning rate")
    parser.add_argument("--ttt_momentum", type=float, default=0.9, help="TTT momentum for SGD")
    parser.add_argument("--ttt_optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="TTT optimizer type")
    parser.add_argument("--ttt_steps", type=int, default=1, help="Number of gradient steps per batch")
    parser.add_argument("--ttt_warmup", type=int, default=0, help="Number of warmup batches before adaptation")
    parser.add_argument("--ttt_entropy_weight", type=float, default=1.0, help="Weight for entropy loss")
    parser.add_argument("--ttt_temperature", type=float, default=1.0, help="Temperature for softmax in entropy calculation")
    
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
        func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))
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
    # 这边的数据集只需要用来推理，不需要进行数据增强，但是后续需要根据选择来决定四种转换的其中一种
    dataset_train = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_train, ys_train, subj_ids_train)], use_aug=False)
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
    
    return dataset_train, dataset_validation, dataset_test

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

def collect_logits_and_compute_accuracy(model, dataset, device):
    """收集模型在数据集上的logits并同时计算准确率"""
    model.eval()
    all_logits = []
    all_preds = []
    all_true_labels = []
    all_subj_ids = []
    
    with torch.no_grad():
        for batch in dataset:
            batch_i = [
                batch["X"].to(device=device),
                batch["y"].to(device=device),
                batch["subj_id"].to(device=device),
            ]
            logits, _, _ = model(batch_i)
            
            # 收集logits
            all_logits.append(logits.cpu().numpy())
            
            # 收集预测和真实标签
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = torch.argmax(batch_i[1], dim=-1).cpu().numpy()
            subj_ids = torch.argmax(batch_i[2], dim=-1).cpu().numpy()
            
            all_preds.append(preds)
            all_true_labels.append(true_labels)
            all_subj_ids.append(subj_ids)
    
    # 拼接所有结果
    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_subj_ids = np.concatenate(all_subj_ids, axis=0)
    
    # 计算整体准确率
    overall_acc = (all_preds == all_true_labels).mean()
    
    # 按subject计算准确率
    accuracy_per_subj = []
    subj_ids = sorted(set(all_subj_ids))
    for subj_idx in subj_ids:
        subj_mask = all_subj_ids == subj_idx
        subj_acc = (all_preds[subj_mask] == all_true_labels[subj_mask]).mean()
        accuracy_per_subj.append(subj_acc)
    
    return all_logits, overall_acc, accuracy_per_subj

def _ttt(dataset_test, model, device, params, ttt_args):
    """
    TENT: Test-Time Adaptation by Entropy Minimization
    """
    # 关键：保持模型在eval模式，这样BN层不会更新running statistics
    model.eval()
    
    # 冻结所有参数，只保留BN仿射参数可训练
    for n, p in model.named_parameters():
        p.requires_grad = False
    
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
    
    bn_params = collect_bn_params(model)
    
    print(f"TTT (TENT): Found {len(bn_params)} BN parameters to optimize")
    print(f"TTT (TENT): Model in eval mode, only updating BN affine parameters")
    
    # 获取TTT参数，提供默认值
    lr = getattr(ttt_args, 'ttt_lr', 1e-3)
    momentum = getattr(ttt_args, 'ttt_momentum', 0.9)
    temperature = getattr(ttt_args, 'ttt_temperature', 1.0)
    entropy_weight = getattr(ttt_args, 'ttt_entropy_weight', 1.0)
    optimizer_type = getattr(ttt_args, 'ttt_optimizer', 'sgd')
    steps = getattr(ttt_args, 'ttt_steps', 1)
    warmup = getattr(ttt_args, 'ttt_warmup', 0)
    
    # 创建优化器
    if optimizer_type == "sgd":
        tent_optimizer = torch.optim.SGD(bn_params, lr=lr, momentum=momentum)
    elif optimizer_type == "adam":
        tent_optimizer = torch.optim.Adam(bn_params, lr=lr)
    elif optimizer_type == "adamw":
        tent_optimizer = torch.optim.AdamW(bn_params, lr=lr)
    
    accuracy_test = []
    all_logits = []
    total_loss = 0.0
    batch_count = 0
    
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
                all_logits.append(logits.cpu().numpy())
            continue
        
        # 多步优化
        for step in range(steps):
            # Forward pass
            logits, _, _ = model(batch_i)
            
            # 计算适应损失
            loss = entropy_minimize_loss(logits, temperature) * entropy_weight
            
            if step == 0:  # 只在第一步记录损失
                total_loss += loss.item()
                batch_count += 1
            
            # Backward pass and optimization
            tent_optimizer.zero_grad()
            loss.backward()
            tent_optimizer.step()
        
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
            all_logits.append(logits_updated.cpu().numpy())
    
    # 计算最终准确率
    accuracy_test = np.concatenate(accuracy_test, axis=0)
    subj_ids = sorted(set(accuracy_test[:,1]))
    accs = [accuracy_test[np.where(accuracy_test[:,1] == subj_idx),0].mean() for subj_idx in subj_ids]
    
    # 拼接所有logits
    all_logits = np.concatenate(all_logits, axis=0)
    
    # 打印详细的适应统计信息
    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"TTT Summary: Average entropy loss: {avg_loss:.6f}")
    
    return accs, all_logits

def apply_domain_shift_transform(dataset, transform_type, transform_args):
    """Apply domain shift transforms to the dataset"""
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
            
        elif transform_type == "scaling":
            scale_min = transform_args.get("scale_min", 0.8)
            scale_max = transform_args.get("scale_max", 1.2)
            # Random scaling factor for each sample
            batch_size = X.shape[0]
            scale_factors = torch.rand(batch_size, 1, 1) * (scale_max - scale_min) + scale_min
            X_transformed = X * scale_factors
            print(f"Applied amplitude scaling [{scale_min}, {scale_max}]")
            
        elif transform_type == "shift":
            shift_max = transform_args.get("shift_max", 10)
            X_transformed = torch.zeros_like(X)
            batch_size = X.shape[0]
            for i in range(batch_size):
                shift = torch.randint(-shift_max, shift_max+1, (1,)).item()
                if shift > 0:
                    X_transformed[i, :, shift:] = X[i, :, :-shift]
                elif shift < 0:
                    X_transformed[i, :, :shift] = X[i, :, -shift:]
                else:
                    X_transformed[i] = X[i]
            print(f"Applied temporal shift with max_shift={shift_max}")
            
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



def compute_msp_kl_divergence(msp1, msp2):
    """计算两个MSP分布之间的KL散度"""
    from scipy.stats import entropy
    
    # 创建直方图
    bins = np.linspace(0, 1, 51)  # 50个bins，范围0-1
    hist1, _ = np.histogram(msp1, bins=bins, density=True)
    hist2, _ = np.histogram(msp2, bins=bins, density=True)
    
    # 添加小量避免log(0)
    epsilon = 1e-10
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon
    
    # 归一化
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # 计算KL散度
    kl_div = entropy(hist1, hist2)
    
    return kl_div

def compute_msp_statistics(logits, dataset_name):
    """计算MSP统计"""
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    max_probs = np.max(probs, axis=-1)
    
    stats = {
        'mean': np.mean(max_probs),
        'std': np.std(max_probs),
        'min': np.min(max_probs),
        'max': np.max(max_probs),
        'median': np.median(max_probs),
        'entropy': -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
    }
    
    print(f"{dataset_name} MSP Statistics:")
    print(f"  Mean max prob: {stats['mean']:.4f}")
    print(f"  Std max prob: {stats['std']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean entropy: {stats['entropy']:.4f}")
    
    return max_probs, stats

def test_with_seed(seed, cls_part, use_senet, all_channels, ckpt_path, ttt_args=None):
    """
    Test the model with a specific seed
    """
    global params, paths
    
    # 设置随机种子
    seed=seed
    
    # 初始化设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params.model.device = device
    print(params.model.device); paths.run.logger.summaries.info(params.model.device)
    
    # 与训练脚本完全一致的subject配置
    if params.train.dataset == "seeg_he2023xuanwu":
        duin_path = '/data/seeg/liyangyang/duin'
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
        
        # 加载数据
        for load_params_idx in range(len(load_params)):
            load_params_i = cp.deepcopy(load_params[load_params_idx]); load_params_i.subjs_cfg = subjs_cfg
            load_params_i.cls_part = cls_part  # 添加cls_part参数
            
            msg = (
                "Testing started with experiment {} with {:d} subjects."
            ).format(load_params_i.name, len(load_params_i.subjs_cfg))
            print(msg); paths.run.logger.summaries.info(msg)
            
            # 加载数据
            dataset_train, dataset_validation, dataset_test = load_data(load_params_i)
            
            # 应用domain shift transform (如果有)
            if ttt_args and hasattr(ttt_args, 'test_transform') and ttt_args.test_transform != "none":
                transform_type = getattr(ttt_args, 'test_transform', 'none')
                transform_args = {
                    'noise_sigma': getattr(ttt_args, 'noise_sigma', 0.2),
                    'scale_min': getattr(ttt_args, 'scale_min', 0.8),
                    'scale_max': getattr(ttt_args, 'scale_max', 1.2),
                    'shift_max': getattr(ttt_args, 'shift_max', 10)
                }
                dataset_test = apply_domain_shift_transform(dataset_test, transform_type, transform_args)
            
            # 模型初始化和加载
            model = duin_model(params.model)
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            log = model.load_state_dict(ckpt, strict=True)
            print("Load state dict result:", log)
            model = model.to(device=device)
            
            # 步骤2：收集原始ckpt的logits并计算准确率
            print("\n=== Collecting ckpt logits and computing accuracies ===")
            train_logits, train_acc, train_acc_per_subj = collect_logits_and_compute_accuracy(model, dataset_train, device)
            val_logits, val_acc, val_acc_per_subj = collect_logits_and_compute_accuracy(model, dataset_validation, device)
            test_logits, test_acc, test_acc_per_subj = collect_logits_and_compute_accuracy(model, dataset_test, device)
            
            print(f"Train accuracy: {train_acc:.4f}")
            print(f"Val accuracy: {val_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
            
            # 保存ckpt logits
            ckpt_logits = {
                'train': train_logits,
                'val': val_logits,
                'test': test_logits
            }
            save_pickle(os.path.join(paths.run.save, f'ckpt_logits_seed{seed}.pkl'), ckpt_logits)
            print(f"Saved ckpt logits to ckpt_logits_seed{seed}.pkl")
            
            # 步骤3：TTT (TENT) 测试
            print("\n=== TTT (TENT) Results ===")
            ttt_accs, ttt_logits = _ttt(dataset_test, model, device, params, ttt_args or utils.DotDict())
            print("TTT accuracy by subject:", ttt_accs)
            print("Mean TTT accuracy: {:.4f}".format(np.mean(ttt_accs)))
            
            # 保存ttt logits
            ttt_logits_dict = {
                'test': ttt_logits
            }
            save_pickle(os.path.join(paths.run.save, f'ttt_logits_seed{seed}.pkl'), ttt_logits_dict)
            print(f"Saved ttt logits to ttt_logits_seed{seed}.pkl")
            
            # 步骤4：MSP分布分析
            print("\n=== MSP-based Distribution Analysis ===")
            
            # 计算原始ckpt的MSP统计
            train_msp, train_stats = compute_msp_statistics(train_logits, "Training")
            val_msp, val_stats = compute_msp_statistics(val_logits, "Validation")
            test_msp, test_stats = compute_msp_statistics(test_logits, "Test (before TTT)")
            
            # 计算TTT后的MSP统计
            ttt_test_msp, ttt_test_stats = compute_msp_statistics(ttt_logits, "Test (after TTT)")
            
            # 计算原始ckpt的MSP KL散度
            print("\n=== Computing MSP KL Divergences (before TTT) ===")
            kl_train_val = compute_msp_kl_divergence(train_msp, val_msp)
            kl_train_test = compute_msp_kl_divergence(train_msp, test_msp)
            kl_val_test = compute_msp_kl_divergence(val_msp, test_msp)
            
            print(f"MSP KL divergences (before TTT):")
            print(f"  Train -> Val: {kl_train_val:.4f}")
            print(f"  Train -> Test: {kl_train_test:.4f}")
            print(f"  Val -> Test: {kl_val_test:.4f}")
            
            # 计算TTT后的MSP KL散度
            print("\n=== Computing MSP KL Divergences (after TTT) ===")
            kl_train_val_after = compute_msp_kl_divergence(train_msp, val_msp)  # 保持不变，用于对比
            kl_train_ttt_test = compute_msp_kl_divergence(train_msp, ttt_test_msp)
            kl_val_ttt_test = compute_msp_kl_divergence(val_msp, ttt_test_msp)
            
            print(f"MSP KL divergences (after TTT):")
            print(f"  Train -> Val: {kl_train_val_after:.4f}")
            print(f"  Train -> TTT_Test: {kl_train_ttt_test:.4f}")
            print(f"  Val -> TTT_Test: {kl_val_ttt_test:.4f}")
            
            # logger 写入 - 准确率
            paths.run.logger.summaries.info(f"Train accuracy: {train_acc:.4f}")
            paths.run.logger.summaries.info(f"Val accuracy: {val_acc:.4f}")
            paths.run.logger.summaries.info(f"Test accuracy: {test_acc:.4f}")
            paths.run.logger.summaries.info("Train accuracy by subject: {}".format([f"{acc:.4f}" for acc in train_acc_per_subj]))
            paths.run.logger.summaries.info("Val accuracy by subject: {}".format([f"{acc:.4f}" for acc in val_acc_per_subj]))
            paths.run.logger.summaries.info("Test accuracy by subject: {}".format([f"{acc:.4f}" for acc in test_acc_per_subj]))
            
            # logger 写入 - TTT结果
            paths.run.logger.summaries.info("TTT accuracy by subject: {}".format(ttt_accs))
            paths.run.logger.summaries.info("Mean TTT accuracy: {:.4f}".format(np.mean(ttt_accs)))
            
            # logger 写入 - MSP统计
            paths.run.logger.summaries.info(f"Train MSP - mean: {train_stats['mean']:.4f}, std: {train_stats['std']:.4f}")
            paths.run.logger.summaries.info(f"Val MSP - mean: {val_stats['mean']:.4f}, std: {val_stats['std']:.4f}")
            paths.run.logger.summaries.info(f"Test MSP (before TTT) - mean: {test_stats['mean']:.4f}, std: {test_stats['std']:.4f}")
            paths.run.logger.summaries.info(f"Test MSP (after TTT) - mean: {ttt_test_stats['mean']:.4f}, std: {ttt_test_stats['std']:.4f}")
            
            # logger 写入 - MSP KL散度 (before TTT)
            paths.run.logger.summaries.info(f"MSP KL divergences (before TTT) - Train->Val: {kl_train_val:.4f}, Train->Test: {kl_train_test:.4f}, Val->Test: {kl_val_test:.4f}")
            
            # logger 写入 - MSP KL散度 (after TTT)
            paths.run.logger.summaries.info(f"MSP KL divergences (after TTT) - Train->Val: {kl_train_val_after:.4f}, Train->TTT_Test: {kl_train_ttt_test:.4f}, Val->TTT_Test: {kl_val_ttt_test:.4f}")
            
            # logger 写入 - TTT前后的KL散度对比分析
            print(f"\n=== TTT Impact Analysis ===")
            print(f"KL divergence changes after TTT:")
            print(f"  Train->Test: {kl_train_test:.4f} -> {kl_train_ttt_test:.4f} (change: {kl_train_ttt_test - kl_train_test:+.4f})")
            
            paths.run.logger.summaries.info(f"TTT Impact - Train->Test KL change: {kl_train_test:.4f} -> {kl_train_ttt_test:.4f} (delta: {kl_train_ttt_test - kl_train_test:+.4f})")
            
            return ttt_accs

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
        utils.model.torch.set_seeds(seed_i)
        ttt_accs = test_with_seed(seed_i, args.cls_part, args.use_senet, args.all_channels, args.ckpt, args)

