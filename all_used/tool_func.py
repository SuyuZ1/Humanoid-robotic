import math
import torch
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import pickle
import gzip, pickle

# 角度归一化
def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

# 把四元数（xyzw）转 RPY 欧拉角
def euler_from_quat(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # q: (..., 4) in (x, y, z, w)
    x, y, z, w = q.unbind(-1)
    # roll (x-axis)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    # pitch (y-axis)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)
    # yaw (z-axis)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return roll, pitch, yaw

# 随机四舍五入，但是不严格，为了解决一些系统偏差，比如实际硬件只能支持每秒3.7帧，
# 但是随机3帧和4帧避免出现系统偏差。

def sample_int_from_float(steps_float: torch.Tensor) -> torch.Tensor:
    # steps_float: [nenv] — 可以是浮点，比如 2.7 这样的值

    # 计算向下取整值（floor） 和  计算小数部分 frac  同时生成均匀采样 U ∼ U[0,1)
    flo = torch.floor(steps_float).to(torch.long)  
    frac = (steps_float - flo.to(steps_float.dtype)).clamp(0.0, 1.0) 
    u = torch.rand_like(steps_float)  

    # 根据 u < frac 决定是否加 1
    up = (u < frac).to(torch.long)  # 0 或 1

    out = flo + up  # 随机向上取整的结果 [nenv], torch.LongTensor

    # 确保至少为 1（如果 steps_float < 1 时防止为 0）
    out = torch.clamp(out, min=1)

    return out

def _is_zip(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"  # npz/zip 

def _try_load_npz(path: str):
    try:
        obj = np.load(path, allow_pickle=True)
        if hasattr(obj, "files"):
            return True, obj
        obj.close()
    except Exception:
        pass
    return False, None

def _try_load_joblib(path: str):
    try:
        import joblib
        return True, joblib.load(path)
    except Exception:
        return False, None

def _try_load_pickle_or_gzip(path: str):
    try:
        with open(path, "rb") as f:
            return True, pickle.load(f)
    except Exception:
        pass
    try:
        with gzip.open(path, "rb") as f:
            return True, pickle.load(f)
    except Exception:
        return False, None

def _group_npz_as_sequences(npz_obj) -> Dict[str, Dict[str, np.ndarray]]:
    seqs: Dict[str, Dict[str, np.ndarray]] = {}
    for k in npz_obj.files:
        if "/" in k:
            a, b = k.split("/", 1)
            # 支持 "seq/field" 或 "field/seq" 两种命名
            if a in {"dof","root_trans_offset","root_rot","pose_aa","fps"}:
                field, seq = a, b
            elif b in {"dof","root_trans_offset","root_rot","pose_aa","fps"}:
                seq, field = a, b
            else:
                continue
            seqs.setdefault(seq, {})[field] = npz_obj[k]
    return seqs

def _align_trim_to_min(seq_list: List[torch.Tensor]) -> Tuple[torch.Tensor, int, torch.Tensor]:
    T = min(x.shape[0] for x in seq_list)
    seq_list = [x[:T] for x in seq_list]
    out = torch.stack(seq_list, dim=0)                      # [B, T, ...]
    mask = torch.ones((len(seq_list), T), dtype=torch.bool, device=out.device)
    return out, T, mask

def _align_pad_last(seq_list: List[torch.Tensor]) -> Tuple[torch.Tensor, int, torch.Tensor]:
    B = len(seq_list)
    T = max(x.shape[0] for x in seq_list)
    shape_tail = seq_list[0].shape[1:]
    out, mask = [], torch.zeros((B, T), dtype=torch.bool, device=seq_list[0].device)
    for b, x in enumerate(seq_list):
        L = x.shape[0]
        mask[b, :L] = True
        if L == T:
            out.append(x)
        else:
            pad = x[-1:].expand((T - L,) + shape_tail)     # 用最后一帧复制补齐
            out.append(torch.cat([x, pad], dim=0))
    return torch.stack(out, dim=0), T, mask

def _align_list_and_keep_tensor(seq_list: List[torch.Tensor]) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    为了“list 不报错”，生成一个可 .shape 的张量（默认用 pad_last 逻辑）与 mask，
    同时会把原始 list 也放进 batch["dof_list"] 等字段里。
    """
    return _align_pad_last(seq_list)

def load_all_from_any(
    device: torch.device,
    path: str,
    dof_offset: Optional[torch.Tensor] = None,
    seq_names: Optional[List[str]] = None,
    stack: str = "trim_to_min",   # "trim_to_min" | "pad_last" | "list"
):
    # 1) 识别并加载
    data_obj = None
    if _is_zip(path):
        ok, obj = _try_load_npz(path)
        if ok:
            data_obj = ("npz", obj)
    if data_obj is None:
        ok, obj = _try_load_joblib(path)
        if ok:
            data_obj = ("joblib", obj)
    if data_obj is None:
        ok, obj = _try_load_pickle_or_gzip(path)
        if ok:
            data_obj = ("pickle", obj)
    if data_obj is None:
        raise RuntimeError("无法识别文件格式（非 npz/joblib/pickle），或内容结构与预期不符")

    kind, obj = data_obj
    if kind == "npz":
        data = _group_npz_as_sequences(obj)
        obj.close()
    else:
        data = obj

    if not (isinstance(data, dict) and data and isinstance(next(iter(data.values())), dict)):
        raise TypeError("加载成功，但内容不是 {seq: {field: ...}} 结构")

    # 2) 选择序列
    keys_all = list(data.keys())
    keys = keys_all if seq_names is None else [k for k in seq_names if k in data]
    if not keys:
        raise ValueError("没有匹配到任何序列键")

    # 3) 收集并转 tensor
    dof_list, rt_list, rr_list, paa_list, fps_list, T_list = [], [], [], [], [], []
    for k in keys:
        d = data[k]
        dof = torch.as_tensor(d["dof"], dtype=torch.float32, device=device)                 # [T,19]
        rt  = torch.as_tensor(d["root_trans_offset"], dtype=torch.float32, device=device)   # [T,3]
        rr  = torch.as_tensor(d["root_rot"], dtype=torch.float32, device=device)            # [T,4]
        paa = torch.as_tensor(d["pose_aa"], dtype=torch.float32, device=device)             # [T,22,3]
        if dof_offset is not None:
            dof = dof + dof_offset.view(1, -1)  # 广播加法（PyTorch broadcasting）
        dof_list.append(dof); rt_list.append(rt); rr_list.append(rr); paa_list.append(paa)
        fps_list.append(int(d.get("fps", 30))); T_list.append(dof.shape[0])

    lengths = torch.tensor(T_list, device=device, dtype=torch.long)
    fps_t   = torch.tensor(fps_list, device=device, dtype=torch.int32)

    # 4) 对齐策略（统一返回：张量 + mask；并在 list 模式额外返回 *_list）
    if stack == "trim_to_min":
        dof, T_policy, mask = _align_trim_to_min(dof_list)
        root_trans, _, _ = _align_trim_to_min(rt_list)
        root_rot,   _, _ = _align_trim_to_min(rr_list)
        pose_aa,    _, _ = _align_trim_to_min(paa_list)
        extra_lists = {}
    elif stack == "pad_last":
        dof, T_policy, mask = _align_pad_last(dof_list)
        root_trans, _, _ = _align_pad_last(rt_list)
        root_rot,   _, _ = _align_pad_last(rr_list)
        pose_aa,    _, _ = _align_pad_last(paa_list)
        extra_lists = {}
    elif stack == "list":
        # 仍返回一个可 .shape 的张量（pad_last）+ mask，并保留原始 list
        dof, T_policy, mask = _align_list_and_keep_tensor(dof_list)
        root_trans, _, _ = _align_list_and_keep_tensor(rt_list)
        root_rot,   _, _ = _align_list_and_keep_tensor(rr_list)
        pose_aa,    _, _ = _align_list_and_keep_tensor(paa_list)
        extra_lists = {
            "dof_list": dof_list,
            "root_trans_list": rt_list,
            "root_rot_list": rr_list,
            "pose_aa_list": paa_list,
        }
    else:
        raise ValueError("stack 必须是 'trim_to_min' | 'pad_last' | 'list'")

    batch = {
        "dof": dof,                       # [B, T, 19] —— 即使在 list 模式也可 .shape
        "root_trans": root_trans,         # [B, T, 3]
        "root_rot": root_rot,             # [B, T, 4]
        "pose_aa": pose_aa,               # [B, T, 22, 3]
        "mask": mask,                     # [B, T]  True=有效帧
        "fps": fps_t,                     # [B]
        "names": keys,                    # list[str]
        **extra_lists,                    # 在 list 模式下多出 *_list
    }
    meta = {"names": keys, "T_policy": T_policy, "stack": stack}
    return batch, lengths, meta
