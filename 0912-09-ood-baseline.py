"""
0912-09-ood-baseline.py
================================================================================
对照实验: 特征空间 OOD 警报 (朴素基线) vs MACP 预测空间坍塌警报

动机 (回应 "报警 vs 简单 OOD 检测器" 的质疑):
  概念漂移在特征空间隐身 (测试点在校准支撑内, 特征 MMD 温和), 但在预测空间现身
  (模型输出坍缩为常数)。特征空间 OOD 检测器 (MMD on X, 无纯度门) 预期:
  - 对坍塌对漏检 (Czech->IPVS 特征 MMD 仅 0.21) -> coverage 停留在 0.56
  - 对良性偏移对误报 (跨语种特征分布必然不同) -> 过度弃权

协议与主实验完全一致 (同源划分/同 scaler/同 RF/同 10 seeds/同置换检验),
唯一区别是警报信号: 本脚本 = MMD(X_cal, X_test) 显著即弃权;
MACP = MMD(f(X_cal), f(X_test)) 显著 AND 纯度 > 0.9。

输出: 终端对照表 + alarm_comparison.csv
用法: python 0912-09-ood-baseline.py
================================================================================
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

candidates = sorted(Path(__file__).resolve().parent.glob('0912-02-MACP-v3*.py'),
                    key=lambda f: f.stat().st_mtime, reverse=True)
spec = importlib.util.spec_from_file_location("macp_v3", str(candidates[0]))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
print(f"Loading main module: {candidates[0].name}")

N_RUNS = 10


def alarms_for(Xtr, ytr, Xca, yca, Xte, seed):
    """同一 run 内计算两种警报的触发情况."""
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(Xtr, ytr)
    # 预测空间坍塌门限警报 (MACP)
    macp = M.MACP(M.MACPConfig(alpha=0.1))
    macp.fit(Xca, yca, clf)
    macp.predict(Xte)
    pred_alarm = bool(macp.drift_alarm)
    # 特征空间 MMD 警报 (朴素 OOD 基线, 无纯度门)
    probe = M.MACP(M.MACPConfig())  # 仅借用 _mmd_permutation
    p_feat = probe._mmd_permutation(Xca, Xte)
    feat_alarm = bool(p_feat < 0.05)
    return pred_alarm, feat_alarm


def main():
    loader = M.VoiceDataLoader()
    data = {}
    for name in ['Czech', 'Spanish', 'IPVS', 'MultiModel']:
        X, y, g = loader.load_dataset(name, max_samples=200)
        if len(X) > 0:
            data[name] = (X, y, np.array([f"{name}_{s}" for s in g]))

    rows = []

    def run_block(label, X_src, y_src, g_src, X_te, y_te=None):
        pa, fa = [], []
        for run in range(N_RUNS):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.4,
                                    random_state=42 + run)
            tr, ca = next(gss.split(X_src, y_src, g_src))
            sc = StandardScaler().fit(X_src[tr])
            a, b = alarms_for(sc.transform(X_src[tr]), y_src[tr],
                              sc.transform(X_src[ca]), y_src[ca],
                              sc.transform(X_te), 42 + run)
            pa.append(a)
            fa.append(b)
        rows.append(dict(setting=label, kind='rate',
                         pred=f"{int(sum(pa))}/{len(pa)}",
                         feat=f"{int(sum(fa))}/{len(fa)}"))
        return sum(pa), sum(fa), len(pa)

    names = list(data)
    pairs = [(s, t) for s in names for t in names if s != t]
    print("\n--- Voice single-source pairs ---")
    collapsed = [('Czech', 'IPVS'), ('Czech', 'MultiModel')]
    for s, t in pairs:
        pa, fa, n = run_block(f"{s}->{t}",
                              data[s][0], data[s][1], data[s][2],
                              data[t][0])
        flag = ' [COLLAPSED]' if (s, t) in collapsed else ''
        print(f"  {s:>9} -> {t:<10} pred {pa}/{n}  feat {fa}/{n}{flag}")

    print("\n--- Voice multi-source ---")
    ms_collapsed = ['IPVS', 'MultiModel']
    for target in names:
        srcs = [n for n in names if n != target]
        Xs = np.vstack([data[n][0] for n in srcs])
        ys = np.concatenate([data[n][1] for n in srcs])
        gs = np.concatenate([data[n][2] for n in srcs])
        pa, fa, n = run_block(f"MS->{target}", Xs, ys, gs, data[target][0])
        flag = ' [COLLAPSED]' if target in ms_collapsed else ''
        print(f"  MS -> {target:<10} pred {pa}/{n}  feat {fa}/{n}{flag}")

    print("\n--- PPMI cross-site (1.5T -> 3T) ---")
    cache = Path(__file__).resolve().parent / 'feat_cache' / 'PPMI_resnet_extreme.pkl'
    with open(cache, 'rb') as f:
        X, y, meta = pickle.load(f)
    m15 = meta['domain'].values == '1.5T'
    m3 = meta['domain'].values == '3T'
    Xs, ys = X[m15], y[m15]
    pa, fa = [], []
    for run in range(N_RUNS):
        Xtr, Xca, ytr, yca = train_test_split(
            Xs, ys, test_size=0.4, random_state=42 + run, stratify=ys)
        sc = StandardScaler().fit(Xtr)
        a, b = alarms_for(sc.transform(Xtr), ytr, sc.transform(Xca), yca,
                          sc.transform(X[m3]), 42 + run)
        pa.append(a)
        fa.append(b)
    rows.append(dict(setting='PPMI cross-site', kind='rate',
                     pred=f"{int(sum(pa))}/{len(pa)}",
                     feat=f"{int(sum(fa))}/{len(fa)}"))
    print(f"  pred {sum(pa)}/{len(pa)}  feat {sum(fa)}/{len(fa)}")

    df = pd.DataFrame(rows)
    df.to_csv('alarm_comparison.csv', index=False)
    print("\nSaved alarm_comparison.csv")


if __name__ == "__main__":
    main()
