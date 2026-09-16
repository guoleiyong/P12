"""
0912-04-ppmi-diagnostic.py
================================================================================
PPMI 完整诊断脚本: 快速定位 MRI 实验瓶颈 (任务信号 vs 模型 vs 特征 vs 协议)

直接读取主实验的 MRI 特征缓存, 在 4 种 (模型 x 降维) 组合下对比:
  协议A: within-site 随机划分 (无分布偏移, 测任务本身信号)
  协议B: cross-site 1.5T -> 3T (真实扫描仪偏移)

输出: ppmi_diagnostic.csv + 终端对比表

用法: 与主文件同目录 (已跑过一次主实验生成 feat_cache/PPMI_resnet.pkl):
  python 0912-04-ppmi-diagnostic.py
================================================================================
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---- 动态导入主模块 (自动匹配最新版本) ----
candidates = sorted(Path(__file__).resolve().parent.glob('0912-02-MACP-v3*.py'),
                    key=lambda f: f.stat().st_mtime, reverse=True)
if not candidates:
    raise FileNotFoundError("未找到 0912-02-MACP-v3*.py")
MODULE_PATH = candidates[0]
print(f"Loading main module: {MODULE_PATH.name}")
spec = importlib.util.spec_from_file_location("macp_v3", str(MODULE_PATH))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

N_RUNS = 5
ALPHA = 0.1

MODELS = {
    'MLP+PCA64': dict(model_fn=lambda rs: MLPClassifier(hidden_layer_sizes=(64, 32),
                                                         max_iter=500, random_state=rs),
                      pca=64),
    'MLP raw':   dict(model_fn=lambda rs: MLPClassifier(hidden_layer_sizes=(64, 32),
                                                         max_iter=500, random_state=rs),
                      pca=None),
    'RF raw':    dict(model_fn=lambda rs: RandomForestClassifier(n_estimators=100,
                                                                  random_state=rs),
                      pca=None),
}


def run_one(X_tr, y_tr, X_ca, y_ca, X_te, y_te, cfg, seed):
    """单次训练+评估, 返回 (accuracy, naive_cov, macp_cov, macp_size)."""
    sc = StandardScaler().fit(X_tr)
    Xtr, Xca, Xte = sc.transform(X_tr), sc.transform(X_ca), sc.transform(X_te)
    if cfg['pca']:
        pca = PCA(n_components=min(cfg['pca'], Xtr.shape[0] - 1, Xtr.shape[1]),
                  random_state=seed)
        Xtr, Xca, Xte = pca.fit_transform(Xtr), pca.transform(Xca), pca.transform(Xte)
    clf = cfg['model_fn'](seed)
    clf.fit(Xtr, y_tr)
    acc = clf.score(Xte, y_te)
    auc = roc_auc_score(y_te, clf.predict_proba(Xte)[:, 1])

    naive = M.NaiveCP(alpha=ALPHA).fit(Xca, y_ca, clf)
    r_n = naive.evaluate(Xte, y_te)
    # 关门评估: 读出基座+conformal 的真实水平 (门的影响单独统计)
    macp = M.MACP(M.MACPConfig(alpha=ALPHA, drift_gate=False)).fit(Xca, y_ca, clf)
    r_m = macp.evaluate(Xte, y_te)
    # 门统计 (单独 fit 一次, 利用 predict 时的警报标记)
    macp_g = M.MACP(M.MACPConfig(alpha=ALPHA, drift_gate=True)).fit(Xca, y_ca, clf)
    macp_g.predict(Xte)
    alarm = macp_g.drift_alarm
    return (acc, auc, r_n['coverage'], r_m['coverage'], r_m['mean_set_size'],
            alarm, macp_g.mmd_p)


def main():
    import sys
    label_mode = sys.argv[1] if len(sys.argv) > 1 else 'median'
    assert label_mode in ('median', 'extreme', 'subtype', 'pdhc'), \
        "用法: python 0912-04-ppmi-diagnostic.py [median|extreme|subtype|pdhc]"

    # 缓存路径相对脚本自身解析 (兼容从其他目录运行)
    cache = Path(__file__).resolve().parent / 'feat_cache' / 'PPMI_resnet.pkl'
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} 不存在, 请先跑一次主实验生成 MRI 特征缓存")
    with open(cache, 'rb') as f:
        X, y, metadata = pickle.load(f)

    # 优先使用 partIII_baseline.csv (2026 新 dump, 条目级验证过) 的标签
    base_csv = Path(__file__).resolve().parent / 'partIII_baseline.csv'
    lut = None
    if base_csv.exists():
        lab = pd.read_csv(base_csv).dropna(subset=['PATNO'])
        lab = lab.drop_duplicates('PATNO')
        lut = lab.set_index(lab['PATNO'].astype(int))

    if label_mode == 'extreme':
        col = 'NP3TOT_new'
        metadata = metadata.copy()
        if lut is not None:
            metadata[col] = metadata['PATNO'].astype(int).map(lut['NP3TOT'])
        else:
            col = 'NP3TOT'
        q25, q75 = metadata[col].quantile([0.25, 0.75])
        mask = (metadata[col] <= q25) | (metadata[col] >= q75)
        X, metadata = X[mask.values], metadata[mask].reset_index(drop=True)
        y = (metadata[col] >= q75).astype(int).values
        print(f"Extreme labels ({'new dump' if lut is not None else 'cache'}): "
              f"Q25={q25:.0f}, Q75={q75:.0f}, n={len(y)} "
              f"(Mild={np.sum(y == 0)}, Severe={np.sum(y == 1)})")

    elif label_mode == 'subtype':
        if lut is None:
            raise FileNotFoundError(
                "partIII_baseline.csv 不存在, 请先运行 0912-06 脚本")
        st = metadata['PATNO'].astype(int).map(lut['subtype'])
        mask = st.isin(['TD', 'PIGD']).values
        X, metadata = X[mask], metadata[mask].reset_index(drop=True)
        y = (st[mask] == 'PIGD').astype(int).values
        print(f"Subtype labels: n={len(y)} "
              f"(TD={np.sum(y == 0)}, PIGD={np.sum(y == 1)})")

    elif label_mode == 'pdhc':
        # PD vs HC: NP3TOT >= 10 -> PD(1); NP3TOT == 0 -> HC(0); 中间剔除
        # (HC 的 Part III 得分≈0; 注意 prodromal 携带者可能得分也为 0, 论文中需声明)
        if lut is None:
            raise FileNotFoundError("partIII_baseline.csv 不存在, 请先运行 0912-06 脚本")
        score = metadata['PATNO'].astype(int).map(lut['NP3TOT'])
        mask = (score >= 10) | (score == 0)
        X, metadata = X[mask.values], metadata[mask.values].reset_index(drop=True)
        score = score[mask].reset_index(drop=True)
        y = (score >= 10).astype(int).values
        print(f"PD vs HC labels (NP3TOT>=10 vs ==0): n={len(y)} "
              f"(HC={np.sum(y == 0)}, PD={np.sum(y == 1)})")

    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} features (label={label_mode})")

    if 'domain' not in metadata.columns:
        raise ValueError("缓存的 metadata 缺少 'domain' 列, 请用主文件重跑一次 "
                         "(extract_scanners=True) 以写入扫描仪信息")

    rows = []
    for proto in ['within-site', 'cross-site']:
        for cfg_name, cfg in MODELS.items():
            for run in range(N_RUNS):
                seed = 42 + run
                if proto == 'within-site':
                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                        X, y, test_size=0.5, random_state=seed, stratify=y)
                    X_ca, X_te, y_ca, y_te = train_test_split(
                        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
                else:
                    m15 = metadata['domain'].values == '1.5T'
                    m3 = metadata['domain'].values == '3T'
                    X_s, y_s = X[m15], y[m15]
                    X_tr, X_ca, y_tr, y_ca = train_test_split(
                        X_s, y_s, test_size=0.4, random_state=seed, stratify=y_s)
                    X_te, y_te = X[m3], y[m3]

                acc, auc, cov_n, cov_m, size_m, alarm, mmd_p = run_one(
                    X_tr, y_tr, X_ca, y_ca, X_te, y_te, cfg, seed)
                rows.append(dict(protocol=proto, config=cfg_name, run=run,
                                 accuracy=acc, auc=auc, naive_cov=cov_n,
                                 macp_cov=cov_m, macp_size=size_m,
                                 alarm=alarm, mmd_p=mmd_p))
                flag = ' [ALARM]' if alarm else ''
                print(f"  [{proto:<11s}] {cfg_name:<10s} run{run}: "
                      f"acc={acc:.3f}  auc={auc:.3f}  "
                      f"naive={cov_n:.3f}  macp={cov_m:.3f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(f'ppmi_diagnostic_{label_mode}.csv', index=False)

    print("\n" + "=" * 78)
    print("SUMMARY (mean ± std over runs)")
    print("=" * 78)
    summary = df.groupby(['protocol', 'config']).agg(
        acc=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
        auc=('auc', 'mean'),
        macp=('macp_cov', 'mean'), naive=('naive_cov', 'mean'),
        size=('macp_size', 'mean')).reset_index()
    print(f"{'protocol':<12} {'config':<10} {'accuracy':<16} {'AUC':<10} "
          f"{'MACP cov':<12} {'Naive cov':<12} {'MACP size':<10}")
    print("-" * 92)
    for _, r in summary.iterrows():
        print(f"{r['protocol']:<12} {r['config']:<10} "
              f"{r['acc']:.3f} ± {r['acc_std']:.3f}   "
              f"{r['auc']:.3f}    "
              f"{r['macp']:.3f}       {r['naive']:.3f}       {r['size']:.2f}")

    print("\n漂移警报统计 (Branch 0):")
    for (proto, cfg_name), g in df.groupby(['protocol', 'config']):
        print(f"  [{proto:<11s}] {cfg_name:<10s} 警报率: "
              f"{g['alarm'].mean():.2f}  (p 中位数 {g['mmd_p'].median():.4f})")

    print("\n判读:")
    print("  within-site acc >> cross-site acc  -> 偏移杀死信号, 保任务, 讲 MACP 故事")
    print("  within-site acc ~ 0.55 且各 config 都上不去 -> 任务天花板, 需换 PD vs HC 标签")


if __name__ == "__main__":
    main()