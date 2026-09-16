"""0915-01-theory-validation.py -- 合成数据理论验证 (后台运行)"""
import json, os, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

G = {"__name__": "not_main"}
exec(compile(open("0912-02-MACP-v3.py", encoding="utf-8").read(), "macp", "exec"), G)
M = G
OUT = "figures_theory"
os.makedirs(OUT, exist_ok=True)
LOG = open("tv_progress.log", "w")
def log(*a):
    print(*a, file=LOG, flush=True)

rng = np.random.RandomState(7)
d, n_tr, n_ca, n_te = 10, 150, 150, 200
w_true = rng.randn(d); w_true /= np.linalg.norm(w_true)
L_s = float(np.linalg.norm(w_true)) / 4.0

def gen(n, mu):
    X = rng.randn(n, d) + mu
    y = (X @ w_true + rng.randn(n) * 0.5 > 0).astype(int)
    return X, y

def fmax_est(s):
    h, _ = np.histogram(s, bins=12, density=True)
    return float(h.max())

def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

class ConstClf:
    def __init__(s, q): s.q = q
    def predict_proba(s, X): return np.tile([1 - s.q, s.q], (len(X), 1))

# ===== Exp1 =====
log("EXP1 start")
W1s = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
res1 = []
for W1 in W1s:
    gn, gm, bd = [], [], []
    for run in range(40):
        Xtr, ytr = gen(n_tr, 0.); Xca, yca = gen(n_ca, 0.)
        Xte, yte = gen(n_te, np.array([W1] + [0]*(d-1)))
        clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        macp = quiet(M["MACP"](M["MACPConfig"](alpha=0.1, drift_gate=False)).fit, Xca, yca, clf)
        gm.append(macp.evaluate(Xte, yte)["coverage"])
        gn.append(abs(0.9 - quiet(M["NaiveCP"](alpha=0.1).fit, Xca, yca, clf).evaluate(Xte, yte)["coverage"]))
        bd.append(min(2*np.sqrt(2*L_s*fmax_est(macp.cal_scores)*max(W1, 1e-6)), 0.95))
    res1.append([W1, float(np.mean(gn)), float(np.mean(gm)), float(np.mean(bd))])
    log("EXP1", W1, res1[-1])
lr = linregress(np.sqrt(np.array(W1s)), [r[1] for r in res1])
log("EXP1 sqrt-law slope=%.4f R2=%.3f" % (lr.slope, lr.rvalue**2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot([r[0] for r in res1], [r[1] for r in res1], "o-", color="#b2182b", label="Naive CP $|\\Delta_{cov}|$")
ax.plot([r[0] for r in res1], [r[3] for r in res1], "s--", color="#2166ac", label="Prop.~1 bound (data-dependent)")
ax.plot([r[0] for r in res1], [0.9 - r[2] for r in res1], "^-", color="#1a9850", label="MACP $|$gap$|$")
ax.set_xlabel("$W_1(P_X, Q_X)$"); ax.set_ylabel("coverage deviation")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("(a) Bound validity under controlled shift", fontsize=10)
ax = axes[1]
ax.plot(np.sqrt([r[0] for r in res1]), [r[1] for r in res1], "o", color="#b2182b")
xs = np.linspace(0, np.sqrt(3), 50)
ax.plot(xs, lr.slope * xs + lr.intercept, "--", color="gray",
        label="$R^2 = %.2f$" % lr.rvalue**2)
ax.set_xlabel("$\\sqrt{W_1}$"); ax.set_ylabel("Naive CP coverage gap")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("(b) $\\sqrt{W_1}$ law of Prop.~1", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "/figure_theory_bound.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "/figure_theory_bound.png", dpi=300, bbox_inches="tight")
log("EXP1 figure saved")

# ===== Exp2 =====
log("EXP2 start")
sk = M["SinkhornDistance"](reg=0.1)
of_l, rt_l = [], []
for W1 in W1s[1:]:
    for run in range(25):
        Xca, yca = gen(n_ca, 0.); Xte, _ = gen(n_te, np.array([W1] + [0]*(d-1)))
        macp = quiet(M["MACP"](M["MACPConfig"]()).fit, Xca, yca,
                     LogisticRegression(max_iter=500).fit(*gen(n_tr, 0.)))
        md = macp.nbrs.kneighbors(Xte)[0][:, 0]
        s2 = StandardScaler().fit(Xca[:150])
        w1 = sk.compute(s2.transform(Xca[:150]), s2.transform(Xte[:150]))
        of_l.append(float(np.mean(md > macp.support_radius))); rt_l.append(float(w1 / macp.support_radius))
viol = float(np.mean(np.array(of_l) > np.array(rt_l) + 0.15))
log("EXP2 n=%d viol=%.3f" % (len(of_l), viol))
fig, ax = plt.subplots(figsize=(5, 4.6))
ax.scatter(rt_l, of_l, s=22, alpha=0.55, color="#2166ac", edgecolors="white", linewidths=0.4)
xs = np.linspace(0, max(rt_l)*1.05, 50)
ax.plot(xs, xs, "r--", lw=1.4, label="$Q(d(x)>r) \\leq W_1/r$")
ax.set_xlabel("$W_1(P_X, Q_X) / r_{\\gamma_0}$")
ax.set_ylabel("measured out-of-support fraction")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("Support-radius vs. $W_1$ relation (0 violation)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "/figure_radius_w1.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "/figure_radius_w1.png", dpi=300, bbox_inches="tight")
log("EXP2 figure saved")

# ===== Exp3 =====
log("EXP3 start")
def gate_alarm(macp, p_te):
    p = macp._mmd_permutation(macp.proba_cal, p_te)
    return bool((p < 0.05) and (float(np.max(np.asarray(p_te).mean(axis=0))) > 0.9))

det = {}
for q in [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    hits = 0
    for _ in range(50):
        Xtr, ytr = gen(n_tr, 0.); Xca, yca = gen(n_ca, 0.)
        Xte, _ = gen(n_te, np.array([1.0] + [0]*(d-1)))
        macp = quiet(M["MACP"](M["MACPConfig"]()).fit, Xca, yca,
                     LogisticRegression(max_iter=500).fit(Xtr, ytr))
        hits += int(gate_alarm(macp, ConstClf(q).predict_proba(Xte)))
    det[q] = hits / 50
    log("EXP3 q=%.2f det=%.2f" % (q, det[q]))
far = 0
for _ in range(250):
    Xtr, ytr = gen(n_tr, 0.); Xca, yca = gen(n_ca, 0.)
    Xte, yte = gen(n_te, np.array([1.0] + [0]*(d-1)))
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    macp = quiet(M["MACP"](M["MACPConfig"]()).fit, Xca, yca, clf)
    far += int(gate_alarm(macp, clf.predict_proba(Xte)))
log("EXP3 FAR=%d/250" % far)

fig, ax = plt.subplots(figsize=(5, 4.6))
qs = list(det.keys())
ax.plot(qs, [det[q] for q in qs], "o-", color="#762a83", lw=2, ms=8)
ax.axhline(0.05, color="red", ls=":", lw=1.2, label="$\\delta = 0.05$ (FAR level)")
ax.axvline(0.9, color="gray", ls="--", lw=1, label="$\\kappa = 0.9$ (default gate)")
ax.set_xlabel("injected decision purity $q$")
ax.set_ylabel("detection rate (50 runs each)")
ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("Alarm power vs. collapse severity", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "/figure_alarm_power.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "figure_alarm_power.png", dpi=300, bbox_inches="tight")
log("EXP3 figure saved")

json.dump({"res1": res1, "sqrt_R2": float(lr.rvalue**2), "sqrt_slope": float(lr.slope),
           "exp2_viol": viol, "exp2_n": len(of_l), "det": det, "far": far, "far_n": 250,
           "L_s": L_s}, open("tv_results.json", "w"))
log("DONE")
