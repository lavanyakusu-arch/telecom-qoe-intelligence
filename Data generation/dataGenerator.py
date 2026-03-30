"""
Telecom QoE Dataset Generator — v3
====================================
Changes from v2
---------------
• app_weights updated to values grounded in ITU-T G.1010 / 3GPP QoS
  requirements and published telecom QoE research:
    - Video   : throughput-dominant (buffer stalls > all else)
    - VoLTE   : latency + loss ultra-sensitive (E-model R-factor basis)
    - Gaming  : latency-dominant with sharp loss cliff

• calculate_qoe rewritten to produce realistic MOS-like scores [1,5]:
    - Each KPI mapped through a piecewise-linear degradation function
      derived from ITU-T P.1203 (video), G.107 E-model (voice),
      and published gaming QoE studies
    - Penalty accumulates multiplicatively for severe impairments
      (e.g. VoLTE with both high latency AND loss tanks to Poor)
    - Still uses the same signature: (app, tput, lat, loss, sinr, load, prb)
    - Output still clipped to [1, 5] and categorised the same way

All KPI generation (Sections 3–10) is identical to v2.
Same output columns as original.
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

np.random.seed(42)

n_samples = 5000
apps      = ["video", "volte", "gaming"]

# ──────────────────────────────────────────────────────────
# SECTION 1 — IMSI
# ──────────────────────────────────────────────────────────
def generate_imsi(n):
    mcc, mnc = "310", "150"
    msin = np.random.randint(100_000_000, 999_999_999, n)
    return [mcc + mnc + str(i) for i in msin]

ue_id = generate_imsi(n_samples)

# ──────────────────────────────────────────────────────────
# SECTION 2 — APPLICATION TYPE
# ──────────────────────────────────────────────────────────
app = np.random.choice(apps, n_samples)

# ──────────────────────────────────────────────────────────
# SECTION 3 — MOBILITY  (truncated-normal; pedestrian/vehicular mix)
# ──────────────────────────────────────────────────────────
_a = (0 - 25) / 30
_b = (120 - 25) / 30
mobility = truncnorm.rvs(_a, _b, loc=25, scale=30, size=n_samples)
mob_norm  = mobility / 120.0

# ──────────────────────────────────────────────────────────
# SECTION 4 — CELL LOAD  (bimodal: light vs. congested cells)
# ──────────────────────────────────────────────────────────
low_load_mask = np.random.random(n_samples) < 0.60
cell_load = np.where(
    low_load_mask,
    np.random.normal(35, 12, n_samples),
    np.random.normal(75, 10, n_samples),
)
cell_load = np.clip(cell_load, 10, 100)

# ──────────────────────────────────────────────────────────
# SECTION 5 — PRB UTILIZATION  (correlated with cell_load)
# ──────────────────────────────────────────────────────────
prb_utilization = np.clip(
    0.88 * cell_load + np.random.normal(0, 5, n_samples),
    10, 100
)

# ──────────────────────────────────────────────────────────
# SECTION 6 — SINR  (congestion + mobility penalties)
# ──────────────────────────────────────────────────────────
sinr_base               = np.random.normal(12, 7, n_samples)
congestion_sinr_penalty = 0.12 * (cell_load - 50)
mobility_sinr_penalty   = 8.0  * mob_norm
sinr = np.clip(sinr_base - congestion_sinr_penalty - mobility_sinr_penalty, -5, 30)

# ──────────────────────────────────────────────────────────
# SECTION 7 — RSRP  (SINR-correlated + mobility penalty)
# ──────────────────────────────────────────────────────────
rsrp = np.clip(
    -95 + 0.4 * (sinr - 12) + np.random.normal(0, 6, n_samples) - 10.0 * mob_norm,
    -120, -70
)

# ──────────────────────────────────────────────────────────
# SECTION 8 — THROUGHPUT  (Shannon-like capacity)
# ──────────────────────────────────────────────────────────
sinr_linear    = 10 ** (sinr / 10.0)
capacity_proxy = np.log2(1 + sinr_linear)
throughput = np.clip(
    capacity_proxy * 3.5 - 0.12 * (cell_load - 40) + np.random.normal(0, 1.5, n_samples),
    0.5, 50
)

# ──────────────────────────────────────────────────────────
# SECTION 9 — LATENCY  (M/M/1 queuing + HO interruption)
# ──────────────────────────────────────────────────────────
utilisation_ratio = cell_load / 100.0
mm1_delay = 10.0 / np.clip(1.0 - utilisation_ratio, 0.05, 1.0)
latency = np.clip(
    mm1_delay + 20.0 * mob_norm + np.random.normal(0, 8, n_samples),
    5, 200
)

# ──────────────────────────────────────────────────────────
# SECTION 10 — PACKET LOSS  (SINR + congestion + mobility)
# ──────────────────────────────────────────────────────────
packet_loss = np.clip(
    np.clip(1.5 - 0.08 * sinr, 0, 3)
    + 0.025 * np.clip(cell_load - 50, 0, 50)
    + 1.5 * mob_norm
    + np.random.normal(0, 0.3, n_samples),
    0, 5
)

# ──────────────────────────────────────────────────────────
# SECTION 11 — app_weights  (ITU-T G.1010 / 3GPP TS 22.261
#              / E-model R-factor grounded values)
#
# Video  (ITU-T P.1203 / Netflix VMAF research):
#   throughput dominates — stalls at < 2 Mbps tank MOS sharply
#   latency matters mainly for live/interactive video
#   packet_loss causes blockiness and rebuffering
#
# VoLTE  (ITU-T G.107 E-model):
#   latency: one-way delay > 150 ms → perceptible echo/clipping
#   loss:    even 1 % uncorrected loss clearly audible
#   throughput: ~64–128 kbps is more than enough; barely matters
#
# Gaming  (published QoE studies: Claypool, ITU-T G.1072):
#   latency: < 50 ms "excellent", > 100 ms "poor" for fast games
#   loss:    any loss causes rubber-banding / desync
#   throughput: 1–5 Mbps sufficient; high tput gives little gain
# ──────────────────────────────────────────────────────────
app_weights = {
    # weights normalised so sum ≈ 1.0 per app for comparability
    "video": {
        "throughput": 0.40,   # dominant — stall / bitrate adaptation
        "latency":    0.15,   # matters for live streaming
        "loss":       0.25,   # blockiness, rebuffering
        "sinr":       0.20,   # link quality (drives tput + loss)
    },
    "volte": {
        "throughput": 0.05,   # 64 kbps is enough; barely matters
        "latency":    0.45,   # E-model: delay > 150 ms → R drops fast
        "loss":       0.35,   # audible at > 0.5 %; very sensitive
        "sinr":       0.15,   # affects codec selection
    },
    "gaming": {
        "throughput": 0.10,   # 1–5 Mbps sufficient for most games
        "latency":    0.50,   # #1 factor — input lag, rubber-banding
        "loss":       0.25,   # causes desync / packet retransmit
        "sinr":       0.15,   # underlying link quality
    },
}

# ──────────────────────────────────────────────────────────
# SECTION 12 — QoE SCORING
#
# Design rationale
# ----------------
# 1. Each KPI is mapped to a [0, 1] impairment score via
#    piecewise-linear breakpoints derived from standards:
#      throughput : video stall threshold ~2 Mbps, saturation ~15 Mbps
#      latency    : voip cliff 150 ms / gaming cliff 80 ms / video 100 ms
#      packet_loss: voip audible at 0.5 %, gaming desync at 1 %, video rebuffer at 2 %
#      sinr       : poor link < 0 dB, good link > 15 dB
#
# 2. Weighted sum of impairments → total_impairment ∈ [0, 1]
#
# 3. MOS = 5 × (1 − total_impairment)  remapped to [1, 5]
#    i.e.  MOS = 1 + 4 × (1 − total_impairment)
#
# 4. Multiplicative cliff: if any single KPI is in "catastrophic"
#    range (loss > 3 %, latency > 150 ms for VoLTE/gaming,
#    throughput < 0.5 Mbps for video), apply an extra 30 % penalty
#    to simulate the non-linear perceptual collapse seen in practice.
#
# 5. Small Gaussian jitter (σ = 0.15) to simulate real measurement noise.
# ──────────────────────────────────────────────────────────

def _piecewise_impairment(value, breakpoints):
    """
    Map a scalar `value` to impairment in [0, 1] via linear
    interpolation across (value, impairment) breakpoints.
    breakpoints: list of (value, impairment) tuples, ascending by value.
    """
    xs = [p[0] for p in breakpoints]
    ys = [p[1] for p in breakpoints]
    return float(np.clip(np.interp(value, xs, ys), 0.0, 1.0))


def calculate_qoe(application, tput, lat, loss, sinr_val, load, prb):
    w = app_weights[application]

    # ── per-KPI impairment [0 = perfect, 1 = catastrophic] ──

    if application == "video":
        # throughput: stall risk < 2 Mbps, excellent > 15 Mbps
        tput_imp = _piecewise_impairment(tput, [(0.5, 1.0), (2.0, 0.7),
                                                (5.0, 0.35),(15.0, 0.0),(50.0, 0.0)])
        # latency: live video tolerates up to ~100 ms
        lat_imp  = _piecewise_impairment(lat,  [(5, 0.0),(50, 0.1),(100, 0.4),(200, 1.0)])
        # packet loss: visible blockiness above 1 %
        loss_imp = _piecewise_impairment(loss, [(0, 0.0),(1.0, 0.2),(2.5, 0.6),(5.0, 1.0)])
        # SINR: poor below 0 dB, saturates above 20 dB
        sinr_imp = _piecewise_impairment(sinr_val, [(-5,1.0),(0,0.8),(8,0.3),(20,0.0),(30,0.0)])
        cliff    = (tput < 0.8)    # rebuffering cliff

    elif application == "volte":
        # throughput: 64 kbps = 0.064 Mbps is enough → very forgiving
        tput_imp = _piecewise_impairment(tput, [(0.0, 0.8),(0.1, 0.1),(0.5, 0.0),(50, 0.0)])
        # latency: E-model cliff at ~150 ms one-way
        lat_imp  = _piecewise_impairment(lat,  [(5, 0.0),(50, 0.05),(100, 0.3),(150, 0.7),(200, 1.0)])
        # loss: audible above 0.5 %, very bad above 2 %
        loss_imp = _piecewise_impairment(loss, [(0, 0.0),(0.5, 0.3),(1.5, 0.7),(3.0, 1.0)])
        # SINR: AMR codec adapts down to ~5 dB, but degrades
        sinr_imp = _piecewise_impairment(sinr_val, [(-5,1.0),(0,0.9),(5,0.5),(12,0.1),(30,0.0)])
        cliff    = (lat > 150 or loss > 2.5)   # E-model hard cliff

    else:  # gaming
        # throughput: 1 Mbps sufficient; diminishing returns above 5
        tput_imp = _piecewise_impairment(tput, [(0.5, 0.9),(1.0, 0.3),(5.0, 0.0),(50, 0.0)])
        # latency: sharp cliff — < 50 ms excellent, > 100 ms poor
        lat_imp  = _piecewise_impairment(lat,  [(5, 0.0),(30, 0.05),(80, 0.5),(120, 0.9),(200, 1.0)])
        # loss: rubber-banding above 0.5 %, unplayable > 2 %
        loss_imp = _piecewise_impairment(loss, [(0, 0.0),(0.5, 0.25),(1.5, 0.7),(3.0, 1.0)])
        # SINR: fast-game packets need reliable link
        sinr_imp = _piecewise_impairment(sinr_val, [(-5,1.0),(0,0.85),(7,0.3),(15,0.05),(30,0.0)])
        cliff    = (lat > 120 or loss > 2.0)   # unplayable threshold

    # ── congestion impairment (common to all apps) ──
    # PRB > 90 % → scheduler can't meet delay budgets
    cong_imp = _piecewise_impairment(prb, [(10,0.0),(60,0.05),(80,0.2),(95,0.5),(100,0.7)])

    # ── weighted total impairment ──
    total_imp = (
        w["throughput"] * tput_imp
        + w["latency"]    * lat_imp
        + w["loss"]       * loss_imp
        + w["sinr"]       * sinr_imp
        + 0.10            * cong_imp    # fixed small congestion contribution
    )

    # ── cliff penalty: 30 % extra when a critical threshold is breached ──
    if cliff:
        total_imp = min(1.0, total_imp * 1.30)

    # ── map impairment → MOS [1, 5] ──
    mos = 1.0 + 4.0 * (1.0 - total_imp)
    mos += np.random.normal(0, 0.15)   # realistic measurement jitter
    return float(np.clip(mos, 1.0, 5.0))


# ──────────────────────────────────────────────────────────
# SECTION 13 — GENERATE SCORES & CATEGORIES
# ──────────────────────────────────────────────────────────
qoe_score = [
    calculate_qoe(
        app[i], throughput[i], latency[i], packet_loss[i],
        sinr[i], cell_load[i], prb_utilization[i]
    )
    for i in range(n_samples)
]

def categorize_qoe(score):
    if   score < 2: return "Poor"
    elif score < 3: return "Fair"
    elif score < 4: return "Good"
    else:           return "Excellent"

qoe_category = [categorize_qoe(s) for s in qoe_score]

# ──────────────────────────────────────────────────────────
# SECTION 14 — DATAFRAME  (same columns as original)
# ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    "imsi":            ue_id,
    "app":             app,
    "throughput":      np.round(throughput,      2),
    "latency":         np.round(latency,         2),
    "packet_loss":     np.round(packet_loss,     2),
    "sinr":            np.round(sinr,            2),
    "rsrp":            np.round(rsrp,            2),
    "cell_load":       np.round(cell_load,       2),
    "prb_utilization": np.round(prb_utilization, 2),
    "mobility":        np.round(mobility,        2),
    "qoe_score":       np.round(qoe_score,       2),
    "qoe_category":    qoe_category,
})

df.to_csv("telecom_qoe_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())

print("\nQoE category distribution:")
print(df["qoe_category"].value_counts())

print("\nPer-app QoE mean:")
print(df.groupby("app")["qoe_score"].mean().round(3))

print("\nPer-app QoE category breakdown:")
print(df.groupby(["app","qoe_category"]).size().unstack(fill_value=0))

print("\nCorrelation: KPIs vs qoe_score:")
cols = ["throughput","latency","packet_loss","sinr","cell_load","prb_utilization","mobility","qoe_score"]
print(df[cols].corr()["qoe_score"].drop("qoe_score").round(3))