import os
import json
import copy
import random
from tqdm import tqdm
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from rbo import RankingSimilarity
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Configuration
# =============================================================================
INDEX_NAME     = 'beir-v1.0.0-trec-covid.flat'
TOPICS_NAME    = 'beir-v1.0.0-trec-covid-test'
API_KEY        = ""
MODEL_NAME     = 'gpt-3.5-turbo'
NUM_INPUTS     = 50                 # distinct sampled inputs per query
DETER_RUNS     = 2                  # LLM calls per input
OUTPUT_DIR     = "scenario1_logs"
    
# ============================================================================
#    Functions
# =============================================================================

def rbo_score(list1, list2, p):
    sim = RankingSimilarity(list1, list2)  # just pass the two lists here
    return sim.rbo(p=p)  


def sample_pool(pool, k):
    """Sample k items, with replacement only if pool < k."""
    if len(pool) >= k:
        return random.sample(pool, k)
    else:
        return random.choices(pool, k=k)
    
def mixed_sampler(rel, nonrel):
    head = sample_pool(rel,    5)
    tail = sample_pool(nonrel, 5)
    combo = head + tail
    random.shuffle(combo)
    return combo
    
# ==============================================================================

# Load topics, qrels, and searcher
# =============================================================================

# # Three modes: only relevant, only non‐relevant, and 5+5 mixed
# MODES = [
#     ("only_relevant",    lambda rel, nonrel: sample_pool(rel,    10)),
#     ("only_nonrelevant", lambda rel, nonrel: sample_pool(nonrel, 10)),
#     ("mixed",           mixed_sampler),
# ]


# # Make top-level dirs
# for mode, _ in MODES:
#     os.makedirs(os.path.join(OUTPUT_DIR, mode), exist_ok=True)

    
# Load data/index
topics   = get_topics(TOPICS_NAME)
qrels    = get_qrels(TOPICS_NAME)
searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
qids     = sorted(topics.keys(), key=int)

# #__________________________________________________________________________

# # Main loop
# for mode, sampler in MODES:
#     mode_dir = os.path.join(OUTPUT_DIR, mode)
#     print(f"Running mode: {mode}")

#     for qid in tqdm(qids, desc=f"{mode}"):
#         print("=====================")
#         print(qid)
#         print("=====================")
#         qdir = os.path.join(mode_dir, str(qid))
#         os.makedirs(qdir, exist_ok=True)

#         # split docs
#         rel_docs    = [d for d,r in qrels[qid].items() if int(r)>0 and searcher.doc(d) is not None]
#         nonrel_docs = [d for d,r in qrels[qid].items() if int(r)==0 and searcher.doc(d) is not None]

#         for inp_idx in range(NUM_INPUTS):
#             print(inp_idx)
#             # sample 10 docids
#             docids = sampler(rel_docs, nonrel_docs)
#             # fetch their content
#             docs = [{'docid':d, 'content': searcher.doc(d).raw()} for d in docids]

#             for run_idx in range(DETER_RUNS):
                
#                 item = {'query': topics[qid]['title'], 'hits': copy.deepcopy(docs)}
#                 msgs = create_permutation_instruction(item, 0, len(docs), model_name=MODEL_NAME)
#                 resp = run_llm(msgs, api_key=API_KEY, model_name=MODEL_NAME)
#                 out  = receive_permutation(item, resp, 0, len(docs))

#                 log = {
#                     "qid": qid,
#                     "mode": mode,
#                     "input_index": inp_idx,
#                     "run": run_idx,
#                     "prompt": msgs,
#                     "input_docids": docids,
#                     "output_docids": [h['docid'] for h in out['hits'][:len(docs)]],
#                     "response": resp
#                 }
#                 fname = f"in{inp_idx:02d}_run{run_idx}.json"
#                 with open(os.path.join(qdir, fname), 'w') as f:
#                     json.dump(log, f, indent=2)

# print("Done! Logs are in", OUTPUT_DIR)


# ============================================================================
#   Visialization - General Config
# =============================================================================

OUTPUT_DIR = "scenario1_logs"
MODES      = ["only_relevant", "only_nonrelevant", "mixed"]
QIDS       = qids
NUM_INPUTS = 50

# ─── Read logs & compute per-input τ and RBO ────────────────────────────────
tau_vals = {mode: {qid: [] for qid in QIDS} for mode in MODES}
rbo_vals = {mode: {qid: [] for qid in QIDS} for mode in MODES}

for mode in MODES:
    for qid in QIDS:
        qdir = os.path.join(OUTPUT_DIR, mode, str(qid))
        for inp_idx in range(NUM_INPUTS):
            # load the two runs for this input
            with open(os.path.join(qdir, f"in{inp_idx:02d}_run0.json")) as f:
                out0 = json.load(f)["output_docids"]
            with open(os.path.join(qdir, f"in{inp_idx:02d}_run1.json")) as f:
                out1 = json.load(f)["output_docids"]
            # kendall tau
            tau, _ = kendalltau(out0, out1)
            tau_vals[mode][qid].append(tau)
            # RBO@?
            rbo_score_temp = rbo_score(out0, out1, 0.79)
            rbo_vals[mode][qid].append(rbo_score_temp)

# ============================================================================
#   Visialization -1 boxplot for each query (showes the distribution of 50 rbo\tau per query)
# =============================================================================


########################################## v1 - all three box together ############
# # ─── Plot per-query boxplots ────────────────────────────────────────────────
# fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
# width  = 0.2
# gap    = 0.2
# colors = {
#     "only_relevant":    "lightblue",
#     "mixed":            "lightgreen",
#     "only_nonrelevant": "lightpink"
# }

# for ax, (metric_dict, ylabel, title) in zip(
#     axes,
#     [
#         (tau_vals, "Kendall’s τ", "Per-Query Kendall’s τ Distribution"),
#         (rbo_vals, "RBO@0.9",       "Per-Query RBO@0.9 Distribution"),
#     ]
# ):
#     positions = []
#     data      = []
#     # build data+positions
#     for i, qid in enumerate(QIDS):
#         base = i * (len(MODES) * width + gap)
#         for j, mode in enumerate(MODES):
#             positions.append(base + j * width)
#             data.append(metric_dict[mode][qid])
#     # draw boxplots
#     bp = ax.boxplot(
#         data, positions=positions,
#         widths=width, patch_artist=True,
#         showfliers=False
#     )
#     # color each box
#     for patch, mode in zip(bp["boxes"], MODES * len(QIDS)):
#         patch.set_facecolor(colors[mode])
#         patch.set_alpha(0.7)
#     # axis labels
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.grid(axis="y", linestyle="--", alpha=0.5)

# # configure x‐ticks at cluster centers
# cluster_centers = [
#     i * (len(MODES) * width + gap) + (len(MODES) * width) / 2
#     for i in range(len(QIDS))
# ]
# axes[-1].set_xticks(cluster_centers)
# axes[-1].set_xticklabels(QIDS, rotation=45, ha="right", fontsize=8)

# # single legend
# handles = [plt.Line2D([0],[0], color=colors[m], lw=10) for m in MODES]
# labels  = ["Only Relevant", "Mixed", "Only Non-Relevant"]
# axes[0].legend(handles, labels, title="Mode", loc="upper right")

# plt.tight_layout()

# # Save the figure:
# fig_path = os.path.join(OUTPUT_DIR, "per_query_stability.png")
# fig.savefig(fig_path, dpi=300)
# fig_path = os.path.join(OUTPUT_DIR, "per_query_stability.pdf")
# fig.savefig(fig_path)
# print(f"Saved per-query stability plot to {fig_path}")

# plt.show()


############################# v2 - each box plot in diffrent figure ##################

import os
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "scenario1_logs"
MODES      = ["only_relevant", "only_nonrelevant", "mixed"]
QIDS       = qids              # your list of 50 query IDs
NUM_INPUTS = 50
# assume you already have:
#   rbo_vals = { mode: { qid: [50 RBO values] } }

##6699CC 
colors = {
    "only_relevant":    "lightblue",
    "only_nonrelevant": "lightpink",
    "mixed":            "#C7EA46"
}

# font size for everything
FONT_SIZE = 16

# width of each individual box
BOX_WIDTH = 0.6

# ─── BUILD FIGURE ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)
fig.subplots_adjust(hspace=0.2)

mode_lable= {'mixed': "half of the documents relevant",
             'only_relevant':"all of the documents relevant",
             'only_nonrelevant':"none of the documents relevant"}

for ax, mode in zip(axes, MODES):
    # gather per-query RBO lists in qid order
    data = [rbo_vals[mode][qid] for qid in QIDS]
    positions = list(range(len(QIDS)))
    
    # draw the boxplot
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False
    )
    # color the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(colors[mode])
        patch.set_edgecolor(colors[mode])      # box outline is now black
        patch.set_linewidth(1.5)
        patch.set_alpha(1)
        if colors[mode] == "#C7EA46":
            patch.set_alpha(0.5)
            
            
    for element in ('whiskers','caps'):
        for line in bp[element]:
            # 3B688A
            line.set_color("#3B688A")   # retain your chosen mode color
            line.set_linewidth(1)
            
            
    # Now, on the *first* subplot, add arrows for Q14 & Q39
#     if ax is axes[0]:
#         for q in [14, 39]:
#             idx = QIDS.index(q)
#             # find the top of that box (median + IQR/2) or just use max of data
#             y_top = max(data[idx]) + 0.02
#             ax.annotate(
#                 f"Q{q}",
#                 xy=(idx, max(data[idx])),
#                 xytext=(idx, y_top + 0.05),
#                 arrowprops=dict(arrowstyle="->", color="red", lw=2),
#                 ha="center",
#                 fontsize=FONT_SIZE,
#                 color="red"
#             )
    
    # titles & labels
#     if mode == "only_nonrelevant":
#         mode = "only_non-relevant"
    ax.set_title(mode_lable[mode], fontsize=FONT_SIZE)
    
#     if ax is axes[-1]:
#         ax.set_xlabel("Query index (ascending order)",              fontsize=FONT_SIZE)
#       ax.set_xticks([])
        
    ax.set_ylabel("RBO@0.79",fontsize=FONT_SIZE,labelpad=15)




    if ax is axes[0]:
        ax.set_xlabel("Query index",              fontsize=FONT_SIZE)
       
    if ax is axes[0]:
        # choose which query‐IDs to label
        tick_qids = [5, 10, 20, 30, 40, 50]
        # find their positions in the QIDS list (zero‐based)
        tick_positions = [QIDS.index(q) for q in tick_qids]
        # set only those ticks + labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(q) for q in tick_qids],
             fontsize=16
        )
    
    
    # ticks
#     if ax is axes[2]:
#         ax.set_xticks(positions)
#         ax.set_xticklabels(QIDS, rotation=45, ha="right", fontsize=12)
#     else:
#     ax.set_xticks([])

    ax.tick_params(axis="y", labelsize=16)
    
    # grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.5)

# ─── LEGEND ────────────────────────────────────────────────────────────────
# from matplotlib.lines import Line2D
# handles = [Line2D([0],[0], color=colors[m], lw=10) for m in MODES]
# labels  = ["Only Relevant", "Only Non-Relevant", "Mixed"]
# leg = fig.legend(handles, labels, loc="upper center",
#                  ncol=3, fontsize=14)
# leg.get_title().set_fontsize(14)

# plt.tight_layout(rect=[0,0,1,0.9])

plt.tight_layout()

# ─── SAVE & SHOW ─────────────────────────────────────────────────────────
fig_path = os.path.join(OUTPUT_DIR, "rbo_per_mode_boxplot.png")
fig.savefig(fig_path, dpi=300)
fig_path = os.path.join(OUTPUT_DIR, "rbo_per_mode_boxplot.pdf")
fig.savefig(fig_path)
print(f"Saved figure to {fig_path}")
plt.show()




# ======================================================================================
#   Creating the rbo and tau values based on log files 
# ======================================================================================

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import logging



# ——— Configuration —————————————————————————————————————
OUTPUT_DIR = "scenario1_logs"
MODES      = ["only_relevant", "only_nonrelevant", "mixed"]
NUM_INPUTS = 50
QIDS       = [str(q) for q in sorted(map(int, os.listdir(os.path.join(OUTPUT_DIR, MODES[0]))))]


# ——— 1) Read logs & compute tau and rbo ————————————————————————
records = []
for mode in MODES:
    for qid in QIDS:
        qdir = os.path.join(OUTPUT_DIR, mode, qid)
        if not os.path.isdir(qdir):
            out.write(f"WARNING: Directory not found, skipping: {qdir}\n")
            continue
        for inp_idx in range(NUM_INPUTS):
            f0 = os.path.join(qdir, f"in{inp_idx:02d}_run0.json")
            f1 = os.path.join(qdir, f"in{inp_idx:02d}_run1.json")
            if not os.path.exists(f0) or not os.path.exists(f1):
                out.write(f"WARNING: Missing runs for mode={mode}, qid={qid}, input={inp_idx}\n")
                continue
            out0 = json.load(open(f0))["output_docids"]
            out1 = json.load(open(f1))["output_docids"]
            tau, _ = kendalltau(out0, out1)
            ### if N=5 then p=0.63, if N=10 then p=0.79 and if N=20 then p=0.89
            rbo = rbo_score(out0, out1, 0.79)
            records.append({"mode":mode, "qid":qid, "input":inp_idx, "tau":tau, "rbo":rbo})

df = pd.DataFrame(records)


# ======================================================================================
#   Significance Tests - 1 : is these values significamtly diffrent between queries?
# ======================================================================================


# # ——— Configure logging to file —————————————————————————————
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, "analysis_results.txt")
# out = open(OUTPUT_FILE, "w")

# if df.empty:
#     out.write("ERROR: No data loaded. Check OUTPUT_DIR, MODES, and QIDS.\n")
# else:
#     # ——— 2) Omnibus ANOVA + Tukey HSD —————————————————————
#     for mode in MODES:
#         out.write(f"\n=== Mode: {mode} ===\n")
#         subset = df[df["mode"] == mode]
#         for metric in ["tau", "rbo"]:
#             groups = [
#                 subset[subset["qid"]==qid][metric].values
#                 for qid in QIDS
#                 if not subset[subset["qid"]==qid].empty
#             ]
#             if len(groups) < 2:
#                 out.write(f"Not enough data for mode={mode}, metric={metric}. Skipping.\n")
#                 continue

#             F_stat, p_val = f_oneway(*groups)
#             out.write(f"{metric.upper()} ANOVA: F = {F_stat:.3f}, p = {p_val:.3g}\n")
#             if p_val < 0.05:
#                 out.write("Post-hoc Tukey HSD:\n")
#                 tukey = pairwise_tukeyhsd(
#                     endog=subset[metric],
#                     groups=subset["qid"],
#                     alpha=0.05
#                 )
#                 for line in tukey.summary().as_text().split("\n"):
#                     out.write(line + "\n")
#             else:
#                 out.write("No significant differences among queries (skip post-hoc).\n")

# # ——— Close your file —————————————————————————————————————
# out.write(f"\nAnalysis complete. See results in {OUTPUT_FILE}\n")
# out.close()
# print(f"✅ Done — all results in {OUTPUT_FILE}")


# ################## v2 - assum non normality distribution of rbo #############

# import os
# import itertools
# from scipy.stats import kruskal, mannwhitneyu

# # ─── CONFIG ──────────────────────────────────────────────────────────────
# OUTPUT_DIR   = "scenario1_logs"
# MODES        = ["only_relevant", "only_nonrelevant", "mixed"]
# METRICS      = ["rbo", "tau"]
# QIDS         = sorted(df["qid"].unique())
# NUM_QUERIES  = len(QIDS)
# output_path  = os.path.join(OUTPUT_DIR, "query_significance.txt")

# # ─── RUN & WRITE RESULTS ─────────────────────────────────────────────────
# with open(output_path, "w") as out:
#     out.write("Query‐by‐Query Consistency Significance Tests\n")
#     out.write("="*50 + "\n\n")

#     for mode in MODES:
#         out.write(f"Mode: {mode}\n")
#         mode_df = df[df["mode"] == mode]

#         for metric in METRICS:
#             out.write(f"\n  Metric: {metric.upper()}\n")

#             # 1) Group RBO (or tau) values by query
#             groups = [mode_df[mode_df["qid"] == q][metric].values for q in QIDS]

#             # 2) Omnibus Kruskal–Wallis
#             if len(groups) < 2:
#                 out.write("    Not enough data for omnibus test.\n")
#                 continue

#             H_stat, p_kw = kruskal(*groups)
#             out.write(f"    Kruskal–Wallis H = {H_stat:.3f}, p = {p_kw:.4f}\n")

#             if p_kw >= 0.05:
#                 out.write("    → No significant differences among queries.\n")
#                 continue

#             out.write("    → Significant differences found! Pairwise tests:\n")

#             # 3) Pairwise Mann–Whitney U with Bonferroni correction
#             num_pairs   = NUM_QUERIES * (NUM_QUERIES - 1) / 2
#             bonf_alpha  = 0.05 / num_pairs
#             out.write(f"      Bonferroni‐corrected α = {bonf_alpha:.6f}\n")

#             for i, j in itertools.combinations(range(NUM_QUERIES), 2):
#                 q1, q2 = QIDS[i], QIDS[j]
#                 vals1   = mode_df[mode_df["qid"] == q1][metric]
#                 vals2   = mode_df[mode_df["qid"] == q2][metric]

#                 U_stat, p_u = mannwhitneyu(vals1, vals2, alternative="two-sided")
#                 sig = "yes" if p_u < bonf_alpha else "no"
#                 out.write(
#                     f"      Q{q1} vs Q{q2}: U = {U_stat:.2f}, p = {p_u:.4g}, significant? {sig}\n"
#                 )

#         out.write("\n" + "-"*50 + "\n\n")

# print(f"✅ Done – see detailed results in {output_path}")


# #################### v3 just count the pairs out of all pairs ##########################################################


# from itertools import combinations
# from scipy.stats import ttest_ind

# # … up above you have df built, OUTPUT_FILE opened as `out`, etc. …
# output_path  = os.path.join(OUTPUT_DIR, "pair_query_significance.txt")

# # ─── RUN & WRITE RESULTS ─────────────────────────────────────────────────
# with open(output_path, "w") as out:

#     for mode in MODES:
#         out.write(f"\n=== Mode: {mode} ===\n")
#         subset = df[df["mode"] == mode]

#         for metric in ["tau", "rbo"]:
#             out.write(f"\n  Metric: {metric.upper()}\n")

#             # 1) Gather each query's 50 values into a dict
#             data_by_q = {
#                 qid: subset[subset["qid"] == qid][metric].values
#                 for qid in QIDS
#             }

#             # 2) Do all pairwise tests
#             sig_count = 0
#             total_pairs = 0

#             for q1, q2 in combinations(QIDS, 2):
#                 vals1 = data_by_q[q1]
#                 vals2 = data_by_q[q2]
#                 # skip if any is empty
#                 if len(vals1)==0 or len(vals2)==0:
#                     continue
#                 total_pairs += 1

#                 # Welch's t-test (unequal variances)
#                 stat, pval = ttest_ind(vals1, vals2, equal_var=False)
#                 if pval < 0.05:
#                     sig_count += 1

#             frac = sig_count / total_pairs if total_pairs else 0
#             out.write(f"    Tested {total_pairs} query‐pairs;\n")
#             out.write(f"    {sig_count} ({frac:.1%}) were significantly different at α=0.05\n")


# ======================================================================================
#   Significance Tests - 2 : which mode is statistically significant better?
# ======================================================================================

############## v1 #############################

# import os
# import pandas as pd
# from scipy.stats import friedmanchisquare, wilcoxon

# OUTPUT_DIR = "scenario1_logs"
# output_file = os.path.join(OUTPUT_DIR, "mode_comparison_results.txt")

# # Your df must already exist, with columns: ['mode','qid','input_idx','tau','rbo']

# with open(output_file, "w") as out:
#     out.write("### Mode Consistency Comparison (RBO & Tau) ###\n\n")

#     for metric in ["rbo", "tau"]:
#         out.write(f"--- Metric: {metric.upper()} ---\n")
#         # 1) pivot to get mean per query & mode
#         mean_df = (
#             df
#             .groupby(["qid","mode"])[metric]
#             .mean()
#             .reset_index()
#             .pivot(index="qid", columns="mode", values=metric)
#         )

#         # 2) Friedman omnibus
#         stat, p = friedmanchisquare(
#             mean_df["only_relevant"],
#             mean_df["only_nonrelevant"],
#             mean_df["mixed"]
#         )
#         out.write(f"Friedman χ² = {stat:.3f}, p = {p:.4f}\n")
#         if p >= 0.05:
#             out.write("→ No evidence of differences across modes (p ≥ 0.05)\n\n")
#             continue
#         out.write("→ Significant differences across modes (p < 0.05)\n\n")

#         # 3) Print means so we know direction
#         means = mean_df.mean().to_dict()
#         out.write("Mean values by mode:\n")
#         for m, mval in means.items():
#             out.write(f"  {m:>20}: {mval:.4f}\n")
#         out.write("\n")

#         # 4) Pairwise Wilcoxon with Bonferroni
#         comparisons = [
#             ("only_relevant",    "only_nonrelevant"),
#             ("only_relevant",    "mixed"),
#             ("only_nonrelevant", "mixed"),
#         ]
#         alpha = 0.05
#         bonf = alpha / len(comparisons)
#         out.write(f"Pairwise Wilcoxon (Bonferroni α = {bonf:.4f}):\n")
#         for a, b in comparisons:
#             w_stat, p_w = wilcoxon(mean_df[a], mean_df[b])
#             sig = p_w < bonf
#             out.write(f"- {a:>20} vs {b:<20}: W = {w_stat:.2f}, p = {p_w:.4f}  significant? {sig}\n")
#             if sig:
#                 # interpret direction:
#                 better = a if means[a] > means[b] else b
#                 out.write(f"    => {better} is significantly more consistent ({metric})\n")
#         out.write("\n")

# print(f"✅ Results written to {output_file}")



################# v2 - simple sign test and t-test ##################

# import os
# from scipy.stats import binomtest

# # ─── CONFIG ──────────────────────────────────────────────────────────────
# OUTPUT_DIR = "scenario1_logs"
# log_path   = os.path.join(OUTPUT_DIR, "easy_mode_comparison_results.txt")

# # 1) Build per-query mean RBO table
# mean_df = (
#     df
#     .groupby(['qid','mode'])['rbo']
#     .mean()
#     .unstack('mode')
# )

# # 2) Define the comparisons
# pairs = [
#     ('only_relevant', 'mixed'),
#     ('only_nonrelevant', 'mixed'),
#     ('only_relevant', 'only_nonrelevant'),
# ]

# # ─── SIGN COUNTS & TEST ───────────────────────────────────────────────────
# with open(log_path, "a") as out:
#     out.write("\n### Sign Counts: How often A > B vs B > A ###\n\n")
#     for A, B in pairs:
#         diffs     = mean_df[A] - mean_df[B]
#         wins_ab   = (diffs > 0).sum()   # A above B
#         wins_ba   = (diffs < 0).sum()   # B above A
#         ties      = (diffs == 0).sum()  # equal
#         total     = len(mean_df)

#         # two-sided sign test p-value
#         n = wins_ab + wins_ba
#         if n > 0:
#             pval = binomtest(wins_ab, n, p=0.5, alternative='two-sided').pvalue
#             sig_flag = "YES" if pval < 0.05 else "no"
#         else:
#             pval, sig_flag = float('nan'), "na"

#         out.write(
#             f"{A.replace('_',' ').title()} vs {B.replace('_',' ').title()}:\n"
#             f"  {A} > {B}: {wins_ab}/{total} queries\n"
#             f"  {B} > {A}: {wins_ba}/{total} queries\n"
#             f"  Ties       : {ties}/{total}\n"
#             f"  p(two-sided sign test) = {pval:.4f}, significant? {sig_flag}\n\n"
#         )

# print(f"✅ Sign‐counts results appended to {log_path}")




# ============================================================================
#   Visialization -1 line plot to compare each mix , rel and non rel mode 
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "scenario1_logs"

# ─── Compute per-query mean and standard deviation ─────────────────────────
agg = df.groupby(['qid','mode'])['rbo'].agg(['mean','std']).reset_index()
mean_rbo = agg.pivot(index='qid', columns='mode', values='mean')
std_rbo  = agg.pivot(index='qid', columns='mode', values='std')

# N = NUM_INPUTS  # 50
# # 1) Compute the 95% CI half‐width (1.96 * SEM) instead of raw σ
# ci95 = std_rbo / np.sqrt(N) * 1.96

# 2) Sort as before
sorted_qids = mean_rbo['mixed'].sort_values(ascending=True).index
mean_sorted = mean_rbo.loc[sorted_qids]
# ci_sorted   = ci95.loc[sorted_qids]
x = np.arange(len(sorted_qids))

# ─── Define light (halo) and dark (line) colors ───────────────────────────
light_colors = {
    'only_relevant':    '#C1D9E9',  # Soft Sky Blue
    'only_nonrelevant': '#F6BDC7',  # Rose Quartz
    'mixed':            '#C7EA46'   # Pastel Lime
}
dark_colors = {
    'only_relevant':    '#3B688A',  # Slate Blue
    'only_nonrelevant': '#A33A4B',  # Dusty Rose
    'mixed':            '#6E8B3D'   # Deep Olive
}

mode_lable= {'mixed': "half of the documents relevant",
             'only_relevant':"all of the documents relevant",
             'only_nonrelevant':"none of the documents relevant"}
# 3) Plot with CI band
fig = plt.figure(figsize=(12,6))
for mode in ['mixed','only_relevant','only_nonrelevant']:
    mu = mean_sorted[mode]
#     ci = ci_sorted[mode]
#     # draw the 95% CI band
#     plt.fill_between(x, mu - ci, mu + ci,
#                      color=light_colors[mode], alpha=0.5)
    # draw the mean line on top
    plt.plot(
        x, mu,
        linestyle='-',                      # solid line
        color=light_colors[mode],           # light line color
        linewidth=3,
        marker='o',                         # circle markers
        markerfacecolor=dark_colors[mode],  # fill marker dark
        markeredgecolor=dark_colors[mode],  # outline marker dark
        markersize=6,                       # tweak as desired
        label=mode_lable[mode]
    )

plt.gca().tick_params(
axis='x',
which='both',
bottom=False,
top=False,
labelbottom=False
)

plt.tick_params(axis='y', labelsize=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# plt.xticks(x, sorted_qids, rotation=90, fontsize=8)
plt.xlabel('Query (sorted by half of the documents relevant RBO)', fontsize=16, labelpad=15)
plt.ylabel('Mean RBO@0.79', fontsize=16, labelpad=15)
# plt.title('Mean RBO per Query', fontsize=14)
plt.legend(fontsize=12, loc='upper left')

fig_path = os.path.join(OUTPUT_DIR, "mode_significant.png")
fig.savefig(fig_path, dpi=300)
fig_path = os.path.join(OUTPUT_DIR, "mode_significant.pdf")
fig.savefig(fig_path)

plt.tight_layout()

# ─── Show plot ────────────────────────────────────────────────────────────
plt.show()


