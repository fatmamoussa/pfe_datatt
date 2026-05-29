"""
=============================================================
EDA — Analyse Exploratoire du Dataset chunks.json
Tunisie Telecom — PFE Cycle Ingénieur IA & Data Science
=============================================================

Format : JSON — liste de chunks documentaires
  - 2 821 chunks extraits de PDFs
  - 549 fichiers PDF uniques
  - Période : 2018 → 2025
  - Champs : chunk_id, source, file_name, year, month,
             document_type, chunk_index, total_chunks,
             confidentiality, confidentiality_reason,
             shareable_with_client, chunk_text

Génère 6 graphiques :
  1. Distribution chunks par année
  2. Heatmap activité documentaire année × mois
  3. Distribution confidentialité (shareable vs confidentiel)
  4. Top 20 raisons de confidentialité
  5. Distribution longueurs chunk_text
  6. Distribution taille des documents (total_chunks par fichier)
=============================================================
"""

import json
import os
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
INPUT_FILE = "data/chunks.json"
OUTPUT_DIR = "outputs"

# Couleurs charte Tunisie Telecom
TT_BLUE   = "#003A8C"
TT_BLUE2  = "#0055CC"
TT_BLUE3  = "#5B8FD4"
TT_BLUE4  = "#A8C4F0"
TT_GREEN  = "#1a7a4a"
TT_ORANGE = "#D46B00"
TT_RED    = "#C0392B"
TT_GRAY   = "#7F8C8D"
TT_PURPLE = "#6C3483"

PALETTE = [TT_BLUE, TT_ORANGE, TT_GREEN, TT_RED, TT_PURPLE,
           TT_GRAY, TT_BLUE3, TT_BLUE4, "#17A589", "#D35400",
           "#1F618D", "#7D6608", "#922B21", "#1A5276", "#117A65"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

MONTHS_FR = {
    "01": "Jan", "02": "Fév", "03": "Mar", "04": "Avr",
    "05": "Mai", "06": "Jun", "07": "Jul", "08": "Aoû",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Déc",
}

# Mapping raisons → libellés lisibles
REASON_LABELS = {
    "rule:offre":                "Offre",
    "rule:offres":               "Offres (pl.)",
    "default:shareable":         "Partageable (défaut)",
    "default:confidential":      "Confidentiel (défaut)",
    "rule:client":               "Client",
    "hard_rule:argumentaire":    "Argumentaire (strict)",
    "rule:flash commercial":     "Flash Commercial",
    "rule:fiche commerciale":    "Fiche Commerciale",
    "rule:promo":                "Promotion",
    "rule:forfait":              "Forfait",
    "rule:packs":                "Packs",
    "rule:pack":                 "Pack",
    "rule:mobile":               "Mobile",
    "rule:prix":                 "Prix",
    "rule:tarif":                "Tarif",
    "rule:forfaits":             "Forfaits (pl.)",
    "rule:fiche produit":        "Fiche Produit",
    "rule:roaming":              "Roaming",
    "rule:internet":             "Internet",
    "rule:promotion":            "Promotion (pl.)",
    "rule:adsl":                 "ADSL",
    "rule:gpon":                 "GPON",
    "rule:esim":                 "eSIM",
    "rule:tourist":              "Touriste",
    "rule:sim":                  "SIM",
    "rule:vdsl":                 "VDSL",
    "rule:netbox":               "Netbox",
    "rule:4g":                   "4G",
    "rule:activation":           "Activation",
    "rule:validite":             "Validité",
    "rule:tarifaire":            "Tarifaire",
    "rule:argumentaire de vente":"Argumentaire Vente",
    "rule:segmentation":         "Segmentation",
    "rule:confidentiel":         "Confidentiel",
    "rule:back office":          "Back Office",
    "path_marker:clients":       "Path: clients",
    "path_marker:client":        "Path: client",
    "hard_rule:procedure":       "Procédure (strict)",
    "hard_rule:segmentation":    "Segmentation (strict)",
    "hard_rule:benchmark":       "Benchmark (strict)",
    "hard_rule:use case":        "Use Case (strict)",
}

# Mots-clés dans les noms de fichiers → catégorie thématique
TOPIC_KEYWORDS = {
    "Mobile Prépayé": ["hayya", "mriguel", "trankil", "prepay", "prepaid", "prépayé"],
    "Mobile Postpayé": ["oh!", "oh ", "ohmega", "postpaid", "postpayé", "hybride", "hybrid"],
    "Internet Mobile / Data": ["data", "4g", "5g", "internet mobile", "m2m"],
    "Internet Fixe": ["adsl", "vdsl", "gpon", "netbox", "internet fixe", "fibre", "fixe"],
    "Roaming": ["roaming", "roam"],
    "eSIM / SIM": ["esim", "e-sim", " sim"],
    "Touriste": ["tourist", "touriste", "touristiq"],
    "Entreprise / B2B": ["entreprise", "b2b", "gps", "tracking", "m2m"],
    "Offres Générales": [],  # fallback
}

def detect_topic(file_name: str) -> str:
    fn = file_name.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if kws and any(kw in fn for kw in kws):
            return topic
    return "Offres Générales"


# ═══════════════ CHARGEMENT ═══════════════

def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Chargé : {len(data)} chunks")
    return data


# ═══════════════ CALCUL STATS ═══════════════

def compute_stats(data):
    stats = {}

    years             = []
    months            = []
    year_months       = []
    confidentialities = []
    conf_reasons      = []
    text_lengths      = []
    topics            = []

    # Par fichier unique
    file_info = {}

    for chunk in data:
        year  = chunk.get("year", "?")
        month = chunk.get("month", "?")
        conf  = chunk.get("confidentiality", "unknown")
        reason = chunk.get("confidentiality_reason", "unknown")
        text   = chunk.get("chunk_text", "")
        fname  = chunk.get("file_name", "")
        total_ch = chunk.get("total_chunks", 1)

        years.append(year)
        months.append(month)
        year_months.append((year, month))
        confidentialities.append(conf)
        conf_reasons.append(reason)
        text_lengths.append(len(text))
        topics.append(detect_topic(fname))

        if fname not in file_info:
            file_info[fname] = {
                "year": year, "month": month,
                "total_chunks": total_ch,
                "confidentiality": conf,
                "topic": detect_topic(fname),
            }

    stats["total_chunks"]     = len(data)
    stats["total_files"]      = len(file_info)
    stats["years"]            = Counter(years)
    stats["months"]           = Counter(months)
    stats["year_months"]      = Counter(year_months)
    stats["confidentialities"]= Counter(confidentialities)
    stats["conf_reasons"]     = Counter(conf_reasons)
    stats["text_lengths"]     = text_lengths
    stats["topics"]           = Counter(topics)

    # Distribution du nombre de chunks par document (taille doc)
    total_chunks_per_file = [v["total_chunks"] for v in file_info.values()]
    stats["total_chunks_per_file"] = total_chunks_per_file

    # Fichiers par année (unique)
    files_by_year = defaultdict(set)
    for chunk in data:
        files_by_year[chunk.get("year", "?")].add(chunk.get("file_name", ""))
    stats["files_by_year"] = {y: len(s) for y, s in sorted(files_by_year.items())}

    return stats


# ═══════════════ GRAPHIQUE 1 — Chunks & fichiers par année ═══════════════

def plot_by_year(stats, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    years_sorted = sorted(stats["years"].keys())
    chunks_vals  = [stats["years"][y] for y in years_sorted]
    files_vals   = [stats["files_by_year"].get(y, 0) for y in years_sorted]

    # Barres chunks
    bars = ax1.bar(years_sorted, chunks_vals, color=TT_BLUE, edgecolor="white",
                   linewidth=1.5, zorder=3, width=0.6)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, chunks_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(chunks_vals)*0.01,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_title("Chunks extraits par année", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Année"); ax1.set_ylabel("Nombre de chunks")
    ax1.set_ylim(0, max(chunks_vals) * 1.15)

    # Barres fichiers
    bars2 = ax2.bar(years_sorted, files_vals, color=TT_ORANGE, edgecolor="white",
                    linewidth=1.5, zorder=3, width=0.6)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars2, files_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(files_vals)*0.01,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Fichiers PDF uniques par année", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Année"); ax2.set_ylabel("Nombre de fichiers")
    ax2.set_ylim(0, max(files_vals) * 1.15)

    txt = (f"Total chunks     : {stats['total_chunks']}\n"
           f"Total fichiers   : {stats['total_files']}\n"
           f"Période          : 2018 – 2025")
    ax1.text(0.02, 0.97, txt, transform=ax1.transAxes, va="top",
             fontsize=9, color=TT_GRAY,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))

    plt.suptitle("Volume documentaire par année — Corpus Tunisie Telecom",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "01_volume_par_annee.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK 01_volume_par_annee.png")


# ═══════════════ GRAPHIQUE 2 — Heatmap année × mois ═══════════════

def plot_heatmap_year_month(stats, out_dir):
    fig, ax = plt.subplots(figsize=(13, 5))

    years_list  = sorted(set(ym[0] for ym in stats["year_months"]))
    months_list = [f"{i:02d}" for i in range(1, 13)]
    months_labels = [MONTHS_FR[m] for m in months_list]

    matrix = []
    for y in years_list:
        row = [stats["year_months"].get((y, m), 0) for m in months_list]
        matrix.append(row)
    matrix_np = np.array(matrix, dtype=float)

    im = ax.imshow(matrix_np, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(12)); ax.set_xticklabels(months_labels, fontsize=10)
    ax.set_yticks(range(len(years_list))); ax.set_yticklabels(years_list, fontsize=10)
    ax.set_title("Activité documentaire — Chunks par Année × Mois\n"
                 "(valeur = nombre de chunks extraits)",
                 fontsize=12, fontweight="bold", pad=12)

    vmax = matrix_np.max()
    for i in range(len(years_list)):
        for j in range(12):
            val = int(matrix_np[i, j])
            if val == 0:
                continue
            color = "white" if val > vmax * 0.55 else TT_BLUE
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Nombre de chunks", shrink=0.85, pad=0.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "02_heatmap_annee_mois.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK 02_heatmap_annee_mois.png")


# ═══════════════ GRAPHIQUE 3 — Confidentialité & thèmes ═══════════════

def plot_confidentiality_and_topics(stats, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Camembert confidentialité
    conf = stats["confidentialities"]
    labels_c = []
    values_c = []
    colors_c = []
    label_map = {
        "shareable":    ("Partageable",   TT_GREEN),
        "confidential": ("Confidentiel",  TT_RED),
    }
    for k in ["shareable", "confidential"]:
        if k in conf:
            lbl, col = label_map[k]
            labels_c.append(f"{lbl} ({conf[k]})")
            values_c.append(conf[k])
            colors_c.append(col)

    wedges, texts, autotexts = ax1.pie(
        values_c, labels=None, autopct="%1.1f%%",
        colors=colors_c, startangle=90, pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=2.5)
    )
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight("bold"); at.set_color("white")
    ax1.legend(wedges, labels_c, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), fontsize=11, frameon=False)
    ax1.set_title("Confidentialité des chunks\n(shareable_with_client)",
                  fontsize=12, fontweight="bold")

    # Barres horizontales thèmes
    topics = stats["topics"]
    items  = sorted(topics.items(), key=lambda x: x[1])
    labs   = [p[0] for p in items]
    vals   = [p[1] for p in items]
    colors_t = [PALETTE[i % len(PALETTE)] for i in range(len(labs))]

    bars = ax2.barh(labs, vals, color=colors_t, edgecolor="white",
                    linewidth=1.2, height=0.65, zorder=3)
    ax2.grid(axis="x", alpha=0.3, zorder=0)
    total = sum(vals)
    for bar, val in zip(bars, vals):
        ax2.text(val + max(vals)*0.005, bar.get_y() + bar.get_height()/2,
                 f"{val}  ({100*val/total:.1f}%)", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Nombre de chunks", fontsize=10)
    ax2.set_xlim(0, max(vals) * 1.28)
    ax2.set_title("Répartition thématique\n(détectée via noms de fichiers)",
                  fontsize=12, fontweight="bold")

    plt.suptitle("Confidentialité & Thèmes — Corpus Tunisie Telecom",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "03_confidentialite_themes.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK 03_confidentialite_themes.png")


# ═══════════════ GRAPHIQUE 4 — Top raisons de confidentialité ═══════════════

def plot_top_reasons(stats, out_dir):
    fig, ax = plt.subplots(figsize=(11, 8))

    reasons = stats["conf_reasons"]
    top20   = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:20]
    labels  = [REASON_LABELS.get(k, k) for k, _ in reversed(top20)]
    values  = [v for _, v in reversed(top20)]

    # Couleur : hard_rule en rouge, default en gris, rule en bleu
    def get_color(label_key):
        k = list(reasons.keys())[list(reasons.keys()).index(
            next(k for k, _ in top20 if REASON_LABELS.get(k, k) == label_key or k == label_key)
        )]
        if k.startswith("hard_rule"):   return TT_RED
        if k.startswith("default"):     return TT_GRAY
        if k.startswith("path_marker"): return TT_PURPLE
        return TT_BLUE2

    raw_keys_reversed = [k for k, _ in reversed(top20)]
    colors_b = []
    for k in raw_keys_reversed:
        if k.startswith("hard_rule"):   colors_b.append(TT_RED)
        elif k.startswith("default"):   colors_b.append(TT_GRAY)
        elif k.startswith("path_marker"):colors_b.append(TT_PURPLE)
        else:                           colors_b.append(TT_BLUE2)

    bars = ax.barh(labels, values, color=colors_b, edgecolor="white",
                   linewidth=1.2, height=0.7, zorder=3)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.005, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Nombre de chunks", fontsize=11)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_title("Top 20 raisons de classification (confidentiality_reason)\n"
                 "→ Comprendre les règles de partage du corpus",
                 fontsize=12, fontweight="bold", pad=12)

    # Légende couleurs
    legend_handles = [
        mpatches.Patch(color=TT_BLUE2, label="rule: (règle métier)"),
        mpatches.Patch(color=TT_RED,   label="hard_rule: (règle stricte)"),
        mpatches.Patch(color=TT_GRAY,  label="default: (défaut système)"),
        mpatches.Patch(color=TT_PURPLE,label="path_marker: (chemin fichier)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True,
              framealpha=0.9, edgecolor="#ddd")

    txt = (f"Raisons uniques  : {len(reasons)}\n"
           f"Total chunks     : {stats['total_chunks']}")
    ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=TT_GRAY,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(out_dir, "04_raisons_confidentialite.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK 04_raisons_confidentialite.png")


# ═══════════════ GRAPHIQUE 5 — Distribution longueurs chunk_text ═══════════════

def plot_chunk_lengths(stats, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax1, ax2, ax3 = axes

    lengths = stats["text_lengths"]
    mean_l  = statistics.mean(lengths)
    med_l   = statistics.median(lengths)

    # Histogramme
    ax1.hist(lengths, bins=40, color=TT_BLUE, edgecolor="white", alpha=0.85, zorder=3)
    ax1.axvline(mean_l, color=TT_RED,    linestyle="--", linewidth=1.8,
                label=f"Moy. {mean_l:.0f}")
    ax1.axvline(med_l,  color=TT_ORANGE, linestyle="--", linewidth=1.8,
                label=f"Méd. {med_l:.0f}")
    ax1.set_title("Distribution longueurs\nchunk_text (chars)", fontweight="bold")
    ax1.set_xlabel("Caractères"); ax1.set_ylabel("Fréquence")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, zorder=0)

    # Boxplot
    bp = ax2.boxplot(lengths, vert=True, patch_artist=True,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2.5, color=TT_ORANGE))
    bp["boxes"][0].set_facecolor(TT_BLUE4)
    bp["boxes"][0].set_edgecolor(TT_BLUE)
    for w in bp["whiskers"]: w.set_color(TT_BLUE)
    ax2.set_ylabel("Longueur (chars)")
    ax2.set_title("Boxplot longueurs\nchunk_text", fontweight="bold")
    ax2.set_xticklabels(["chunk_text"])
    ax2.grid(axis="y", alpha=0.3)

    # Tableau stats
    ax3.axis("off")
    stats_data = [
        ["Statistique", "Valeur"],
        ["Total chunks",  f"{len(lengths)}"],
        ["Min",           f"{min(lengths)} chars"],
        ["Max",           f"{max(lengths)} chars"],
        ["Médiane",       f"{med_l:.0f} chars"],
        ["Moyenne",       f"{mean_l:.0f} chars"],
        ["Écart-type",    f"{statistics.stdev(lengths):.0f}"],
        ["P25",           f"{np.percentile(lengths,25):.0f}"],
        ["P75",           f"{np.percentile(lengths,75):.0f}"],
        ["Chunks vides",  f"{sum(1 for l in lengths if l < 20)}"],
        ["Chunks pleins", f"{sum(1 for l in lengths if l >= 500)}"],
    ]
    table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                      cellLoc="center", loc="center",
                      colColours=["#f0f4ff", TT_BLUE4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.6)
    ax3.set_title("Statistiques descriptives", fontweight="bold")

    plt.suptitle("Distribution des longueurs de chunks — Corpus Tunisie Telecom",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "05_longueurs_chunks.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK 05_longueurs_chunks.png")


# ═══════════════ GRAPHIQUE 6 — Taille des documents (total_chunks) ═══════════════

def plot_doc_size_distribution(stats, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    tc = stats["total_chunks_per_file"]
    tc_counter = Counter(tc)

    # Histogramme taille docs
    ax1.hist(tc, bins=range(1, max(tc)+2), color=TT_BLUE3, edgecolor="white",
             alpha=0.9, zorder=3, align="left")
    ax1.axvline(statistics.mean(tc), color=TT_RED, linestyle="--", linewidth=1.8,
                label=f"Moy. {statistics.mean(tc):.1f} chunks/doc")
    ax1.axvline(statistics.median(tc), color=TT_ORANGE, linestyle="--", linewidth=1.8,
                label=f"Méd. {statistics.median(tc):.0f} chunks/doc")
    ax1.set_title("Distribution des tailles de documents\n(total_chunks par fichier PDF)",
                  fontweight="bold")
    ax1.set_xlabel("Nombre de chunks par document")
    ax1.set_ylabel("Nombre de fichiers PDF")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, zorder=0)

    # Répartition par taille de doc (barres groupées)
    bins = [(1, "1 chunk\n(petit)"),
            (2, "2 chunks"),
            (3, "3 chunks"),
            (4, "4 chunks"),
            (5, "5 chunks"),
            (6, "6+ chunks\n(grand)")]
    labels_g = []
    vals_g   = []
    for threshold, label in bins:
        if threshold == 6:
            cnt = sum(v for k, v in tc_counter.items() if k >= 6)
        else:
            cnt = tc_counter.get(threshold, 0)
        labels_g.append(label)
        vals_g.append(cnt)

    colors_g = [TT_BLUE if v == max(vals_g) else TT_BLUE3 for v in vals_g]
    bars = ax2.bar(labels_g, vals_g, color=colors_g, edgecolor="white",
                   linewidth=1.5, zorder=3, width=0.65)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    total_f = sum(vals_g)
    for bar, val in zip(bars, vals_g):
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(vals_g)*0.01,
                 f"{val}\n({100*val/total_f:.0f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("Fichiers par catégorie de taille", fontweight="bold")
    ax2.set_ylabel("Nombre de fichiers PDF")
    ax2.set_ylim(0, max(vals_g) * 1.20)

    txt = (f"Total fichiers PDF      : {stats['total_files']}\n"
           f"Total chunks            : {stats['total_chunks']}\n"
           f"Moy. chunks/doc         : {statistics.mean(tc):.1f}\n"
           f"Max chunks dans un doc  : {max(tc)}")
    ax1.text(0.97, 0.97, txt, transform=ax1.transAxes, ha="right", va="top",
             fontsize=9, color=TT_GRAY,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.9))

    plt.suptitle("Taille des documents PDF — Corpus Tunisie Telecom",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "06_taille_documents.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK 06_taille_documents.png")


# ═══════════════ MAIN ═══════════════

def main():
    print("=" * 60)
    print("  EDA — Corpus chunks.json Tunisie Telecom")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nChargement : {INPUT_FILE}")
    data = load_data()

    print("\nCalcul des statistiques...")
    stats = compute_stats(data)

    print(f"\nRÉSUMÉ DU CORPUS :")
    print(f"  Total chunks            : {stats['total_chunks']}")
    print(f"  Total fichiers PDF      : {stats['total_files']}")
    print(f"  Période                 : {min(stats['years'])} – {max(stats['years'])}")
    print(f"  Longueur moy. chunk     : {statistics.mean(stats['text_lengths']):.0f} chars")
    print(f"  Confidentiels           : {stats['confidentialities'].get('confidential',0)}")
    print(f"  Partageables            : {stats['confidentialities'].get('shareable',0)}")
    print(f"  Raisons uniques         : {len(stats['conf_reasons'])}")
    print(f"  Thèmes détectés         : {len(stats['topics'])}")

    print(f"\nGénération des graphiques dans : {OUTPUT_DIR}\n")

    plot_by_year(stats, OUTPUT_DIR)
    plot_heatmap_year_month(stats, OUTPUT_DIR)
    plot_confidentiality_and_topics(stats, OUTPUT_DIR)
    plot_top_reasons(stats, OUTPUT_DIR)
    plot_chunk_lengths(stats, OUTPUT_DIR)
    plot_doc_size_distribution(stats, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"  6 GRAPHIQUES GÉNÉRÉS")
    print(f"{'='*60}")
    print("  01_volume_par_annee.png          → Chunks & fichiers par année")
    print("  02_heatmap_annee_mois.png        → Activité documentaire (heatmap)")
    print("  03_confidentialite_themes.png    → Confidentialité + thèmes")
    print("  04_raisons_confidentialite.png   → Top 20 raisons de classification")
    print("  05_longueurs_chunks.png          → Distribution longueurs chunk_text")
    print("  06_taille_documents.png          → Taille des PDFs (nb chunks)")
    print("\n  Pour la soutenance : 02 + 03 en priorité")


if __name__ == "__main__":
    main()