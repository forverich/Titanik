import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Titanic Survival Analysis")
st.caption("Business IT 2 (60SCI006) · Vietnamese-German University · Prof. Do Duc Tan")
st.markdown("---")

# ── Load & prepare data (mirrors the .qmd setup chunk) ────────────────────────
@st.cache_data
def load_data():
    titanic = sns.load_dataset("titanic")
    titanic["Sex"]      = titanic["sex"].str.capitalize()
    titanic["Survived"] = titanic["survived"]
    titanic["Pclass"]   = titanic["pclass"].astype(str)
    titanic["Age"]      = titanic["age"].fillna(titanic["age"].median())
    titanic["Fare"]     = titanic["fare"]
    titanic["SibSp"]    = titanic["sibsp"]
    titanic["Parch"]    = titanic["parch"]
    titanic["Embarked"] = titanic["embarked"].fillna("S").str.upper()
    titanic = titanic[titanic["Embarked"].notna() & (titanic["Embarked"] != "")]
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] + 1
    return titanic

titanic = load_data()

survival_colors = ["#B22222", "#1F78B4"]
class_colors    = ["#1F78B4", "#33A02C", "#E31A1C"]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.autolayout": True,
})

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio(
    "Go to figure:",
    [
        "Fig 1 · Survival by Gender",
        "Fig 2 · Survival by Class",
        "Fig 3 · Survival by Port & Class",
        "Fig 4 · Age Distribution",
        "Fig 5 · Age Density by Survival",
        "Fig 6 · Age by Passenger Class",
        "Fig 7 · Fare Distribution (Log)",
        "Fig 8 · Family Size Distribution",
        "Fig 9 · Survival by Family Size",
        "Fig 10 · Age vs Fare (Scatter)",
        "Fig 11 · Age vs Fare (Faceted)",
        "Fig 12 · Spearman Correlation Heatmap",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Group members**")
st.sidebar.markdown(
    "Ly Tam Anh · Pham Huynh Dam\n\n"
    "Nguyen Khanh Huan · Nguyen Thi Minh Tuong\n\n"
    "Huynh Dong Nghi"
)

# ── Helper: show figure ────────────────────────────────────────────────────────
def show_fig(title, insight_md):
    st.subheader(title)
    st.pyplot(plt.gcf())
    plt.close()
    with st.expander("💡 Discussion & Insights"):
        st.markdown(insight_md)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1
# ══════════════════════════════════════════════════════════════════════════════
if section == "Fig 1 · Survival by Gender":
    df_fig1 = pd.crosstab(titanic["Sex"], titanic["Survived"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    df_fig1.plot(kind="bar", stacked=True, color=survival_colors,
                 edgecolor="white", width=0.5, ax=ax)
    for p in ax.patches:
        w, h = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if h > 0:
            ax.text(x + w/2, y + h/2, f"{h:.1f}%",
                    ha="center", va="center", color="white", weight="bold")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(["No", "Yes"], title="Survived", loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    show_fig("Figure 1 · Survival Rate by Gender",
        "- **High Female Survival Rate:** ~**74.2%** of female passengers survived, strongly supporting the *women and children first* protocol.\n"
        "- **Low Male Survival Rate:** Only ~**18.9%** of male passengers survived. Gender was a critical determining factor.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 2 · Survival by Class":
    df_counts = pd.crosstab(titanic["Pclass"], titanic["Survived"]).reindex(["1","2","3"])
    df_pct    = pd.crosstab(titanic["Pclass"], titanic["Survived"],
                            normalize="index").reindex(["1","2","3"]) * 100

    fig, ax = plt.subplots(figsize=(6, 6))
    df_counts.plot(kind="bar", stacked=False, color=survival_colors,
                   edgecolor="white", width=0.6, ax=ax)
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height > 0:
                pct = df_pct.iloc[j, i]
                ax.text(bar.get_x() + bar.get_width()/2, height/2,
                        f"{pct:.1f}%", ha="center", va="center",
                        color="white", weight="bold", fontsize=9)
                ax.annotate(f"{int(height)}",
                            (bar.get_x() + bar.get_width()/2, height),
                            ha="center", va="bottom", fontsize=12, color="black",
                            xytext=(0, 3), textcoords="offset points")
    ax.set_xlabel("Passenger Class (Pclass)")
    ax.set_ylabel("Passenger Count")
    ax.set_xticklabels(["1","2","3"], rotation=0)
    ax.legend(["No","Yes"], title="Survived")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    show_fig("Figure 2 · Survival Rate by Passenger Class",
        "- **First Class privilege:** 63.0% of Class 1 passengers survived — socio-economic status and proximity to lifeboats were decisive.\n"
        "- **Third Class toll:** Despite being the largest group, 75.8% of Class 3 passengers died.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 3 · Survival by Port & Class":
    g = sns.FacetGrid(titanic, row="Pclass", row_order=["1","2","3"], height=3, aspect=2.5)
    g.map_dataframe(sns.countplot, y="Embarked", hue="Survived",
                    palette=survival_colors, edgecolor="white", order=["C","Q","S"])
    g.set_axis_labels("Passenger Count", "Port of Embarkation")
    g.add_legend(title="Survived")
    for text, label in zip(g._legend.get_texts(), ["No","Yes"]):
        text.set_text(label)
    for ax in g.axes.flat:
        for container in ax.containers:
            for bar in container:
                width = bar.get_width()
                if width > 0:
                    ax.annotate(f"{int(width)}",
                                (width, bar.get_y() + bar.get_height()/2),
                                ha="left", va="center", fontsize=11,
                                color="black", weight="bold",
                                xytext=(5, 0), textcoords="offset points")

    st.subheader("Figure 3 · Survival by Port, Faceted by Class")
    st.pyplot(g.figure)
    plt.close()
    with st.expander("💡 Discussion & Insights"):
        st.markdown(
            "- Southampton (S) contributed the most passengers in all classes and showed high absolute death counts.\n"
            "- Cherbourg (C) passengers, concentrated in Class 1, had relatively better survival outcomes.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 4 · Age Distribution":
    age_bins = list(range(0, 86, 5))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(titanic["Age"].dropna(), bins=age_bins, color="#1F78B4", edgecolor="white")
    ax.set_xlabel("Age", fontsize=11, weight="bold")
    ax.set_ylabel("Frequency", fontsize=11, weight="bold")
    ax.set_xticks(range(0, 81, 20))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 4 · Age Distribution of Passengers",
        "- **Dominant young adults:** Dense concentration between ages 20–35 (unimodal, right-skewed).\n"
        "- **Children bump:** Secondary accumulation at 0–5 reflects traveling families.\n"
        "- **Sparse elderly:** Sharp drop after age 50.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 5 · Age Density by Survival":
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=titanic.dropna(subset=["Age"]), x="Age",
                hue="Survived", fill=True, palette=survival_colors,
                alpha=0.5, common_norm=False, ax=ax)
    ax.set_xlabel("Age", fontsize=11, weight="bold")
    ax.set_ylabel("Density", fontsize=11, weight="bold")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    legend = ax.get_legend()
    if legend:
        legend.set_title("Survived")
        for text, label in zip(legend.get_texts(), ["No","Yes"]):
            text.set_text(label)
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 5 · Age Distribution by Survival Status",
        "- **Children first:** Ages 0–10 show a higher survival density — institutional prioritization in action.\n"
        "- **Young adult risk:** 20–30 age bracket had peak non-survival density.\n"
        "- **Convergence at 45+:** Age neutralized structural advantages at advanced age.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 6 · Age by Passenger Class":
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=titanic.dropna(subset=["Age"]), x="Pclass", y="Age",
                   hue="Pclass", palette=class_colors, order=["1","2","3"],
                   inner=None, ax=ax)
    if ax.get_legend():
        ax.get_legend().remove()
    for collection in ax.collections:
        collection.set_alpha(0.5)
    ax.set_xlabel("Passenger Class", fontsize=11, weight="bold")
    ax.set_ylabel("Age", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 6 · Age Distribution by Passenger Class",
        "- **Class 1 — Affluence & maturity:** Widest body between 30–50; established, wealthy adults.\n"
        "- **Class 2 — Middle-class workforce:** Symmetrical, compact, centered at 25–38.\n"
        "- **Class 3 — Youth & migration:** Bottom-heavy, peaks sharply in late teens/20s.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 7 · Fare Distribution (Log)":
    df_fare = titanic[titanic["Fare"] > 0].dropna(subset=["Fare"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_fare, x="Pclass", y="Fare", hue="Pclass",
                palette=class_colors, order=["1","2","3"],
                boxprops=dict(alpha=0.8),
                flierprops=dict(alpha=0.4, marker="o",
                                markerfacecolor="grey", markeredgecolor="none"),
                ax=ax)
    ax.set_yscale("log")
    ax.set_yticks([10, 30, 100, 300])
    ax.set_yticklabels(["10","30","100","300"])
    if ax.get_legend():
        ax.get_legend().remove()
    ax.set_xlabel("Passenger Class", fontsize=11, weight="bold")
    ax.set_ylabel("Fare (Log Scale)", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 7 · Fare Distribution by Class (Log Scale)",
        "- **Extreme stratification:** Class 1 median (~£60) vastly exceeds Class 3 median (~£8).\n"
        "- **Class 3 outliers:** A few steerage passengers paid inflated prices (late bookings, large groups).\n"
        "- **Overlap exists:** Cheapest Class 1 berths overlap with premium Class 2/3 prices.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 8 · Family Size Distribution":
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=titanic, x="FamilySize", color="#1F78B4",
                  edgecolor="white", ax=ax)
    ax.set_xlabel("Family Size", fontsize=11, weight="bold")
    ax.set_ylabel("Frequency", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}",
                            (bar.get_x() + bar.get_width()/2, height),
                            ha="center", va="bottom", fontsize=10,
                            color="black", weight="bold",
                            xytext=(0, 3), textcoords="offset points")
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 8 · Family Size Distribution",
        "- **Solo travelers dominate:** Family size = 1 is by far the most common (migrant workforce).\n"
        "- **Nuclear families:** Sizes 2–4 maintain a stable presence.\n"
        "- **Large families (7+):** Rare but disproportionately lethal — coordination failure in crisis.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 9 · Survival by Family Size":
    family_survival = titanic.groupby("FamilySize")["Survived"].mean().reset_index()
    family_survival["SurvivalRate"] = family_survival["Survived"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=family_survival, x="FamilySize", y="SurvivalRate",
                color="#2CA02C", edgecolor="white", ax=ax)
    ax.set_xlabel("Family Size", fontsize=11, weight="bold")
    ax.set_ylabel("Survival Rate (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    (p.get_x() + p.get_width()/2, height),
                    ha="center", va="bottom", fontsize=9,
                    color="black", weight="bold",
                    xytext=(0, 3), textcoords="offset points")
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 9 · Survival Rate by Family Size",
        "- **Solo penalty:** Travelers alone had only ~30% survival.\n"
        "- **Sweet spot:** Sizes 3–4 exceeded 50–72% survival — coordinated + financially advantaged.\n"
        "- **Catastrophic threshold:** Families of 7+ had near-zero survival; language barriers & refusal to split.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 10 · Age vs Fare (Scatter)":
    df_fig10 = titanic.dropna(subset=["Age","Fare"]).copy()
    df_fig10 = df_fig10[df_fig10["Fare"] > 0]
    np.random.seed(42)
    df_fig10["Age_jitter"] = df_fig10["Age"] + np.random.uniform(-0.5, 0.5, size=len(df_fig10))
    surv_colors_map = {0: "#D62728", 1: "#1F78B4"}

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_fig10, x="Age_jitter", y="Fare", hue="Survived",
                    palette=surv_colors_map, alpha=0.3, s=35, linewidth=0,
                    legend=True, ax=ax)
    for surv_val in [0, 1]:
        subset = df_fig10[df_fig10["Survived"] == surv_val]
        sns.regplot(data=subset, x="Age", y="Fare", scatter=False,
                    lowess=True, color=surv_colors_map[surv_val],
                    line_kws={"linewidth": 2}, ax=ax)
    ax.set_yscale("log")
    ax.set_yticks([10, 30, 100, 300])
    ax.set_yticklabels(["10","30","100","300"])
    ax.set_xlabel("Age", fontsize=11, weight="bold")
    ax.set_ylabel("Fare (Log Scale)", fontsize=11, weight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:2], labels=["Died","Survived"],
              title="Survived", bbox_to_anchor=(1.02, 0.7),
              loc="upper left", frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()

    show_fig("Figure 10 · Age vs Fare, Colored by Survival",
        "- **Economic shield:** Survival trend line floats consistently above the death line across all ages.\n"
        "- **Poverty-age cluster:** Deaths concentrated at low fares (£7–10) for ages 18–45.\n"
        "- **Children anomaly:** Protocol temporarily overrode economic status for ages 0–10.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 11 · Age vs Fare (Faceted)":
    df_fig10 = titanic.dropna(subset=["Age","Fare"]).copy()
    df_fig10 = df_fig10[df_fig10["Fare"] > 0]
    np.random.seed(42)
    df_fig10["Age_jitter"] = df_fig10["Age"] + np.random.uniform(-0.5, 0.5, size=len(df_fig10))
    surv_colors_map = {0: "#D62728", 1: "#1F78B4"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, surv_val in zip(axes, [0, 1]):
        subset = df_fig10[df_fig10["Survived"] == surv_val]
        sns.scatterplot(data=subset, x="Age_jitter", y="Fare",
                        color=surv_colors_map[surv_val], alpha=0.3,
                        s=35, linewidth=0, ax=ax)
        sns.regplot(data=subset, x="Age", y="Fare", scatter=False,
                    lowess=True, color=surv_colors_map[surv_val],
                    line_kws={"linewidth": 2}, ax=ax)
        ax.set_yscale("log")
        ax.set_yticks([10, 30, 100, 300])
        ax.set_yticklabels(["10","30","100","300"])
        ax.set_xlabel("Age", fontsize=11, weight="bold")
        ax.set_title(str(surv_val), fontsize=11, weight="bold", pad=10)
        ax.grid(alpha=0.3, linestyle="--")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Fare (Log Scale)", fontsize=11, weight="bold")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F78B4", markersize=8),
    ]
    fig.legend(handles=handles, labels=["0 = Died","1 = Survived"],
               title="Survival Status", bbox_to_anchor=(1.02, 1.0),
               loc="upper left", frameon=False)
    plt.tight_layout()

    show_fig("Figure 11 · Age vs Fare, Faceted by Survival",
        "- **Shape of mortality (Panel 0):** Dense pooling at low fares (£7–10) for ages 20–40.\n"
        "- **Shape of rescue (Panel 1):** Survivors spread vertically across fare scale; higher fares = linear advantage at every age.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Fig 12 · Spearman Correlation Heatmap":
    corr_data  = titanic[["Age","Fare","SibSp","Parch"]].dropna()
    cor_matrix = corr_data.corr(method="spearman").iloc[::-1]
    cmap = LinearSegmentedColormap.from_list("custom_diverging", ["#1F78B4","white","#E31A1C"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0,
                linewidths=0.5, cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    show_fig("Figure 12 · Spearman Correlation Heatmap",
        "- **SibSp ↔ Parch (0.43):** Strongest link — cohesive multi-generational family travel.\n"
        "- **SibSp/Parch ↔ Fare (0.41–0.42):** Larger families incurred higher collective ticket costs.\n"
        "- **Age ↔ Parch (−0.25):** Younger passengers had more parents aboard (they were the children).\n"
        "- **Age ↔ Fare (0.14):** Wealth was not concentrated exclusively in older passengers.")
