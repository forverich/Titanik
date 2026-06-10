import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Titanic Survival Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# COLOR SYSTEM  (vivid / sặc sỡ)
# ══════════════════════════════════════════════════════════════════════════════
C_SURV   = "#00C2A8"   # teal-green  → survived
C_DIED   = "#FF5470"   # coral-pink  → died
C_CLASS  = {"1": "#7B5CFF", "2": "#FF9F1C", "3": "#2EC4F0"}   # vivid per class
VIVID    = ["#7B5CFF", "#FF5470", "#00C2A8", "#FF9F1C", "#2EC4F0", "#FF6FD8", "#A0E548"]
SEQ_SCALE = "Plasma"   # vivid sequential

SURV_MAP = {"Died": C_DIED, "Survived": C_SURV}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#2b2d42"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=46, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                font=dict(color="#2b2d42", size=12), bgcolor="rgba(255,255,255,0)"),
    xaxis=dict(title_font=dict(color="#2b2d42", size=13), tickfont=dict(color="#4a4d6e")),
    yaxis=dict(title_font=dict(color="#2b2d42", size=13), tickfont=dict(color="#4a4d6e")),
    colorway=VIVID,
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — bright, colorful theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg,#f6f7ff 0%, #fdf2f8 100%); }

h1,h2,h3,h4 { font-family:'Plus Jakarta Sans',sans-serif !important; color:#1f2140 !important; }

/* High-contrast text in the MAIN area (sidebar handled separately below) */
[data-testid="stMain"] label,
[data-testid="stMain"] label *,
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
section.main label { color:#1f2140 !important; font-weight:600 !important; }
[data-testid="stMain"] .stMarkdown p,
[data-testid="stMain"] .stMarkdown li,
section.main .stMarkdown p { color:#2b2d42 !important; }
[data-testid="stMain"] [data-baseweb="radio"] *,
[data-testid="stMain"] [data-baseweb="checkbox"] *,
[data-testid="stMain"] [data-baseweb="select"] *,
[data-testid="stMain"] [data-testid="stExpander"] *,
section.main [data-baseweb="radio"] *,
section.main [data-baseweb="select"] * { color:#2b2d42 !important; }
[data-testid="stMain"] summary, .streamlit-expanderHeader { color:#1f2140 !important; font-weight:700 !important; }

/* Hero */
.hero{
  border-radius:22px; padding:2.6rem 2.4rem; margin-bottom:1.6rem; color:#fff;
  background: linear-gradient(120deg,#7B5CFF 0%,#FF5470 50%,#FF9F1C 100%);
  box-shadow:0 18px 40px -16px rgba(123,92,255,.5);
}
.hero h1{ color:#fff !important; font-size:2.9rem; font-weight:800; margin:.2rem 0 .4rem; line-height:1.08; }
.hero p{ color:rgba(255,255,255,.92) !important; font-size:1.05rem; font-weight:300; margin:0; }
.pill{ display:inline-block; background:rgba(255,255,255,.22); border:1px solid rgba(255,255,255,.4);
  padding:.25rem .8rem; border-radius:30px; font-size:.74rem; font-weight:600;
  letter-spacing:.6px; text-transform:uppercase; margin-right:.5rem; }

/* Stat cards */
.stat{ border-radius:18px; padding:1.3rem 1.4rem; color:#fff; height:100%;
  box-shadow:0 12px 26px -14px rgba(0,0,0,.4); }
.stat .num{ font-family:'Plus Jakarta Sans'; font-size:2rem; font-weight:800; line-height:1; }
.stat .lab{ font-size:.78rem; font-weight:600; letter-spacing:.6px; text-transform:uppercase; opacity:.9; margin-top:.3rem;}
.g1{ background:linear-gradient(135deg,#7B5CFF,#A18BFF); }
.g2{ background:linear-gradient(135deg,#00C2A8,#46E0C8); }
.g3{ background:linear-gradient(135deg,#FF9F1C,#FFC75F); }
.g4{ background:linear-gradient(135deg,#FF5470,#FF8BA0); }

/* Section heading */
.sec{ font-size:1.5rem; font-weight:800; margin:2rem 0 .2rem;
  background:linear-gradient(90deg,#7B5CFF,#FF5470); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; }
.sec-sub{ color:#4a4d6e; font-size:.92rem; margin-bottom:1rem; }

/* Control panel */
.panel{ background:#fff; border:1px solid #eceaf6; border-radius:16px;
  padding:1rem 1.2rem; box-shadow:0 8px 22px -18px rgba(0,0,0,.4); margin-bottom:.6rem; }

/* Insight chip */
.chip{ background:#fff; border-left:5px solid #7B5CFF; border-radius:0 12px 12px 0;
  padding:.9rem 1.1rem; margin:.6rem 0; font-size:.9rem; color:#3a3d5c; line-height:1.6;
  box-shadow:0 6px 18px -16px rgba(0,0,0,.5);}
.chip b{ color:#7B5CFF; }

/* Sidebar */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#1f2140,#2b2356) !important; }
[data-testid="stSidebar"] *{ color:#eceaff !important; }

/* Member cards */
.mem{ background:#fff; border:1px solid #eceaf6; border-radius:14px; padding:.9rem;
  text-align:center; box-shadow:0 6px 18px -16px rgba(0,0,0,.5); }
.mem .nm{ font-family:'Plus Jakarta Sans'; font-weight:800; font-size:.9rem; color:#1f2140; }
.mem .id{ font-size:.72rem; color:#7B5CFF; margin:.2rem 0 .4rem; }
.mem .rl{ font-size:.74rem; color:#50536e; line-height:1.45; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    t = sns.load_dataset("titanic")
    df = pd.DataFrame({
        "Survived":   t["survived"],
        "Pclass":     t["pclass"].astype(str),
        "Sex":        t["sex"].str.capitalize(),
        "Age":        t["age"].fillna(t["age"].median()),
        "Fare":       t["fare"],
        "SibSp":      t["sibsp"],
        "Parch":      t["parch"],
        "Embarked":   t["embarked"].fillna("S").str.upper(),
    })
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["AgeGroup"]   = pd.cut(df["Age"], [0, 12, 19, 29, 39, 49, 59, 120],
                              labels=["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"])
    df["Status"]     = df["Survived"].map({0: "Died", 1: "Survived"})
    df["Port"]       = df["Embarked"].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS  (pure functions → return plotly figures, easy to test)
# ══════════════════════════════════════════════════════════════════════════════
def survival_chart(df, dim_col, dim_label, mode):
    """Bar chart of survival, grouped by a categorical dimension."""
    if mode == "Survival rate (%)":
        g = (df.groupby(dim_col, observed=True)["Survived"].mean() * 100).reset_index()
        g.columns = [dim_col, "Rate"]
        g = g.sort_values("Rate", ascending=False)
        fig = px.bar(g, x=dim_col, y="Rate", color="Rate",
                     color_continuous_scale=SEQ_SCALE, text=g["Rate"].round(1).astype(str) + "%")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Survival rate (%)",
                          xaxis_title=dim_label, yaxis_range=[0, 105])
    else:
        g = df.groupby([dim_col, "Status"], observed=True).size().reset_index(name="Count")
        fig = px.bar(g, x=dim_col, y="Count", color="Status", barmode="group",
                     color_discrete_map=SURV_MAP, text="Count")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(xaxis_title=dim_label, yaxis_title="Passengers")
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def dist_chart(df, var, kind, split):
    """Distribution of a numeric variable, optionally split by survival."""
    color = "Status" if split else None
    cmap = SURV_MAP if split else None
    if kind == "Histogram":
        fig = px.histogram(df, x=var, color=color, nbins=30, opacity=.8,
                           color_discrete_map=cmap, barmode="overlay" if split else "relative")
        if not split:
            fig.update_traces(marker_color="#7B5CFF")
    else:
        # box / violin need an x category, otherwise the axis renders "undefined"
        x = "Status" if split else "_grp"
        pdf = df if split else df.assign(_grp="All passengers")
        if kind == "Box":
            fig = px.box(pdf, x=x, y=var, color=color, color_discrete_map=cmap, points="suspectedoutliers")
        else:  # Violin
            fig = px.violin(pdf, x=x, y=var, color=color, color_discrete_map=cmap, box=True, points=False)
        if not split:
            fig.update_traces(fillcolor="rgba(123,92,255,0.55)", line_color="#7B5CFF", marker_color="#7B5CFF")
        fig.update_xaxes(title="")
        if var == "Fare":
            fig.update_yaxes(type="log")
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def scatter_chart(df, x, y, color_by, log_y):
    cmap = SURV_MAP if color_by == "Status" else (C_CLASS if color_by == "Pclass" else None)
    fig = px.scatter(df, x=x, y=y, color=color_by, opacity=.55,
                     color_discrete_map=cmap, color_continuous_scale=SEQ_SCALE,
                     hover_data=["Sex", "Pclass", "Age", "Fare"])
    fig.update_traces(marker=dict(size=8, line=dict(width=0)))
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def corr_heatmap(df):
    cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Survived"]
    m = df[cols].corr(method="spearman").round(2)
    fig = px.imshow(m, text_auto=True, aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r")
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
data = load_data()

# ── Sidebar filters (apply to every chart) ──
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    f_class = st.multiselect("Passenger class", ["1", "2", "3"], default=["1", "2", "3"])
    f_sex   = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    f_age   = st.slider("Age range", 0, 80, (0, 80))
    st.markdown("---")
    st.caption("Dataset · seaborn `titanic` (891 passengers)\n\nBusiness IT 2 · 60SCI006 · Prof. Do Duc Tan")

df = data[
    data["Pclass"].isin(f_class) &
    data["Sex"].isin(f_sex) &
    data["Age"].between(*f_age)
].copy()

# ── Hero ──
st.markdown("""
<div class='hero'>
  <span class='pill'>Python · Data Science</span><span class='pill'>Plotly Interactive</span>
  <h1>Titanic Survival Explorer 🚢</h1>
  <p>Tune the variables below — every chart updates live. No iceberg required.</p>
</div>
""", unsafe_allow_html=True)

# ── Stat cards (reflect current filters) ──
total = len(df)
surv  = int(df["Survived"].sum())
rate  = (surv / total * 100) if total else 0
avg_fare = df["Fare"].mean() if total else 0
cards = [
    ("g1", f"{total}", "Passengers"),
    ("g2", f"{surv}", "Survivors"),
    ("g3", f"{rate:.1f}%", "Survival rate"),
    ("g4", f"£{avg_fare:.0f}", "Avg fare"),
]
cols = st.columns(4)
for col, (g, num, lab) in zip(cols, cards):
    col.markdown(f"<div class='stat {g}'><div class='num'>{num}</div><div class='lab'>{lab}</div></div>",
                 unsafe_allow_html=True)

if total == 0:
    st.warning("No passengers match the current filters — widen them in the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 1 · SURVIVAL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec'>1 · Survival Explorer</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-sub'>Pick a variable and see how survival breaks down by it.</div>", unsafe_allow_html=True)

DIMS = {"Gender": "Sex", "Passenger class": "Pclass", "Port of embarkation": "Port",
        "Family size": "FamilySize", "Age group": "AgeGroup"}
cc1, cc2 = st.columns([2, 1])
dim_label = cc1.selectbox("Break survival down by", list(DIMS.keys()))
mode      = cc2.radio("Show", ["Survival rate (%)", "Passenger count"], horizontal=False)
st.plotly_chart(survival_chart(df, DIMS[dim_label], dim_label, mode), use_container_width=True)

st.markdown("<div class='chip'><b>Tip:</b> switch to <b>Gender</b> or <b>Passenger class</b> in rate mode — "
            "the gap between groups is the single clearest signal in the whole dataset.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2 · DISTRIBUTION EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec'>2 · Distribution Explorer</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-sub'>How is a numeric variable spread out — and does survival shift it?</div>", unsafe_allow_html=True)

d1, d2, d3 = st.columns([1.4, 1.4, 1])
var   = d1.selectbox("Variable", ["Age", "Fare", "FamilySize"])
kind  = d2.radio("Chart type", ["Histogram", "Box", "Violin"], horizontal=True)
split = d3.toggle("Split by survival", value=True)
st.plotly_chart(dist_chart(df, var, kind, split), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 · RELATIONSHIP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec'>3 · Relationship Explorer</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-sub'>Plot any two numbers against each other, colored by a third variable.</div>", unsafe_allow_html=True)

NUMS = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
r1, r2, r3, r4 = st.columns(4)
x_var = r1.selectbox("X axis", NUMS, index=0)
y_var = r2.selectbox("Y axis", NUMS, index=1)
color_by = r3.selectbox("Color by", ["Status", "Sex", "Pclass"])
log_y = r4.toggle("Log Y axis", value=(y_var == "Fare"))
st.plotly_chart(scatter_chart(df, x_var, y_var, color_by, log_y), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec'>4 · Correlation Heatmap</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-sub'>Spearman correlation between the numeric variables (hover for exact values).</div>", unsafe_allow_html=True)
st.plotly_chart(corr_heatmap(df), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# KEY TAKEAWAYS + TEAM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec'>Key Takeaways</div>", unsafe_allow_html=True)
takeaways = [
    ("🚺", "Gender", "≈74% female vs ≈19% male — the starkest single predictor."),
    ("🎫", "Class", "1st 63% · 2nd 47% · 3rd 24% — wealth was literally life-saving."),
    ("💷", "Fare", "Higher fare = upper deck = closer to lifeboats."),
    ("👨‍👩‍👧", "Family size", "Sweet spot 3–4; solo ~30%, very large families near 0%."),
]
tc = st.columns(4)
for col, (ic, t, d) in zip(tc, takeaways):
    col.markdown(f"<div class='mem'><div style='font-size:1.6rem'>{ic}</div>"
                 f"<div class='nm'>{t}</div><div class='rl'>{d}</div></div>", unsafe_allow_html=True)

with st.expander("📖 Project intro, conclusion & group members"):
    st.markdown("""
**Introduction.** On 15 April 1912 the RMS Titanic sank after striking an iceberg, killing more than
1,500 people. The seaborn `titanic` dataset records 891 passengers; missing ages were imputed with the
median and missing ports with Southampton.

**Conclusion.** Survival was not random — it was structured by gender, class, fare (deck position) and
family size acting together. "Women and children first" plus economic proximity to the lifeboats
explains most of the variation you can explore in the charts above.
""")
    members = [
        ("Ly Tam Anh", "102240293", "Leader · Report design"),
        ("Pham Huynh Dam", "10625070", "Data repair"),
        ("Nguyen Khanh Huan", "10625062", "Exercise summary"),
        ("Nguyen Thi Minh Tuong", "10620511", "Report design"),
        ("Huynh Dong Nghi", "10625085", "Report design"),
    ]
    mc = st.columns(len(members))
    for col, (nm, sid, rl) in zip(mc, members):
        col.markdown(f"<div class='mem'><div class='nm'>{nm}</div>"
                     f"<div class='id'>{sid}</div><div class='rl'>{rl}</div></div>", unsafe_allow_html=True)
