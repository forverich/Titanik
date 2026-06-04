import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(178,34,34,0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 20% 80%, rgba(31,120,180,0.10) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: #f5f0e8;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(245,240,232,0.55);
    margin: 0 0 1.5rem 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(178,34,34,0.25);
    border: 1px solid rgba(178,34,34,0.5);
    color: #e87575;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 0.5rem;
}
.hero-badge.blue {
    background: rgba(31,120,180,0.2);
    border-color: rgba(31,120,180,0.5);
    color: #74b3e0;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 140px;
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #B22222);
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f5f0e8;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-label {
    font-size: 0.78rem;
    color: rgba(245,240,232,0.45);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f5f0e8;
    margin: 2.5rem 0 0.25rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.section-sub {
    color: rgba(245,240,232,0.45);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    font-weight: 300;
}

/* Insight cards */
.insight {
    background: #111827;
    border-left: 3px solid #B22222;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.92rem;
    color: rgba(245,240,232,0.8);
    line-height: 1.65;
}
.insight strong { color: #e87575; }
.insight.blue { border-color: #1F78B4; }
.insight.blue strong { color: #74b3e0; }

/* Introduction / Conclusion boxes */
.prose-box {
    background: #0f1623;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 2rem 2.2rem;
    font-size: 0.97rem;
    line-height: 1.8;
    color: rgba(245,240,232,0.75);
    margin-bottom: 1.5rem;
}
.prose-box h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #f5f0e8;
    margin-bottom: 0.8rem;
}

/* Streamlit overrides */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: rgba(245,240,232,0.8) !important; }
.stSelectbox label, .stMultiSelect label { color: rgba(245,240,232,0.6) !important; font-size: 0.82rem !important; }
.stApp { background: #080d17; }
h1, h2, h3 { color: #f5f0e8 !important; }
p, li { color: rgba(245,240,232,0.75) !important; }

/* Plot container */
.plot-wrap {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load & prep data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    titanic = sns.load_dataset('titanic')
    titanic['Sex']      = titanic['sex'].str.capitalize()
    titanic['Survived'] = titanic['survived']
    titanic['Pclass']   = titanic['pclass'].astype(str)
    titanic['Age']      = titanic['age'].fillna(titanic['age'].median())
    titanic['Fare']     = titanic['fare']
    titanic['SibSp']    = titanic['sibsp']
    titanic['Parch']    = titanic['parch']
    titanic['Embarked'] = titanic['embarked'].fillna('S').str.upper()
    titanic = titanic[titanic['Embarked'].notna() & (titanic['Embarked'] != '')]
    titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
    return titanic

titanic = load_data()

# ── Plot theme ────────────────────────────────────────────────────────────────
survival_colors = ['#B22222', '#1F78B4']
class_colors    = ['#1F78B4', '#33A02C', '#E31A1C']

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.facecolor': '#111827',
    'axes.facecolor': '#111827',
    'axes.edgecolor': '#2a3040',
    'axes.labelcolor': '#d4ccc0',
    'xtick.color': '#9a9080',
    'ytick.color': '#9a9080',
    'grid.color': '#1e2535',
    'text.color': '#d4ccc0',
    'legend.facecolor': '#1a2236',
    'legend.edgecolor': '#2a3040',
})

def style_fig(fig):
    fig.patch.set_facecolor('#111827')
    return fig

def add_trend(ax, x, y, color, lw=2):
    """statsmodels-free trend line: rolling median over x-sorted data.
    Replaces seaborn's lowess (which needs statsmodels). Stays positive,
    so it is safe to draw on a log y-axis."""
    s = pd.DataFrame({'x': np.asarray(x, float), 'y': np.asarray(y, float)}).dropna()
    s = s.sort_values('x')
    if len(s) < 8:
        return
    win = max(5, len(s) // 6)
    s['smooth'] = s['y'].rolling(win, center=True, min_periods=3).median()
    s = s.dropna(subset=['smooth'])
    if len(s) >= 2:
        ax.plot(s['x'].to_numpy(), s['smooth'].to_numpy(),
                color=color, linewidth=lw, zorder=6)

# ── Sidebar (filters only — no section navigation) ─────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    sel_class = st.multiselect("Passenger Class", ["1","2","3"], default=["1","2","3"])
    sel_sex   = st.multiselect("Gender", ["Male","Female"], default=["Male","Female"])
    age_range = st.slider("Age Range", 0, 80, (0, 80))
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:rgba(245,240,232,0.3); line-height:1.6;'>
    <strong style='color:rgba(245,240,232,0.5)!important'>Dataset</strong><br>
    Titanic passenger data via seaborn<br><br>
    <strong style='color:rgba(245,240,232,0.5)!important'>Course</strong><br>
    Business IT 2 · 60SCI006<br><br>
    <strong style='color:rgba(245,240,232,0.5)!important'>Instructor</strong><br>
    Prof. Do Duc Tan
    </div>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
df = titanic[
    titanic['Pclass'].isin(sel_class) &
    titanic['Sex'].isin(sel_sex) &
    titanic['Age'].between(age_range[0], age_range[1])
].copy()

total = len(df)
survived = df['Survived'].sum()
surv_rate = survived / total * 100 if total > 0 else 0
avg_age = df['Age'].mean() if total > 0 else 0
avg_fare = df['Fare'].mean() if total > 0 else 0

# ═══════════════════════════════════════════════════════════════════════════════
# HERO + INTRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <div style='position:relative;z-index:1;'>
        <span class='hero-badge'>Python for Data Science</span>
        <span class='hero-badge blue'>Business IT 2</span>
        <h1 class='hero-title'>Titanic<br>Survival Analysis</h1>
        <p class='hero-subtitle'>A comprehensive data science study of the 1912 maritime disaster</p>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='stat-card' style='--accent:#B22222'>
        <div class='stat-number'>{len(titanic)}</div>
        <div class='stat-label'>Total Passengers</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='stat-card' style='--accent:#1F78B4'>
        <div class='stat-number'>{titanic['Survived'].sum()}</div>
        <div class='stat-label'>Survivors</div></div>""", unsafe_allow_html=True)
with c3:
    rate = titanic['Survived'].mean()*100
    st.markdown(f"""<div class='stat-card' style='--accent:#33A02C'>
        <div class='stat-number'>{rate:.1f}%</div>
        <div class='stat-label'>Survival Rate</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='stat-card' style='--accent:#E31A1C'>
        <div class='stat-number'>11</div>
        <div class='stat-label'>Visualizations</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""<div class='prose-box'>
<h3>📖 Project Introduction</h3>
<p>On April 15, 1912, the RMS Titanic — once heralded as the world's most luxurious and "unsinkable" ocean liner — sank in the North Atlantic Ocean after striking an iceberg, claiming the lives of more than 1,500 passengers and crew. This catastrophic event left behind not only a profound human tragedy but also one of the most extensively documented passenger datasets in maritime history.</p>
<p>This project applies Python-based data science methodologies to rigorously investigate the structural patterns of survival and mortality aboard the Titanic. Through a series of eleven carefully engineered visualizations, we systematically dissect the multi-dimensional factors — including gender, socio-economic class, age demographics, fare distribution, port of embarkation, family size, and inter-variable correlations — that determined a passenger's probability of survival on that fateful night.</p>
<p>The dataset, sourced from the widely-used seaborn <code>titanic</code> sample dataset, contains records for 891 passengers. Missing values in the <em>Age</em> column were imputed using the median value, while missing <em>Embarked</em> entries were filled with the modal port of Southampton. These preprocessing steps ensure analytical integrity without introducing significant statistical bias.</p>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class='prose-box'>
<h3>👥 Group Members</h3>
</div>""", unsafe_allow_html=True)

members = [
    ("Ly Tam Anh", "102240293", "Leader · Report design · Figures 5, 6 · Ch 3 & 4"),
    ("Pham Huynh Dam", "10625070", "Data repair · Figures 1, 2 · Ch 1 & 2"),
    ("Nguyen Khanh Huan", "10625062", "Exercise summary · Figures 9, 10 · Ch 5"),
    ("Nguyen Thi Minh Tuong", "10620511", "Report design · Figures 7, 8 · Ch 5"),
    ("Huynh Dong Nghi", "10625085", "Report design · Figures 3, 4, 11 · Ch 4"),
]
cols = st.columns(len(members))
for col, (name, sid, role) in zip(cols, members):
    col.markdown(f"""
    <div style='background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:1rem;text-align:center;'>
        <div style='font-family:Playfair Display,serif;font-size:0.95rem;color:#f5f0e8;font-weight:700;margin-bottom:0.3rem;'>{name}</div>
        <div style='font-size:0.72rem;color:#74b3e0;margin-bottom:0.5rem;'>ID: {sid}</div>
        <div style='font-size:0.75rem;color:rgba(245,240,232,0.45);line-height:1.5;'>{role}</div>
    </div>
    """, unsafe_allow_html=True)

# Filtered snapshot (reflects the sidebar filters, applies to every figure below)
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filtered Passengers", total)
m2.metric("Survivors", int(survived))
m3.metric("Survival Rate", f"{surv_rate:.1f}%")
m4.metric("Avg Age", f"{avg_age:.1f} yrs")

# ═══════════════════════════════════════════════════════════════════════════════
# ALL VISUALIZATIONS — Figures 1 → 12, in order, one continuous page
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 class='section-header'>📊 Visualizations</h1>", unsafe_allow_html=True)
st.markdown("<p class='section-sub'>All twelve figures, in numerical order</p>", unsafe_allow_html=True)

# ── Figure 1: Survival Rate by Gender ──
st.markdown("#### Figure 1 · Survival Rate by Gender")
if len(df) > 0:
    df_fig1 = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df_fig1.plot(kind='bar', stacked=True, color=survival_colors, edgecolor='#111827', width=0.5, ax=ax1)
    for p in ax1.patches:
        h = p.get_height()
        if h > 2:
            ax1.text(p.get_x()+p.get_width()/2, p.get_y()+h/2, f'{h:.1f}%',
                     ha='center', va='center', color='white', weight='bold', fontsize=10)
    ax1.set_xlabel('Gender')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend(['No', 'Yes'], title='Survived', loc='upper left', bbox_to_anchor=(1,1))
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax1)
    st.pyplot(style_fig(fig1))
    plt.close(fig1)

# ── Figure 2: Survival by Passenger Class ──
st.markdown("#### Figure 2 · Survival by Passenger Class")
if len(df) > 0:
    present_cls = [c for c in ['1','2','3'] if c in df['Pclass'].values]
    df_counts = pd.crosstab(df['Pclass'], df['Survived']).reindex(present_cls)
    df_pct    = pd.crosstab(df['Pclass'], df['Survived'], normalize='index').reindex(present_cls) * 100
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df_counts.plot(kind='bar', stacked=False, color=survival_colors, edgecolor='#111827', width=0.6, ax=ax2)
    for i, container in enumerate(ax2.containers):
        for j, bar in enumerate(container):
            h = bar.get_height()
            if h > 0:
                try:
                    pct = df_pct.iloc[j, i]
                    ax2.text(bar.get_x()+bar.get_width()/2, h/2, f'{pct:.1f}%',
                             ha='center', va='center', color='white', weight='bold', fontsize=8)
                except: pass
                ax2.annotate(f'{int(h)}', (bar.get_x()+bar.get_width()/2, h),
                             ha='center', va='bottom', fontsize=9, color='#d4ccc0',
                             xytext=(0,3), textcoords='offset points')
    ax2.set_xlabel('Passenger Class')
    ax2.set_ylabel('Passenger Count')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.legend(['No','Yes'], title='Survived')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax2)
    st.pyplot(style_fig(fig2))
    plt.close(fig2)

st.markdown("""<div class='insight'>
<strong>~74.2% of female passengers survived</strong> vs only <strong>~18.9% of males</strong> — a stark validation of the "women and children first" protocol.
<strong>Class 1 passengers had a 63% survival rate</strong>, while Class 3 suffered a 75.8% mortality rate — socio-economic status was a life-or-death divide.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 3: Survival by Port of Embarkation, Faceted by Class ──
st.markdown("#### Figure 3 · Survival by Port of Embarkation, Faceted by Class")
if len(df) > 0:
    present_cls = [c for c in ['1','2','3'] if c in df['Pclass'].values]
    g = sns.FacetGrid(df, row="Pclass", row_order=present_cls, height=2.5, aspect=3,
                      gridspec_kws={'hspace': 0.4})
    g.figure.patch.set_facecolor('#111827')
    g.map_dataframe(sns.countplot, y="Embarked", hue="Survived",
                    palette=survival_colors, edgecolor='#111827', order=['C','Q','S'])
    g.set_axis_labels("Passenger Count", "Port")
    g.add_legend(title="Survived")
    for text, label in zip(g._legend.get_texts(), ['No','Yes']):
        text.set_text(label)
    for ax in g.axes.flat:
        ax.set_facecolor('#111827')
        for container in ax.containers:
            for bar in container:
                w = bar.get_width()
                if w > 0:
                    ax.annotate(f'{int(w)}', (w, bar.get_y()+bar.get_height()/2.),
                                ha='left', va='center', fontsize=9, color='#d4ccc0',
                                weight='bold', xytext=(4,0), textcoords='offset points')
    st.pyplot(g.figure)
    plt.close()

st.markdown("""<div class='insight blue'>
<strong>Cherbourg (C) Class 1 passengers</strong> had the best survival profile — affluent continental travelers with upper-deck proximity.
<strong>Southampton (S) Class 3</strong> carries the heaviest toll in absolute numbers — working-class passengers bore the disaster's full weight.
<strong>Queenstown (Q)</strong> passengers were almost entirely in Third Class, suffering severe survival barriers.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 4: Age Distribution of Passengers ──
st.markdown("#### Figure 4 · Age Distribution of Passengers")
fig4, ax4 = plt.subplots(figsize=(8, 4))
age_bins = list(range(0, 86, 5))
ax4.hist(df['Age'].dropna(), bins=age_bins, color="#1F78B4", edgecolor='#111827', linewidth=0.5)
ax4.set_xlabel('Age', fontsize=11, weight='bold')
ax4.set_ylabel('Frequency', fontsize=11, weight='bold')
ax4.set_xticks(range(0, 81, 10))
ax4.grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(ax=ax4)
st.pyplot(style_fig(fig4))
plt.close(fig4)

st.markdown("""<div class='insight'>
The distribution is <strong>unimodal and right-skewed</strong>, with a dense concentration between ages 20–35.
A secondary bump at ages 0–5 reflects families traveling with young children.
Beyond age 50, frequency drops sharply — international travel in 1912 was dominated by younger demographics.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 5: Age Distribution by Survival Status (KDE) ──
st.markdown("#### Figure 5 · Age Distribution by Survival Status")
fig5, ax5 = plt.subplots(figsize=(9, 4))
sns.kdeplot(data=df.dropna(subset=['Age']), x="Age", hue="Survived",
            fill=True, palette=survival_colors, alpha=0.5, common_norm=False, ax=ax5)
ax5.set_xlabel('Age', fontsize=11, weight='bold')
ax5.set_ylabel('Density', fontsize=11, weight='bold')
ax5.grid(linestyle='--', alpha=0.3)
legend = ax5.get_legend()
if legend:
    legend.set_title("Survived")
    for text, label in zip(legend.get_texts(), ['No','Yes']):
        text.set_text(label)
sns.despine(ax=ax5)
st.pyplot(style_fig(fig5))
plt.close(fig5)

st.markdown("""<div class='insight'>
Children aged <strong>0–10</strong> show a clear survival peak (blue &gt; red) — direct evidence of the "women and children first" protocol.
The <strong>20–30 age bracket</strong> shows the highest non-survival density — young male passengers and crew absorbed the greatest loss.
Beyond <strong>age 45</strong>, curves converge, suggesting age neutralized structural advantages in the chaos.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 6: Age Distribution by Passenger Class (Violin) ──
st.markdown("#### Figure 6 · Age Distribution by Passenger Class")
fig6, ax6 = plt.subplots(figsize=(8, 4))
violin_data = df.dropna(subset=['Age'])
if len(violin_data) > 0 and len(violin_data['Pclass'].unique()) > 0:
    present_classes = [c for c in ['1','2','3'] if c in violin_data['Pclass'].unique()]
    palette = {c: class_colors[int(c)-1] for c in present_classes}
    sns.violinplot(data=violin_data, x="Pclass", y="Age", hue="Pclass",
                   palette=palette, order=present_classes, inner=None, ax=ax6)
    if ax6.get_legend(): ax6.get_legend().remove()
    for coll in ax6.collections: coll.set_alpha(0.55)
ax6.set_xlabel('Passenger Class', fontsize=11, weight='bold')
ax6.set_ylabel('Age', fontsize=11, weight='bold')
ax6.grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(ax=ax6)
st.pyplot(style_fig(fig6))
plt.close(fig6)

st.markdown("""<div class='insight blue'>
<strong>Class 1</strong> skews older (30–50), reflecting wealthy, established travelers.
<strong>Class 2</strong> shows a symmetrical distribution around young adults (25–38).
<strong>Class 3</strong> is bottom-heavy with teens and twenties — predominantly young migrant laborers.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 7: Fare Distribution by Class (Log Scale) ──
st.markdown("#### Figure 7 · Fare Distribution by Class (Log Scale)")
df_fare = df[df['Fare'] > 0].dropna(subset=['Fare'])
if len(df_fare) > 0:
    present_cls = [c for c in ['1','2','3'] if c in df_fare['Pclass'].values]
    palette = {c: class_colors[int(c)-1] for c in present_cls}
    fig7, ax7 = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df_fare, x="Pclass", y="Fare", hue="Pclass", palette=palette,
                order=present_cls,
                boxprops=dict(alpha=0.8),
                flierprops=dict(alpha=0.3, marker='o', markerfacecolor='grey', markeredgecolor='none'),
                ax=ax7)
    ax7.set_yscale('log')
    ax7.set_yticks([10, 30, 100, 300])
    ax7.set_yticklabels(['10','30','100','300'])
    if ax7.get_legend(): ax7.get_legend().remove()
    ax7.set_xlabel('Passenger Class', fontsize=11, weight='bold')
    ax7.set_ylabel('Fare (Log Scale)', fontsize=11, weight='bold')
    ax7.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax7)
    st.pyplot(style_fig(fig7))
    plt.close(fig7)

st.markdown("""<div class='insight'>
<strong>Class 1 fares span a massive range</strong> (£10–£500+), revealing everything from economy first-class to luxury suites.
<strong>Class 3 fares are tightly compressed</strong> near £8–15 for the vast majority.
The log scale reveals subtle <strong>overlap</strong> between cheap first-class berths and premium third-class tickets.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 8: Family Size Distribution ──
st.markdown("#### Figure 8 · Family Size Distribution")
fig8, ax8 = plt.subplots(figsize=(8, 4))
sns.countplot(data=df, x="FamilySize", color="#1F78B4", edgecolor="#111827", ax=ax8)
ax8.set_xlabel('Family Size', fontsize=11, weight='bold')
ax8.set_ylabel('Frequency', fontsize=11, weight='bold')
ax8.grid(axis='y', linestyle='--', alpha=0.3)
for container in ax8.containers:
    for bar in container:
        h = bar.get_height()
        if h > 0:
            ax8.annotate(f'{int(h)}', (bar.get_x()+bar.get_width()/2., h),
                         ha='center', va='bottom', fontsize=9, color='#d4ccc0', weight='bold',
                         xytext=(0,3), textcoords='offset points')
sns.despine(ax=ax8)
st.pyplot(style_fig(fig8))
plt.close(fig8)

st.markdown("""<div class='insight'>
The distribution is heavily <strong>right-skewed</strong>: solo travelers (size=1) dominate, reflecting migrant workforce patterns.
Small nuclear families of 2–4 are stable. Beyond size 5, counts dwindle — but these large groups faced catastrophic coordination barriers during evacuation.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 9: Survival Rate by Family Size ──
st.markdown("#### Figure 9 · Survival Rate by Family Size")
if len(df) > 0:
    family_survival = df.groupby('FamilySize')['Survived'].mean().reset_index()
    family_survival['SurvivalRate'] = family_survival['Survived'] * 100
    fig9, ax9 = plt.subplots(figsize=(9, 4))
    sns.barplot(data=family_survival, x="FamilySize", y="SurvivalRate",
                color="#2CA02C", edgecolor="#111827", ax=ax9)
    ax9.set_xlabel('Family Size', fontsize=11, weight='bold')
    ax9.set_ylabel('Survival Rate (%)', fontsize=11, weight='bold')
    ax9.set_ylim(0, 115)
    ax9.grid(axis='y', linestyle='--', alpha=0.3)
    for p in ax9.patches:
        h = p.get_height()
        ax9.annotate(f'{h:.1f}%', (p.get_x()+p.get_width()/2., h),
                     ha='center', va='bottom', fontsize=9, color='#d4ccc0', weight='bold',
                     xytext=(0,3), textcoords='offset points')
    sns.despine(ax=ax9)
    st.pyplot(style_fig(fig9))
    plt.close(fig9)

st.markdown("""<div class='insight blue'>
Solo travelers (size=1) have only ~<strong>30% survival</strong>.
<strong>Family sizes 3–4</strong> hit the "sweet spot" — small enough to coordinate, often in higher classes.
Beyond size 5, survival <strong>collapses toward zero</strong> — large immigrant families in steerage faced locked gates and language barriers.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 10: Age vs Fare, Colored by Survival Status ──
st.markdown("#### Figure 10 · Age vs Fare, Colored by Survival Status")
df_fig10 = df.dropna(subset=['Age','Fare']).copy()
df_fig10 = df_fig10[df_fig10['Fare'] > 0]
if len(df_fig10) > 5:
    np.random.seed(42)
    df_fig10['Age_jitter'] = df_fig10['Age'] + np.random.uniform(-0.5, 0.5, size=len(df_fig10))
    surv_colors_map = {0: '#D62728', 1: '#1F78B4'}
    fig10, ax10 = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df_fig10, x='Age_jitter', y='Fare', hue='Survived',
                    palette=surv_colors_map, alpha=0.3, s=35, linewidth=0, ax=ax10)
    for surv_val in df_fig10['Survived'].unique():
        subset = df_fig10[df_fig10['Survived'] == surv_val]
        if len(subset) > 10:
            add_trend(ax10, subset['Age'], subset['Fare'], surv_colors_map[surv_val])
    ax10.set_yscale('log')
    ax10.set_yticks([10, 30, 100, 300])
    ax10.set_yticklabels(['10','30','100','300'])
    ax10.set_xlabel('Age', fontsize=11, weight='bold')
    ax10.set_ylabel('Fare (Log Scale)', fontsize=11, weight='bold')
    ax10.grid(alpha=0.3, linestyle='--')
    handles, labels = ax10.get_legend_handles_labels()
    ax10.legend(handles=handles[:2], labels=['Died','Survived'], title='Survival',
                bbox_to_anchor=(1.02, 0.7), loc='upper left', frameon=False)
    sns.despine(ax=ax10)
    plt.tight_layout()
    st.pyplot(style_fig(fig10))
    plt.close(fig10)

st.markdown("""<div class='insight blue'>
The <strong>blue trend line (survivors) consistently floats above red</strong> across all ages — higher fares = upper decks = lifeboat access.
The <strong>red mass clusters at £7–10 across ages 18–45</strong> — young, poor passengers faced the highest mortality.
At ages <strong>0–10, even the red line dips</strong> — the "children first" protocol temporarily overrode economic status.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 11: Age vs Fare, Faceted by Survival Status ──
st.markdown("#### Figure 11 · Age vs Fare, Faceted by Survival Status")
df_fig11 = df.dropna(subset=['Age','Fare']).copy()
df_fig11 = df_fig11[df_fig11['Fare'] > 0]
if len(df_fig11) > 5:
    np.random.seed(42)
    df_fig11['Age_jitter'] = df_fig11['Age'] + np.random.uniform(-0.5, 0.5, size=len(df_fig11))
    surv_vals = sorted(df_fig11['Survived'].unique())
    fig11, axes11 = plt.subplots(1, len(surv_vals), figsize=(10, 5), sharey=True)
    if len(surv_vals) == 1: axes11 = [axes11]
    surv_colors_map = {0: '#D62728', 1: '#1F78B4'}
    for ax, sv in zip(axes11, surv_vals):
        subset = df_fig11[df_fig11['Survived'] == sv]
        sns.scatterplot(data=subset, x='Age_jitter', y='Fare',
                        color=surv_colors_map.get(sv, '#888'), alpha=0.3, s=35, linewidth=0, ax=ax)
        if len(subset) > 10:
            add_trend(ax, subset['Age'], subset['Fare'], surv_colors_map.get(sv, '#888'))
        ax.set_yscale('log')
        ax.set_yticks([10,30,100,300])
        ax.set_yticklabels(['10','30','100','300'])
        ax.set_xlabel('Age', fontsize=11, weight='bold')
        ax.set_title('Died' if sv == 0 else 'Survived', fontsize=12, weight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#111827')
        sns.despine(ax=ax)
    axes11[0].set_ylabel('Fare (Log Scale)', fontsize=11, weight='bold')
    fig11.patch.set_facecolor('#111827')
    plt.tight_layout()
    st.pyplot(fig11)
    plt.close(fig11)

st.markdown("""<div class='insight'>
<strong>Panel 0 (Died)</strong>: heavy pooling at low fares; trend dips in childhood then curves upward — age played secondary role to poverty.
<strong>Panel 1 (Survived)</strong>: much more even vertical spread; trend line rises steadily — higher fares provided consistent linear advantage across all ages.
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Figure 12: Spearman Correlation Heatmap ──
st.markdown("#### Figure 12 · Spearman Correlation Heatmap")
corr_data = df[['Age','Fare','SibSp','Parch']].dropna()
if len(corr_data) > 5:
    cor_matrix = corr_data.corr(method='spearman').iloc[::-1]
    cmap = LinearSegmentedColormap.from_list('custom', ['#1F78B4', 'white', '#E31A1C'])
    fig12, ax12 = plt.subplots(figsize=(7, 5))
    sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0,
                linewidths=0.5, cbar_kws={'label': 'Correlation'}, ax=ax12,
                annot_kws={'size': 12, 'weight': 'bold'})
    ax12.set_xticklabels(ax12.get_xticklabels(), rotation=45, ha='right')
    ax12.set_yticklabels(ax12.get_yticklabels(), rotation=0)
    ax12.set_xlabel('')
    ax12.set_ylabel('')
    plt.tight_layout()
    st.pyplot(style_fig(fig12))
    plt.close(fig12)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""<div class='insight'>
    <strong>SibSp ↔ Parch (0.43)</strong>: Strongest correlation — passengers traveled as cohesive multi-generational family units.
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class='insight blue'>
    <strong>SibSp/Parch ↔ Fare (~0.42)</strong>: Larger families incurred higher collective ticket costs — family size links to economic footprint.
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class='insight'>
    <strong>Age ↔ SibSp/Parch (negative)</strong>: Younger passengers traveled with more family members; older passengers traveled alone or in smaller groups.
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class='insight blue'>
    <strong>Age ↔ Fare (0.14)</strong>: Near-zero correlation — wealth was distributed across all age groups, not concentrated in elderly demographics.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Interactive summary (bonus) ──
st.markdown("#### Survival Rate by Variable (Interactive Summary)")
metric = st.selectbox("Group by", ["Pclass", "Sex", "Embarked", "FamilySize"])
if len(df) > 0:
    grp = df.groupby(metric)['Survived'].mean().reset_index()
    grp['SurvivalRate'] = grp['Survived'] * 100
    grp = grp.sort_values('SurvivalRate', ascending=False)
    fig_s, ax_s = plt.subplots(figsize=(9, 3.5))
    colors_s = [survival_colors[1] if v > 50 else survival_colors[0] for v in grp['SurvivalRate']]
    bars = ax_s.bar(grp[metric].astype(str), grp['SurvivalRate'], color=colors_s, edgecolor='#111827', width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax_s.annotate(f'{h:.1f}%', (bar.get_x()+bar.get_width()/2., h),
                      ha='center', va='bottom', fontsize=10, color='#d4ccc0', weight='bold',
                      xytext=(0,3), textcoords='offset points')
    ax_s.set_xlabel(metric, fontsize=11, weight='bold')
    ax_s.set_ylabel('Survival Rate (%)', fontsize=11, weight='bold')
    ax_s.set_ylim(0, 115)
    ax_s.axhline(y=50, color='rgba(255,255,255,0.2)', linestyle='--', linewidth=1)
    ax_s.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax_s)
    st.pyplot(style_fig(fig_s))
    plt.close(fig_s)

# ═══════════════════════════════════════════════════════════════════════════════
# CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <div style='position:relative;z-index:1;'>
        <h1 class='hero-title'>Conclusions &<br>Key Findings</h1>
        <p class='hero-subtitle'>What the data tells us about survival aboard the Titanic</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""<div class='prose-box'>
<h3>📝 Project Conclusion</h3>
<p>This project has demonstrated that survival aboard the Titanic was not a matter of chance, but a product of deeply entrenched structural inequalities operating simultaneously across multiple demographic and socio-economic dimensions. Through eleven rigorous Python visualizations, we have systematically uncovered the multi-layered architecture of life and death on the night of April 14–15, 1912.</p>
<p>The analysis conclusively shows that <strong>gender</strong> was the most powerful single predictor of survival: female passengers survived at a rate of approximately 74.2%, compared to only 18.9% for males — a direct statistical imprint of the "women and children first" maritime protocol. <strong>Passenger class</strong> operated as a secondary but equally decisive force: First Class passengers benefited from physical proximity to the boat deck and superior evacuation logistics, achieving a 63% survival rate, while Third Class passengers — despite constituting the largest segment of the vessel's population — suffered a catastrophic 75.8% mortality rate.</p>
<p>Age analysis revealed a critical non-linear dynamic: infants and children under 10 were successfully prioritized during evacuation, while young adults aged 20–30 bore the heaviest proportional fatality burden. The violin plot analysis exposed how each passenger class attracted distinct demographic cohorts, with Third Class dominated by young migrant laborers whose age profiles aligned with those most likely to perish.</p>
<p>Economic analysis confirmed that fare price — as a direct proxy for cabin class and deck position — provided a consistent survival advantage across all age groups. Passengers paying higher fares were systematically positioned closer to lifeboats, creating an inescapable economic shield that operated independently of age. Family size analysis revealed a non-linear relationship: small nuclear families of 3–4 members achieved the highest survival rates through coordinated evacuation, while large multi-generational units collapsed toward zero survival probability due to coordination failures, steerage gate restrictions, and language barriers.</p>
<p>The Spearman correlation heatmap provided final structural confirmation, revealing that family cohesion, economic capacity, and demographic youth were deeply intertwined variables that collectively determined a passenger's structural position in the survival hierarchy.</p>
<p>In summary, the Titanic disaster serves as a timeless case study in the fatal intersection of institutional protocol, structural inequality, and human survival probability — a story told not merely in history books, but now, conclusively, in data.</p>
</div>""", unsafe_allow_html=True)

st.markdown("### Key Takeaways")
findings = [
    ("🚺", "Gender", "74.2% female vs 18.9% male survival — the starkest single predictor in the dataset"),
    ("🎫", "Passenger Class", "Class 1: 63% survival · Class 2: 47% · Class 3: 24% — wealth was literally life-saving"),
    ("👶", "Age", "Children under 10 were prioritized; young adults 20–30 suffered the highest absolute mortality"),
    ("💷", "Fare", "Higher fares = upper deck cabin = lifeboat proximity — a consistent survival advantage at all ages"),
    ("👨‍👩‍👧", "Family Size", "Sweet spot: 3–4 members. Solo travelers ~30%, large families (~7+) near 0% survival"),
    ("⚓", "Port", "Cherbourg Class 1 survivors vs Southampton Class 3 fatalities — origin shaped destination"),
]
cols = st.columns(3)
for i, (icon, title, desc) in enumerate(findings):
    with cols[i % 3]:
        st.markdown(f"""
        <div style='background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                    padding:1.2rem;margin-bottom:1rem;'>
            <div style='font-size:1.8rem;margin-bottom:0.5rem;'>{icon}</div>
            <div style='font-family:Playfair Display,serif;font-size:1rem;color:#f5f0e8;
                        font-weight:700;margin-bottom:0.4rem;'>{title}</div>
            <div style='font-size:0.82rem;color:rgba(245,240,232,0.55);line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
