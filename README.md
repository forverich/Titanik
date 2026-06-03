# 🚢 Titanic Survival Analysis — Streamlit Dashboard

> **Business IT 2 (60SCI006) · Vietnamese-German University**  
> Instructor: Prof. Do Duc Tan

An interactive data visualization dashboard built with **Streamlit**, **Matplotlib**, and **Seaborn**, reproducing all 12 figures from the group's Python for Data Science report on the Titanic dataset.

---

## 👥 Group Members

| No | Name | Student ID |
|----|------|------------|
| 1 | Ly Tam Anh | 102240293 |
| 2 | Pham Huynh Dam | 10625070 |
| 3 | Nguyen Khanh Huan | 10625062 |
| 4 | Nguyen Thi Minh Tuong | 10620511 |
| 5 | Huynh Dong Nghi | 10625085 |

---

## 📊 Figures Included

| Figure | Type | Description |
|--------|------|-------------|
| Fig 1 | Stacked Bar | Survival rate by gender |
| Fig 2 | Grouped Bar | Survival count & % by passenger class |
| Fig 3 | Faceted Count | Survival by port, faceted by class |
| Fig 4 | Histogram | Age distribution (bin = 5 years) |
| Fig 5 | KDE Plot | Age density by survival status |
| Fig 6 | Violin Plot | Age distribution by passenger class |
| Fig 7 | Boxplot (log) | Fare distribution by passenger class |
| Fig 8 | Count Plot | Family size distribution |
| Fig 9 | Bar Chart | Survival rate by family size |
| Fig 10 | Scatter + LOWESS | Age vs Fare, colored by survival |
| Fig 11 | Faceted Scatter | Age vs Fare, faceted by survival |
| Fig 12 | Heatmap | Spearman correlation matrix |

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `app.py` and `requirements.txt` are at the root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy** — done!

> The Titanic dataset is loaded automatically via `seaborn.load_dataset('titanic')` — no file upload needed.

---

## 🗂️ Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web framework
- [Seaborn](https://seaborn.pydata.org/) — statistical visualization & dataset
- [Matplotlib](https://matplotlib.org/) — figure rendering
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [NumPy](https://numpy.org/) — numerical operations
