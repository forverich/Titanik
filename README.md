# 🚢 Titanic Survival Analysis Dashboard

An interactive Streamlit web application that visualizes survival patterns aboard the RMS Titanic using Python data science tools.

**Course:** Business IT 2 · 60SCI006  
**Instructor:** Prof. Do Duc Tan

---

## 👥 Group Members

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Ly Tam Anh | 102240293 | Leader · Report design · Figures 5, 6 · Ch 3 & 4 |
| Pham Huynh Dam | 10625070 | Data repair · Figures 1, 2 · Ch 1 & 2 |
| Nguyen Khanh Huan | 10625062 | Exercise summary · Figures 9, 10 · Ch 5 |
| Nguyen Thi Minh Tuong | 10620511 | Report design · Figures 7, 8 · Ch 5 |
| Huynh Dong Nghi | 10625085 | Report design · Figures 3, 4, 11 · Ch 4 |

---

## 📊 Visualizations Included

| Figure | Description | Chart Type |
|--------|-------------|------------|
| 1 | Survival Rate by Gender | Stacked Bar |
| 2 | Survival Rate by Passenger Class | Grouped Bar |
| 3 | Survival by Port, Faceted by Class | Facet Grid |
| 4 | Age Distribution of Passengers | Histogram |
| 5 | Age Distribution by Survival Status | KDE Plot |
| 6 | Age Distribution by Passenger Class | Violin Plot |
| 7 | Fare Distribution by Class (Log Scale) | Box Plot |
| 8 | Family Size Distribution | Count Plot |
| 9 | Survival Rate by Family Size | Bar Plot |
| 10 | Age vs Fare with Survival Overlay | Scatter + LOWESS |
| 11 | Age vs Fare, Faceted by Survival | Faceted Scatter |
| 12 | Spearman Correlation Heatmap | Heatmap |

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-analysis.git
cd titanic-survival-analysis
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

The app will open at `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (ensure `app.py` and `requirements.txt` are at the root)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New app**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — done!

---

## 🗂️ Project Structure

```
titanic-survival-analysis/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

> **Note:** The Titanic dataset is loaded directly via `seaborn.load_dataset('titanic')` — no external data file is needed.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — web framework
- **Pandas** — data manipulation
- **Seaborn / Matplotlib** — visualizations
- **NumPy** — numerical operations

---

## 📄 License

For academic use only — Business IT 2, 2024–2025.
