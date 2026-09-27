import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Career Success Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (1, 0)
W = {"width": "stretch"} if _v >= (1, 49) else {"use_container_width": True}

PALETTE = px.colors.sequential.Viridis
CORR_SCALE = "RdBu_r"

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DASHBOARD_DIR)

DEFAULT_PATH = os.path.join(BASE_DIR, "dataset", "education_career_success_cleaned.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model_career_prediction2.pkl")


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File model tidak ditemukan pada lokasi: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError(
            f"File .pkl terdeteksi sebagai {type(bundle).__name__}, bukan dictionary."
        )

    model = (
        bundle.get("model")
        or bundle.get("pipeline")
        or bundle.get("best_model")
    )

    if model is None:
        raise KeyError(
            f"Key 'model' tidak ditemukan/bernilai None! Key yang tersedia di .pkl Anda: {list(bundle.keys())}"
        )

    categories = bundle.get("categories", {})
    features = bundle.get("features", [])

    return model, categories, features

EXPECTED_COLS = [
    "Age", "Gender", "High_School_GPA", "SAT_Score",
    "University_GPA", "Field_of_Study", "Internships_Completed",
    "Projects_Completed", "Certifications", "Soft_Skills_Score",
    "Networking_Score", "Job_Offers", "Starting_Salary",
    "Career_Satisfaction", "Years_to_Promotion", "Current_Job_Level",
    "Work_Life_Balance"
]

@st.cache_data(show_spinner=False)
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["categories"], bundle["features"]

@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in ["Student_ID", "Entrepreneurship"] if c in df.columns])

    for src, dst in [("Starting_Salary", "salary_scaled"),
                     ("Work_Life_Balance", "worklife_scaled")]:
        if src in df.columns:
            lo, hi = df[src].min(), df[src].max()
            df[dst] = (df[src] - lo) / (hi - lo) if hi > lo else 0.0

    if {"salary_scaled", "worklife_scaled"} <= set(df.columns):
        df["salary_worklife_index"] = (df["salary_scaled"] + df["worklife_scaled"]) / 2

    edu_cols = [c for c in ["High_School_GPA", "SAT_Score", "University_GPA"] if c in df.columns]
    skill_cols = [c for c in ["Soft_Skills_Score", "Certifications"] if c in df.columns]
    career_cols = [c for c in ["Starting_Salary", "Job_Offers", "Career_Satisfaction"] if c in df.columns]

    if edu_cols:
        df["Education"] = df[edu_cols].mean(axis=1)
    if skill_cols:
        df["Soft Skills + Certifications"] = df[skill_cols].mean(axis=1)
    if career_cols:
        df["Career_Success_Score"] = df[career_cols].mean(axis=1)

    return df


def sample_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0, 1, n)
    noise = lambda s: rng.normal(0, s, n)
    df = pd.DataFrame({
        "Student_ID": [f"S{i:05d}" for i in range(n)],
        "Age": rng.integers(18, 31, n),
        "Gender": rng.choice(["Male", "Female", "Other"], n, p=[0.47, 0.47, 0.06]),
        "High_School_GPA": np.clip(2 + latent * 2 + noise(0.1), 2, 4).round(2),
        "SAT_Score": np.clip(900 + latent * 700 + noise(30), 900, 1600).astype(int),
        "University_GPA": np.clip(2 + latent * 2 + noise(0.1), 2, 4).round(2),
        "Field_of_Study": rng.choice(
            ["Computer Science", "Engineering", "Business", "Medicine", "Arts", "Mathematics", "Law"], n),
        "Internships_Completed": np.clip((latent * 4 + noise(0.3)).round(), 0, 4).astype(int),
        "Projects_Completed": np.clip((latent * 9 + noise(0.6)).round(), 0, 9).astype(int),
        "Certifications": np.clip((latent * 5 + noise(0.3)).round(), 0, 5).astype(int),
        "Soft_Skills_Score": np.clip((1 + latent * 9 + noise(0.5)).round(), 1, 10).astype(int),
        "Networking_Score": np.clip((1 + latent * 9 + noise(0.8)).round(), 1, 10).astype(int),
        "Job_Offers": np.clip((latent * 5 + noise(0.3)).round(), 0, 5).astype(int),
        "Starting_Salary": (25000 + latent * 125000 + noise(4000)).round(-2),
        "Career_Satisfaction": np.clip((1 + latent * 9 + noise(0.6)).round(), 1, 10).astype(int),
        "Years_to_Promotion": np.clip((6 - latent * 5 + noise(0.4)).round(), 1, 6).astype(int),
        "Current_Job_Level": rng.choice(["Entry", "Mid", "Senior", "Executive"], n),
        "Work_Life_Balance": np.clip((9 - latent * 6 + noise(0.7)).round(), 1, 10).astype(int),
        "Entrepreneurship": "No",
    })
    return df


def regplot(df, x, y, title, line_color="#2ca02c", height=380):
    d = df[[x, y]].dropna()
    fig = px.scatter(d, x=x, y=y, opacity=0.55, height=height,
                     color_discrete_sequence=["#4C78A8"])
    if len(d) >= 2 and d[x].nunique() > 1:
        slope, intercept = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 100)
        r = d[x].corr(d[y])
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines",
            line=dict(color=line_color, width=3),
            name=f"Tren (r = {r:.3f})",
        ))
    fig.update_layout(title=title, margin=dict(t=55, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return fig


def corr_heatmap(df, cols, method="spearman", title="", fmt=".3f", zmin=None, zmax=None, height=420):
    cols = [c for c in cols if c in df.columns]
    m = df[cols].corr(method).round(4)
    fig = px.imshow(m, text_auto=fmt, color_continuous_scale=CORR_SCALE,
                    zmin=zmin if zmin is not None else -1,
                    zmax=zmax if zmax is not None else 1,
                    aspect="auto", height=height)
    fig.update_layout(title=title, margin=dict(t=55, b=10, l=10, r=10))
    return fig, m


def mean_bar(df, x, y, title, xlabel, ylabel, color_scale="Viridis", height=400):
    g = df.groupby(x, observed=True)[y].mean().reset_index()
    fig = px.bar(g, x=x, y=y, color=y, color_continuous_scale=color_scale,
                 text=g[y].round(2), height=height)
    fig.update_traces(textposition="outside")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                      coloraxis_showscale=False, margin=dict(t=55, b=10, l=10, r=10))
    return fig


def narrative(text: str):
    with st.expander("Insight", expanded=False):
        st.markdown(text)


st.sidebar.title("Career Success")
st.sidebar.caption("Career Success Dashboard")

try:
    raw = load_data(DEFAULT_PATH)
    source_note = f"`{DEFAULT_PATH}` (Folder Lokal)"
except Exception as e:
    st.error(f"File `{DEFAULT_PATH}` tidak ditemukan di folder lokal. Pastikan file CSV berada di direktori yang sama dengan script.")
    st.stop()

missing = [c for c in EXPECTED_COLS if c not in raw.columns]
if missing:
    st.sidebar.warning("Kolom tidak ditemukan: " + ", ".join(missing))

df_all = prepare(raw)

st.sidebar.subheader("Navigation")
selected_page = st.sidebar.selectbox(
    "Pages:",
    [
        "Exploratory Data Analysis (EDA)",
        "1. Recruitment Trends",
        "2. Career Well-Being",
        "3. Skills Gap",
        "4. Gender Parity",
        "5. Unequal Opportunities",
        "6. Conclusion",
        "7. Job Offers Predictions"
    ]
)

st.sidebar.subheader("Global Filter")

genders = sorted(df_all["Gender"].dropna().unique()) if "Gender" in df_all else []
sel_gender = st.sidebar.multiselect("Gender", genders, default=genders)

fields = sorted(df_all["Field_of_Study"].dropna().unique()) if "Field_of_Study" in df_all else []
sel_field = st.sidebar.multiselect("Field of Study", fields, default=fields)

mask = pd.Series(True, index=df_all.index)
if genders:
    mask &= df_all["Gender"].isin(sel_gender)
if fields:
    mask &= df_all["Field_of_Study"].isin(sel_field)

if "University_GPA" in df_all:
    lo, hi = float(df_all["University_GPA"].min()), float(df_all["University_GPA"].max())
    g_lo, g_hi = st.sidebar.slider("University GPA", lo, hi, (lo, hi), 0.05)
    mask &= df_all["University_GPA"].between(g_lo, g_hi)

if "Age" in df_all:
    a_lo, a_hi = int(df_all["Age"].min()), int(df_all["Age"].max())
    s_lo, s_hi = st.sidebar.slider("Usia", a_lo, a_hi, (a_lo, a_hi))
    mask &= df_all["Age"].between(s_lo, s_hi)

if "Starting_Salary" in df_all:
    s0, s1 = float(df_all["Starting_Salary"].min()), float(df_all["Starting_Salary"].max())
    sal = st.sidebar.slider("Starting Salary ($)", s0, s1, (s0, s1), step=500.0)
    mask &= df_all["Starting_Salary"].between(*sal)

df = df_all[mask].copy()

if df.empty:
    st.error("No data passed the filter. Loosen the filter in the sidebar.")
    st.stop()

st.title("Factors That Influence Career Success")
st.markdown("---")


if selected_page == "Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown(
        "This page presents a preliminary overview of academic characteristics, "
        "the distribution of fields of study, and a portfolio of student experiences such as projects, internships, certifications"
    )

    k = st.columns(5)
    k[0].metric("Number Of Students", f"{len(df):,}")
    if "University_GPA" in df:
        k[1].metric("Average GPA", f"{df['University_GPA'].mean():.2f}")
    if "Starting_Salary" in df:
        k[2].metric("Average Starting Salary", f"${df['Starting_Salary'].mean():,.0f}")
    if "Job_Offers" in df:
        k[3].metric("Average Job Offers", f"{df['Job_Offers'].mean():.2f}")
    if "Career_Satisfaction" in df:
        k[4].metric("Career Satisfaction", f"{df['Career_Satisfaction'].mean():.2f}/10")

    st.markdown("### 1. Student Academic Profile & GPA")
    e1, e2 = st.columns(2)
    with e1:
        if "University_GPA" in df.columns:
            fig_gpa = px.histogram(
                df, x="University_GPA", nbins=20, marginal="box",
                title="Distribution of Student GPAs (University GPA)",
                color_discrete_sequence=["#2b5c8f"]
            )
            fig_gpa.add_vline(x=df["University_GPA"].mean(), line_dash="dash", line_color="red",
                              annotation_text=f"Rata-rata: {df['University_GPA'].mean():.2f}")
            st.plotly_chart(fig_gpa, **W)

    with e2:
        if "University_GPA" in df.columns and "Field_of_Study" in df.columns:
            gpa_field = df.groupby("Field_of_Study")["University_GPA"].mean().reset_index()
            fig_gpa_field = px.bar(
                gpa_field, x="Field_of_Study", y="University_GPA",
                title="Average GPA by Field of Study",
                color="University_GPA", color_continuous_scale="Blues",
                text=gpa_field["University_GPA"].round(2)
            )
            fig_gpa_field.update_traces(textposition="outside")
            fig_gpa_field.update_yaxes(range=[0, 4.0])
            st.plotly_chart(fig_gpa_field, **W)

    st.markdown("### 2. Distribution of Fields of Study")
    if "Field_of_Study" in df.columns:
        f_counts = df["Field_of_Study"].value_counts().reset_index()
        f_counts.columns = ["Field_of_Study", "Jumlah Mahasiswa"]
        fig_field = px.pie(
            f_counts, names="Field_of_Study", values="Jumlah Mahasiswa",
            title="Student Composition by Field of Study",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_field, **W)

    st.markdown("### 3. Range of Student Experience & Portfolios")
    st.markdown("Mapping the distribution of the number of **Projects**, **Internships**, and **Certifications** completed by students:")
    
    p1, p2, p3 = st.columns(3)
    with p1:
        if "Projects_Completed" in df.columns:
            fig_proj = px.histogram(
                df, x="Projects_Completed",
                title="Range of Completed Projects",
                color_discrete_sequence=["#27ae60"], text_auto=True
            )
            st.plotly_chart(fig_proj, **W)
    with p2:
        if "Internships_Completed" in df.columns:
            fig_intern = px.histogram(
                df, x="Internships_Completed",
                title="Range of Completed Internships",
                color_discrete_sequence=["#e67e22"], text_auto=True
            )
            st.plotly_chart(fig_intern, **W)
    with p3:
        if "Certifications" in df.columns:
            fig_cert = px.histogram(
                df, x="Certifications",
                title="Range of Completed Certifications",
                color_discrete_sequence=["#8e44ad"], text_auto=True
            )
            st.plotly_chart(fig_cert, **W)

elif selected_page == "1. Recruitment Trends":
    st.subheader("1. Shifting Recruitment Trends")

    st.markdown("#### a. How do soft skills compare to a GPA in terms of their impact on the likelihood of receiving a job offer?")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(regplot(df, "University_GPA", "Job_Offers", "GPA vs Job Offers"),
                        **W)
    with c2:
        st.plotly_chart(regplot(df, "Soft_Skills_Score", "Job_Offers", "Soft Skills vs Job Offers"),
                        **W)

    fig, m = corr_heatmap(df, ["Job_Offers", "Soft_Skills_Score", "University_GPA"],
                          "spearman", "Correlation between GPA, Soft Skills, and Job Offers",
                          height=400)
    c3, c4 = st.columns([1.2, 1])
    with c3:
        st.plotly_chart(fig, **W)
    with c4:
        st.markdown("**Spearman's Correlation Coefficient**")
        st.dataframe(m, **W)

    narrative(
        "Both scatter plots show an upward trend: both GPA and soft skills are accompanied by "
        "an increase in the number of job offers. The heat map shows a very strong positive correlation"
        "(±0.95–0.97) among the three, meaning that both academic and interpersonal skills"
        "play a major role in employment opportunities."
    )

    st.markdown("---")
    st.markdown("#### b. Do individuals with strong skills but a low GPA still receive job offers?")
    hi_skill = st.slider("'High' soft skills threshold (≥)", 1.0, 10.0,
                         float(df["Soft_Skills_Score"].quantile(0.75)), 0.5)
    lo_gpa = st.slider("'Low' GPA threshold (≤)", float(df["University_GPA"].min()),
                       float(df["University_GPA"].max()),
                       float(df["University_GPA"].quantile(0.25)), 0.05)

    fig = px.scatter(df, x="University_GPA", y="Soft_Skills_Score",
                     size="Job_Offers", color="Job_Offers", size_max=34,
                     color_continuous_scale="Viridis", opacity=0.75, height=520,
                     hover_data=[c for c in ["Gender", "Field_of_Study", "Starting_Salary"] if c in df])
    fig.add_vline(x=lo_gpa, line_dash="dash", line_color="crimson")
    fig.add_hline(y=hi_skill, line_dash="dash", line_color="crimson")
    fig.update_layout(title="GPA vs Soft Skills (ukuran & warna = Job Offers)",
                      xaxis_title="University GPA", yaxis_title="Soft Skills Score",
                      margin=dict(t=55, b=10, l=10, r=10))
    st.plotly_chart(fig, **W)

    seg = df[(df["Soft_Skills_Score"] >= hi_skill) & (df["University_GPA"] <= lo_gpa)]
    m1, m2, m3 = st.columns(3)
    m1.metric("The number of students with 'high skills but low GPAs' ", len(seg))
    m2.metric("Average job offers in this segment",
              f"{seg['Job_Offers'].mean():.2f}" if len(seg) else "—")
    m3.metric("Those who received ≥1 offer",
              f"{(seg['Job_Offers'] >= 1).mean() * 100:.0f}%" if len(seg) else "—")
    if len(seg):
        st.dataframe(seg.head(20), **W)

    narrative(
        "Since both GPA and soft skills are strongly correlated with job offers,"
        "the 'high soft skills but low GPA' quadrant is nearly empty in this dataset — "
        "feel free to adjust both thresholds to see for yourself."
    )


elif selected_page == "2. Career Well-Being":
    st.subheader("2. Demands for Career Well-Being")

    st.markdown("#### a. How do salary and work-life balance together influence career satisfaction?")
    w = st.slider("The weight of wages in the composite index", 0.0, 1.0, 0.5, 0.05,
                  help="Indeks = w × salary_scaled + (1-w) × worklife_scaled. "
                       "Notebook memakai bobot 0.5 (rata-rata sederhana).")
    d = df.copy()
    d["salary_worklife_index"] = w * d["salary_scaled"] + (1 - w) * d["worklife_scaled"]

    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig = px.scatter(d, x="salary_worklife_index", y="Career_Satisfaction",
                         color="Career_Satisfaction", color_continuous_scale="Viridis",
                         opacity=0.8, height=480,
                         hover_data=[c for c in ["Starting_Salary", "Work_Life_Balance",
                                                 "Field_of_Study"] if c in d])
        fig.update_layout(title="Salary + Work-Life Index vs Career Satisfaction",
                          margin=dict(t=55, b=10, l=10, r=10))
        fig.update_xaxes(showgrid=True)
        st.plotly_chart(fig, **W)
    with c2:
        fig2, m2_ = corr_heatmap(d, ["salary_worklife_index", "Career_Satisfaction"],
                                 "spearman", "Correlation Index vs Satisfaction", height=480)
        st.plotly_chart(fig2, **W)

    narrative(
        "The distribution varies widely: at low satisfaction indices, moderate levels dominate,"
        "while scores of 9–10 only appear consistently once the index exceeds ~0.38."
        "The correlation is only weak to moderate, so the combination of salary and work-life balance is not the sole"
        "determinant of career satisfaction."
    )

    st.markdown("---")
    st.markdown("#### b. Career satisfaction: the 'salary seekers' vs. the 'work-life balance seekers' ")
    q = st.slider("Quantiles defining priority groups", 0.50, 0.95, 0.75, 0.05)
    salary_p = df[df["Starting_Salary"] >= df["Starting_Salary"].quantile(q)]
    balance_p = df[df["Work_Life_Balance"] >= df["Work_Life_Balance"].quantile(q)]

    stats = pd.DataFrame({
        "Salary Prioritizers": {
            "Count": len(salary_p),
            "Average Career Satisfaction": salary_p["Career_Satisfaction"].mean(),
            "Average Salary": salary_p["Starting_Salary"].mean(),
            "Average Work-Life Balance": salary_p["Work_Life_Balance"].mean(),
        },
        "Work-Life Balance Prioritizers": {
            "Count": len(balance_p),
            "Average Career Satisfaction": balance_p["Career_Satisfaction"].mean(),
            "Average Salary": balance_p["Starting_Salary"].mean(),
            "Average Work-Life Balance": balance_p["Work_Life_Balance"].mean(),
        },
    }).round(2)
    st.dataframe(stats, **W)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(regplot(df, "Starting_Salary", "Career_Satisfaction",
                                "Salary Trends vs Job Satisfaction", "#d62728"),
                        **W)
    with c2:
        st.plotly_chart(regplot(df, "Work_Life_Balance", "Career_Satisfaction",
                                "Work-Life Balance vs Satisfaction", "#2ca02c"),
                        **W)

    c3, c4 = st.columns(2)
    with c3:
        means = pd.DataFrame({
            "Groups": ["Salary Focus", "Work-Life Balance Focus"],
            "Average Satisfaction": [salary_p["Career_Satisfaction"].mean(),
                                     balance_p["Career_Satisfaction"].mean()],
        })
        fig = px.bar(means, x="Groups", y="Average Satisfaction", color="Groups",
                     color_discrete_sequence=["#440154", "#21918c"],
                     text=means["Average Satisfaction"].round(2), height=440)
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 10])
        fig.update_layout(title="Average Career Satisfaction by Primary Focus",
                          showlegend=False, margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(fig, **W)
    with c4:
        fig, cm = corr_heatmap(df, ["Starting_Salary", "Work_Life_Balance", "Career_Satisfaction"],
                               "pearson", "Matrix Correlation", height=440)
        st.plotly_chart(fig, **W)

    narrative(
        "The group with high salaries reported significantly higher career satisfaction "
        "than the group with high work-life balance. A systemic trade-off was also observed:"
        "high salaries are inversely correlated with work-life balance scores."
    )


elif selected_page == "3. Skills Gap":
    st.subheader("3. Skills Gap")

    st.markdown("#### a.  The Impact of Internships and Projects on Job Offers")
    fig, cm = corr_heatmap(df, ["Internships_Completed", "Projects_Completed", "Job_Offers"],
                           "pearson", "Matrix Correlation", height=380)
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.plotly_chart(fig, **W)
    with c2:
        st.dataframe(cm, **W)
        st.caption("The correlation is nearly perfect: a portfolio of practical work is a key factor in the hiring process.")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(mean_bar(df, "Internships_Completed", "Job_Offers",
                                 "The Impact of Internships on Job Offers",
                                 "Number of Completed Internships",
                                 "Average Number of Job Offers", "Viridis"),
                        **W)
    with c4:
        st.plotly_chart(mean_bar(df, "Projects_Completed", "Job_Offers",
                                 "Impact of Projects on Job Offers",
                                 "Number of Completed Projects",
                                 "Average Number of Job Offers", "Magma"),
                        **W)

    pivot = df.pivot_table(values="Job_Offers", index="Internships_Completed",
                           columns="Projects_Completed")
    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="YlGnBu",
                    aspect="auto", height=470,
                    labels=dict(x="Projects Completed", y="Internships Completed",
                                color="Avg Job Offers"))
    fig.update_layout(title="Average Job Offers: Internships × Projects",
                      margin=dict(t=55, b=10, l=10, r=10))
    st.plotly_chart(fig, **W)

    narrative(
        "There is a minimum threshold: candidates with no internships or very few projects"
        "hardly ever receive job offers. A combination of many internships + many projects"
        "results in the highest number of job offers (multiplier effect)."
    )

    st.markdown("---")
    st.markdown("#### b. How significant is the impact of soft skills and certifications compared to formal education?")
    agg_cols = ["Education", "Soft Skills + Certifications", "Career_Success_Score"]
    fig, corr_sp = corr_heatmap(df, agg_cols, "spearman",
                                "Correlation Heatmap (Aggregated Columns)", fmt=".4f", height=430)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig, **W)
    with c2:
        target = "Career_Success_Score"
        tc = corr_sp[target].drop(target).sort_values(key=abs, ascending=False)
        
        bar = pd.DataFrame({"Category": tc.index, "Korelasi": tc.values})
        
        # 2. Sesuaikan parameter x dan color ke 'Category'
        fig = px.bar(bar, x="Category", y="Korelasi", color="Category",
                     color_discrete_sequence=["orange", "skyblue"],
                     text=bar["Korelasi"].round(4), height=430)
        
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[max(0, bar["Korelasi"].min() - 0.01), 1.0])
        fig.update_layout(title="Average Impact on Career Success", showlegend=False,
                          xaxis_title="Category",  # Mengamankan judul x-axis
                          yaxis_title="Average Absolute Correlation",
                          margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(fig, **W)

    st.caption("Note: `Education` = the average of High_School_GPA, SAT_Score, and University_GPA. "
               "`Career_Success_Score` = the average of Starting_Salary, Job_Offers, and Career_Satisfaction.")

    narrative(
        "Both have a very strong influence, but Soft Skills + Certifications are slightly"
        "more important than Education—in line with the trend toward skills-based hiring. The two are"
        "also highly correlated, indicating that students with strong academic performance"
        "tend to earn certifications as well.."
    )


elif selected_page == "4. Gender Parity":
    st.subheader("4. Gender Parity in Professional Outcomes")

    st.markdown("#### a. Differences in Starting Salaries and Promotion Timelines by Gender in the Same Field of Study")
    avail = sorted(df["Field_of_Study"].dropna().unique())
    default_fields = [f for f in ["Engineering", "Business"] if f in avail] or avail[:2]
    pick_fields = st.multiselect("Fields of study compared", avail, default=default_fields)
    dff = df[df["Field_of_Study"].isin(pick_fields)]

    if dff.empty:
        st.info("Select at least one field of study.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            g = dff.groupby(["Field_of_Study", "Gender"], observed=True)["Starting_Salary"].mean().reset_index()
            fig = px.bar(g, x="Field_of_Study", y="Starting_Salary", color="Gender",
                         barmode="group", text=g["Starting_Salary"].map(lambda v: f"${v:,.0f}"),
                         color_discrete_sequence=px.colors.sequential.Blues_r, height=470)
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Average Starting Salary by Gender",
                              xaxis_title="Field of Study",
                              yaxis_title="Average Starting Salary ($)",
                              margin=dict(t=55, b=10, l=10, r=10))
            st.plotly_chart(fig, **W)
        with c2:
            g = dff.groupby(["Field_of_Study", "Gender"], observed=True)["Years_to_Promotion"].mean().reset_index()
            fig = px.bar(g, x="Field_of_Study", y="Years_to_Promotion", color="Gender",
                         barmode="group", text=g["Years_to_Promotion"].map(lambda v: f"{v:.1f} thn"),
                         color_discrete_sequence=px.colors.sequential.Oranges_r, height=470)
            fig.update_traces(textposition="outside")
            fig.update_layout(title="Average Time to Promotion by Gender",
                              xaxis_title="Field of Study",
                              yaxis_title="Average Time to Promotion (Years)",
                              margin=dict(t=55, b=10, l=10, r=10))
            st.plotly_chart(fig, **W)

        st.markdown("**Summary of Data Comparison**")
        summary = (dff.groupby(["Field_of_Study", "Gender"], observed=True)
                   [["Starting_Salary", "Years_to_Promotion"]].mean().round(2))
        st.dataframe(summary, **W)

    narrative(
        "In this dataset, women outperform men in both areas—in terms of both starting salary "
        "and promotion speed—with the most striking difference in the Business field."
    )

    st.markdown("---")
    st.markdown("#### b. The Effect of Gender on the Number of Job Offers Among Groups with Equivalent GPAs")
    edges = st.slider("GPA Group Thresholds", 2.0, 4.0, (3.3, 3.6), 0.1)
    lo_e, hi_e = edges
    bins = [0, lo_e, hi_e, 4.0]
    labels = [f"GPA ≤ {lo_e:.1f}", f"GPA {lo_e:.1f} – {hi_e:.1f}", f"GPA > {hi_e:.1f}"]
    dg = df.copy()
    dg["GPA_Group"] = pd.cut(dg["University_GPA"], bins=bins, labels=labels)
    g = dg.dropna(subset=["GPA_Group"]).groupby(["GPA_Group", "Gender"], observed=True)["Job_Offers"].mean().reset_index()

    fig = px.bar(g, x="GPA_Group", y="Job_Offers", color="Gender", barmode="group",
                 text=g["Job_Offers"].round(2), color_discrete_sequence=px.colors.sequential.Viridis,
                 height=500)
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Average Number of Job Offers by Gender Within Comparable GPA Groups",
                      xaxis_title="Academic Performance Group (GPA)",
                      yaxis_title="Average Number of Job Offers",
                      margin=dict(t=55, b=10, l=10, r=10))
    st.plotly_chart(fig, **W)

    narrative(
        "A higher GPA always means more job offers for all genders."
        "In the lower-to-middle GPA group, women tend to outperform men, whereas in the"
        "highest GPA group, the gap virtually disappears."
    )


elif selected_page == "5. Unequal Opportunities":
    st.subheader("5. Unequal Early-Career Opportunities Among Graduates")

    st.markdown("#### a. Same GPA, different work experience, different career outcomes?")
    color_by = st.selectbox("Color the dots based on",
                            [c for c in ["Projects_Completed", "Certifications",
                                         "Internships_Completed", "Networking_Score"] if c in df])
    fig = px.scatter(df, x="University_GPA", y="Starting_Salary", color=color_by,
                     color_continuous_scale="Viridis", opacity=0.8, height=520,
                     hover_data=[c for c in ["Field_of_Study", "Gender", "Job_Offers"] if c in df])
    fig.update_layout(title=f"GPA vs Starting Salary based on {color_by}",
                      xaxis_title="University GPA", yaxis_title="Starting Salary ($)",
                      margin=dict(t=55, b=10, l=10, r=10))
    st.plotly_chart(fig, **W)

    gpa_pick = st.slider("Examine the distribution of salaries at a specific GPA (±0.05)",
                         float(df["University_GPA"].min()), float(df["University_GPA"].max()),
                         float(round(df["University_GPA"].median(), 1)), 0.05)
    same = df[df["University_GPA"].between(gpa_pick - 0.05, gpa_pick + 0.05)]
    if len(same) > 1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Students with this GPA", len(same))
        c2.metric("Salary range",
                  f"${same['Starting_Salary'].max() - same['Starting_Salary'].min():,.0f}")
        c3.metric("Average salary", f"${same['Starting_Salary'].mean():,.0f}")

    narrative(
        "Identical academic achievements do not guarantee identical career outcomes: even with"
        "identical GPAs, the difference in starting salaries can be significant. What sets them apart is a portfolio of"
        "practical experience and professional certifications."
    )

    st.markdown("---")
    st.markdown("#### b. Do certain fields of study lead to significantly better career outcomes?")
    fig = px.scatter(df, x="University_GPA", y="Starting_Salary", color="Field_of_Study",
                     opacity=0.75, height=540,
                     hover_data=[c for c in ["Gender", "Job_Offers", "Certifications"] if c in df])
    fig.update_layout(title="Distribution of Starting Salaries by GPA and Field of Study",
                      xaxis_title="University GPA", yaxis_title="Starting Salary ($)",
                      margin=dict(t=55, b=10, l=10, r=10))
    st.plotly_chart(fig, **W)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="Field_of_Study", y="Starting_Salary", color="Field_of_Study",
                     points="outliers", height=470)
        fig.update_layout(title="Distribution of Starting Salaries by Field of Study", showlegend=False,
                          xaxis_title="", margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(fig, **W)
    with c2:
        rank = (df.groupby("Field_of_Study", observed=True)
                .agg(Rata_Gaji=("Starting_Salary", "mean"),
                     Rata_Job_Offers=("Job_Offers", "mean"),
                     Rata_Kepuasan=("Career_Satisfaction", "mean"),
                     Jumlah=("Starting_Salary", "size"))
                .round(2).sort_values("Rata_Gaji", ascending=False))
        st.markdown("**Rankings by Field of Study**")
        st.dataframe(rank, **W, height=430)

    fig, _ = corr_heatmap(df, ["University_GPA", "Projects_Completed", "Certifications",
                               "Starting_Salary", "Job_Offers"], "pearson",
                          "Correlation Heatmap: Academic and Practical Factors",
                          zmin=0, zmax=1, height=560)
    st.plotly_chart(fig, **W)

    narrative(
        "A high GPA is not the only determinant of financial success. The vertical spread of "
        "data points indicates significant differences in salary even among those with the same GPA, and"
        "technical fields such as computer science tend to offer higher financial rewards."
    )


elif selected_page == "6. Conclusion":
    st.subheader("E. Conclusion")
    st.markdown(
        """
Early career success is no longer determined solely by academic achievements, but rather
by a combination of academic achievements, practical experience, soft skills, and external factors
such as gender and field of study.

- **Practical experience is key.** Projects, internships, and certifications correlate almost
  perfectly with the number of job offers, they are more decisive than grades alone.
- **Same GPA, different outcomes.** With identical GPAs, starting salaries can vary widely,
  indicating an inequality of opportunity driven by practical portfolios.
- **Fields of study create structural inequality.** High-paying technical majors
  provide more opportunities to build portfolios that hold value in the job market.
- **There’s a trade-off between salary and work-life balance.** Groups that prioritize salary
  report significantly higher career satisfaction than those prioritizing work-life balance (WLB).
- **Gender equality is improving.** In Business and Engineering, female graduates
  earn higher starting salaries and receive promotions faster.

**Implications:** Universities need to shift toward project-based learning and certification
pathways, while students need to combine academic achievements with industry experience
and the development of soft skills.
        """
    )
    st.markdown("---")
    st.markdown("**Summary of figures for the currently filtered data**")
    summary_cols = [c for c in ["University_GPA", "Soft_Skills_Score", "Internships_Completed",
                                "Projects_Completed", "Certifications", "Job_Offers",
                                "Starting_Salary", "Career_Satisfaction", "Work_Life_Balance",
                                "Years_to_Promotion"] if c in df]
    st.dataframe(df[summary_cols].describe().T.round(2), **W)
    
elif selected_page == "7. Job Offers Predictions":
    st.subheader("7. Job Offer Predictions")
    st.markdown(
        "Fill out the following questionnaire based on your profile, then click **Predict**"
        " to see an estimate of how many job offers you might receive."
        "The model was trained on the same dataset as this dashboard."
    )
 
    try:
        model, categories, features = load_model()
    except Exception as e:
        st.error(
            f"**Gagal memuat model:** File model `{MODEL_PATH}` tidak ditemukan "
            f"atau terjadi kesalahan pembacaan.\n\nDetail error: `{e}`"
        )
        st.stop()
 
    with st.form("questionnaire_job_offers"):
        col1, col2 = st.columns(2)
 
        with col1:
            q_age = st.number_input("Age", min_value=17, max_value=60, value=22, step=1)
            q_gender = st.selectbox("Gender", categories["Gender"])
            q_university_gpa = st.number_input(
                "University GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01, format="%.2f"
            )
            q_field_of_study = st.selectbox("Field of Study", categories["Field_of_Study"])
            q_internships = st.number_input("Internships Completed", min_value=0, max_value=20, value=2, step=1)
 
        with col2:
            q_projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=5, step=1)
            q_certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2, step=1)
            q_soft_skills = st.slider("Soft Skills Score", min_value=0, max_value=10, value=7)
            q_networking = st.slider("Networking Score", min_value=0, max_value=10, value=6)
            q_starting_salary = st.number_input(
                "Starting Salary (Expectation)", min_value=0, max_value=1_000_000, value=60000, step=1000
            )
 
        q_submitted = st.form_submit_button("Predict")
 
    if q_submitted:
        input_df = pd.DataFrame([{
            "Age": q_age,
            "Gender": q_gender,
            "University_GPA": q_university_gpa,
            "Field_of_Study": q_field_of_study,
            "Internships_Completed": q_internships,
            "Projects_Completed": q_projects,
            "Certifications": q_certifications,
            "Soft_Skills_Score": q_soft_skills,
            "Networking_Score": q_networking,
            "Starting_Salary": q_starting_salary,
        }])[features]
 
        prediction = model.predict(input_df)[0]
        prediction = max(0, round(prediction))
 
        st.success(f"### Estimated Number of Job Offers: **{prediction}**")
        st.caption(
            "Note: These results are statistical estimates from a machine learning model, "
            "not a guarantee of actual results."
        )