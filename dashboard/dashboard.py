import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

DEFAULT_PATH = "../dataset/education_career_success_cleaned.csv"

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
    st.error(f"❌ File `{DEFAULT_PATH}` tidak ditemukan di folder lokal. Pastikan file CSV berada di direktori yang sama dengan script.")
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
        "6. Conclusion"
    ]
)

st.markdown("---")

st.sidebar.subheader("Filter global")

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

# st.sidebar.markdown("---")
# st.sidebar.metric("Baris terpakai", f"{len(df):,}", f"{len(df) - len(df_all):,} vs total")
# if st.sidebar.button("Reset filter"):
#     st.rerun()
# st.sidebar.download_button("Unduh data terfilter",
#                             df.to_csv(index=False).encode(), "filtered_data.csv", "text/csv")
# st.sidebar.caption(f"Sumber: {source_note}")

if df.empty:
    st.error("Tidak ada data yang lolos filter. Longgarkan filter di sidebar.")
    st.stop()

st.title("Factors That Influence Career Success")
st.markdown("---")


if selected_page == "Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown(
        "Halaman ini menampilkan gambaran pemetaan awal terhadap karakteristik akademik, "
        "sebaran bidang studi, serta portofolio pengalaman mahasiswa (proyek, magang, sertifikasi)."
    )

    k = st.columns(5)
    k[0].metric("Jumlah mahasiswa", f"{len(df):,}")
    if "University_GPA" in df:
        k[1].metric("Rata-rata GPA", f"{df['University_GPA'].mean():.2f}")
    if "Starting_Salary" in df:
        k[2].metric("Rata-rata gaji awal", f"${df['Starting_Salary'].mean():,.0f}")
    if "Job_Offers" in df:
        k[3].metric("Rata-rata job offers", f"{df['Job_Offers'].mean():.2f}")
    if "Career_Satisfaction" in df:
        k[4].metric("Kepuasan karier", f"{df['Career_Satisfaction'].mean():.2f}/10")

    st.markdown("### 1. Profil Academic & IPK Mahasiswa")
    e1, e2 = st.columns(2)
    with e1:
        if "University_GPA" in df.columns:
            fig_gpa = px.histogram(
                df, x="University_GPA", nbins=20, marginal="box",
                title="Sebaran IPK Mahasiswa (University GPA)",
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
                title="Rata-rata IPK per Bidang Studi",
                color="University_GPA", color_continuous_scale="Blues",
                text=gpa_field["University_GPA"].round(2)
            )
            fig_gpa_field.update_traces(textposition="outside")
            fig_gpa_field.update_yaxes(range=[0, 4.0])
            st.plotly_chart(fig_gpa_field, **W)

    st.markdown("### 2. Sebaran Bidang Studi (Field of Study)")
    if "Field_of_Study" in df.columns:
        f_counts = df["Field_of_Study"].value_counts().reset_index()
        f_counts.columns = ["Field_of_Study", "Jumlah Mahasiswa"]
        fig_field = px.pie(
            f_counts, names="Field_of_Study", values="Jumlah Mahasiswa",
            title="Komposisi Mahasiswa Berdasarkan Field of Study",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_field, **W)

    st.markdown("### 3. Rentang Pengalaman & Portofolio Mahasiswa")
    st.markdown("Pemetaan distribusi jumlah **Proyek**, **Magang (Internship)**, dan **Sertifikasi** yang diikuti oleh mahasiswa:")
    
    p1, p2, p3 = st.columns(3)
    with p1:
        if "Projects_Completed" in df.columns:
            fig_proj = px.histogram(
                df, x="Projects_Completed",
                title="Rentang Proyek Selesai",
                color_discrete_sequence=["#27ae60"], text_auto=True
            )
            st.plotly_chart(fig_proj, **W)
    with p2:
        if "Internships_Completed" in df.columns:
            fig_intern = px.histogram(
                df, x="Internships_Completed",
                title="Rentang Magang Selesai",
                color_discrete_sequence=["#e67e22"], text_auto=True
            )
            st.plotly_chart(fig_intern, **W)
    with p3:
        if "Certifications" in df.columns:
            fig_cert = px.histogram(
                df, x="Certifications",
                title="Rentang Sertifikasi Selesai",
                color_discrete_sequence=["#8e44ad"], text_auto=True
            )
            st.plotly_chart(fig_cert, **W)

elif selected_page == "1. Recruitment Trends":
    st.subheader("1. Shifting Recruitment Trends")

    st.markdown("#### a. Bagaimana pengaruh soft skills dibandingkan GPA terhadap peluang job offer?")
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
        st.markdown("**Nilai korelasi Spearman**")
        st.dataframe(m, **W)

    narrative(
        "Kedua scatter menunjukkan tren naik: GPA maupun soft skill sama-sama diikuti "
        "kenaikan jumlah job offer. Heatmap memperlihatkan korelasi positif sangat kuat "
        "(±0.95–0.97) di antara ketiganya, artinya kemampuan akademik dan interpersonal "
        "sama-sama berperan besar pada peluang kerja."
    )

    st.markdown("---")
    st.markdown("#### b. Apakah individu dengan skill kuat tetapi GPA rendah tetap mendapat job offer?")
    hi_skill = st.slider("Ambang soft skill 'tinggi' (≥)", 1.0, 10.0,
                         float(df["Soft_Skills_Score"].quantile(0.75)), 0.5)
    lo_gpa = st.slider("Ambang GPA 'rendah' (≤)", float(df["University_GPA"].min()),
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
    m1.metric("Jumlah 'skill tinggi, GPA rendah'", len(seg))
    m2.metric("Rata-rata job offers segmen ini",
              f"{seg['Job_Offers'].mean():.2f}" if len(seg) else "—")
    m3.metric("Yang dapat ≥1 offer",
              f"{(seg['Job_Offers'] >= 1).mean() * 100:.0f}%" if len(seg) else "—")
    if len(seg):
        st.dataframe(seg.head(20), **W)

    narrative(
        "Karena GPA dan soft skill sama-sama berkorelasi sangat kuat dengan job offers, "
        "kuadran 'soft skill tinggi tetapi GPA rendah' nyaris kosong pada dataset ini — "
        "silakan geser kedua ambang untuk memeriksa sendiri."
    )


elif selected_page == "2. Career Well-Being":
    st.subheader("2. Demands for Career Well-Being")

    st.markdown("#### a. Bagaimana gaji dan work-life balance bersama-sama memengaruhi kepuasan karier?")
    w = st.slider("Bobot gaji dalam indeks gabungan", 0.0, 1.0, 0.5, 0.05,
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
                                 "spearman", "Korelasi Index vs Satisfaction", height=480)
        st.plotly_chart(fig2, **W)

    narrative(
        "Sebaran sangat bervariasi: pada indeks rendah kepuasan didominasi level sedang, "
        "sementara skor 9–10 baru muncul konsisten setelah indeks melewati ~0.38. "
        "Korelasinya hanya lemah–moderat, jadi kombinasi gaji dan WLB bukan satu-satunya "
        "penentu kepuasan karier."
    )

    st.markdown("---")
    st.markdown("#### b. Kepuasan karier: kelompok pengejar gaji vs pengejar work-life balance")
    q = st.slider("Kuantil pembatas kelompok prioritas", 0.50, 0.95, 0.75, 0.05)
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
        "Kelompok yang gajinya tinggi melaporkan kepuasan karier jauh lebih tinggi "
        "dibanding kelompok dengan WLB tinggi. Terlihat pula trade-off sistemik: "
        "gaji tinggi berbanding terbalik dengan skor work-life balance."
    )


elif selected_page == "3. Skills Gap":
    st.subheader("3. Skills Gap")

    st.markdown("#### a. Dampak magang dan proyek terhadap job offers")
    fig, cm = corr_heatmap(df, ["Internships_Completed", "Projects_Completed", "Job_Offers"],
                           "pearson", "Matrix Correlation", height=380)
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.plotly_chart(fig, **W)
    with c2:
        st.dataframe(cm, **W)
        st.caption("Korelasi mendekati sempurna: portofolio praktik sangat menentukan rekrutmen.")

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
        "Ada ambang minimum: kandidat tanpa magang atau dengan proyek sangat sedikit "
        "hampir tidak menerima tawaran. Kombinasi magang banyak + proyek banyak "
        "menghasilkan jumlah tawaran tertinggi (efek pengganda)."
    )

    st.markdown("---")
    st.markdown("#### b. Seberapa besar dampak soft skill & sertifikasi dibanding pendidikan formal?")
    agg_cols = ["Education", "Soft Skills + Certifications", "Career_Success_Score"]
    fig, corr_sp = corr_heatmap(df, agg_cols, "spearman",
                                "Correlation Heatmap (Aggregated Columns)", fmt=".4f", height=430)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig, **W)
    with c2:
        target = "Career_Success_Score"
        tc = corr_sp[target].drop(target).sort_values(key=abs, ascending=False)
        bar = pd.DataFrame({"Kategori": tc.index, "Korelasi": tc.values})
        fig = px.bar(bar, x="Kategori", y="Korelasi", color="Kategori",
                     color_discrete_sequence=["orange", "skyblue"],
                     text=bar["Korelasi"].round(4), height=430)
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[max(0, bar["Korelasi"].min() - 0.01), 1.0])
        fig.update_layout(title="Average Impact on Career Success", showlegend=False,
                          yaxis_title="Average Absolute Correlation",
                          margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(fig, **W)

    st.caption("Catatan: `Education` = rata-rata High_School_GPA, SAT_Score, University_GPA. "
               "`Career_Success_Score` = rata-rata Starting_Salary, Job_Offers, Career_Satisfaction.")

    narrative(
        "Keduanya berpengaruh sangat kuat, tetapi Soft Skills + Certifications sedikit "
        "unggul dibanding Education — sejalan dengan tren skills-based hiring. Keduanya "
        "juga saling berkorelasi tinggi, menandakan mahasiswa berprestasi akademik "
        "cenderung juga mengumpulkan sertifikasi."
    )


elif selected_page == "4. Gender Parity":
    st.subheader("4. Gender Parity in Professional Outcomes")

    st.markdown("#### a. Perbedaan gaji awal & waktu promosi antar gender pada bidang studi yang sama")
    avail = sorted(df["Field_of_Study"].dropna().unique())
    default_fields = [f for f in ["Engineering", "Business"] if f in avail] or avail[:2]
    pick_fields = st.multiselect("Bidang studi yang dibandingkan", avail, default=default_fields)
    dff = df[df["Field_of_Study"].isin(pick_fields)]

    if dff.empty:
        st.info("Pilih minimal satu bidang studi.")
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
        "Pada dataset ini perempuan unggul di kedua bidang, baik dari sisi gaji awal "
        "maupun kecepatan promosi, dengan selisih paling mencolok di bidang Business."
    )

    st.markdown("---")
    st.markdown("#### b. Pengaruh gender terhadap jumlah job offer pada kelompok GPA setara")
    edges = st.slider("Batas kelompok GPA", 2.0, 4.0, (3.3, 3.6), 0.1)
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
        "GPA lebih tinggi selalu berarti lebih banyak tawaran kerja untuk semua gender. "
        "Pada kelompok GPA bawah–menengah perempuan cenderung unggul, sedangkan pada "
        "kelompok GPA tertinggi selisihnya nyaris hilang."
    )


elif selected_page == "5. Unequal Opportunities":
    st.subheader("5. Unequal Early-Career Opportunities Among Graduates")

    st.markdown("#### a. GPA setara, pengalaman praktik berbeda — hasil kariernya berbeda?")
    color_by = st.selectbox("Warnai titik berdasarkan",
                            [c for c in ["Projects_Completed", "Certifications",
                                         "Internships_Completed", "Networking_Score"] if c in df])
    fig = px.scatter(df, x="University_GPA", y="Starting_Salary", color=color_by,
                     color_continuous_scale="Viridis", opacity=0.8, height=520,
                     hover_data=[c for c in ["Field_of_Study", "Gender", "Job_Offers"] if c in df])
    fig.update_layout(title=f"GPA vs Starting Salary berdasarkan {color_by}",
                      xaxis_title="University GPA", yaxis_title="Starting Salary ($)",
                      margin=dict(t=55, b=10, l=10, r=10))
    st.plotly_chart(fig, **W)

    gpa_pick = st.slider("Periksa sebaran gaji pada GPA tertentu (±0.05)",
                         float(df["University_GPA"].min()), float(df["University_GPA"].max()),
                         float(round(df["University_GPA"].median(), 1)), 0.05)
    same = df[df["University_GPA"].between(gpa_pick - 0.05, gpa_pick + 0.05)]
    if len(same) > 1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mahasiswa dengan GPA ini", len(same))
        c2.metric("Rentang gaji",
                  f"${same['Starting_Salary'].max() - same['Starting_Salary'].min():,.0f}")
        c3.metric("Rata-rata gaji", f"${same['Starting_Salary'].mean():,.0f}")

    narrative(
        "Prestasi akademik yang sama tidak menjamin hasil karier yang sama: pada GPA "
        "identik, selisih gaji awal bisa sangat lebar. Pembedanya adalah portofolio "
        "praktik dan sertifikasi profesional."
    )

    st.markdown("---")
    st.markdown("#### b. Apakah bidang studi tertentu memberi hasil karier jauh lebih tinggi?")
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
        fig.update_layout(title="Sebaran gaji awal per bidang studi", showlegend=False,
                          xaxis_title="", margin=dict(t=55, b=10, l=10, r=10))
        st.plotly_chart(fig, **W)
    with c2:
        rank = (df.groupby("Field_of_Study", observed=True)
                .agg(Rata_Gaji=("Starting_Salary", "mean"),
                     Rata_Job_Offers=("Job_Offers", "mean"),
                     Rata_Kepuasan=("Career_Satisfaction", "mean"),
                     Jumlah=("Starting_Salary", "size"))
                .round(2).sort_values("Rata_Gaji", ascending=False))
        st.markdown("**Peringkat bidang studi**")
        st.dataframe(rank, **W, height=430)

    fig, _ = corr_heatmap(df, ["University_GPA", "Projects_Completed", "Certifications",
                               "Starting_Salary", "Job_Offers"], "pearson",
                          "Correlation Heatmap: Academic and Practical Factors",
                          zmin=0, zmax=1, height=560)
    st.plotly_chart(fig, **W)

    narrative(
        "GPA tinggi bukan satu-satunya penentu keberhasilan finansial. Sebaran vertikal "
        "titik menunjukkan perbedaan gaji besar pada GPA yang sama, dan bidang studi "
        "teknis seperti Computer Science cenderung mendapat imbalan finansial lebih tinggi."
    )


elif selected_page == "6. Conclusion":
    st.subheader("E. Conclusion")
    st.markdown(
        """
Kesuksesan karier awal tidak lagi ditentukan semata oleh capaian akademik, melainkan
oleh kombinasi prestasi akademik, pengalaman praktik, soft skill, serta faktor eksternal
seperti gender dan bidang studi.

- **Pengalaman praktik menentukan.** Proyek, magang, dan sertifikasi berkorelasi hampir
  sempurna dengan jumlah tawaran kerja — lebih menentukan daripada nilai semata.
- **GPA sama, hasil berbeda.** Pada GPA identik, gaji awal bisa berbeda sangat jauh,
  menandakan ketimpangan peluang yang digerakkan oleh portofolio praktik.
- **Bidang studi menciptakan ketimpangan struktural.** Jurusan teknis bergaji tinggi
  memberi lebih banyak kesempatan membangun portofolio yang bernilai di pasar kerja.
- **Ada trade-off gaji vs work-life balance.** Kelompok yang memprioritaskan gaji
  melaporkan kepuasan karier jauh lebih tinggi dibanding kelompok WLB.
- **Kesetaraan gender membaik.** Pada bidang Business dan Engineering, lulusan perempuan
  memperoleh gaji awal lebih tinggi dan promosi lebih cepat.

**Implikasi:** perguruan tinggi perlu bergerak ke pembelajaran berbasis proyek dan jalur
sertifikasi, sementara mahasiswa perlu memadukan capaian akademik dengan pengalaman
industri dan pengasahan soft skill.
        """
    )
    st.markdown("---")
    st.markdown("**Ringkasan angka pada data yang sedang difilter**")
    summary_cols = [c for c in ["University_GPA", "Soft_Skills_Score", "Internships_Completed",
                                "Projects_Completed", "Certifications", "Job_Offers",
                                "Starting_Salary", "Career_Satisfaction", "Work_Life_Balance",
                                "Years_to_Promotion"] if c in df]
    st.dataframe(df[summary_cols].describe().T.round(2), **W)
    st.caption("Dashboard dibuat dengan Streamlit + Plotly · Group 2 · SDG 4 & 8")