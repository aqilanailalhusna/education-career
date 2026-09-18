import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Factors Influencing Career Success",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../dataset/education_career_success.csv')
        columns_to_drop = [col for col in ['Student_ID', 'Entrepreneurship'] if col in df.columns]
    except FileNotFoundError:
        st.error("Dataset file not found.")
        return None, None
    df_cleaned = df.drop(columns=columns_to_drop)
    return df, df_cleaned

raw_df, df = load_data()

st.sidebar.header("🔍 Filter Data")
fields_of_study = ["All"] + list(df['Field_of_Study'].unique())
selected_field = st.sidebar.selectbox("Select Field of Study:", fields_of_study)

if selected_field != "All":
    filtered_df = df[df['Field_of_Study'] == selected_field]
else:
    filtered_df = df.copy()

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini menganalisis faktor-faktor akademis dan non-akademis yang memengaruhi kesuksesan karir lulusan (SDG 4 & 8).")

st.title("Factors That Influence Career Success (SDG 4 & 8)")
st.markdown("---")

with st.expander("A. Introduction & Background", expanded=False):
    st.write("""
    The shift from college to the job market has become more competitive and varied in recent times. 
    Employers have begun to look beyond just academic success and are now focusing on real-world experience, 
    interpersonal skills, the ability to build connections, and relevant certifications.
    
    This collection of data includes **400 anonymous student records** reflecting academic, professional, and personal growth metrics. 
    By investigating the connections between academic success, hands-on experiences, networking skills, and career results, 
    this research seeks to uncover which factors have the strongest impact on early career outcomes.
    """)

st.header("Data Overview & Quality Assessment")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(filtered_df))
col2.metric("Avg University GPA", f"{filtered_df['University_GPA'].mean():.2f}")
col3.metric("Avg Soft Skills Score", f"{filtered_df['Soft_Skills_Score'].mean():.1f}/10")
col4.metric("Avg Job Offers", f"{filtered_df['Job_Offers'].mean():.2f}")

tab1, tab2, tab3 = st.tabs(["Filtered Data", "Summary Statistics", "Data Cleaning Summary"])

with tab1:
    st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.dataframe(filtered_df.describe(), use_container_width=True)

with tab3:
    st.markdown("""
    **Summary of Data Cleaning:**
    - **Missing Values:** No missing values found in the dataset.
    - **Dropped Columns:**
      - `Student_ID`: Removed because it serves only as a personal identifier without analytical value.
      - `Entrepreneurship`: Removed because all records contain the value `'No'`, providing no variance or insights.
    """)

st.markdown("---")


st.header("Analysis & Visualization")
st.subheader("1. Shifting Recruitment Trends")
st.markdown("**Question:** *How do soft skills, compared to GPA, affect your chances of receiving a job offer?*")

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    fig_gpa = px.scatter(
        filtered_df,
        x='University_GPA',
        y='Job_Offers',
        trendline="ols",
        trendline_color_override="green",
        title="University GPA vs Job Offers",
        labels={'University_GPA': 'University GPA', 'Job_Offers': 'Job Offers Received'},
        opacity=0.7
    )
    fig_gpa.update_layout(template="plotly_white")
    st.plotly_chart(fig_gpa, use_container_width=True)

with fig_col2:
    fig_soft = px.scatter(
        filtered_df,
        x='Soft_Skills_Score',
        y='Job_Offers',
        trendline="ols",
        trendline_color_override="green",
        title="Soft Skills Score vs Job Offers",
        labels={'Soft_Skills_Score': 'Soft Skills Score', 'Job_Offers': 'Job Offers Received'},
        opacity=0.7
    )
    fig_soft.update_layout(template="plotly_white")
    st.plotly_chart(fig_soft, use_container_width=True)

st.subheader("Correlation Heatmap: GPA, Soft Skills & Job Offers")
corr_df = filtered_df[['Job_Offers', 'Soft_Skills_Score', 'University_GPA']].corr(method='spearman')

fig_corr = px.imshow(
    corr_df,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Spearman Correlation Matrix",
    aspect="auto"
)
fig_corr.update_layout(template="plotly_white")

col_corr, col_desc = st.columns([1, 1])

with col_corr:
    st.plotly_chart(fig_corr, use_container_width=True)

with col_desc:
    st.markdown("### Key Findings")
    st.write("""
    1. **University GPA & Job Offers:** Shows a strong positive correlation (~0.95–0.97). Students with higher academic performance consistently receive more job offers.
    2. **Soft Skills Score & Job Offers:** Demonstrates an equally strong positive linear relationship. Interpersonal communication, teamwork, and leadership heavily influence employer hiring decisions.
    3. **Conclusion:** Both academic excellence (University GPA) and non-academic capabilities (Soft Skills) are equally vital drivers for early-career job placement success.
    """)