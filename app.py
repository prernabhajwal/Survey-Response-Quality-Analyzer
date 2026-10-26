from pathlib import Path
import pandas as pd
import streamlit as st

from preprocessing import preprocess_data
from feature_engineering import extract_features
from response_analysis import analyze_responses
from anomaly_detection import detect_anomalies
from quality_scoring import score_quality
from database import save_results, get_results

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULT_FILE = OUTPUT_DIR / "final_quality_results.csv"

st.set_page_config(
    page_title="SurveyIQ | Survey Quality Analyzer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: #F6F8FC; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3E8F0; }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 26px; }
    .hero-label { color: #1A73E8; font-size: 0.78rem; font-weight: 700; letter-spacing: .08em; }
    .hero-title { color: #172B4D; font-size: 2rem; font-weight: 750; margin: 5px 0; }
    .hero-text { color: #627087; font-size: 1rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #E2E8F0; padding: 16px; border-radius: 12px; }
    .stButton > button[kind="primary"] { background: #1A73E8; border-color: #1A73E8; }
    h1, h2, h3 { color: #172B4D; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 📊 SurveyIQ")
st.sidebar.caption("Survey Response Quality Analyzer")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Home", "Upload & Analyze", "Results", "Saved Records"]
)
st.sidebar.divider()
st.sidebar.caption("Python · Pandas · Scikit-learn · SQLite · Streamlit")

if "result_df" not in st.session_state:
    st.session_state["result_df"] = None
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None

def run_pipeline(raw_df, question_columns):
    """Run each project module in the correct order and save its output."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    cleaned = preprocess_data(raw_df)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_survey_data.csv", index=False)

    featured = extract_features(cleaned, question_columns)
    featured.to_csv(OUTPUT_DIR / "survey_features.csv", index=False)

    analyzed = analyze_responses(featured, question_columns)
    analyzed.to_csv(OUTPUT_DIR / "response_analysis.csv", index=False)

    detected = detect_anomalies(analyzed)
    detected.to_csv(OUTPUT_DIR / "anomaly_results.csv", index=False)

    final_df = score_quality(detected)
    final_df.to_csv(RESULT_FILE, index=False)
    save_results(final_df)

    return final_df

def show_results(df):
    if df is None or df.empty:
        st.info("No analysis results yet. Upload a CSV and run an analysis first.")
        return

    total = len(df)
    average_score = df["Quality_Score"].mean() if "Quality_Score" in df else 0
    flagged = int(df["Review_Recommended"].fillna(False).astype(bool).sum()) if "Review_Recommended" in df else 0
    anomalies = int((df["Anomaly_Status"] == "Anomaly").sum()) if "Anomaly_Status" in df else 0

    a, b, c, d = st.columns(4)
    a.metric("Total responses", f"{total:,}")
    b.metric("Average quality score", f"{average_score:.1f}/100")
    c.metric("Recommended for review", f"{flagged:,}")
    d.metric("ML anomalies", f"{anomalies:,}")

    st.subheader("Quality category breakdown")
    if "Quality_Category" in df:
        category_counts = (
            df["Quality_Category"].value_counts()
            .reindex(["High Quality", "Medium Quality", "Low Quality"], fill_value=0)
        )
        st.bar_chart(category_counts)

    st.subheader("Response-level results")
    preferred = [
        "Response_ID", "Completion_Rate", "Missing_Answers",
        "Review_Recommended", "Anomaly_Status", "Quality_Score", "Quality_Category"
    ]
    visible = [column for column in preferred if column in df.columns]
    st.dataframe(df[visible] if visible else df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download final results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="surveyiq_final_results.csv",
        mime="text/csv"
    )
    st.caption("Scores and flags are review aids, not proof that a response is invalid.")

if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-label">SURVEY RESPONSE ANALYTICS</div>
        <div class="hero-title">Understand your survey data.</div>
        <div class="hero-text">
            Upload a survey dataset, review response patterns, identify unusual records,
            and explore response-quality scores in one place.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.subheader("How it works")
    steps = st.columns(3)
    steps[0].markdown("**01 · Upload**\n\nChoose a survey CSV and select the answer columns.")
    steps[1].markdown("**02 · Analyze**\n\nClean data, calculate features, check patterns, and detect anomalies.")
    steps[2].markdown("**03 · Review**\n\nExplore scores, flags, charts, and downloadable results.")
    st.info("Start with **Upload & Analyze** in the sidebar. The app accepts CSV files; analysis quality depends on selecting the correct answer columns.")
    if RESULT_FILE.exists():
        try:
            old_results = pd.read_csv(RESULT_FILE)
            st.subheader("Latest saved analysis")
            show_results(old_results)
        except Exception:
            pass

elif page == "Upload & Analyze":
    st.title("Upload a survey")
    st.write("Upload a CSV file. You can use rating, multiple-choice, numeric, or text-answer columns.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            st.subheader("Data preview")
            st.caption(f"{len(preview_df):,} rows · {len(preview_df.columns)} columns")
            st.dataframe(preview_df.head(8), use_container_width=True, hide_index=True)

            metadata_names = {
                "response_id", "respondent_id", "student_id", "record_id",
                "completion_status", "status", "submitted_at", "timestamp",
                "date", "email", "name", "response_time_seconds",
                "duration_seconds", "time_taken"
            }
            default_questions = [
                column for column in preview_df.columns
                if column.lower() not in metadata_names
                and not column.lower().endswith("_id")
            ]
            selected_questions = st.multiselect(
                "Select survey answer columns",
                options=list(preview_df.columns),
                default=default_questions,
                help="Choose the columns containing respondents' answers. Exclude IDs, names, timestamps and other metadata."
            )
            st.caption("Tip: For text questions, include the text-answer column if you want it considered in completion and pattern checks.")

            if st.button("Run survey analysis", type="primary", use_container_width=True):
                if not selected_questions:
                    st.error("Please select at least one survey answer column.")
                else:
                    with st.spinner("Running all analysis modules..."):
                        final_df = run_pipeline(preview_df, selected_questions)
                        st.session_state["result_df"] = final_df
                    st.success(f"Analysis complete. Processed {len(final_df):,} responses.")
                    st.session_state["result_df"] = final_df
                    st.rerun()
        except Exception as error:
            st.error(f"Could not read or analyze this CSV: {error}")

elif page == "Results":
    st.title("Analysis results")
    current_df = st.session_state.get("result_df")
    if current_df is None and RESULT_FILE.exists():
        try:
            current_df = pd.read_csv(RESULT_FILE)
        except Exception:
            current_df = None
    show_results(current_df)

elif page == "Saved Records":
    st.title("Saved records")
    st.write("These records are stored in the local SQLite database.")
    search = st.text_input("Search by response ID", placeholder="Enter part or all of an ID")
    try:
        records = get_results()
        if search.strip():
            records = records[
                records["response_id"].astype(str).str.contains(search.strip(), case=False, na=False)
            ]
        st.metric("Matching saved records", len(records))
        st.dataframe(records, use_container_width=True, hide_index=True)
        st.download_button(
            "Download saved records",
            data=records.to_csv(index=False).encode("utf-8"),
            file_name="surveyiq_saved_records.csv",
            mime="text/csv"
        )
    except Exception as error:
        st.error(f"Could not load the database: {error}")
#python streamlit run app.py