import os
import sqlite3
import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------
# SurveyIQ — Survey Response Quality Analyzer
# ---------------------------------------------------------
st.set_page_config(
    page_title="SurveyIQ | Response Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATABASE = Path("survey_database.db")
RESULT_FILE = OUTPUT_DIR / "final_quality_results.csv"

QUESTION_COLUMNS = [
    "Q1_Ease_of_Use",
    "Q2_Satisfaction",
    "Q3_Speed",
    "Q4_Support",
    "Q5_Clarity",
    "Q6_Reliability",
    "Q7_Recommend",
    "Q8_Overall_Experience",
]

# -------------------------- Styling -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
      --bg:#0b1120; --surface:#172235; --surface-soft:#1d2b40; --line:#2b3b52;
      --text:#f3f6fc; --muted:#9aa9bd; --blue:#4f8cff; --blue-soft:#1b3154;
      --purple:#a78bfa; --teal:#20c5b5; --green:#22c55e; --orange:#f59e0b;
    }
    html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--text); }
    .stApp { background:var(--bg); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(11,17,32,.94); }
    [data-testid="stSidebar"] { background:#111c2e; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--text); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.65rem .75rem; border-radius:10px; margin:.12rem 0; font-weight:600; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#20314a; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] { background:#0f766e; color:#ffffff; }
    .block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1580px; }
    h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; color:#f4f7fc !important; letter-spacing:-.025em; }
    p, li, label { color:#c3cede; }
    .brand { font-family:'Manrope',sans-serif; font-size:1.65rem; font-weight:800; letter-spacing:-.05em; color:#20c5b5; }
    .brand span { color:#f3f6fc; }
    .eyebrow { color:#8fa2bc; font-size:.70rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    .muted { color:var(--muted); }
    .hero { position:relative; overflow:hidden; background:linear-gradient(115deg,#17263b 0%,#182b45 65%,#123a3d 100%); border:1px solid #2b4561; border-radius:20px; padding:1.6rem 1.8rem; margin-bottom:1.3rem; }
    .hero:after { content:'✦'; position:absolute; right:5%; top:-55px; font-size:13rem; color:rgba(32,197,181,.08); transform:rotate(12deg); pointer-events:none; }
    .hero-title { font-family:'Manrope',sans-serif; font-size:2.05rem; line-height:1.15; font-weight:800; color:#f8fbff; margin:.42rem 0 .55rem; letter-spacing:-.04em; }
    .hero-sub { color:#b8c8dc; font-size:.98rem; max-width:760px; line-height:1.65; }
    .metric-card { background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:1.1rem 1.15rem; min-height:120px; box-shadow:0 3px 12px rgba(0,0,0,.12); }
    .metric-card:hover { border-color:#496888; }
    .metric-label { color:#9aabc1; font-size:.72rem; font-weight:800; letter-spacing:.06em; }
    .metric-value { color:#f5f8ff; font-family:'Manrope',sans-serif; font-size:1.9rem; font-weight:800; margin-top:.48rem; line-height:1.1; }
    .metric-note { color:#20c5b5; font-size:.77rem; margin-top:.5rem; font-weight:600; }
    .section-head { font-family:'Manrope',sans-serif; color:#e5edf8; font-size:1.08rem; font-weight:800; margin:.65rem 0 .85rem; }
    .pill { display:inline-block; border-radius:999px; padding:.38rem .72rem; font-size:.73rem; font-weight:700; background:#1b2c42; color:#c8d8ee; border:1px solid #344a64; }
    div[data-testid="stFileUploader"] { background:#172235; border:1.5px dashed #496888; border-radius:14px; padding:.8rem; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#172235; }
    div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); padding:1rem; border-radius:14px; }
    div[data-testid="stMetricLabel"] { color:#a7b6ca; }
    div[data-testid="stMetricValue"] { color:#f5f8ff; }
    .stButton button,.stDownloadButton button { border-radius:9px; border:1px solid #167d78; background:#0f766e; color:#fff; font-weight:700; min-height:2.55rem; transition:all .16s ease; }
    .stButton button:hover,.stDownloadButton button:hover { border-color:#20c5b5; background:#0e8b80; color:#fff; }
    .stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] { background:#1b2c42; color:#e8f0fc; border-color:#3a506b; }
    div[data-baseweb="select"] > div { background:#172235; border-color:#344a64; border-radius:9px; }
    input,textarea { background:#172235 !important; color:#f3f6fc !important; border-color:#344a64 !important; }
    [data-testid="stExpander"] { background:#172235; border:1px solid var(--line); border-radius:12px; }
    [data-testid="stAlert"] { border-radius:11px; }
    [data-testid="stTabs"] button { font-weight:700; color:#b8c8dc; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:#20c5b5; }
    hr { border-color:#2b3b52; }
    #MainMenu, footer { visibility:hidden; }
    @media(max-width:700px) { .block-container{padding-left:1rem;padding-right:1rem;} h1{font-size:1.75rem!important;} .hero{padding:1.25rem;} .hero-title{font-size:1.65rem;} .metric-value{font-size:1.65rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- Analysis pipeline -------------
def infer_question_columns(df: pd.DataFrame):
    """Infer answer/question fields while excluding common respondent metadata."""
    metadata_tokens = (
        "response_id", "respondent_id", "observation", "timestamp", "submitted_at",
        "date", "time_seconds", "duration", "completion_status", "email", "name",
        "phone", "ip_address", "start_time", "end_time", "session_id",
        "completion_rate", "unique_answers", "response_variation", "straight_lining",
        "average_rating", "comment_length", "missing_answers", "very_short_response_time",
        "incomplete_response", "fast_response_flag", "repeated_pattern_flag",
        "short_comment_flag", "review_recommended", "anomaly_status", "quality_score",
        "quality_category", "_pattern_key"
    )
    candidates = []
    for col in df.columns:
        name = str(col).strip().lower().replace(" ", "_")
        if any(token in name for token in metadata_tokens):
            continue
        # Keep columns with actual response data; exclude fully empty columns.
        if df[col].notna().any():
            candidates.append(col)
    return candidates


def run_analysis(raw_df: pd.DataFrame, question_columns=None) -> pd.DataFrame:
    """Analyze general survey CSVs without requiring fixed question names."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or len(df.columns) == 0:
        raise ValueError("The uploaded CSV is empty. Please upload a file containing survey responses.")
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "cleaned_survey_data.csv", index=False)

    questions = list(question_columns) if question_columns else infer_question_columns(df)
    questions = [c for c in questions if c in df.columns]
    if not questions:
        raise ValueError("No response columns could be detected. Select your survey answer columns and try again.")

    if "Response_ID" not in df.columns:
        df.insert(0, "Response_ID", [f"RESP{i:04d}" for i in range(1, len(df) + 1)])
    if "Response_Time_Seconds" not in df.columns:
        # Support common alternative duration column names.
        time_col = next((c for c in df.columns if any(x in c.lower() for x in ["response_time", "duration_seconds", "time_taken"])), None)
        df["Response_Time_Seconds"] = pd.to_numeric(df[time_col], errors="coerce") if time_col else float("nan")
    else:
        df["Response_Time_Seconds"] = pd.to_numeric(df["Response_Time_Seconds"], errors="coerce")

    comment_col = next((c for c in df.columns if any(x in c.lower() for x in ["comment", "feedback", "open_ended", "text_response", "suggestion"])), None)
    if comment_col:
        comments = df[comment_col].fillna("").astype(str)
    else:
        comments = pd.Series([""] * len(df), index=df.index)
        # If no obvious comment field exists, don't mislabel arbitrary categorical answers as comments.
    answers = df[questions].copy()
    # A cell is unanswered only when blank/NA. Values such as "Not selected" are valid recorded answers.
    normalized = answers.replace(r"^\s*$", pd.NA, regex=True)
    answered_count = normalized.notna().sum(axis=1)
    total_questions = max(len(questions), 1)

    # Encode each answer column only for numeric pattern/ML calculations; retain original answers in output.
    encoded = pd.DataFrame(index=df.index)
    numeric_cols = []
    for col in questions:
        numeric = pd.to_numeric(normalized[col], errors="coerce")
        # Use numeric values when the column is substantially numeric; otherwise categorical codes.
        nonmissing = normalized[col].notna().sum()
        if nonmissing and numeric.notna().sum() >= max(1, int(nonmissing * 0.7)):
            encoded[col] = numeric
            numeric_cols.append(col)
        else:
            categories = normalized[col].astype("category")
            encoded[col] = categories.cat.codes.replace(-1, float("nan")).astype(float)

    df["Completion_Rate"] = answered_count / total_questions * 100
    df["Unique_Answers"] = normalized.nunique(axis=1, dropna=True)
    df["Response_Variation"] = encoded.std(axis=1).fillna(0)
    df["Straight_Lining"] = (df["Unique_Answers"] == 1) & (answered_count >= 2)
    # Average rating is meaningful only when there are numeric answer columns.
    df["Average_Rating"] = encoded[numeric_cols].mean(axis=1) if numeric_cols else float("nan")
    df["Comment_Length"] = comments.str.split().str.len().fillna(0).astype(int)
    df["Missing_Answers"] = total_questions - answered_count
    df["Very_Short_Response_Time"] = df["Response_Time_Seconds"].lt(60).fillna(False)
    df["_Pattern_Key"] = normalized.astype("string").fillna("<MISSING>").agg("|".join, axis=1)
    feature_cols = ["Response_ID"] + questions + ["Response_Time_Seconds", "Completion_Rate", "Unique_Answers", "Response_Variation", "Straight_Lining", "Average_Rating", "Comment_Length", "Missing_Answers", "Very_Short_Response_Time"]
    df[[c for c in feature_cols if c in df.columns]].to_csv(OUTPUT_DIR / "survey_features.csv", index=False)

    df["Incomplete_Response"] = df["Missing_Answers"] > 0
    df["Straight_Lining_Flag"] = df["Straight_Lining"].astype(bool)
    df["Fast_Response_Flag"] = df["Very_Short_Response_Time"].astype(bool)
    df["Repeated_Pattern_Flag"] = df["_Pattern_Key"].duplicated(keep=False)
    df["Short_Comment_Flag"] = (df["Comment_Length"] <= 2) if comment_col else False
    flag_cols = ["Incomplete_Response", "Straight_Lining_Flag", "Fast_Response_Flag", "Repeated_Pattern_Flag", "Short_Comment_Flag"]
    df["Review_Recommended"] = df[flag_cols].any(axis=1)
    df.drop(columns=["_Pattern_Key"], inplace=True, errors="ignore")
    df.to_csv(OUTPUT_DIR / "response_analysis.csv", index=False)

    ml_features = ["Completion_Rate", "Unique_Answers", "Response_Variation", "Average_Rating", "Comment_Length", "Response_Time_Seconds"]
    X = df[ml_features].apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median()).fillna(0)
    if len(df) >= 5:
        predictions = IsolationForest(n_estimators=100, contamination="auto", random_state=42).fit_predict(X)
        df["Anomaly_Status"] = pd.Series(predictions, index=df.index).map({1: "Normal", -1: "Anomaly"})
    else:
        df["Anomaly_Status"] = "Not evaluated (need 5+ responses)"
    df.to_csv(OUTPUT_DIR / "anomaly_results.csv", index=False)

    df["Quality_Score"] = 100
    for flag, points in [("Incomplete_Response", 20), ("Straight_Lining_Flag", 15), ("Fast_Response_Flag", 15), ("Repeated_Pattern_Flag", 15), ("Short_Comment_Flag", 10)]:
        df.loc[df[flag].fillna(False).astype(bool), "Quality_Score"] -= points
    df.loc[df["Anomaly_Status"] == "Anomaly", "Quality_Score"] -= 15
    df["Quality_Score"] = df["Quality_Score"].clip(0, 100)
    df["Quality_Category"] = df["Quality_Score"].apply(lambda score: "High Quality" if score >= 75 else ("Medium Quality" if score >= 50 else "Low Quality"))
    return df


def save_to_database(df: pd.DataFrame) -> None:
    """Save latest results to SQLite using the existing project table schema."""
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS survey_responses (
                response_id TEXT PRIMARY KEY,
                completion_rate REAL,
                average_rating REAL,
                quality_score REAL,
                quality_category TEXT,
                anomaly_status TEXT
            )
            """
        )
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO survey_responses
                (response_id, completion_rate, average_rating, quality_score, quality_category, anomaly_status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["Response_ID"]),
                    _number(row.get("Completion_Rate")),
                    _number(row.get("Average_Rating")),
                    _number(row.get("Quality_Score")),
                    str(row.get("Quality_Category", "")),
                    str(row.get("Anomaly_Status", "")),
                ),
            )


def _number(value):
    try:
        return float(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------- Sidebar -----------------------
with st.sidebar:
    st.markdown('<div class="brand">Survey<span>IQ</span> ✦</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Response intelligence workspace</p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Module pipeline", "Explore responses", "Question insights", "Upload & analyze", "Database"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown('<div class="eyebrow">PROJECT STATUS</div>', unsafe_allow_html=True)
    st.markdown("🟢 Analysis engine")
    st.markdown("🗄️ Results database")
    st.markdown("🧠 Anomaly detection")
    st.markdown("<br><span class='muted'>Flags suggest a review; they do not prove a response is invalid.</span>", unsafe_allow_html=True)

# Load the most recent results, if available.
df = None
if "analysis_df" in st.session_state:
    df = st.session_state["analysis_df"].copy()
elif RESULT_FILE.exists():
    try:
        df = pd.read_csv(RESULT_FILE)
    except Exception:
        df = None

# -------------------------- Header ------------------------
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 .85rem 0;">
      <div class="eyebrow">SURVEYIQ  /  ANALYTICS WORKSPACE</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span class="pill">● Engine ready</span><span class="pill">SQLite connected</span><span class="pill">ML enabled</span>
      </div>
    </div>
    <div class="hero">
      <div class="eyebrow">SURVEY RESPONSE INTELLIGENCE</div>
      <div class="hero-title">Clarity in every response.</div>
      <div class="hero-sub">Upload survey responses to see data quality, suspicious patterns, ML anomaly signals, and a clear quality score for every response.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------- Pages -------------------------
if page == "Module pipeline":
    st.header("Module pipeline")
    st.write("Follow each stage of the analysis. Every card explains what the module does, which output it creates, and how its result is used by the next stage.")

    modules = [
        {"name": "1 · Data Preprocessing", "script": "preprocessing.py", "output": OUTPUT_DIR / "cleaned_survey_data.csv", "purpose": "Loads the raw survey, checks data quality, and removes exact duplicate rows.", "columns": ["Response_ID", "Completion_Status", "Response_Time_Seconds"]},
        {"name": "2 · Feature Engineering", "script": "feature_engineering.py", "output": OUTPUT_DIR / "survey_features.csv", "purpose": "Creates completion rate, unique answers, variation, straight-line indicator, average rating, comment length, and missing-answer features.", "columns": ["Response_ID", "Completion_Rate", "Unique_Answers", "Response_Variation", "Straight_Lining", "Average_Rating", "Comment_Length", "Missing_Answers"]},
        {"name": "3 · Response Pattern Analysis", "script": "response_analysis.py", "output": OUTPUT_DIR / "response_analysis.csv", "purpose": "Checks incomplete answers, straight-lining, very fast responses, repeated rating patterns, and short comments.", "columns": ["Response_ID", "Incomplete_Response", "Straight_Lining_Flag", "Fast_Response_Flag", "Repeated_Pattern_Flag", "Short_Comment_Flag", "Review_Recommended"]},
        {"name": "4 · ML Anomaly Detection", "script": "anomaly_detection.py", "output": OUTPUT_DIR / "anomaly_results.csv", "purpose": "Uses Isolation Forest to identify unusual combinations of response features for further review.", "columns": ["Response_ID", "Anomaly_Status", "Completion_Rate", "Average_Rating", "Response_Time_Seconds"]},
        {"name": "5 · Quality Scoring", "script": "quality_scoring.py", "output": RESULT_FILE, "purpose": "Combines configured rule deductions and anomaly signals into a 0–100 quality score and category.", "columns": ["Response_ID", "Quality_Score", "Quality_Category", "Anomaly_Status", "Review_Recommended"]},
        {"name": "6 · SQLite Database", "script": "database.py", "output": DATABASE, "purpose": "Stores the latest response-level scores and statuses in SQLite for searching and record management.", "columns": []},
    ]

    ready_count = sum(1 for module in modules if module["output"].exists())
    m1, m2, m3 = st.columns(3)
    m1.metric("Project modules", len(modules))
    m2.metric("Outputs available", f"{ready_count}/{len(modules)}")
    m3.metric("Pipeline status", "Ready" if ready_count == len(modules) else "Partially ready")
    st.progress(ready_count / len(modules))

    tabs = st.tabs(["Preprocessing", "Features", "Pattern checks", "Anomaly ML", "Quality score", "Database"])
    for tab, module in zip(tabs, modules):
        with tab:
            st.subheader(module["name"])
            st.write(module["purpose"])
            output_path = module["output"]
            if output_path.exists():
                st.success(f"Output available: {output_path.as_posix()}")
                if output_path.suffix.lower() == ".csv":
                    try:
                        module_df = pd.read_csv(output_path)
                        k1, k2 = st.columns(2)
                        k1.metric("Rows", f"{len(module_df):,}")
                        k2.metric("Columns", f"{len(module_df.columns):,}")
                        if module["columns"]:
                            visible = [c for c in module["columns"] if c in module_df.columns]
                            if visible:
                                st.markdown("**Key output fields**")
                                st.dataframe(module_df[visible].head(30), use_container_width=True, hide_index=True)
                        with st.expander("View full module output"):
                            st.dataframe(module_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            f"Download {module['name']} output",
                            module_df.to_csv(index=False).encode("utf-8"),
                            file_name=output_path.name,
                            mime="text/csv",
                            key=f"download_{module['name']}"
                        )
                    except Exception as exc:
                        st.error(f"Could not read module output: {exc}")
                else:
                    st.info("SQLite database file exists. Open the Database page to inspect records and manage them.")
            else:
                st.warning(f"Output not created yet: {output_path.as_posix()}")
                st.caption("Run a full analysis from Upload & analyze to generate this module's output.")
            if Path(module["script"]).exists():
                with st.expander("Module script details"):
                    st.code(module["script"], language="python")
                    st.caption("This script exists in the project folder. The dashboard displays the module's output file; use the main analysis workflow to keep all module outputs synchronized.")
            else:
                st.caption(f"Standalone script not found in the current folder: {module['script']}. The dashboard's integrated pipeline can still produce the corresponding output when supported.")

    st.divider()
    st.markdown("**Pipeline flow**")
    st.code("Raw CSV → Preprocessing → Feature Engineering → Response Pattern Analysis → Isolation Forest → Quality Scoring → SQLite", language="text")
    st.caption("Anomaly and rule flags are review signals, not proof that a response is invalid.")

if page == "Upload & analyze":
    st.header("Upload a survey dataset")
    st.write("Upload a survey CSV in different formats. SurveyIQ automatically detects likely answer columns; you can adjust the selection before analysis.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="survey_upload")
    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            with st.expander("Preview uploaded data", expanded=True):
                st.caption(f"{len(preview_df):,} rows · {len(preview_df.columns)} columns")
                st.dataframe(preview_df.head(8), use_container_width=True, hide_index=True)
            detected = infer_question_columns(preview_df)
            selected_questions = st.multiselect(
                "Survey answer columns (auto-detected — adjust if needed)",
                options=list(preview_df.columns), default=detected,
                help="Choose the columns containing respondents' answers. Exclude IDs, timestamps and other metadata. For multi-select exports, each option column can be treated as an answer field."
            )
            st.caption(f"Detected {len(detected)} likely response columns out of {len(preview_df.columns)} total columns.")
            if st.button("✦ Run full analysis", type="primary", use_container_width=True):
                with st.spinner("Preparing features, checking response patterns, and running Isolation Forest…"):
                    result = run_analysis(preview_df, selected_questions)
                    st.session_state["analysis_df"] = result
                    result.to_csv(RESULT_FILE, index=False)
                    save_to_database(result)
                st.success(f"Analysis complete — {len(result):,} responses processed and saved.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    st.info("SurveyIQ now supports varied CSV survey formats. Review the auto-detected answer columns before analysis. Numeric questions can be summarized as ratings; categorical and multi-select exports are analyzed using response patterns. Scores are heuristic indicators, not validated measures of truthfulness.")

elif page == "Database":
    st.header("Database workspace")
    st.caption("Browse stored results, search a response, or remove a record from the SQLite table.")
    if DATABASE.exists():
        try:
            with sqlite3.connect(DATABASE) as conn:
                db_df = pd.read_sql_query("SELECT * FROM survey_responses ORDER BY response_id", conn)
            st.metric("Stored records", f"{len(db_df):,}")
            query = st.text_input("Search stored response ID", placeholder="Type an ID, e.g. RESP0001")
            shown = db_df
            if query.strip():
                shown = db_df[db_df["response_id"].astype(str).str.contains(query.strip(), case=False, na=False)]
            st.dataframe(shown, use_container_width=True, hide_index=True)
            b1, b2 = st.columns([1, 1])
            with b1:
                st.download_button(
                    "Download database view as CSV",
                    shown.to_csv(index=False).encode("utf-8"),
                    file_name="surveyiq_database.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with b2:
                if not db_df.empty:
                    delete_id = st.selectbox("Select a record to delete", db_df["response_id"].astype(str).tolist())
                    if st.button("Delete selected record", use_container_width=True):
                        with sqlite3.connect(DATABASE) as conn:
                            conn.execute("DELETE FROM survey_responses WHERE response_id = ?", (delete_id,))
                        st.success(f"Deleted {delete_id} from the database.")
                        st.rerun()
            st.info("Database delete affects the SQLite record only; it does not delete the source CSV or analysis output.")
        except Exception as exc:
            st.error(f"Could not read the database: {exc}")
    else:
        st.info("No database has been created yet. Upload a dataset and run analysis first.")

elif df is None:
    st.header("Welcome to SurveyIQ")
    st.write("Your workspace is ready. Load the demo dataset to see the dashboard, or upload a CSV to run the analysis.")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Start with your demo dataset")
        st.markdown("Use the sample CSV in your project's `data` folder to create the first analysis.")
        demo_path = Path("data/survey_response_quality_demo.csv")
        if demo_path.exists() and st.button("Analyze demo dataset", type="primary", use_container_width=True):
            with st.spinner("Running the analysis pipeline…"):
                result = run_analysis(pd.read_csv(demo_path))
                st.session_state["analysis_df"] = result
                result.to_csv(RESULT_FILE, index=False)
                save_to_database(result)
            st.success("Demo analysis completed.")
            st.rerun()
        else:
            st.caption("If the demo file is not found, use Upload & analyze in the sidebar.")
    with right:
        st.markdown("#### What SurveyIQ checks")
        st.markdown("- Missing answers and completion rate\n- Straight-line and repeated answer patterns\n- Very short response times and comments\n- Isolation Forest anomaly signals\n- Rule-based quality score and category")
elif page == "Question insights" and df is not None:
    st.header("Question insights")
    st.write("Compare average ratings and missing-answer rates across survey questions to identify items that may need clearer wording or follow-up.")
    qcols = infer_question_columns(df) if df is not None else []
    if not qcols:
        st.info("No response columns detected in the current results. Upload a survey CSV and select its answer columns.")
    else:
        qdata = df[qcols].copy()
        numeric_data = qdata.apply(pd.to_numeric, errors="coerce")
        summary = pd.DataFrame({
            "Question / answer field": qcols,
            "Average numeric value": [numeric_data[c].mean() if numeric_data[c].notna().any() else float("nan") for c in qcols],
            "Missing answers": [int(qdata[c].isna().sum() + qdata[c].astype(str).str.strip().eq("").sum()) for c in qcols],
            "Missing rate (%)": [float((qdata[c].isna() | qdata[c].astype(str).str.strip().eq("")).mean() * 100) for c in qcols],
            "Unique answers": [int(qdata[c].nunique(dropna=True)) for c in qcols],
        })
        summary["Question / answer field"] = summary["Question / answer field"].astype(str).str.replace("_", " ")
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="section-head">Average rating by question</div>', unsafe_allow_html=True)
            numeric_summary = summary.dropna(subset=["Average numeric value"])
            if not numeric_summary.empty:
                st.bar_chart(numeric_summary.set_index("Question / answer field")[["Average numeric value"]], color="#BDA8FF")
            else:
                st.info("These answers are categorical, so an average numeric rating is not applicable.")
        with b:
            st.markdown('<div class="section-head">Missing answers by question</div>', unsafe_allow_html=True)
            st.bar_chart(summary.set_index("Question / answer field")[["Missing answers"]], color="#63D6C6")
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
        st.download_button("Download question insights", summary.to_csv(index=False).encode("utf-8"), file_name="surveyiq_question_insights.csv", mime="text/csv")
        st.caption("Average ratings describe the observed dataset; missingness or low ratings alone do not prove that a question is poorly designed.")

else:
    # Common filters for Overview and Explore responses
    st.header("Overview" if page == "Overview" else "Explore responses")
    if df.empty:
        st.warning("The current results file has no rows.")
    else:
        work = df.copy()
        if "Quality_Category" not in work.columns:
            st.warning("This results file does not contain quality scores. Upload a dataset and run analysis.")
        else:
            category_options = sorted(work["Quality_Category"].dropna().unique().tolist())
            anomaly_options = sorted(work["Anomaly_Status"].dropna().unique().tolist()) if "Anomaly_Status" in work.columns else []
            with st.container():
                f1, f2, f3 = st.columns([1, 1, 1.4])
                with f1:
                    selected_categories = st.multiselect("Quality category", category_options, default=category_options)
                with f2:
                    selected_anomalies = st.multiselect("Model status", anomaly_options, default=anomaly_options)
                with f3:
                    search_text = st.text_input("Search response ID", placeholder="e.g. SRQ0001")
            if selected_categories:
                work = work[work["Quality_Category"].isin(selected_categories)]
            else:
                work = work.iloc[0:0]
            if selected_anomalies and "Anomaly_Status" in work.columns:
                work = work[work["Anomaly_Status"].isin(selected_anomalies)]
            elif "Anomaly_Status" in work.columns:
                work = work.iloc[0:0]
            if search_text.strip() and "Response_ID" in work.columns:
                work = work[work["Response_ID"].astype(str).str.contains(search_text.strip(), case=False, na=False)]

            total = len(work)
            avg_score = work["Quality_Score"].mean() if total and "Quality_Score" in work.columns else 0
            anomaly_count = (work["Anomaly_Status"] == "Anomaly").sum() if total and "Anomaly_Status" in work.columns else 0
            review_count = work["Review_Recommended"].sum() if total and "Review_Recommended" in work.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("RESPONSES IN VIEW", f"{total:,}", "Matches current filters")
            with c2:
                metric_card("AVERAGE QUALITY", f"{avg_score:.1f}/100", "Rule-based score")
            with c3:
                metric_card("ANOMALY SIGNALS", f"{anomaly_count:,}", "Isolation Forest")
            with c4:
                metric_card("REVIEW SUGGESTED", f"{review_count:,}", "One or more checks flagged")

            st.markdown("<br>", unsafe_allow_html=True)
            if page == "Overview":
                chart1, chart2 = st.columns([1.15, 1])
                with chart1:
                    st.markdown('<div class="section-head">Quality score distribution</div>', unsafe_allow_html=True)
                    if total:
                        score_df = work[["Quality_Score"]].copy()
                        st.bar_chart(score_df["Quality_Score"].value_counts().sort_index(), color="#BDA8FF")
                    else:
                        st.info("No responses match these filters.")
                with chart2:
                    st.markdown('<div class="section-head">Quality categories</div>', unsafe_allow_html=True)
                    if total:
                        category_chart = work["Quality_Category"].value_counts().rename_axis("Category").reset_index(name="Responses")
                        st.bar_chart(category_chart.set_index("Category"), color="#63D6C6")
                    else:
                        st.info("No category data to display.")
                st.markdown('<div class="section-head">Latest response snapshot</div>', unsafe_allow_html=True)
                display_cols = [c for c in ["Response_ID", "Quality_Score", "Quality_Category", "Anomaly_Status", "Review_Recommended"] if c in work.columns]
                st.dataframe(work[display_cols].head(12), use_container_width=True, hide_index=True)
                question_cols = [c for c in infer_question_columns(work) if c in work.columns and c not in {"Response_ID", "Quality_Score", "Quality_Category", "Anomaly_Status", "Completion_Rate", "Unique_Answers", "Response_Variation", "Straight_Lining", "Average_Rating", "Comment_Length", "Missing_Answers", "Very_Short_Response_Time", "Incomplete_Response", "Straight_Lining_Flag", "Fast_Response_Flag", "Repeated_Pattern_Flag", "Short_Comment_Flag", "Review_Recommended"}]
                if question_cols:
                    st.markdown('<div class="section-head">Question-level rating overview</div>', unsafe_allow_html=True)
                    qnumeric = work[question_cols].apply(pd.to_numeric, errors="coerce")
                    qmeans = qnumeric.mean().dropna().rename_axis("Question").reset_index(name="Average numeric value")
                    if not qmeans.empty:
                        qmeans["Question"] = qmeans["Question"].astype(str).str.replace("_", " ")
                        st.bar_chart(
                            category_chart.set_index("Category"),
                            height=350,
                            use_container_width=True,
                            color="#63D6C6"
                        )
                    else:
                        st.caption("This survey uses categorical answers; numeric rating averages are not shown.")
            else:
                st.markdown('<div class="section-head">Response explorer</div>', unsafe_allow_html=True)
                display_cols = [c for c in [
                    "Response_ID", "Completion_Rate", "Average_Rating", "Quality_Score",
                    "Quality_Category", "Anomaly_Status", "Incomplete_Response",
                    "Straight_Lining_Flag", "Fast_Response_Flag", "Repeated_Pattern_Flag",
                    "Short_Comment_Flag", "Review_Recommended"
                ] if c in work.columns]
                st.dataframe(work[display_cols], use_container_width=True, hide_index=True)
                if len(work):
                    response_ids = work["Response_ID"].astype(str).tolist()
                    chosen_id = st.selectbox("Inspect a response", response_ids)
                    selected = work[work["Response_ID"].astype(str) == chosen_id].iloc[0]
                    st.markdown('<div class="section-head">Response details</div>', unsafe_allow_html=True)
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Quality score", f"{selected.get('Quality_Score', 0):.0f}/100")
                    d2.metric("Category", str(selected.get("Quality_Category", "—")))
                    d3.metric("Model signal", str(selected.get("Anomaly_Status", "—")))
                    flags = [name for name in [
                        "Incomplete_Response", "Straight_Lining_Flag", "Fast_Response_Flag",
                        "Repeated_Pattern_Flag", "Short_Comment_Flag"
                    ] if bool(selected.get(name, False))]
                    if flags:
                        st.warning("Checks to review: " + ", ".join(flags))
                    else:
                        st.success("No configured rule checks flagged this response.")
                    comment = selected.get("Free_Text_Comment", "")
                    if pd.notna(comment) and str(comment).strip():
                        st.caption("Respondent comment")
                        st.write(str(comment))
                    st.markdown('<div class="section-head">Quality score breakdown</div>', unsafe_allow_html=True)
                    deductions = [
                        ("Incomplete answers", "Incomplete_Response", 20),
                        ("Straight-line pattern", "Straight_Lining_Flag", 15),
                        ("Very short response time", "Fast_Response_Flag", 15),
                        ("Repeated answer pattern", "Repeated_Pattern_Flag", 15),
                        ("Very short comment", "Short_Comment_Flag", 10),
                        ("Isolation Forest anomaly", "Anomaly_Status", 15),
                    ]
                    breakdown_rows = []
                    for label, column, points in deductions:
                        flagged = (str(selected.get(column, "")) == "Anomaly") if column == "Anomaly_Status" else bool(selected.get(column, False))
                        breakdown_rows.append({"Check": label, "Result": "Flagged" if flagged else "Clear", "Points deducted": points if flagged else 0})
                    st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)
            st.divider()
            st.download_button(
                "↓ Download filtered results",
                data=work.to_csv(index=False).encode("utf-8"),
                file_name="surveyiq_filtered_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Interpretation note: anomaly and rule flags are review signals, not proof that a response is invalid.")

# Footer
st.markdown("<br><hr><div style='color:#858597;font-size:.8rem'>SurveyIQ · Survey Response Quality Analyzer · Python / Pandas / scikit-learn / SQLite / Streamlit</div>", unsafe_allow_html=True)

#python -m streamlit run app.py