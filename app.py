from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing import preprocess_data
from feature_engineering import extract_features
from response_analysis import analyze_responses
from anomaly_detection import detect_anomalies
from quality_scoring import score_quality

from database import (
    save_results,
    get_results,
    search_data,
    update_data,
    delete_data
)


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "final_quality_results.csv"

st.set_page_config(
    page_title="SurveyIQ | Survey Quality Analyzer",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# 2. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 📊 SurveyIQ")
st.sidebar.caption("Survey Response Quality Analyzer")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Upload & Analyze",
        "Results",
        "Saved Records"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Python · Pandas · Scikit-learn · SQLite · Streamlit"
)


# =====================================================
# 3. SESSION STATE
# =====================================================

if "result_df" not in st.session_state:
    st.session_state["result_df"] = None

if "response_page" not in st.session_state:
    st.session_state["response_page"] = 0

if "active_filter" not in st.session_state:
    st.session_state["active_filter"] = "All Responses"


# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def get_response_id(row, index=None):
    """Return the response ID using available column names."""

    if "Response_ID" in row.index:
        return row["Response_ID"]

    if "response_id" in row.index:
        return row["response_id"]

    if index is not None:
        return f"Response {index + 1}"

    return "Unknown"


def get_boolean_series(series):
    """Convert common true/false representations into Boolean values."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# =====================================================
# 5. INTEGRATED ANALYSIS PIPELINE
# =====================================================

def run_pipeline(raw_df, question_columns):
    """
    Run all project modules in sequence.

    Preprocessing
        -> Feature Engineering
        -> Response Analysis
        -> Anomaly Detection
        -> Quality Scoring
        -> Save CSV and SQLite records
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Preprocessing
    cleaned = preprocess_data(raw_df)

    cleaned.to_csv(
        OUTPUT_DIR / "cleaned_survey_data.csv",
        index=False
    )

    # Step 2: Feature Engineering
    featured = extract_features(
        cleaned,
        question_columns
    )

    featured.to_csv(
        OUTPUT_DIR / "survey_features.csv",
        index=False
    )

    # Step 3: Rule-Based Response Analysis
    analyzed = analyze_responses(
        featured,
        question_columns
    )

    analyzed.to_csv(
        OUTPUT_DIR / "response_analysis.csv",
        index=False
    )

    # Step 4: ML-Based Anomaly Detection
    detected = detect_anomalies(analyzed)

    detected.to_csv(
        OUTPUT_DIR / "anomaly_results.csv",
        index=False
    )

    # Step 5: Quality Scoring
    final_df = score_quality(detected)

    # Step 6: Save final results
    final_df.to_csv(
        RESULT_FILE,
        index=False
    )

    # Step 7: Save records in SQLite
    save_results(final_df)

    return final_df


# =====================================================
# 6. RESULTS DASHBOARD
# =====================================================

def show_results(df):

    if df is None or df.empty:
        st.info(
            "No analysis results yet. "
            "Upload a CSV file and run the analysis first."
        )
        return

    df = df.copy()

    # -------------------------------------------------
    # QUALITY SUMMARY
    # -------------------------------------------------

    st.subheader("Quality Summary")

    total_responses = len(df)

    if "Quality_Score" in df.columns:
        average_score = pd.to_numeric(
            df["Quality_Score"],
            errors="coerce"
        ).mean()

        if pd.isna(average_score):
            average_score = 0
    else:
        average_score = 0

    if "Quality_Category" in df.columns:

        high_count = (
            df["Quality_Category"] == "High Quality"
        ).sum()

        medium_count = (
            df["Quality_Category"] == "Medium Quality"
        ).sum()

        low_count = (
            df["Quality_Category"] == "Low Quality"
        ).sum()

    else:
        high_count = 0
        medium_count = 0
        low_count = 0

    if "Review_Recommended" in df.columns:
        review_flags = get_boolean_series(
            df["Review_Recommended"]
        )

        review_count = int(review_flags.sum())
    else:
        review_flags = pd.Series(
            False,
            index=df.index
        )

        review_count = 0

    if "Anomaly_Status" in df.columns:
        anomaly_count = int(
            df["Anomaly_Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("anomaly")
            .sum()
        )
    else:
        anomaly_count = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Responses",
        f"{total_responses:,}"
    )

    col2.metric(
        "Average Quality Score",
        f"{average_score:.1f}/100"
    )

    col3.metric(
        "Responses Requiring Review",
        f"{review_count:,}"
    )

    col4.metric(
        "ML-Detected Anomalies",
        f"{anomaly_count:,}"
    )

    # -------------------------------------------------
    # QUALITY CATEGORY DISTRIBUTION
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("Quality Distribution")

    if "Quality_Category" in df.columns:

        category_counts = (
            df["Quality_Category"]
            .value_counts()
            .reindex(
                [
                    "High Quality",
                    "Medium Quality",
                    "Low Quality"
                ],
                fill_value=0
            )
        )

        fig, ax = plt.subplots()

        ax.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Survey Response Quality")
        ax.axis("equal")

        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------
    # QUALITY SCORE HISTOGRAM
    # -------------------------------------------------

    st.subheader("Quality Score Distribution")

    if "Quality_Score" in df.columns:

        scores = pd.to_numeric(
            df["Quality_Score"],
            errors="coerce"
        ).dropna()

        if not scores.empty:

            fig, ax = plt.subplots()

            ax.hist(
                scores,
                bins=10,
                edgecolor="black"
            )

            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Number of Responses")
            ax.set_title("Distribution of Quality Scores")

            st.pyplot(fig)
            plt.close(fig)

    # -------------------------------------------------
    # INDIVIDUAL RESPONSE VIEWER
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("Individual Response Viewer")

    filter_options = [
        "All Responses",
        "Responses Requiring Review",
        "High Quality",
        "Medium Quality",
        "Low Quality",
        "Anomalies"
    ]

    selected_filter = st.selectbox(
        "Filter Responses",
        filter_options,
        key="results_filter"
    )

    # Reset navigation when the filter changes
    if st.session_state["active_filter"] != selected_filter:

        st.session_state["active_filter"] = selected_filter
        st.session_state["response_page"] = 0

    filtered_df = df.copy()

    if selected_filter == "Responses Requiring Review":

        if "Review_Recommended" in filtered_df.columns:

            mask = get_boolean_series(
                filtered_df["Review_Recommended"]
            )

            filtered_df = filtered_df.loc[mask]

        else:
            filtered_df = filtered_df.iloc[0:0]

    elif selected_filter in [
        "High Quality",
        "Medium Quality",
        "Low Quality"
    ]:

        if "Quality_Category" in filtered_df.columns:

            filtered_df = filtered_df[
                filtered_df["Quality_Category"]
                == selected_filter
            ]

        else:
            filtered_df = filtered_df.iloc[0:0]

    elif selected_filter == "Anomalies":

        if "Anomaly_Status" in filtered_df.columns:

            mask = (
                filtered_df["Anomaly_Status"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("anomaly")
            )

            filtered_df = filtered_df.loc[mask]

        else:
            filtered_df = filtered_df.iloc[0:0]

    filtered_df = filtered_df.reset_index(drop=True)

    if filtered_df.empty:

        st.info(
            "No responses found for the selected filter."
        )

    else:

        total_filtered = len(filtered_df)

        current_index = st.session_state["response_page"]

        if current_index >= total_filtered or current_index < 0:

            current_index = 0
            st.session_state["response_page"] = 0

        response = filtered_df.iloc[current_index]

        response_id = get_response_id(
            response,
            current_index
        )

        st.caption(
            f"Showing response {current_index + 1} "
            f"of {total_filtered}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Response ID",
            str(response_id)
        )

        if "Quality_Score" in response.index:

            col2.metric(
                "Quality Score",
                str(response["Quality_Score"])
            )
        else:
            col2.metric(
                "Quality Score",
                "Not available"
            )

        if "Quality_Category" in response.index:

            col3.metric(
                "Quality Category",
                str(response["Quality_Category"])
            )
        else:
            col3.metric(
                "Quality Category",
                "Not available"
            )

        with st.expander("View Full Response Details"):

            st.dataframe(
                response.to_frame(name="Value"),
                use_container_width=True
            )

        previous_col, next_col = st.columns(2)

        with previous_col:

            if st.button(
                "⬅ Previous",
                disabled=(current_index == 0),
                use_container_width=True
            ):

                st.session_state["response_page"] = (
                    current_index - 1
                )

                st.rerun()

        with next_col:

            if st.button(
                "Next ➡",
                disabled=(current_index == total_filtered - 1),
                use_container_width=True
            ):

                st.session_state["response_page"] = (
                    current_index + 1
                )

                st.rerun()

    # -------------------------------------------------
    # RESPONSES REQUIRING REVIEW
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("⚠️ Responses Requiring Review")

    st.write(
        "These responses have been flagged by the "
        "implemented rule-based response checks."
    )

    if "Review_Recommended" in df.columns:

        review_df = df.loc[
            get_boolean_series(df["Review_Recommended"])
        ].copy()

        if not review_df.empty:

            st.write(
                f"Total responses requiring review: "
                f"{len(review_df):,}"
            )

            review_columns = [
                "Response_ID",
                "Quality_Score",
                "Quality_Category",
                "Incomplete_Response",
                "Straight_Lining_Flag",
                "Fast_Response_Flag",
                "Repeated_Pattern_Flag",
                "Short_Comment_Flag",
                "Contradictory_Response_Flag"
            ]

            review_columns = [
                column
                for column in review_columns
                if column in review_df.columns
            ]

            if not review_columns:

                review_df = review_df.copy()

            else:

                review_df = review_df[review_columns]

            st.dataframe(
                review_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Responses Requiring Review",
                data=review_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="responses_requiring_review.csv",
                mime="text/csv"
            )

        else:

            st.success(
                "No responses currently require review."
            )

    else:

        st.info(
            "Review flags are not available in the current results."
        )

    # -------------------------------------------------
    # DOWNLOAD ALL RESULTS
    # -------------------------------------------------

    st.markdown("---")

    st.download_button(
        "Download All Analysis Results",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="final_quality_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Quality scores and flags help identify responses for review; "
        "they do not prove that a response is invalid."
    )


# =====================================================
# 7. HOME PAGE
# =====================================================

if page == "Home":

    st.title("SurveyIQ")

    st.subheader(
        "Survey Response Quality Analyzer"
    )

    st.write(
        "Analyze survey responses, identify potentially unreliable "
        "response patterns, and calculate response-quality scores."
    )

    st.markdown("---")
    st.subheader("How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 01 · Upload")
        st.write(
            "Upload your survey dataset in CSV format."
        )

    with col2:
        st.markdown("### 02 · Analyze")
        st.write(
            "Clean the data, extract features, check response patterns, "
            "and detect anomalies."
        )

    with col3:
        st.markdown("### 03 · Review")
        st.write(
            "Explore quality scores, review flags, charts, "
            "and downloadable results."
        )

    st.info(
        "Start with Upload & Analyze in the sidebar."
    )

    if RESULT_FILE.exists():

        try:

            old_results = pd.read_csv(RESULT_FILE)

            st.markdown("---")
            st.subheader("Latest Saved Analysis")

            show_results(old_results)

        except Exception as error:

            st.warning(
                f"Could not load the latest analysis: {error}"
            )


# =====================================================
# 8. UPLOAD & ANALYZE PAGE
# =====================================================

elif page == "Upload & Analyze":

    st.title("Upload & Analyze")

    st.write(
        "Upload a CSV file containing survey responses."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            preview_df = pd.read_csv(uploaded_file)

            st.subheader("Data Preview")

            st.caption(
                f"{len(preview_df):,} rows · "
                f"{len(preview_df.columns)} columns"
            )

            st.dataframe(
                preview_df.head(8),
                use_container_width=True,
                hide_index=True
            )

            metadata_names = {
                "response_id",
                "respondent_id",
                "student_id",
                "record_id",
                "completion_status",
                "status",
                "submitted_at",
                "timestamp",
                "date",
                "email",
                "name",
                "response_time_seconds",
                "duration_seconds",
                "time_taken"
            }

            default_questions = [
                column
                for column in preview_df.columns
                if column.lower() not in metadata_names
                and not column.lower().endswith("_id")
            ]

            selected_questions = st.multiselect(
                "Select survey answer columns",
                options=list(preview_df.columns),
                default=default_questions,
                help=(
                    "Select the columns containing survey answers. "
                    "Exclude IDs, names, timestamps, and unrelated metadata."
                )
            )

            st.caption(
                "Tip: Include text-answer columns if you want them "
                "considered by the implemented response checks."
            )

            if st.button(
                "Run Survey Analysis",
                type="primary",
                use_container_width=True
            ):

                if not selected_questions:

                    st.error(
                        "Please select at least one survey answer column."
                    )

                else:

                    with st.spinner(
                        "Running all analysis modules..."
                    ):

                        final_df = run_pipeline(
                            preview_df,
                            selected_questions
                        )

                        st.session_state["result_df"] = final_df
                        st.session_state["response_page"] = 0

                    st.success(
                        f"Analysis complete! Processed "
                        f"{len(final_df):,} responses."
                    )

                    st.info(
                        "Open Results in the sidebar to view the analysis."
                    )

        except Exception as error:

            st.error(
                f"Could not read or analyze this CSV: {error}"
            )


# =====================================================
# 9. RESULTS PAGE
# =====================================================

elif page == "Results":

    st.title("Analysis Results")

    current_df = st.session_state.get("result_df")

    if current_df is None and RESULT_FILE.exists():

        try:

            current_df = pd.read_csv(RESULT_FILE)

        except Exception:

            current_df = None

    show_results(current_df)


# =====================================================
# 10. SAVED RECORDS PAGE
# =====================================================

elif page == "Saved Records":

    st.title("Saved Records")

    st.write(
        "These records are stored in the local SQLite database."
    )

    try:

        records = get_results()

        if records is None:
            records = pd.DataFrame()

        search_text = st.text_input(
            "Search by Response ID",
            placeholder="Enter part or all of an ID"
        )

        if not records.empty:

            if search_text.strip():

                id_column = (
                    "response_id"
                    if "response_id" in records.columns
                    else "Response_ID"
                    if "Response_ID" in records.columns
                    else None
                )

                if id_column:

                    records = records[
                        records[id_column]
                        .astype(str)
                        .str.contains(
                            search_text.strip(),
                            case=False,
                            na=False
                        )
                    ]

            st.metric(
                "Matching Saved Records",
                len(records)
            )

            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Saved Records",
                data=records.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="surveyiq_saved_records.csv",
                mime="text/csv"
            )

        else:

            st.info(
                "No saved records found in the database."
            )

        # ---------------------------------------------
        # SEARCH A SINGLE RECORD
        # ---------------------------------------------

        st.markdown("---")
        st.subheader("Find a Specific Record")

        response_id_input = st.text_input(
            "Enter exact Response ID",
            key="saved_record_search"
        )

        if st.button("Search Record"):

            if not response_id_input.strip():

                st.warning(
                    "Please enter a Response ID."
                )

            else:

                try:

                    found_record = search_data(
                        response_id_input.strip()
                    )

                    if found_record is not None:

                        st.success("Record found.")

                        st.write(found_record)

                    else:

                        st.info(
                            "No record found with that Response ID."
                        )

                except Exception as error:

                    st.error(
                        f"Could not search the record: {error}"
                    )

        # ---------------------------------------------
        # UPDATE QUALITY CATEGORY
        # ---------------------------------------------

        st.markdown("---")
        st.subheader("Update Quality Category")

        update_id = st.text_input(
            "Response ID to update",
            key="update_response_id"
        )

        new_category = st.selectbox(
            "New Quality Category",
            [
                "High Quality",
                "Medium Quality",
                "Low Quality"
            ]
        )

        if st.button("Update Category"):

            if not update_id.strip():

                st.warning(
                    "Please enter a Response ID."
                )

            else:

                try:

                    updated = update_data(
                        update_id.strip(),
                        new_category
                    )

                    if updated:

                        st.success(
                            "Quality category updated successfully."
                        )

                    else:

                        st.warning(
                            "Record not found or no change was made."
                        )

                except Exception as error:

                    st.error(
                        f"Could not update the record: {error}"
                    )

        # ---------------------------------------------
        # DELETE RECORD
        # ---------------------------------------------

        st.markdown("---")
        st.subheader("Delete a Saved Record")

        delete_id = st.text_input(
            "Response ID to delete",
            key="delete_response_id"
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this record."
        )

        if st.button("Delete Record"):

            if not delete_id.strip():

                st.warning(
                    "Please enter a Response ID."
                )

            elif not confirm_delete:

                st.warning(
                    "Please confirm before deleting the record."
                )

            else:

                try:

                    deleted = delete_data(
                        delete_id.strip()
                    )

                    if deleted:

                        st.success(
                            "Record deleted successfully."
                        )

                        st.rerun()

                    else:

                        st.warning(
                            "Record not found."
                        )

                except Exception as error:

                    st.error(
                        f"Could not delete the record: {error}"
                    )

    except Exception as error:

        st.error(
            f"Could not load the database: {error}"
        )

