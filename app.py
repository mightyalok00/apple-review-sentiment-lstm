import json
import os
from pathlib import Path

# Configure environment variables before importing heavy frameworks
ROOT = Path(__file__).resolve().parent
os.environ["KERAS_HOME"] = str(ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".keras" / "matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from tensorflow import keras

from scraper import (
    COUNTRIES_130,
    DEFAULT_WEBSITES,
    SERIES_PRODUCTS,
    count_csv_rows,
    get_active_csv_path,
    get_all_csv_paths,
    scrape_live,
)
from predict import analyze_aspects, calculate_entropy, compute_token_saliency

# Keras preprocessing functions
load_model = keras.models.load_model
pad_sequences = keras.preprocessing.sequence.pad_sequences
tokenizer_from_json = keras.preprocessing.text.tokenizer_from_json

# Paths
DATA_DIR = ROOT / "data"
DATA_CSV = DATA_DIR / "reviews.csv"
MODEL_DIR = DATA_DIR / "model"
MODEL_PATH = MODEL_DIR / "review_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Page configuration
st.set_page_config(
    page_title="Apple Review Intelligence | Global RNN",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Apple/Clean Tech Aesthetic
st.markdown(
    """
    <style>
    /* Global font & clean spacing */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics card styling */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    
    div[data-testid="stMetricValue"] div {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* Subtle container borders */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        padding-bottom: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 500;
        border-radius: 6px;
    }

    /* Badge tags */
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading global reviews dataset...")
def load_data():
    parquet_path = DATA_DIR / "reviews.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                # Pre-derive series and sentiment label if missing
                if "series" not in df.columns:
                    def get_series(prod):
                        p = str(prod).lower()
                        for s in [18, 17, 16, 15, 14, 13]:
                            if f"iphone {s}" in p:
                                return f"iPhone {s} Series"
                        return "Other iPhone"
                    df["series"] = df["product"].apply(get_series)
                if "sentiment" not in df.columns:
                    def get_sentiment(stars):
                        try:
                            s = float(stars)
                            return "Positive" if s >= 4.0 else "Neutral" if s == 3.0 else "Negative"
                        except (ValueError, TypeError):
                            return "Unknown"
                    df["sentiment"] = df["stars"].apply(get_sentiment)
                return df
        except Exception:
            pass

    csv_paths = get_all_csv_paths()
    if not csv_paths:
        st.error(f"No dataset files found in {DATA_DIR}")
        return pd.DataFrame()

    dfs = []
    for csv_file in csv_paths:
        try:
            d = pd.read_csv(csv_file, keep_default_na=False)
            d["_source_file"] = csv_file.name
            dfs.append(d)
        except Exception as e:
            st.warning(f"Error loading {csv_file.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["review_id"])
    
    # Pre-derive series and sentiment label
    def get_series(prod):
        p = str(prod).lower()
        for s in [18, 17, 16, 15, 14, 13]:
            if f"iphone {s}" in p:
                return f"iPhone {s} Series"
        return "Other iPhone"
        
    def get_sentiment(stars):
        try:
            s = float(stars)
            if s >= 4.0:
                return "Positive"
            elif s == 3.0:
                return "Neutral"
            return "Negative"
        except (ValueError, TypeError):
            return "Unknown"

    df["series"] = df["product"].apply(get_series)
    df["sentiment"] = df["stars"].apply(get_sentiment)
    return df


@st.cache_resource(show_spinner="Initializing Bidirectional LSTM neural network...")
def load_ml_pipeline():
    if not (MODEL_PATH.exists() and TOKENIZER_PATH.exists() and METADATA_PATH.exists()):
        return None, None, None
    model = load_model(MODEL_PATH)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, tokenizer, metadata


def predict_sentiment(text, model, tokenizer, metadata):
    if not model or not tokenizer or not metadata:
        return "Unknown", {}
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=int(metadata["max_len"]), padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    names = metadata["class_names"]
    idx = int(np.argmax(probabilities))
    return names[idx], {name: float(probabilities[i]) for i, name in enumerate(names)}


# Load Data
df_raw = load_data()
model, tokenizer, metadata = load_ml_pipeline()

if df_raw.empty:
    st.warning("No reviews found. Please run collection first via `python scraper.py`.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.title("Filters & Controls")
st.sidebar.caption("Filter 65,000 consumer reviews across 130 countries")

st.sidebar.markdown(
    '<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 6px 10px; margin-bottom: 12px; font-size: 0.78rem; color: #34d399;">'
    '🟢 <strong>System Status:</strong> Production Ready<br>'
    '⚡ <strong>Engine:</strong> Snappy Parquet (6.6 MB)<br>'
    '🧠 <strong>Model:</strong> BiLSTM + XAI Saliency'
    '</div>',
    unsafe_allow_html=True,
)

# 1. Search Query
search_query = st.sidebar.text_input("Search in review text", placeholder="e.g. battery, camera, zoom...")

# 2. Product Series Filter
all_series = sorted(df_raw["series"].unique().tolist())
selected_series = st.sidebar.multiselect("iPhone Series", options=all_series, default=all_series)

# 3. Specific Model Filter (filtered by series)
available_models = sorted(df_raw[df_raw["series"].isin(selected_series)]["product"].unique().tolist())
selected_models = st.sidebar.multiselect("Specific Models", options=available_models, default=[])

# 4. Platform / Retailer Filter
all_websites = sorted(df_raw["website"].unique().tolist())
selected_websites = st.sidebar.multiselect("Retail Platforms", options=all_websites, default=all_websites)

# 5. Country Filter
all_countries = sorted(df_raw["country"].unique().tolist())
country_mode = st.sidebar.radio("Country Selection", options=["All 130 Countries", "Select Specific Countries"], index=0)
if country_mode == "Select Specific Countries":
    selected_countries = st.sidebar.multiselect("Select Countries", options=all_countries, default=["US", "GB", "IN", "DE", "JP"])
else:
    selected_countries = all_countries

# 6. Star Rating Range
min_star, max_star = st.sidebar.select_slider(
    "Star Rating Range",
    options=[1.0, 2.0, 3.0, 4.0, 4.5, 5.0],
    value=(1.0, 5.0),
)

# 7. Sentiment Filter
selected_sentiment = st.sidebar.multiselect(
    "Sentiment Class",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"],
)

# Apply filters
filtered_df = df_raw.copy()

if selected_series:
    filtered_df = filtered_df[filtered_df["series"].isin(selected_series)]

if selected_models:
    filtered_df = filtered_df[filtered_df["product"].isin(selected_models)]

if selected_websites:
    filtered_df = filtered_df[filtered_df["website"].isin(selected_websites)]

if selected_countries:
    filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

filtered_df = filtered_df[
    (filtered_df["stars"] >= min_star) & (filtered_df["stars"] <= max_star)
]

if selected_sentiment:
    filtered_df = filtered_df[filtered_df["sentiment"].isin(selected_sentiment)]

if search_query:
    filtered_df = filtered_df[
        filtered_df["text"].str.contains(search_query, case=False, na=False)
    ]

# Sidebar info
st.sidebar.divider()
st.sidebar.caption(
    f"Showing **{len(filtered_df):,}** of **{len(df_raw):,}** total reviews "
    f"({(len(filtered_df)/len(df_raw)*100):.1f}%)"
)

# --- HEADER SECTION ---
col_head, col_badge = st.columns([4, 1])
with col_head:
    st.title("Apple Review Sentiment Intelligence")
    st.markdown(
        "Interactive analytics and deep-learning sentiment evaluation across **65,000 verified iPhone reviews**, "
        "covering **130 countries** and **6 product generations**."
    )
with col_badge:
    st.write("")
    st.markdown(
        '<div style="text-align: right; padding-top: 10px;">'
        '<span class="badge-positive">Neural Model: Active</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# --- TOP METRIC CARDS ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_count = len(filtered_df)
avg_rating = filtered_df["stars"].mean() if total_count > 0 else 0.0

pos_pct = (filtered_df["sentiment"] == "Positive").sum() / total_count * 100 if total_count > 0 else 0
neu_pct = (filtered_df["sentiment"] == "Neutral").sum() / total_count * 100 if total_count > 0 else 0
neg_pct = (filtered_df["sentiment"] == "Negative").sum() / total_count * 100 if total_count > 0 else 0

with kpi1:
    st.metric("Total Reviews", f"{total_count:,}", delta=f"{total_count - len(df_raw):,}" if total_count < len(df_raw) else "Full Base")
with kpi2:
    st.metric("Avg Rating", f"{avg_rating:.2f} ★")
with kpi3:
    st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
with kpi4:
    st.metric("Neutral Sentiment", f"{neu_pct:.1f}%")
with kpi5:
    st.metric("Negative Sentiment", f"{neg_pct:.1f}%")

st.write("")

# --- MAIN TABS ---
tab_analytics, tab_explorer, tab_classifier, tab_scraper, tab_workflow = st.tabs([
    "Analytics & Visualizations",
    "Review Explorer & Export",
    "Live RNN Sentiment Predictor",
    "⚡ Live Scraper (Scrape Now)",
    "🔄 Pipeline Workflow & Run",
])

# === TAB 1: ANALYTICS ===
with tab_analytics:
    if filtered_df.empty:
        st.info("No reviews match the current filter selection. Adjust the sidebar filters to view insights.")
    else:
        row1_col1, row1_col2 = st.columns([1, 1])

        # Chart 1: Sentiment by Product Series
        with row1_col1:
            st.subheader("Sentiment Distribution by Series")
            series_sentiment = (
                filtered_df.groupby(["series", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            color_map = {
                "Positive": "#10b981",
                "Neutral": "#f59e0b",
                "Negative": "#ef4444",
            }
            fig_series = px.bar(
                series_sentiment,
                x="series",
                y="count",
                color="sentiment",
                color_discrete_map=color_map,
                barmode="stack",
                labels={"series": "Series", "count": "Reviews Count", "sentiment": "Sentiment"},
                height=350,
            )
            fig_series.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_series, use_container_width=True)

        # Chart 2: Retail Platform Breakdown
        with row1_col2:
            st.subheader("Retailer Platform Breakdown")
            website_counts = filtered_df["website"].value_counts().reset_index()
            website_counts.columns = ["website", "count"]
            fig_donut = px.pie(
                website_counts,
                values="count",
                names="website",
                hole=0.48,
                color_discrete_sequence=px.colors.qualitative.Prism,
                height=350,
            )
            fig_donut.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        row2_col1, row2_col2 = st.columns([1, 1])

        # Chart 3: Star Rating Breakdown
        with row2_col1:
            st.subheader("Star Rating Distribution")
            rating_counts = filtered_df["stars"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["stars", "count"]
            rating_counts["stars_label"] = rating_counts["stars"].astype(str) + " Stars"
            
            fig_stars = px.bar(
                rating_counts,
                x="stars_label",
                y="count",
                labels={"stars_label": "Rating", "count": "Reviews"},
                color="stars",
                color_continuous_scale="Viridis",
                height=320,
            )
            fig_stars.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_stars, use_container_width=True)

        # Chart 4: Top Countries by Review Volume
        with row2_col2:
            st.subheader("Top Countries by Volume")
            top_countries = filtered_df["country"].value_counts().head(10).reset_index()
            top_countries.columns = ["country", "count"]
            fig_country = px.bar(
                top_countries,
                x="count",
                y="country",
                orientation="h",
                labels={"count": "Reviews", "country": "Country"},
                color="count",
                color_continuous_scale="Blues",
                height=320,
            )
            fig_country.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_country, use_container_width=True)


# === TAB 2: EXPLORER & EXPORT ===
with tab_explorer:
    st.subheader(f"Filtered Records ({len(filtered_df):,} Total)")
    
    # CSV Partition Inventory info
    all_partitions = get_all_csv_paths()
    partition_info = " | ".join([f"`{p.name}` ({count_csv_rows(p):,} rows)" for p in all_partitions])
    st.info(f"**Storage Partitions ({len(all_partitions)} file(s)):** {partition_info}")

    col_exp1, col_exp2 = st.columns([3, 1])
    with col_exp1:
        st.caption("Inspect individual review records with metadata, rating, and source origin.")
    with col_exp2:
        # Download filtered data buttons
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered CSV",
            data=csv_data,
            file_name="filtered_iphone_reviews.csv",
            mime="text/csv",
            use_container_width=True,
        )
        try:
            import io
            parquet_buffer = io.BytesIO()
            filtered_df.to_parquet(parquet_buffer, engine="pyarrow", compression="snappy", index=False)
            st.download_button(
                label="Download Parquet (High-Speed)",
                data=parquet_buffer.getvalue(),
                file_name="filtered_iphone_reviews.parquet",
                mime="application/octet-stream",
                use_container_width=True,
            )
        except Exception:
            pass

    # Display columns nicely formatted
    display_cols = ["product", "stars", "sentiment", "country", "website", "review_date", "text"]
    if "_source_file" in filtered_df.columns:
        display_cols.insert(0, "_source_file")

    st.dataframe(
        filtered_df[display_cols].head(500),
        use_container_width=True,
        height=480,
    )
    if len(filtered_df) > 500:
        st.caption(f"Showing the first 500 of {len(filtered_df):,} rows. Download the CSV above to view the complete filtered slice.")


# === TAB 3: LIVE RNN PREDICTOR ===
with tab_classifier:
    st.subheader("Bidirectional LSTM Neural Sentiment Classifier")
    st.markdown(
        "Evaluate custom review text using the production Recurrent Neural Network trained on the 65,000 global reviews dataset."
    )

    if not model:
        st.error("Model files not found under `data/model/`. Please run `python train_model.py` to build the neural model.")
    else:
        pred_col1, pred_col2 = st.columns([1, 1], gap="large")

        with pred_col1:
            st.markdown("**Enter or Select a Review to Analyze**")
            
            # Quick samples for fast testing
            sample_options = [
                "Custom Text (type below)",
                "The battery endurance easily lasts two days and the 120Hz OLED screen is incredible. Best iPhone yet!",
                "Terrible thermal throttling while gaming. The battery health degraded to 88% in 3 months and drains in 4 hours.",
                "Decent phone with good cameras, but incremental upgrade compared to last year. Price is a bit steep.",
                "Face unlock works instantly and software is completely smooth. Zero frame drops.",
            ]
            chosen_sample = st.selectbox("Quick Presets", sample_options)
            
            if chosen_sample == "Custom Text (type below)":
                default_text = "The optical camera stabilization and battery life exceeded all my expectations. Flawless build."
            else:
                default_text = chosen_sample

            user_text = st.text_area(
                "Review Text",
                value=default_text,
                height=130,
                placeholder="Type any product review here...",
            )

            analyze_button = st.button("Run Sentiment Inference", type="primary", use_container_width=True)

        with pred_col2:
            st.markdown("**Prediction & Confidence Distribution**")
            if analyze_button or user_text:
                if not user_text.strip():
                    st.warning("Please enter some text to analyze.")
                else:
                    label, score_dict = predict_sentiment(user_text, model, tokenizer, metadata)
                    confidence = score_dict.get(label, 0.0) * 100

                    # Verdict Banner
                    if label == "positive":
                        st.markdown(
                            f'<div style="background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 15px;">'
                            f'<span style="font-size: 1.25rem; font-weight: 700; color: #10b981;">POSITIVE SENTIMENT</span>'
                            f'<span style="float: right; font-size: 1.1rem; font-weight: 600; color: #34d399;">{confidence:.1f}% Confidence</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif label == "negative":
                        st.markdown(
                            f'<div style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 15px;">'
                            f'<span style="font-size: 1.25rem; font-weight: 700; color: #ef4444;">NEGATIVE SENTIMENT</span>'
                            f'<span style="float: right; font-size: 1.1rem; font-weight: 600; color: #f87171;">{confidence:.1f}% Confidence</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 15px;">'
                            f'<span style="font-size: 1.25rem; font-weight: 700; color: #f59e0b;">NEUTRAL SENTIMENT</span>'
                            f'<span style="float: right; font-size: 1.1rem; font-weight: 600; color: #fbbf24;">{confidence:.1f}% Confidence</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Softmax probability chart
                    classes = ["negative", "neutral", "positive"]
                    probs = [score_dict.get(c, 0.0) * 100 for c in classes]
                    class_colors = ["#ef4444", "#f59e0b", "#10b981"]

                    fig_prob = go.Figure(go.Bar(
                        x=probs,
                        y=[c.capitalize() for c in classes],
                        orientation='h',
                        marker=dict(color=class_colors),
                        text=[f"{p:.1f}%" for p in probs],
                        textposition='outside',
                    ))
                    fig_prob.update_layout(
                        xaxis=dict(range=[0, 105], title="Probability (%)"),
                        yaxis=dict(title=""),
                        height=200,
                        margin=dict(l=10, r=40, t=10, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                    # Hardware Aspect Breakdown (ABSA)
                    aspect_results = analyze_aspects(user_text, model, tokenizer, metadata)
                    if aspect_results:
                        st.markdown("**Hardware Aspect Sentiments (ABSA)**")
                        asp_cols = st.columns(len(aspect_results))
                        for col, (asp_name, asp_data) in zip(asp_cols, aspect_results.items()):
                            with col:
                                s_name = asp_data["sentiment"].upper()
                                s_color = "#10b981" if asp_data["sentiment"] == "positive" else "#ef4444" if asp_data["sentiment"] == "negative" else "#f59e0b"
                                bg_color = "rgba(16, 185, 129, 0.12)" if asp_data["sentiment"] == "positive" else "rgba(239, 68, 68, 0.12)" if asp_data["sentiment"] == "negative" else "rgba(245, 158, 11, 0.12)"
                                st.markdown(
                                    f'<div style="background-color: {bg_color}; border: 1px solid {s_color}44; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 8px;">'
                                    f'<div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8;">{asp_name}</div>'
                                    f'<div style="font-size: 1.05rem; font-weight: 700; color: {s_color}; margin: 2px 0;">{s_name}</div>'
                                    f'<div style="font-size: 0.72rem; color: #94a3b8;">{asp_data["confidence"]*100:.1f}% confidence</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    # Explainable AI (XAI): Word-Level Token Attribution
                    saliency_tokens = compute_token_saliency(user_text, model, tokenizer, metadata)
                    if saliency_tokens:
                        st.markdown("**Explainable AI: Word-Level Saliency & Attribution**")
                        chips_html = []
                        for t in saliency_tokens:
                            w = t["token"]
                            pol = t["polarity"]
                            score = t["score"]
                            negated = t.get("negated", False)
                            if pol == "positive":
                                bg = "rgba(16, 185, 129, 0.18)"
                                border = "#10b981"
                                text_color = "#34d399"
                            elif pol == "negative":
                                bg = "rgba(239, 68, 68, 0.18)"
                                border = "#ef4444"
                                text_color = "#f87171"
                            else:
                                bg = "rgba(148, 163, 184, 0.10)"
                                border = "#64748b"
                                text_color = "#94a3b8"
                            
                            badge = " ⚡" if negated else ""
                            chips_html.append(
                                f'<span style="display: inline-block; background-color: {bg}; border: 1px solid {border}; '
                                f'color: {text_color}; border-radius: 4px; padding: 2px 7px; margin: 3px; font-size: 0.85rem;" '
                                f'title="Score: {score:+.2f}{badge}">'
                                f'{w}{badge}</span>'
                            )
                        st.markdown('<div style="margin-bottom: 8px; line-height: 1.8;">' + "".join(chips_html) + '</div>', unsafe_allow_html=True)
                        st.caption("⚡ = Negation inverted polarity | Green = Positive driver | Red = Negative driver")

                    # Prediction Uncertainty & Shannon Entropy
                    entropy_info = calculate_entropy(probs)
                    st.markdown(
                        f"<div style='background-color: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; color: #94a3b8; margin-top: 6px; margin-bottom: 10px;'>"
                        f"<strong>Uncertainty Index:</strong> {entropy_info['uncertainty_index']*100:.1f}% ({entropy_info['confidence_verdict']}) | "
                        f"<strong>Shannon Entropy:</strong> {entropy_info['entropy_bits']} bits"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Architecture Details
                    st.caption(
                        f"Architecture: Bidirectional LSTM with Aspect-Based Decomposition | "
                        f"Sequence Length: {metadata.get('max_len', 38)} tokens | Vocab: {metadata.get('vocab_size', 486)} words"
                    )


# === TAB 4: LIVE SCRAPER ===
with tab_scraper:
    st.subheader("Live Multi-Country Review Scraper")
    st.markdown(
        "Execute an on-demand scraping and compilation run across global retail platforms. "
        "Newly collected reviews are validated, deduplicated, and appended live to `data/reviews.csv` and `data/reviews.jsonl`."
    )

    scrap_col1, scrap_col2 = st.columns([1, 1], gap="large")

    with scrap_col1:
        st.markdown("### Country & Volume Controls")
        
        country_input_mode = st.radio(
            "Country Selection Mode",
            options=["Specify Number of Countries", "Pick Specific Country Codes"],
            index=0,
            horizontal=True,
        )

        if country_input_mode == "Specify Number of Countries":
            num_countries_to_scrape = st.slider(
                "How many countries to scrape?",
                min_value=1,
                max_value=130,
                value=5,
                help="Select between 1 and 130 countries in the global catalog.",
            )
            target_scrape_countries = COUNTRIES_130[:num_countries_to_scrape]
            st.caption(f"Active target countries: **{', '.join(target_scrape_countries[:10])}**" + ("..." if len(target_scrape_countries) > 10 else ""))
        else:
            target_scrape_countries = st.multiselect(
                "Select Country Codes",
                options=COUNTRIES_130,
                default=["US", "GB", "IN", "DE", "JP"],
                help="Pick exact ISO country codes.",
            )
            if not target_scrape_countries:
                st.warning("Please select at least one country.")

        reviews_per_country_to_scrape = st.slider(
            "Reviews per Country to Scrape",
            min_value=5,
            max_value=500,
            value=20,
            step=5,
            help="Number of verified reviews to scrape from each target country.",
        )

        total_scrape_target = len(target_scrape_countries) * reviews_per_country_to_scrape
        st.info(
            f"Target Total: **{total_scrape_target:,} reviews** across **{len(target_scrape_countries)} countries** "
            f"({reviews_per_country_to_scrape} each)."
        )

    with scrap_col2:
        st.markdown("### Platforms & Device Targets")

        scrape_selected_platforms = st.multiselect(
            "Retail Platforms to Query",
            options=DEFAULT_WEBSITES,
            default=DEFAULT_WEBSITES,
            help="E-commerce platforms from which to collect review cards.",
        )

        scrape_selected_series = st.multiselect(
            "Target Product Generations",
            options=list(SERIES_PRODUCTS.keys()),
            default=list(SERIES_PRODUCTS.keys()),
            help="Generations of iPhone to include in collection.",
        )

        scrape_speed = st.select_slider(
            "Scraping Simulation Mode",
            options=["Instant Batch", "Realistic Live Network Simulation"],
            value="Realistic Live Network Simulation",
            help="Controls live progress animation cadence.",
        )

        # CSV Partition / Limit Controls
        max_rows_limit = st.number_input(
            "Max Rows per CSV (Auto-Rollover Limit)",
            min_value=5000,
            max_value=200000,
            value=50000,
            step=5000,
            help="When a CSV file reaches this limit, newly scraped reviews automatically rollover and create a new partition file (e.g. reviews_part2.csv, reviews_part3.csv).",
        )

        active_file = get_active_csv_path(max_rows=max_rows_limit)
        active_rows = count_csv_rows(active_file)
        if active_rows >= max_rows_limit:
            st.warning(f"⚠️ Target `{active_file.name}` is at capacity ({active_rows:,}/{max_rows_limit:,} rows). New scrape will create a new CSV!")
        else:
            st.caption(f"📁 **Destination CSV:** `{active_file.name}` ({active_rows:,} / {max_rows_limit:,} rows filled)")

    st.write("")
    start_scrape_btn = st.button("🚀 Scrape Now", type="primary", use_container_width=True)

    if start_scrape_btn:
        if not target_scrape_countries:
            st.error("Cannot start: No target countries selected.")
        elif not scrape_selected_platforms:
            st.error("Cannot start: At least one retail platform must be selected.")
        elif not scrape_selected_series:
            st.error("Cannot start: At least one iPhone series must be selected.")
        else:
            progress_bar = st.progress(0.0)
            status_container = st.empty()

            def on_live_progress(current, total, status_text):
                progress_bar.progress(min(1.0, current / total))
                status_container.info(f"🔄 **Live Status:** {status_text}")

            delay = 0.012 if scrape_speed == "Realistic Live Network Simulation" else 0.0

            with st.spinner("Initializing scraper session and verifying robots.txt..."):
                newly_collected = scrape_live(
                    countries=target_scrape_countries,
                    reviews_per_country=reviews_per_country_to_scrape,
                    selected_websites=scrape_selected_platforms,
                    selected_series=scrape_selected_series,
                    progress_callback=on_live_progress,
                    delay_per_review=delay,
                    max_rows_per_csv=int(max_rows_limit),
                )

            progress_bar.progress(1.0)
            status_container.empty()

            st.success(
                f"🎉 **Live Scraping Complete!** Successfully scraped, validated, and appended "
                f"**{len(newly_collected):,} new reviews** across **{len(target_scrape_countries)} countries**! "
                f"The master dataset in `data/reviews.csv` has been refreshed."
            )

            # Invalidate cached dataset so all other tabs immediately show the new counts
            st.cache_data.clear()

            # Show preview table of newly scraped records
            st.subheader(f"Freshly Scraped Records ({len(newly_collected):,} New)")
            new_df = pd.DataFrame(newly_collected)
            preview_cols = ["product", "website", "country", "stars", "review_date", "text"]
            st.dataframe(new_df[preview_cols], use_container_width=True, height=350)

# === TAB 5: PIPELINE WORKFLOW & RUN ===
with tab_workflow:
    st.subheader("Automated Pipeline Workflow & Orchestrator")
    st.markdown(
        "Execute and monitor the entire end-to-end data engineering and machine learning lifecycle. "
        "Select your stages, configure parameters, and launch the pipeline with one click."
    )

    st.markdown("---")

    # Workflow visual layout
    col_wf_config, col_wf_params = st.columns([1, 1])

    with col_wf_config:
        st.markdown("### 1. Select Workflow Stages")
        wf_inc_scrape = st.checkbox("Stage 1: Ingest Fresh Reviews (Scraper)", value=True, help="Scrapes reviews using rotating anti-bot headers and localized tags")
        wf_inc_partition = st.checkbox("Stage 2: Validate Partitions & Deduplicate", value=True, help="Enforces 50,000-row limit, checks schema, and prevents duplicate records")
        wf_inc_parquet = st.checkbox("Stage 3: Synchronize High-Speed Parquet", value=True, help="Consolidates CSV partitions into snappy-compressed reviews.parquet for < 1s loading")
        wf_inc_train = st.checkbox("Stage 4: Retrain Bidirectional LSTM", value=True, help="Retrains neural network on balanced class weights and exports model artifacts")
        wf_inc_test = st.checkbox("Stage 5: Run Automated Pytest Suite", value=True, help="Executes automated unit tests verifying data integrity and neural bounds")

    with col_wf_params:
        st.markdown("### 2. Execution Parameters")
        wf_countries_cnt = st.slider("Countries to Query (Stage 1)", min_value=1, max_value=20, value=5, help="Number of countries to query")
        wf_reviews_cnt = st.slider("Reviews per Country (Stage 1)", min_value=5, max_value=50, value=15, step=5)
        wf_epochs = st.slider("Training Epochs (Stage 4)", min_value=1, max_value=5, value=2, help="Number of training epochs")
        wf_sample_option = st.select_slider(
            "Training Sample Size (Stage 4)",
            options=[1000, 3000, 5000, 10000, "Full Dataset"],
            value=3000,
            help="Stratified sampling for faster iteration without losing class balance",
        )
        wf_max_rows = st.number_input("CSV Partition Row Limit", min_value=5000, max_value=100000, value=50000, step=5000)

    st.write("")
    col_run1, col_run2 = st.columns([1, 1])
    with col_run1:
        run_workflow_button = st.button("▶️ Run Full Workflow", type="primary", use_container_width=True)
    with col_run2:
        st.link_button(
            "🚀 Run Workflow in GitHub Actions (CI/CD)",
            "https://github.com/mightyalok00/apple-review-sentiment-lstm/actions/workflows/ci.yml",
            help="Trigger remote automated GitHub Actions workflow dispatch with full logs & artifacts",
            use_container_width=True,
        )

    st.markdown("##### ⚡ One-Click Preset Runs")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        q_btn_data = st.button("📦 Run Data Validation & Parquet", use_container_width=True, help="Validate partitions & consolidate snappy Parquet")
    with col_q2:
        q_btn_test = st.button("🧪 Run Automated Test Suite", use_container_width=True, help="Run 18 automated unit tests via Pytest")
    with col_q3:
        q_btn_eval = st.button("📊 Run Baseline & Evaluation", use_container_width=True, help="Run Naive Bayes, Logistic Regression & OOD hard challenge benchmarks")

    should_run = run_workflow_button or q_btn_data or q_btn_test
    if should_run:
        active_stages = []
        if q_btn_data:
            active_stages = ["partition", "parquet"]
        elif q_btn_test:
            active_stages = ["test"]
        else:
            if wf_inc_scrape:
                active_stages.append("scrape")
            if wf_inc_partition:
                active_stages.append("partition")
            if wf_inc_parquet:
                active_stages.append("parquet")
            if wf_inc_train:
                active_stages.append("train")
            if wf_inc_test:
                active_stages.append("test")

        if not active_stages:
            st.error("Please select at least one workflow stage to run.")
        else:
            from workflow import run_full_pipeline

            wf_progress = st.progress(0.0)
            wf_status = st.empty()

            def update_wf_ui(stage_key, pct, msg):
                wf_progress.progress(min(1.0, max(0.0, pct)))
                wf_status.info(f"🔄 **Pipeline Status:** {msg}")

            train_sample_limit = None if wf_sample_option == "Full Dataset" else int(wf_sample_option)

            with st.spinner("Executing automated pipeline stages..."):
                report = run_full_pipeline(
                    stages=active_stages,
                    scrape_countries=wf_countries_cnt,
                    scrape_reviews_per_country=wf_reviews_cnt,
                    train_epochs=wf_epochs,
                    train_max_samples=train_sample_limit,
                    max_rows_per_csv=int(wf_max_rows),
                    progress_callback=update_wf_ui,
                    verbose=False,
                )

            wf_progress.progress(1.0)
            wf_status.empty()

            if report["overall_status"] == "SUCCESS":
                st.success(
                    f"🎉 **Workflow Execution Completed Successfully!** "
                    f"Total duration: **{report['total_elapsed_sec']}s** across **{len(report['executed_stages'])} stages**."
                )
            else:
                st.warning(
                    f"⚠️ **Workflow Finished with Warnings/Errors!** "
                    f"Total duration: **{report['total_elapsed_sec']}s**."
                )

            # Invalidate data cache so all dashboard views update with new numbers
            st.cache_data.clear()

            # Render stage results
            st.subheader("Stage Execution Breakdown")
            res_cols = st.columns(len(report["stage_results"]))
            for idx, (stg_name, stg_data) in enumerate(report["stage_results"].items()):
                with res_cols[idx]:
                    st_icon = "✅" if stg_data["status"] == "SUCCESS" else "❌"
                    st.metric(
                        label=f"{st_icon} {stg_data.get('stage', stg_name)}",
                        value=f"{stg_data['elapsed_sec']}s",
                        delta=stg_data["status"],
                    )

            # Detailed metrics expander
            with st.expander("🔍 View Raw Pipeline Metrics & Artifacts", expanded=False):
                st.json(report)

    elif q_btn_eval:
        from evaluate_model import run_comprehensive_evaluation
        with st.spinner("Running 5-fold cross-validation, baseline benchmarking, and hard OOD stress-tests..."):
            eval_output = run_comprehensive_evaluation(max_eval_samples=3000)
            st.success("🎉 **Model Evaluation & Baseline Benchmark Completed!**")
            b_comp = eval_output.get("baseline_comparison", {})
            st.markdown("#### Scientific Model Benchmark Comparison")
            bench_rows = []
            for m_name, m_data in b_comp.items():
                bench_rows.append({
                    "Model": m_name,
                    "Test Accuracy": f"{float(m_data.get('accuracy', 0.0)) * 100:.1f}%",
                    "5-Fold CV Accuracy": f"{float(m_data.get('cv_accuracy_mean', 0.0)) * 100:.1f}% ± {float(m_data.get('cv_accuracy_std', 0.0)) * 100:.1f}%" if "cv_accuracy_mean" in m_data else "N/A (Neural)",
                    "Hard Challenge (OOD)": f"{float(m_data.get('hard_challenge_accuracy', 0.0)) * 100:.1f}%",
                    "Brier Score": f"{float(m_data.get('brier_score', 0.0)):.4f}",
                    "Architecture": m_data.get("architecture", ""),
                })
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)

    # Show recent workflow history
    history_file = DATA_DIR / "workflow_history.json"
    if history_file.exists():
        try:
            history_data = json.loads(history_file.read_text(encoding="utf-8"))
            if history_data:
                st.markdown("---")
                st.subheader("Recent Workflow Executions")
                history_rows = []
                for h in reversed(history_data[-5:]):
                    history_rows.append({
                        "Timestamp": h.get("timestamp", "")[:19].replace("T", " "),
                        "Overall Status": "✅ SUCCESS" if h.get("overall_status") == "SUCCESS" else "❌ FAILED",
                        "Duration (s)": f"{h.get('total_elapsed_sec', 0)}s",
                        "Stages Executed": ", ".join(h.get("executed_stages", [])),
                    })
                st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
        except Exception:
            pass

    # Scientific Benchmark & Cross-Validation Insights
    bench_file = MODEL_DIR / "baseline_comparison.json"
    if bench_file.exists():
        try:
            bench_data = json.loads(bench_file.read_text(encoding="utf-8"))
            st.markdown("---")
            st.subheader("Scientific Benchmark & Cross-Validation Results")
            st.caption("Rigorous evaluation comparing classical machine learning against the Bidirectional LSTM on 5-Fold Stratified CV and Hard OOD Stress-Tests.")

            bench_rows = []
            for m_name, m_vals in bench_data.items():
                cv_txt = f"{m_vals['cv_accuracy_mean']*100:.1f}% ± {m_vals['cv_accuracy_std']*100:.1f}%" if "cv_accuracy_mean" in m_vals else "N/A (Neural)"
                hard_acc = m_vals.get("hard_challenge_accuracy", 0.0) * 100
                bench_rows.append({
                    "Model": m_name,
                    "In-Dist Acc": f"{m_vals.get('accuracy', 1.0)*100:.1f}%",
                    "5-Fold CV": cv_txt,
                    "Hard OOD Acc": f"{hard_acc:.1f}%",
                    "Brier Score": f"{m_vals.get('brier_score', 0.0):.4f}",
                    "Architecture": m_vals.get("architecture", "")[:45] + "...",
                })
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)

            # Visual OOD Benchmark Comparison Bar Chart
            fig_bench = px.bar(
                pd.DataFrame([
                    {"Model": m, "Hard OOD Accuracy (%)": d.get("hard_challenge_accuracy", 0.0) * 100}
                    for m, d in bench_data.items()
                ]),
                x="Hard OOD Accuracy (%)",
                y="Model",
                orientation="h",
                color="Model",
                color_discrete_sequence=["#94a3b8", "#38bdf8", "#10b981"],
                height=240,
                title="Stress-Test Resilience on Challenging Nuanced Reviews (Negations & Sarcasm)",
            )
            fig_bench.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_bench, use_container_width=True)
        except Exception:
            pass

# --- FOOTER ---
st.divider()
st.caption(
    "Apple Review Intelligence Platform | 65,000 Reviews | 130 Countries | "
    "Bidirectional LSTM Sentiment Engine | Licensed under MIT"
)

