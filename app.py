
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="TuneWise AI",
    page_icon="🎵",
    layout="wide"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1DB954;
        margin-bottom: 0;
    }

    .subtitle {
        color: #b3b3b3;
        font-size: 18px;
    }

    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b3b3b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/genre_classifier.joblib")


saved = load_model()
model = saved["model"]
features = saved["features"]

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("🎵 TuneWise AI")
    st.write("Your AI-powered music genre explorer.")

    st.divider()

    st.subheader("📌 About the Project")
    st.write(
        "TuneWise AI uses a Random Forest classification "
        "model to predict a music genre from audio features."
    )

    st.subheader("💡 How to Use")
    st.write("1. Adjust the audio feature controls.")
    st.write("2. Click Predict Genre.")
    st.write("3. Explore the Top 5 model predictions.")

    st.divider()

    st.caption("Built with Python, Scikit-learn and Streamlit.")
    st.caption("Machine Learning • Music Analysis")

# -------------------------------
# Main Header
# -------------------------------
st.markdown(
    '<p class="main-title">🎵 TuneWise AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Discover a song’s genre using Machine Learning</p>',
    unsafe_allow_html=True
)

st.write(
    "Enter a song's audio features below to predict its "
    "most likely music genre."
)

# -------------------------------
# Model Information Cards
# -------------------------------
st.divider()

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Model Type", "Random Forest")

with metric2:
    st.metric("Genres Supported", "114")

with metric3:
    st.metric("Baseline Accuracy", "23.53%")

st.caption(
    "Accuracy measured on a held-out test dataset. "
    "Individual predictions may be incorrect."
)

st.divider()

# -------------------------------
# Audio Feature Input Form
# -------------------------------
with st.form("genre_form"):
    st.subheader("🎧 Song Audio Features")
    st.write("Adjust the values to describe the song.")

    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider(
            "Danceability", 0.0, 1.0, 0.5
        )
        energy = st.slider(
            "Energy", 0.0, 1.0, 0.5
        )
        key = st.selectbox(
            "Key", list(range(12))
        )
        loudness = st.slider(
            "Loudness (dB)", -60.0, 5.0, -8.0
        )

    with col2:
        mode = st.selectbox(
            "Mode", [0, 1]
        )
        speechiness = st.slider(
            "Speechiness", 0.0, 1.0, 0.1
        )
        acousticness = st.slider(
            "Acousticness", 0.0, 1.0, 0.5
        )
        instrumentalness = st.slider(
            "Instrumentalness", 0.0, 1.0, 0.0
        )

    with col3:
        liveness = st.slider(
            "Liveness", 0.0, 1.0, 0.2
        )
        valence = st.slider(
            "Valence", 0.0, 1.0, 0.5
        )
        tempo = st.slider(
            "Tempo (BPM)", 30.0, 250.0, 120.0
        )
        time_signature = st.selectbox(
            "Time Signature", [3, 4, 5, 6, 7]
        )

    submitted = st.form_submit_button(
        "🎵 Predict Genre",
        use_container_width=True
    )

# -------------------------------
# Genre Prediction
# -------------------------------
if submitted:
    input_data = pd.DataFrame(
        [{
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "time_signature": time_signature,
        }],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("🎯 Prediction Result")

    st.success(
        f"Predicted genre: **{prediction.title()}**"
    )

    # Display the five highest model scores
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        top_indices = probabilities.argsort()[-5:][::-1]

        top_results = pd.DataFrame({
            "Genre": model.classes_[top_indices],
            "Model score": probabilities[top_indices]
        })

        st.subheader("📊 Top 5 Model Predictions")

        chart_data = top_results.set_index("Genre")
        st.bar_chart(chart_data)

        st.caption(
            "Scores are estimates from the model, not guarantees "
            "that a genre is correct."
        )

    st.info(
        "Tip: Try changing the audio features and predicting again "
        "to explore how the model responds."
    )

# -------------------------------
# Footer
# -------------------------------
st.divider()

st.caption(
    "TuneWise AI | Machine Learning Mini Project | "
    "Random Forest Genre Classification"
)
