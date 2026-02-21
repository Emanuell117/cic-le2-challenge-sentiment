import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HISTORY_PATH = os.getenv("HISTORY_PATH", "/data/history.csv")

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖")

st.title("Sentiment Analyzer")
st.markdown("Microservice example for LE1 and LE2 from the module cic.")

# --- Sidebar: Status Check ---
st.sidebar.header("System Status")
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
    st.sidebar.success(f"Backend Online ✅\nModel: {health.get('model')}")
except Exception as e:
    st.sidebar.error(f"Backend Offline ❌\nURL: {BACKEND_URL}")
    st.sidebar.info("Tip: Check if the service is started.")

# --- Main Area: Inferenz ---
st.subheader("Analyze new text")
text_input = st.text_area("Insert English text:", height=100)

if st.button("Analyze"):
    if text_input:
        try:
            with st.spinner("Request API..."):
                response = requests.post(
                    f"{BACKEND_URL}/predict", json={"text": text_input}
                )

            if response.status_code == 200:
                result = response.json()
                label = result["label"]
                score = round(result["score"] * 100, 2)

                if label == "POSITIVE":
                    st.success(f"Result: POSITIV ({score}%)")
                else:
                    st.error(f"Result: NEGATIV ({score}%)")

                # --- Persistenz (Speichern in History) ---
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_entry = pd.DataFrame(
                    [[timestamp, text_input, label, score]],
                    columns=["Time", "Text", "Label", "Score"],
                )

                os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

                write_header = not os.path.exists(HISTORY_PATH)
                new_entry.to_csv(
                    HISTORY_PATH, mode="a", header=write_header, index=False
                )

            else:
                st.error(f"Backend error: {response.status_code}")

        except Exception as e:
            st.error(f"Connection error: {e}")

# --- History Area ---
st.divider()
st.subheader("History")
st.caption(f"Loading data from: `{HISTORY_PATH}`")

if os.path.exists(HISTORY_PATH):
    try:
        df = pd.read_csv(HISTORY_PATH)
        st.dataframe(df.sort_values(by="Time", ascending=False), width="stretch")
    except Exception as e:
        st.error(f"Error loading history data: {e}")
else:
    st.warning("No history found.")
