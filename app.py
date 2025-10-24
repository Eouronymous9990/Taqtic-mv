import streamlit as st
import pandas as pd
import cv2
from PIL import Image
import io
import os
import time

from ShootingAnalyzer import main as analyze_shooting
from PassingAnalyzer import main as analyze_passing
from ReceivingAnalyzer import main as analyze_receiving

st.set_page_config(page_title="Tactiq", layout="wide", initial_sidebar_state="collapsed", page_icon="icon.png")

if 'results' not in st.session_state: st.session_state.results = None
if 'skill_analyzed' not in st.session_state: st.session_state.skill_analyzed = None
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

st.title("Football Skill Analysis")
st.markdown("Upload your videos and select the skill you want to analyze")

col1, col2 = st.columns(2)
with col1: side_video = st.file_uploader("Upload Side View Video (Required)", type=["mp4"], key=f"side_video_{st.session_state.uploader_key}")
with col2: optional_video = st.file_uploader("Upload Front/Back View Video (Optional)", type=["mp4"], key=f"front_back_video_{st.session_state.uploader_key}")

st.subheader("Select the Skill to Analyze")
skill_choice = st.selectbox("Choose the skill:", ["Receiving", "Shooting", "Passing"], key="skill_selector")

if side_video and (st.session_state.skill_analyzed != skill_choice):
    st.session_state.results = None
    st.session_state.skill_analyzed = None

col_start, col_clear = st.columns([1, 1])
with col_start: start_pressed = st.button("Start Analysis", use_container_width=True)
with col_clear: clear_pressed = st.button("Clear", use_container_width=True, type="primary")

if start_pressed:
    if not side_video: st.warning("Please upload the side view video to proceed.")
    else:
        with st.spinner(f"Analyzing {skill_choice.lower()} skill... please wait"):
            temp_side_path = "temp_side.mp4"
            temp_optional_path = None
            with open(temp_side_path, "wb") as f: f.write(side_video.read())

            if optional_video:
                temp_optional_path = "temp_optional.mp4"
                with open(temp_optional_path, "wb") as f: f.write(optional_video.read())
                has_optional = True
            else:
                temp_optional_path = temp_side_path
                has_optional = False

            results = None
            try:
                if skill_choice == "Shooting":
                    results = analyze_shooting(temp_side_path, temp_optional_path)
                elif skill_choice == "Passing":
                    results = analyze_passing(temp_side_path, temp_optional_path)
                elif skill_choice == "Receiving":
                    results = analyze_receiving(temp_side_path, temp_optional_path)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                results = []

            st.session_state.results = results
            st.session_state.skill_analyzed = skill_choice
            st.session_state.has_optional = has_optional

            if os.path.exists(temp_side_path):
                os.remove(temp_side_path)
            if temp_optional_path and os.path.exists(temp_optional_path):
                os.remove(temp_optional_path)

if clear_pressed:
    st.session_state.results = None
    st.session_state.skill_analyzed = None
    st.session_state.has_optional = False
    st.session_state.uploader_key += 1
    st.rerun()

if st.session_state.results is not None:
    results = st.session_state.results
    has_optional = st.session_state.get("has_optional", True)
    
    if results:
        st.success(f"Analysis complete! {len(results)} detection(s) found.")

        for idx, detection in enumerate(results, 1):
            st.markdown(f"---")
            st.header(f"Detection {idx}")

            side_frame = detection.get("Side Frame")
            other_frame = detection.get("Front Frame") if detection.get("Front Frame") is not None else detection.get("Back Frame")

            colA, colB = st.columns(2)
            with colA:
                if side_frame is not None:
                    side_img = Image.fromarray(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB))
                    st.image(side_img, caption="Side View", use_container_width=True)
                    side_buffer = io.BytesIO()
                    side_img.save(side_buffer, format="JPEG")
                    side_bytes = side_buffer.getvalue()
                    st.download_button(label="⬇ Download Side View", data=side_bytes, file_name=f"detection_{idx}_side.jpg", mime="image/jpeg", key=f"side_download_{idx}")

            with colB:
                if has_optional and other_frame is not None:
                    other_img = Image.fromarray(cv2.cvtColor(other_frame, cv2.COLOR_BGR2RGB))
                    st.image(other_img, caption="Front/Back View", use_container_width=True)
                    other_buffer = io.BytesIO()
                    other_img.save(other_buffer, format="JPEG")
                    other_bytes = other_buffer.getvalue()
                    st.download_button(label="⬇ Download Front/Back View", data=other_bytes, file_name=f"detection_{idx}_other.jpg",mime="image/jpeg", key=f"other_download_{idx}") 

            if skill_choice == "Shooting" or skill_choice == "Passing":
                exclude_keys = ["Side Frame", "Back Frame"]
                table_data = [{"Metric": str(k), "Value": str(v)} for k, v in detection.items() if k not in exclude_keys]
                if not has_optional and len(table_data) > 1: table_data = table_data[:-1]
                st.subheader("Biomechanical Metrics")
                df = pd.DataFrame(table_data)
                st.table(df)

            elif skill_choice == "Receiving":
                exclude_keys = ["Side Frame", "Front Frame"]
                table_data = [{"Metric": str(k), "Value": str(v)} for k, v in detection.items() if k not in exclude_keys]
                if not has_optional and len(table_data) > 1: table_data = table_data[:-6]
                st.subheader("Biomechanical Metrics")
                df = pd.DataFrame(table_data)
                st.table(df)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode()
            st.download_button(label="⬇ Download Metrics", data=csv_bytes, file_name=f"detection_{idx}_metrics.csv", mime="text/csv", key=f"csv_download_{idx}")

    else:
        st.error("No detections found or analysis failed.")