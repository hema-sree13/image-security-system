import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import cv2

st.set_page_config(
    page_title="Image Authenticity Detector",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e8edf5; }
    h1 { color: #00d4ff !important; }
    h2, h3 { color: #e8edf5 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔍 Image Authenticity Detector")
st.markdown("**Upload any image to check if it is real or manipulated/morphed.**")
st.markdown("---")

def perform_ela(image, quality=90):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    ela_image = ImageChops.difference(image, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def calculate_ela_score(ela_image):
    ela_array = np.array(ela_image.convert('L'))
    mean_val = np.mean(ela_array)
    std_val = np.std(ela_array)
    score = (mean_val / 255.0) * 60 + (std_val / 128.0) * 40
    return min(score, 100)

def analyze_noise(image):
    img_array = np.array(image.convert('L')).astype(np.float32)
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    noise = img_array - blurred
    h, w = noise.shape
    block_size = max(h // 4, w // 4, 32)
    noise_levels = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = noise[i:i+block_size, j:j+block_size]
            noise_levels.append(np.std(block))
    if len(noise_levels) < 2:
        return 20.0
    noise_variation = np.std(noise_levels) / (np.mean(noise_levels) + 1e-6)
    return min(noise_variation * 50, 100)

def analyze_metadata(image):
    try:
        exif_data = image._getexif()
        if exif_data is None:
            return 70.0, "No EXIF metadata found — suspicious"
        important_tags = [271, 272, 306, 36867]
        found = sum(1 for tag in important_tags if tag in exif_data)
        if found >= 3:
            return 10.0, "Full camera metadata present — looks authentic"
        elif found >= 1:
            return 40.0, "Partial metadata — some fields missing"
        else:
            return 65.0, "Metadata stripped — common in edited images"
    except Exception:
        return 55.0, "Could not read metadata — file may have been re-saved"

def detect_manipulation(image):
    results = {}
    ela_img = perform_ela(image)
    ela_score = calculate_ela_score(ela_img)
    results['ela'] = ela_score
    results['ela_image'] = ela_img
    results['noise'] = analyze_noise(image)
    meta_score, meta_msg = analyze_metadata(image)
    results['metadata'] = meta_score
    results['metadata_msg'] = meta_msg
    img_array = np.array(image.convert('RGB'))
    channel_stds = [np.std(img_array[:,:,c]) for c in range(3)]
    avg_std = np.mean(channel_stds)
    results['stats'] = 30.0 if avg_std > 20 else 60.0
    final_score = (
        results['ela']      * 0.40 +
        results['noise']    * 0.25 +
        results['metadata'] * 0.20 +
        results['stats']    * 0.15
    )
    results['final_score'] = final_score
    if final_score < 35:
        results['verdict'] = 'AUTHENTIC'
        results['confidence'] = round(100 - final_score, 1)
        results['color'] = 'real'
        results['emoji'] = '✅'
    elif final_score < 60:
        results['verdict'] = 'UNCERTAIN'
        results['confidence'] = round(70 - abs(final_score - 47), 1)
        results['color'] = 'uncertain'
        results['emoji'] = '⚠️'
    else:
        results['verdict'] = 'MANIPULATED'
        results['confidence'] = round(final_score, 1)
        results['color'] = 'fake'
        results['emoji'] = '🚨'
    return results

uploaded_file = st.file_uploader(
    "📂 Upload an image to analyze",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown("### 📸 Uploaded Image")
    st.image(image, caption=f"File: {uploaded_file.name}", use_column_width=True)
    st.markdown(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
    st.markdown("---")

    with st.spinner("🔬 Analyzing image... Please wait..."):
        img_rgb = image.convert('RGB')
        results = detect_manipulation(img_rgb)

    st.markdown("### 🎯 Detection Result")

    if results['color'] == 'real':
        st.success(f"{results['emoji']} {results['verdict']} — Confidence: {results['confidence']}%")
    elif results['color'] == 'fake':
        st.error(f"{results['emoji']} {results['verdict']} — Confidence: {results['confidence']}%")
    else:
        st.warning(f"{results['emoji']} {results['verdict']} — Confidence: {results['confidence']}%")

    st.markdown("---")
    st.markdown("### 🧮 ELA Analysis Map")
    st.markdown("*Brighter areas = more suspicious = likely edited regions*")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original Image", use_column_width=True)
    with col2:
        st.image(results['ela_image'], caption="ELA Map", use_column_width=True)

    st.markdown("---")
    st.markdown("### 📊 Detailed Scores *(higher = more suspicious)*")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧮 ELA Score", f"{results['ela']:.1f} / 100")
        st.metric("📊 Noise Score", f"{results['noise']:.1f} / 100")
    with col2:
        st.metric("🗂️ Metadata Score", f"{results['metadata']:.1f} / 100")
        st.metric("🔬 Pixel Stats", f"{results['stats']:.1f} / 100")

    st.markdown("---")
    st.markdown("### 📋 Findings")
    st.info(f"**Metadata:** {results['metadata_msg']}")
    if results['ela'] > 60:
        st.error("**ELA:** High error levels — strong sign of editing.")
    elif results['ela'] > 35:
        st.warning("**ELA:** Moderate error levels — possible editing.")
    else:
        st.success("**ELA:** Low error levels — looks original.")
    if results['noise'] > 60:
        st.error("**Noise:** Inconsistent — suggests copy-paste or blending.")
    elif results['noise'] > 35:
        st.warning("**Noise:** Slightly uneven — possible filter or edit.")
    else:
        st.success("**Noise:** Consistent — looks like original camera output.")

else:
    st.markdown("### 👆 How to use")
    st.markdown("""
    1. Click **Browse files** above
    2. Select any image (JPG, PNG, etc.)
    3. The app will automatically analyze it
    4. You will see: ✅ AUTHENTIC, ⚠️ UNCERTAIN, or 🚨 MANIPULATED
    """)