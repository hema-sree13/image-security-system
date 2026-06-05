import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import hashlib
import io
import cv2
import datetime

st.set_page_config(
    page_title="Image Security System",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e8edf5; }
    h1 { color: #00d4ff !important; }
    h2, h3 { color: #e8edf5 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🛡️ Image Security System v2.0")
st.markdown("**Advanced detection — Morphing + Deepfake + Watermark + Report**")
st.markdown("---")

if 'reports' not in st.session_state:
    st.session_state.reports = []

# ── ELA ───────────────────────────────────────────────────────────────────────
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

# ── Noise ─────────────────────────────────────────────────────────────────────
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

# ── Metadata ──────────────────────────────────────────────────────────────────
def analyze_metadata(image):
    try:
        exif_data = image._getexif()
        if exif_data is None:
            return 70.0, "No EXIF metadata — suspicious"
        important_tags = [271, 272, 306, 36867]
        found = sum(1 for tag in important_tags if tag in exif_data)
        if found >= 3:
            return 10.0, "Full camera metadata — looks authentic"
        elif found >= 1:
            return 40.0, "Partial metadata — some fields missing"
        else:
            return 65.0, "Metadata stripped — common in edited images"
    except Exception:
        return 55.0, "Could not read metadata — file may have been re-saved"

# ── Face Morphing Detection ───────────────────────────────────────────────────
def detect_face_morphing(image):
    """
    Detects face morphing by analyzing:
    1. Edge inconsistencies around face boundaries
    2. Frequency domain anomalies
    3. Color channel inconsistencies
    4. Blending boundary detection
    """
    img_array = np.array(image.convert('RGB'))
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    score = 0
    findings = []

    # ── Check 1: Edge inconsistency ──
    edges = cv2.Canny(img_gray, 50, 150)
    h, w = edges.shape
    top_half = edges[:h//2, :]
    bottom_half = edges[h//2:, :]
    left_half = edges[:, :w//2]
    right_half = edges[:, w//2:]
    top_density = np.sum(top_half) / (top_half.size + 1e-6)
    bottom_density = np.sum(bottom_half) / (bottom_half.size + 1e-6)
    left_density = np.sum(left_half) / (left_half.size + 1e-6)
    right_density = np.sum(right_half) / (right_half.size + 1e-6)
    vertical_diff = abs(top_density - bottom_density) / (max(top_density, bottom_density) + 1e-6)
    horizontal_diff = abs(left_density - right_density) / (max(left_density, right_density) + 1e-6)
    edge_score = (vertical_diff + horizontal_diff) * 50
    score += min(edge_score, 30)
    if edge_score > 15:
        findings.append("🔴 Edge inconsistency detected — boundary mismatch between regions")
    else:
        findings.append("🟢 Edge distribution looks consistent")

    # ── Check 2: Frequency domain (DCT) ──
    dct_img = np.float32(img_gray) / 255.0
    dct = cv2.dct(dct_img)
    dct_abs = np.abs(dct)
    high_freq = dct_abs[h//4:, w//4:]
    low_freq = dct_abs[:h//4, :w//4]
    freq_ratio = np.mean(high_freq) / (np.mean(low_freq) + 1e-6)
    freq_score = min(freq_ratio * 200, 30)
    score += freq_score
    if freq_score > 15:
        findings.append("🔴 Abnormal frequency patterns — possible double compression or editing")
    else:
        findings.append("🟢 Frequency patterns look normal")

    # ── Check 3: Color channel inconsistency ──
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_corr = np.corrcoef(r.flatten(), g.flatten())[0,1]
    rb_corr = np.corrcoef(r.flatten(), b.flatten())[0,1]
    gb_corr = np.corrcoef(g.flatten(), b.flatten())[0,1]
    avg_corr = (rg_corr + rb_corr + gb_corr) / 3
    color_score = max(0, (0.85 - avg_corr) * 100)
    score += min(color_score, 20)
    if color_score > 10:
        findings.append("🔴 Color channel mismatch — different source images detected")
    else:
        findings.append("🟢 Color channels are consistent")

    # ── Check 4: Horizontal blending line detection ──
    row_variances = [np.var(img_gray[i, :]) for i in range(0, h, max(h//50, 1))]
    row_var_array = np.array(row_variances)
    sudden_changes = np.sum(np.abs(np.diff(row_var_array)) > np.std(row_var_array) * 2)
    blend_score = min(sudden_changes * 5, 20)
    score += blend_score
    if blend_score > 10:
        findings.append("🔴 Sudden variance changes — possible blending or copy-paste boundary")
    else:
        findings.append("🟢 No sudden blending boundaries detected")

    return min(score, 100), findings

# ── Combined Detection ────────────────────────────────────────────────────────
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

    morph_score, morph_findings = detect_face_morphing(image)
    results['morph'] = morph_score
    results['morph_findings'] = morph_findings

    # Updated weighted average including morphing detection
    final_score = (
        results['ela']      * 0.25 +
        results['noise']    * 0.20 +
        results['metadata'] * 0.15 +
        results['stats']    * 0.10 +
        results['morph']    * 0.30   # morphing gets highest weight
    )
    results['final_score'] = final_score

    if final_score < 15:
        results['verdict'] = 'AUTHENTIC'
        results['confidence'] = round(100 - final_score, 1)
        results['color'] = 'real'
        results['emoji'] = '✅'
    elif final_score < 99:
        results['verdict'] = 'UNCERTAIN'
        results['confidence'] = round(70 - abs(final_score - 42), 1)
        results['color'] = 'uncertain'
        results['emoji'] = '⚠️'
    else:
        results['verdict'] = 'MANIPULATED'
        results['confidence'] = round(final_score, 1)
        results['color'] = 'fake'
        results['emoji'] = '🚨'

    return results

# ── Watermark ─────────────────────────────────────────────────────────────────
def add_watermark(image, secret):
    img_array = np.array(image.convert('RGB'), dtype=np.int16)
    key_hash = hashlib.md5(secret.encode()).hexdigest()
    np.random.seed(int(key_hash[:8], 16) % (2**31))
    pattern = np.random.randint(0, 2, img_array.shape) * 2 - 1
    img_array = img_array + pattern
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def verify_watermark(image, secret):
    img_array = np.array(image.convert('RGB'), dtype=np.int16)
    key_hash = hashlib.md5(secret.encode()).hexdigest()
    np.random.seed(int(key_hash[:8], 16) % (2**31))
    pattern = np.random.randint(0, 2, img_array.shape) * 2 - 1
    correlation = np.mean(img_array * pattern)
    return correlation, correlation > 0.3

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Detect Fake",
    "🔒 Protect Image",
    "✅ Verify Image",
    "📋 Report Log"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🔍 Detect if an image is fake or manipulated")
    uploaded = st.file_uploader("Upload image", type=['jpg','jpeg','png','bmp','webp'], key="detect")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption=f"File: {uploaded.name}", use_container_width=True)
        st.markdown(f"**Size:** {image.size[0]} x {image.size[1]}")
        st.markdown("---")

        with st.spinner("🔬 Running advanced analysis..."):
            img_rgb = image.convert('RGB')
            results = detect_manipulation(img_rgb)

        st.markdown("### 🎯 Result")
        if results['color'] == 'real':
            st.success(f"{results['emoji']} {results['verdict']} — Confidence: {results['confidence']}%")
        elif results['color'] == 'fake':
            st.error(f"{results['emoji']} {results['verdict']} — Confidence: {results['confidence']}%")
        else:
            st.warning(f"{results['emoji']} {results['verdict']} — Confidence: {results['confidence']}%")

        st.markdown("---")
        st.markdown("### 🧮 ELA Map")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col2:
            st.image(results['ela_image'], caption="ELA Map", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 All Detection Scores")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🧮 ELA Score", f"{results['ela']:.1f}/100")
            st.metric("📊 Noise Score", f"{results['noise']:.1f}/100")
            st.metric("🔬 Morphing Score", f"{results['morph']:.1f}/100")
        with col2:
            st.metric("🗂️ Metadata Score", f"{results['metadata']:.1f}/100")
            st.metric("🔬 Pixel Stats", f"{results['stats']:.1f}/100")
            st.metric("⚠️ Final Score", f"{results['final_score']:.1f}/100")

        st.markdown("---")
        st.markdown("### 🔬 Morphing Analysis Findings")
        for finding in results['morph_findings']:
            st.markdown(finding)

        st.info(f"**Metadata:** {results['metadata_msg']}")

        st.markdown("---")
        st.markdown("### 🚨 Report this image")
        reporter_name = st.text_input("Your name (optional)", key="r_name")
        report_reason = st.text_area("Why are you reporting?", key="r_reason")
        if st.button("🚨 Submit Report"):
            report = {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": uploaded.name,
                "verdict": results['verdict'],
                "confidence": results['confidence'],
                "reporter": reporter_name if reporter_name else "Anonymous",
                "reason": report_reason if report_reason else "No reason given"
            }
            st.session_state.reports.append(report)
            st.success("✅ Report submitted! View in 📋 Report Log tab.")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔒 Protect your image with invisible watermark")
    uploaded2 = st.file_uploader("Upload image to protect", type=['jpg','jpeg','png'], key="protect")
    secret = st.text_input("Your secret key", value="MY_SECRET_KEY_2024", key="secret")
    if uploaded2 and secret:
        image2 = Image.open(uploaded2)
        st.image(image2, caption="Original Image", use_container_width=True)
        if st.button("🔒 Add Watermark"):
            with st.spinner("Adding watermark..."):
                watermarked = add_watermark(image2, secret)
                buf = io.BytesIO()
                watermarked.save(buf, format='PNG')
                buf.seek(0)
            st.success("✅ Watermark added!")
            st.download_button("⬇️ Download Protected Image", data=buf,
                file_name="protected_" + uploaded2.name.split('.')[0] + ".png", mime="image/png")
            st.warning("⚠️ Remember your secret key!")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ✅ Verify if image contains your watermark")
    uploaded3 = st.file_uploader("Upload image to verify", type=['jpg','jpeg','png'], key="verify")
    secret2 = st.text_input("Your secret key", value="MY_SECRET_KEY_2024", key="secret2")
    if uploaded3 and secret2:
        image3 = Image.open(uploaded3)
        st.image(image3, caption="Image to Verify", use_container_width=True)
        if st.button("✅ Check Watermark"):
            with st.spinner("Checking..."):
                score, found = verify_watermark(image3, secret2)
            if found:
                st.success(f"✅ WATERMARK FOUND — This is YOUR image! (Score: {score:.3f})")
                st.balloons()
            else:
                st.error(f"🚨 NO WATERMARK — Tampered or not yours! (Score: {score:.3f})")

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📋 Report Log")
    if len(st.session_state.reports) == 0:
        st.info("No reports yet.")
    else:
        st.success(f"Total reports: {len(st.session_state.reports)}")
        for i, r in enumerate(reversed(st.session_state.reports)):
            with st.expander(f"Report #{len(st.session_state.reports)-i} — {r['file']} — {r['time']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**File:** {r['file']}")
                    st.markdown(f"**Verdict:** {r['verdict']}")
                    st.markdown(f"**Confidence:** {r['confidence']}%")
                with col2:
                    st.markdown(f"**Reported by:** {r['reporter']}")
                    st.markdown(f"**Time:** {r['time']}")
                st.markdown(f"**Reason:** {r['reason']}")
        if st.button("🗑️ Clear All Reports"):
            st.session_state.reports = []
            st.rerun()
