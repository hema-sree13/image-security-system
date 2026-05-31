import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import io

st.set_page_config(
    page_title="Image Watermark System",
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

st.markdown("# 🛡️ Image Watermark System")
st.markdown("**Protect your images with an invisible watermark**")
st.markdown("---")

SECRET_KEY = "MY_SECRET_IMAGE_KEY_2024"

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

tab1, tab2 = st.tabs(["🔒 Protect My Image", "🔍 Verify Image"])

with tab1:
    st.markdown("### Step 1 — Upload your original image")
    uploaded = st.file_uploader("Choose image to protect", type=['jpg','jpeg','png'])
    secret = st.text_input("Enter your secret key (remember this!)", value=SECRET_KEY)

    if uploaded and secret:
        image = Image.open(uploaded)
        st.image(image, caption="Your Original Image", use_column_width=True)

        if st.button("🔒 Add Invisible Watermark"):
            with st.spinner("Adding watermark..."):
                watermarked = add_watermark(image, secret)
                buf = io.BytesIO()
                watermarked.save(buf, format='PNG')
                buf.seek(0)

            st.success("✅ Watermark added successfully!")
            st.markdown("**Download your protected image below and use this instead of the original:**")
            st.download_button(
                label="⬇️ Download Protected Image",
                data=buf,
                file_name="protected_" + uploaded.name.split('.')[0] + ".png",
                mime="image/png"
            )
            st.warning("⚠️ Remember your secret key — you need it to verify later!")

with tab2:
    st.markdown("### Step 2 — Verify if image is yours")
    uploaded2 = st.file_uploader("Upload image to verify", type=['jpg','jpeg','png'], key="verify")
    secret2 = st.text_input("Enter your secret key", value=SECRET_KEY, key="secret2")

    if uploaded2 and secret2:
        image2 = Image.open(uploaded2)
        st.image(image2, caption="Image to Verify", use_column_width=True)

        if st.button("🔍 Check Watermark"):
            with st.spinner("Checking watermark..."):
                score, found = verify_watermark(image2, secret2)

            if found:
                st.success(f"✅ WATERMARK FOUND — This is YOUR original image! (Score: {score:.3f})")
            else:
                st.error(f"🚨 NO WATERMARK — This image has been tampered or is not yours! (Score: {score:.3f})")