# ======================================================
# app.py — Wiki Image Matcher (Minimal Working Demo)
# ======================================================

import streamlit as st
import torch
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
import pandas as pd

# ======================================================
# Basic Config
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_MODEL_NAME = "vit_base_patch16_siglip_384"
TEXT_MODEL_NAME  = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
MAX_LEN = 39

# ======================================================
# Load models from Hugging Face / timm
# ======================================================
@st.cache_resource
def load_models():
    image_encoder = timm.create_model(
        IMAGE_MODEL_NAME,
        pretrained=True,
        num_classes=0
    ).to(DEVICE).eval()

    text_encoder = AutoModel.from_pretrained(
        TEXT_MODEL_NAME
    ).to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    return image_encoder, text_encoder, tokenizer

image_encoder, text_encoder, tokenizer = load_models()

# ======================================================
# Image preprocessing
# ======================================================
image_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ======================================================
# Load caption list
# ======================================================
@st.cache_data
def load_captions():
    df = pd.read_csv("test_caption_list.csv")
    return df["caption_title_and_reference_description"].astype(str).tolist()

candidate_texts = load_captions()

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Wiki Image Matcher", layout="wide")
st.title("影像標題配對｜Wiki Image Matcher")

left, right = st.columns([1.2, 1])

# ---------------- LEFT PANEL ----------------
with left:
    st.subheader("上傳圖像")
    uploaded = st.file_uploader(
        "支援 JPG / PNG / WebP",
        type=["jpg", "png", "webp"]
    )

    st.subheader("設定")
    topk = st.number_input(
        "返回結果數量 (Top-K)",
        min_value=1,
        max_value=10,
        value=5
    )

    start_btn = st.button("開始匹配")

# ---------------- RIGHT PANEL ----------------
with right:
    st.subheader("匹配結果")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="輸入影像", use_column_width=True)

    if start_btn and uploaded:
        with st.spinner("模型推論中..."):

            # --- Encode image ---
            img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_emb = image_encoder(img_tensor)
                img_emb = F.normalize(img_emb, dim=1)

            # --- Encode all captions ---
            tokens = tokenizer(
                candidate_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            with torch.no_grad():
                txt_emb = text_encoder(
                    input_ids=tokens["input_ids"].to(DEVICE),
                    attention_mask=tokens["attention_mask"].to(DEVICE)
                ).last_hidden_state[:, 0]
                txt_emb = F.normalize(txt_emb, dim=1)

            # --- Similarity & Top-K ---
            sims = (img_emb @ txt_emb.T).squeeze(0)
            scores, indices = sims.topk(topk)

        st.success("匹配完成 ✅")

        for rank, (idx, score) in enumerate(
            zip(indices.tolist(), scores.tolist()), start=1
        ):
            st.markdown(
                f"""
                **{rank}. {candidate_texts[idx]}**  
                <span style="color:gray">Similarity: {score:.4f}</span>
                """,
                unsafe_allow_html=True
            )

    if not uploaded:
        st.info("請先上傳一張圖片，然後點擊「開始匹配」")
