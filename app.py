# ======================================================
# app.py — Wiki Image Matcher (Streamlit Demo)
# ======================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import math

# ======================================================
# Config
# ======================================================
class Config:
    image_model_name = "vit_base_patch16_siglip_384"
    text_model_name  = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    embed_dim = 512
    max_len = 39
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Model definition (must match training)
# ======================================================
class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        self.image_encoder = timm.create_model(
            Config.image_model_name,
            pretrained=False,
            num_classes=0
        )
        self.img_projection = nn.Linear(
            self.image_encoder.num_features,
            Config.embed_dim
        )

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(
            Config.text_model_name
        )
        self.txt_projection = nn.Linear(
            self.text_encoder.config.hidden_size,
            Config.embed_dim
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, x):
        x = self.image_encoder(x)
        x = self.img_projection(x)
        return F.normalize(x, dim=1)

    def encode_text(self, ids, mask):
        x = self.text_encoder(
            input_ids=ids,
            attention_mask=mask
        ).last_hidden_state[:, 0]
        x = self.txt_projection(x)
        return F.normalize(x, dim=1)

# ======================================================
# Load model weights (.pth)
# ======================================================
@st.cache_resource
def load_model():
    model = DualEncoder()

    # load backbones
    model.image_encoder.load_state_dict(
        torch.load("image_backbone.pth", map_location="cpu")
    )
    model.text_encoder.load_state_dict(
        torch.load("text_backbone.pth", map_location="cpu")
    )

    # load projection heads
    proj = torch.load("projection_heads.pth", map_location="cpu")
    model.img_projection.load_state_dict(proj["img_projection"])
    model.txt_projection.load_state_dict(proj["txt_projection"])
    model.logit_scale.data = proj["logit_scale"]

    model.eval().to(Config.device)
    return model

# ======================================================
# Preprocess
# ======================================================
image_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

tokenizer = AutoTokenizer.from_pretrained(Config.text_model_name)

# ======================================================
# Load caption list (CSV)
# ======================================================
@st.cache_data
def load_candidate_texts():
    df = pd.read_csv("test_caption_list.csv")
    texts = df["caption_title_and_reference_description"].astype(str).tolist()
    return texts

candidate_texts = load_candidate_texts()

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Wiki Image Matcher", layout="wide")
st.title("影像標題配對｜Wiki Image Matcher")

model = load_model()

left_col, right_col = st.columns([1.2, 1])

# ---------------- LEFT PANEL ----------------
with left_col:
    st.subheader("上傳圖像")
    uploaded_file = st.file_uploader(
        "支援 JPG / PNG / WebP",
        type=["jpg", "png", "webp"]
    )

    st.subheader("設定")
    topk = st.number_input(
        "返回結果數量 (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    start_btn = st.button("開始匹配")

# ---------------- RIGHT PANEL ----------------
with right_col:
    st.subheader("匹配結果")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="輸入影像", use_column_width=True)

    if start_btn and uploaded_file:
        with st.spinner("模型推論中，請稍候..."):

            # preprocess image
            img_tensor = image_transform(img).unsqueeze(0).to(Config.device)

            with torch.no_grad():
                # encode image
                img_emb = model.encode_image(img_tensor)

                # encode all captions
                tokens = tokenizer(
                    candidate_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=Config.max_len,
                    return_tensors="pt"
                )

                txt_emb = model.encode_text(
                    tokens["input_ids"].to(Config.device),
                    tokens["attention_mask"].to(Config.device)
                )

                # similarity
                sims = (img_emb @ txt_emb.T).squeeze(0)
                scores, indices = sims.topk(topk)

        st.success("匹配完成 ✅")

        for rank, (idx, score) in enumerate(
            zip(indices.tolist(), scores.tolist()), start=1
        ):
            st.markdown(
                f"""
                **{rank}. {candidate_texts[idx]}**  
                <span style="color:gray">Similarity score: {score:.4f}</span>
                """,
                unsafe_allow_html=True
            )

    if not uploaded_file:
        st.info("請先上傳一張影像，然後點擊「開始匹配」")
