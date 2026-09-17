import streamlit as st
import requests
from PIL import Image, ImageDraw

API_URL = "http://localhost:8001"

st.title("Vehicle ReID — демо")

uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Исходное изображение", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        x = st.number_input("x", min_value=0, value=0)
        y = st.number_input("y", min_value=0, value=0)
    with col2:
        w = st.number_input("w", min_value=1, value=100)
        h = st.number_input("h", min_value=1, value=100)

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    st.image(preview, caption="BBox preview", use_container_width=True)

    top_k = st.slider("Top-K", 1, 20, 10)

    if st.button("Найти похожие ТС"):
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
        data = {"x": x, "y": y, "w": w, "h": h, "top_k": top_k}
        resp = requests.post(f"{API_URL}/search", files=files, data=data)

        if resp.status_code != 200:
            st.error(f"Ошибка: {resp.json().get('detail')}")
        else:
            result = resp.json()
            if result["rejected"]:
                st.warning("Отказ: уверенность ниже порога — совпадений не найдено.")
            else:
                st.success(f"Найдено {len(result['results'])} кандидатов")
                for item in result["results"]:
                    st.write(f"gallery_id={item['gallery_id']}  confidence={item['confidence']:.4f}")
