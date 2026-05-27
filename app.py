import streamlit as st
import torch
import timm
import pandas as pd
from PIL import Image
from torchvision import transforms

st.set_page_config(
    page_title="Klasifikasi Roasting Kopi",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best_vit_coffee_roast_final.pth"
CLASS_NAMES = ["Dark", "Light", "Medium"]

@st.cache_resource
def load_model():
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=3
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "class_to_idx" in checkpoint:
            idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
            global CLASS_NAMES
            CLASS_NAMES = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    prob_data = {
        "Kelas": CLASS_NAMES,
        "Probabilitas": [float(probs[i]) * 100 for i in range(len(CLASS_NAMES))]
    }

    return pred_class, confidence, pd.DataFrame(prob_data)

ROAST_INFO = {
    "Light": {
        "emoji": "🟤",
        "title": "Light Roast",
        "desc": "Biji kopi berwarna cokelat muda, permukaan relatif kering, dan karakter visual belum terlalu gelap.",
        "notes": ["Warna lebih terang", "Minyak belum tampak jelas", "Cocok untuk karakter rasa yang lebih asam dan ringan"],
    },
    "Medium": {
        "emoji": "☕",
        "title": "Medium Roast",
        "desc": "Biji kopi berada pada warna cokelat sedang, menjadi titik tengah antara Light Roast dan Dark Roast.",
        "notes": ["Warna cokelat seimbang", "Karakter visual lebih matang", "Umumnya memiliki keseimbangan aroma dan body"],
    },
    "Dark": {
        "emoji": "⚫",
        "title": "Dark Roast",
        "desc": "Biji kopi terlihat lebih gelap dan pekat, biasanya memiliki tampilan permukaan yang lebih intens.",
        "notes": ["Warna cokelat tua sampai gelap", "Kontras visual lebih kuat", "Cenderung berkarakter bold dan pahit"],
    },
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, rgba(145, 83, 38, 0.22), transparent 32%),
                linear-gradient(135deg, #090B10 0%, #11141A 48%, #17110D 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1D2029 0%, #252832 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.main-title {
    font-size: 48px;
    line-height: 1.08;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #F6C177, #B77945, #F4E7D3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #C9B7A6;
    max-width: 980px;
    line-height: 1.7;
}

.hero {
    padding: 34px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(77, 45, 25, 0.85), rgba(19, 21, 27, 0.88));
    border: 1px solid rgba(246, 193, 119, 0.18);
    box-shadow: 0 18px 60px rgba(0,0,0,0.28);
    margin-bottom: 26px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(246, 193, 119, 0.13);
    color: #F6C177;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid rgba(246, 193, 119, 0.25);
    margin-bottom: 14px;
}

.card {
    min-height: 175px;
    padding: 24px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(255, 247, 237, 0.98), rgba(232, 212, 190, 0.92));
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow: 0 12px 32px rgba(0,0,0,0.18);
    color: #2D1B12;
}

.card h3 {
    color: #3B2418;
    font-size: 24px;
    margin-bottom: 12px;
}

.card p, .card li {
    color: #5E4634;
    font-size: 15px;
    line-height: 1.7;
}

.dark-card {
    padding: 24px;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.09);
    box-shadow: 0 12px 32px rgba(0,0,0,0.18);
}

.dark-card h3, .dark-card h4 {
    color: #F4E7D3;
}

.dark-card p, .dark-card li {
    color: #D4C3B4;
    line-height: 1.7;
}

.metric-card {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(246, 193, 119, 0.16), rgba(255,255,255,0.05));
    border: 1px solid rgba(246, 193, 119, 0.18);
    text-align: center;
}

.metric-number {
    font-size: 32px;
    font-weight: 900;
    color: #F6C177;
}

.metric-label {
    color: #C9B7A6;
    font-size: 14px;
    font-weight: 700;
}

.step-box {
    padding: 18px;
    border-radius: 20px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    height: 100%;
}

.step-number {
    width: 36px;
    height: 36px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #F6C177;
    color: #24150D;
    font-weight: 900;
    margin-bottom: 10px;
}

.result-box {
    padding: 26px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(255, 247, 237, 0.98), rgba(246, 193, 119, 0.18));
    border: 2px solid rgba(246, 193, 119, 0.58);
    box-shadow: 0 16px 40px rgba(0,0,0,0.22);
    color: #2D1B12;
}

.small-muted {
    color: #9F9186;
    font-size: 14px;
}

hr {
    border-color: rgba(255,255,255,0.09) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">☕ Web Demo Tugas Akhir • Computer Vision • Deep Learning</div>
    <div class="main-title">Klasifikasi Tingkat Roasting Biji Kopi</div>
    <div class="subtitle">
        Sistem ini menerapkan model <b>Vision Transformer</b> untuk mengenali tingkat roasting biji kopi melalui citra digital.
        Pengguna cukup mengunggah gambar biji kopi, lalu sistem akan menampilkan kelas prediksi beserta probabilitasnya.
    </div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "🔍 Klasifikasi", "📊 Informasi Model", "📚 Tentang Penelitian"]
)

st.sidebar.markdown("---")
st.sidebar.success("Model: Vision Transformer")
st.sidebar.write("Accuracy: **99.83%**")
st.sidebar.write("Precision: **99.83%**")
st.sidebar.write("Recall: **99.83%**")
st.sidebar.write("F1-Score: **99.83%**")
st.sidebar.markdown("---")
st.sidebar.caption("Tips: gunakan foto biji kopi yang jelas, cukup cahaya, dan tidak blur. Jangan upload foto mantan, modelnya trauma.")

if menu == "🏠 Beranda":
    st.markdown("### Gambaran Umum Sistem")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-number">3.989</div><div class="metric-label">Total Citra</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-number">3</div><div class="metric-label">Kelas Roasting</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-number">224×224</div><div class="metric-label">Ukuran Input</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-number">99.83%</div><div class="metric-label">Akurasi Model</div></div>', unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📦 Dataset</h3>
        <p>Total citra: <b>3.989</b></p>
        <p>Train: <b>2.791</b><br>Validation: <b>599</b><br>Test: <b>599</b></p>
        <p>Dataset dipakai untuk melatih, memvalidasi, dan menguji performa model klasifikasi.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🧠 Metode</h3>
        <p>Model menggunakan <b>Vision Transformer</b> dengan pendekatan transfer learning.</p>
        <p>Citra diproses menjadi patch, lalu pola visualnya dipelajari menggunakan mekanisme self-attention.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
        <h3>🏷️ Kelas</h3>
        <p>Model mengenali tiga tingkat roasting:</p>
        <ul>
            <li>Light Roast</li>
            <li>Medium Roast</li>
            <li>Dark Roast</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Alur Kerja Aplikasi")
    s1, s2, s3, s4, s5 = st.columns(5)
    steps = [
        ("1", "Upload Citra", "Pengguna mengunggah gambar biji kopi berformat JPG, JPEG, atau PNG."),
        ("2", "Preprocessing", "Gambar diubah ke RGB, resize 224×224, lalu dinormalisasi."),
        ("3", "Ekstraksi Pola", "Vision Transformer membaca pola warna, tekstur, dan area visual biji kopi."),
        ("4", "Prediksi", "Model menghasilkan probabilitas untuk setiap kelas roasting."),
        ("5", "Interpretasi", "Sistem menampilkan kelas akhir dan pembahasan singkat."),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3, s4, s5], steps):
        with col:
            st.markdown(f"""
            <div class="step-box">
                <div class="step-number">{num}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("""
        <div class="dark-card">
        <h3>Ringkasan Sistem</h3>
        <p>
        Sistem ini dibuat untuk membantu proses identifikasi tingkat roasting biji kopi secara otomatis berdasarkan citra digital.
        Pada proses manual, penentuan roasting sering bergantung pada pengamatan visual sehingga hasilnya bisa berbeda antar orang.
        Dengan pendekatan deep learning, sistem dapat memberikan prediksi yang lebih konsisten berdasarkan pola visual pada gambar.
        </p>
        <p>
        Aplikasi ini cocok digunakan sebagai media demonstrasi hasil penelitian karena menampilkan proses input, hasil prediksi,
        nilai confidence, grafik probabilitas, serta penjelasan singkat dari kelas yang diprediksi.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="dark-card">
        <h3>Kenapa Vision Transformer?</h3>
        <ul>
            <li>Mampu mempelajari hubungan global pada citra.</li>
            <li>Membaca gambar melalui pembagian patch.</li>
            <li>Cocok untuk klasifikasi gambar berbasis pola visual.</li>
            <li>Dapat digunakan untuk membedakan karakter warna dan tekstur roasting.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Karakteristik Visual Kelas Roasting")
    r1, r2, r3 = st.columns(3)
    for col, key in zip([r1, r2, r3], ["Light", "Medium", "Dark"]):
        info = ROAST_INFO[key]
        notes_html = "".join([f"<li>{item}</li>" for item in info["notes"]])
        with col:
            st.markdown(f"""
            <div class="dark-card">
                <h3>{info['emoji']} {info['title']}</h3>
                <p>{info['desc']}</p>
                <ul>{notes_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.info("Buat nyoba sistemnya, masuk ke menu **Klasifikasi**, upload gambar biji kopi, lalu lihat hasil prediksi dan probabilitasnya. Jangan lupa gambar harus jelas, bukan foto gelap kayak masa depan kalau TA nggak kelar.")

elif menu == "🔍 Klasifikasi":
    st.markdown("### Upload Citra Biji Kopi")
    st.write("Unggah gambar biji kopi, lalu sistem akan memproses citra dan menampilkan hasil klasifikasi roasting.")

    uploaded_file = st.file_uploader(
        "Pilih gambar biji kopi",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Citra Input")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            st.caption("Pastikan citra tidak terlalu blur, objek biji kopi terlihat jelas, dan pencahayaan cukup.")

        with col2:
            pred_class, confidence, prob_df = predict(image)
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### Hasil Prediksi")
            st.metric("Kelas Roasting", pred_class)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Probabilitas Tiap Kelas")
            st.bar_chart(prob_df.set_index("Kelas"))
            st.dataframe(
                prob_df.style.format({"Probabilitas": "{:.2f}%"}),
                use_container_width=True
            )

            st.markdown("#### Pembahasan Singkat")
            roast_key = pred_class
            if roast_key in ROAST_INFO:
                info = ROAST_INFO[roast_key]
                st.info(f"{info['title']} terdeteksi dengan confidence {confidence:.2f}%. {info['desc']}")
            else:
                st.info("Kelas terdeteksi, tetapi deskripsi kelas belum tersedia.")

            if confidence < 70:
                st.warning("Confidence masih relatif rendah. Coba gunakan gambar yang lebih terang, tajam, dan objek kopinya lebih dominan. Model bukan cenayang, bro.")
            else:
                st.success("Confidence cukup tinggi. Hasil prediksi terlihat meyakinkan berdasarkan gambar yang diunggah.")
    else:
        st.warning("Upload gambar dulu, bro. Masa modelnya disuruh nebak angin doang 😭")
        st.markdown("""
        <div class="dark-card">
        <h3>Tips Foto yang Bagus</h3>
        <ul>
            <li>Gunakan pencahayaan yang cukup.</li>
            <li>Pastikan biji kopi menjadi objek utama pada gambar.</li>
            <li>Hindari gambar blur atau terlalu gelap.</li>
            <li>Gunakan format JPG, JPEG, atau PNG.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif menu == "📊 Informasi Model":
    st.markdown("### Informasi Model dan Evaluasi")

    m1, m2, m3, m4 = st.columns(4)
    for col, label in zip([m1, m2, m3, m4], ["Accuracy", "Precision", "Recall", "F1-Score"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-number">99.83%</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.write("")
    metrics = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Nilai": [0.9983, 0.9983, 0.9983, 0.9983]
    })
    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics.set_index("Metrik"))

    st.markdown("""
    <div class="dark-card">
    <h3>Spesifikasi Singkat</h3>
    <ul>
        <li>Arsitektur model: <b>Vision Transformer Base Patch 16 224</b></li>
        <li>Ukuran input citra: <b>224 × 224 piksel</b></li>
        <li>Jumlah kelas: <b>3 kelas roasting</b></li>
        <li>Evaluasi menggunakan data uji sebanyak <b>599 citra</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Distribusi Dataset")
    dataset_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Jumlah": [2791, 599, 599]
    })
    st.dataframe(dataset_df, use_container_width=True)
    st.bar_chart(dataset_df.set_index("Split"))

elif menu == "📚 Tentang Penelitian":
    st.markdown("### Tentang Penelitian")
    st.markdown("""
    <div class="dark-card">
    <p>
    Penelitian ini berfokus pada klasifikasi tingkat roasting biji kopi menggunakan pendekatan deep learning.
    Permasalahan utama yang diangkat adalah proses identifikasi tingkat roasting yang masih sering dilakukan secara manual
    berdasarkan pengamatan visual, sehingga berpotensi menimbulkan subjektivitas dan inkonsistensi.
    </p>
    <p>
    Vision Transformer digunakan karena mampu mempelajari hubungan global antarbagian citra melalui mekanisme self-attention.
    Citra biji kopi diproses menjadi patch, kemudian dipelajari oleh model untuk mengenali perbedaan karakteristik visual
    pada kelas Light Roast, Medium Roast, dan Dark Roast.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Tujuan Sistem")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown('<div class="card"><h3>Otomatisasi</h3><p>Membantu proses identifikasi roasting secara otomatis dari citra digital.</p></div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="card"><h3>Konsistensi</h3><p>Mengurangi subjektivitas penilaian visual yang dilakukan secara manual.</p></div>', unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="card"><h3>Implementasi</h3><p>Menyediakan web demo yang mudah digunakan untuk menguji model klasifikasi.</p></div>', unsafe_allow_html=True)

    st.markdown("### Alur Sistem")
    st.markdown("""
    1. Pengguna mengunggah citra biji kopi.
    2. Citra diubah ukurannya menjadi 224 × 224 piksel.
    3. Citra dinormalisasi sesuai standar ImageNet.
    4. Model Vision Transformer melakukan prediksi.
    5. Sistem menampilkan kelas prediksi, confidence, grafik probabilitas, dan pembahasan singkat.
    """)

st.divider()
st.caption("Tugas Akhir - Klasifikasi Tingkat Roasting Biji Kopi Menggunakan Vision Transformer")
