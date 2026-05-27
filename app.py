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

SAMPLE_IMAGES = {
    "Light": "light (2).png",
    "Medium": "medium (5).png",
    "Dark": "dark (2).png",
}

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
    --bg:#0B0D12;
    --panel:#141820;
    --panel-2:#19130F;
    --coffee:#C47A3D;
    --coffee-2:#F2B66D;
    --cream:#F8E7D1;
    --muted:#BBA895;
    --line:rgba(255,255,255,.10);
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
}

.stApp{
    background:
        radial-gradient(circle at 8% 0%, rgba(196,122,61,.25), transparent 28%),
        radial-gradient(circle at 90% 12%, rgba(242,182,109,.12), transparent 26%),
        linear-gradient(135deg, #090B10 0%, #11151D 48%, #120D0A 100%);
    color:#F8E7D1;
}

.block-container{
    max-width:1180px;
    padding-top:2rem;
    padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#171A23 0%,#20242E 100%);
    border-right:1px solid var(--line);
}

section[data-testid="stSidebar"] *{
    color:#F6EBDD;
}

[data-testid="stSidebar"] .stRadio label{
    font-weight:700;
}

h1,h2,h3,h4{
    color:#FFF4E6;
    letter-spacing:-.03em;
}

p, li{
    font-size:15px;
    line-height:1.7;
}

.hero{
    padding:34px 36px;
    border-radius:30px;
    background:
        linear-gradient(135deg, rgba(82,48,28,.92), rgba(18,21,28,.94)),
        radial-gradient(circle at 80% 20%, rgba(242,182,109,.22), transparent 30%);
    border:1px solid rgba(246,193,119,.20);
    box-shadow:0 24px 70px rgba(0,0,0,.28);
    margin-bottom:26px;
}

.hero-grid{
    display:grid;
    grid-template-columns:1.45fr .75fr;
    gap:26px;
    align-items:center;
}

.hero-badge{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:8px 14px;
    border-radius:999px;
    background:rgba(246,193,119,.13);
    color:#F6C177;
    font-weight:800;
    font-size:13px;
    border:1px solid rgba(246,193,119,.24);
    margin-bottom:14px;
}

.main-title{
    font-size:clamp(34px, 5vw, 54px);
    line-height:1.05;
    font-weight:900;
    letter-spacing:-1.8px;
    background:linear-gradient(90deg,#FFE4B8,#F2B66D,#B77945);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:14px;
}

.subtitle{
    font-size:17px;
    color:#D8C3AF;
    max-width:820px;
    line-height:1.75;
}

.hero-side{
    padding:22px;
    border-radius:24px;
    background:rgba(255,255,255,.055);
    border:1px solid rgba(255,255,255,.11);
}

.hero-side .big{
    font-size:40px;
    font-weight:900;
    color:#F6C177;
    line-height:1;
    white-space:nowrap;
}

.hero-side .label{
    margin-top:8px;
    color:#D6C7B8;
    font-weight:700;
}

.section-title{
    margin:26px 0 14px 0;
    font-size:28px;
    font-weight:900;
    color:#FFF4E6;
}

.metric-card{
    min-height:132px;
    padding:20px 16px;
    border-radius:24px;
    background:linear-gradient(145deg, rgba(255,255,255,.075), rgba(246,193,119,.11));
    border:1px solid rgba(246,193,119,.18);
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
    box-shadow:0 14px 34px rgba(0,0,0,.20);
}

.metric-number{
    font-size:clamp(26px, 3vw, 38px);
    line-height:1.05;
    font-weight:900;
    color:#FFC477;
    white-space:nowrap;
}

.metric-label{
    margin-top:10px;
    color:#D8C3AF;
    font-size:14px;
    font-weight:800;
    white-space:nowrap;
}

.info-card{
    min-height:300px;
    padding:26px 26px 24px 26px;
    border-radius:28px;
    background:linear-gradient(145deg, rgba(255,248,238,.98), rgba(224,203,181,.96));
    border:1px solid rgba(255,255,255,.55);
    box-shadow:0 16px 36px rgba(0,0,0,.22);
    color:#2B190F;
    display:flex;
    flex-direction:column;
}

.info-card h3{
    color:#2B190F;
    font-size:24px;
    margin:0 0 16px 0;
    white-space:nowrap;
}

.info-card p,
.info-card li{
    color:#5A412F;
    font-size:15px;
    line-height:1.75;
}

.info-card ul{
    margin-top:6px;
    padding-left:20px;
}

.dark-card{
    min-height:100%;
    padding:25px;
    border-radius:26px;
    background:linear-gradient(145deg, rgba(255,255,255,.075), rgba(255,255,255,.038));
    border:1px solid rgba(255,255,255,.10);
    box-shadow:0 16px 34px rgba(0,0,0,.20);
}

.dark-card h3,
.dark-card h4{
    color:#FFF1DD;
    margin-top:0;
}

.dark-card p,
.dark-card li{
    color:#D3C2B1;
}

.step-box{
    min-height:210px;
    padding:20px;
    border-radius:24px;
    background:linear-gradient(145deg, rgba(255,255,255,.070), rgba(246,193,119,.055));
    border:1px solid rgba(255,255,255,.095);
    box-shadow:0 12px 28px rgba(0,0,0,.16);
}

.step-number{
    width:38px;
    height:38px;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:#F6C177;
    color:#23140B;
    font-weight:900;
    margin-bottom:12px;
}

.step-box h4{
    margin:0 0 8px 0;
    font-size:16px;
    color:#FFF1DD;
}

.step-box p{
    margin:0;
    color:#D1BDAA;
    font-size:13.5px;
    line-height:1.65;
}

.roast-card{
    min-height:245px;
    padding:24px;
    border-radius:26px;
    background:linear-gradient(145deg, rgba(28,31,40,.95), rgba(25,18,13,.95));
    border:1px solid rgba(246,193,119,.14);
    box-shadow:0 16px 34px rgba(0,0,0,.20);
}

.roast-card h3{
    color:#FFE9C8;
    margin-top:0;
}

.roast-card p,
.roast-card li{
    color:#D6C4B4;
}

.result-box{
    padding:26px;
    border-radius:24px;
    background:linear-gradient(145deg, rgba(255,247,237,.98), rgba(246,193,119,.18));
    border:2px solid rgba(246,193,119,.58);
    box-shadow:0 16px 40px rgba(0,0,0,.22);
    color:#2D1B12;
}

.small-muted{
    color:#A99A8B;
    font-size:14px;
}

hr{
    border-color:rgba(255,255,255,.09)!important;
}

@media (max-width: 900px){
    .hero-grid{grid-template-columns:1fr;}
    .metric-card{min-height:112px;}
    .info-card,.step-box,.roast-card{min-height:auto;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-grid">
        <div>
            <div class="hero-badge">☕ Web Demo Tugas Akhir • Vision Transformer</div>
            <div class="main-title">Klasifikasi Tingkat Roasting Biji Kopi</div>
            <div class="subtitle">
                Sistem ini menerapkan model <b>Vision Transformer</b> untuk mengenali tingkat roasting biji kopi melalui citra digital.
                Pengguna cukup mengunggah gambar biji kopi, lalu sistem menampilkan kelas prediksi, nilai confidence, dan probabilitas tiap kelas.
            </div>
        </div>
        <div class="hero-side">
            <div class="big">99.83%</div>
            <div class="label">Akurasi Model</div>
            <p style="margin-top:14px;color:#CDBBAA;font-size:14px;line-height:1.6;">
                Model mengenali tiga kelas roasting: Light, Medium, dan Dark Roast.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "🔍 Klasifikasi", "📊 Informasi Model", "📚 Tentang Penelitian"]
)

st.sidebar.markdown("---")
st.sidebar.success("Model: Vision Transformer")

st.sidebar.markdown("""
### 🧠 Sejarah Singkat

Vision Transformer (ViT) diperkenalkan oleh Google Research pada tahun 2020.  
Model ini memproses citra dalam bentuk patch dan menggunakan mekanisme self-attention untuk mengenali pola visual.
""")

st.sidebar.markdown("---")
st.sidebar.caption("Tips: gunakan foto biji kopi yang jelas, cukup cahaya, dan tidak blur. Jangan upload foto mantan, modelnya trauma.")

if menu == "🏠 Beranda":
    st.markdown('<div class="section-title">Gambaran Umum Sistem</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics_home = [
        ("3.989", "Total Citra"),
        ("3", "Kelas Roasting"),
        ("224×224 px", "Ukuran Input"),
        ("99.83%", "Akurasi Model"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4], metrics_home):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{num}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📦 Dataset</h3>
            <p><b>Total citra:</b> 3.989</p>
            <p><b>Train:</b> 2.791<br><b>Validation:</b> 599<br><b>Test:</b> 599</p>
            <p>Dataset digunakan untuk proses pelatihan, validasi, dan pengujian performa model klasifikasi.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Metode</h3>
            <p>Model menggunakan <b>Vision Transformer</b> dengan pendekatan transfer learning.</p>
            <p>Citra diubah menjadi patch, lalu pola visualnya dipelajari melalui mekanisme <i>self-attention</i>.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>🏷️ Kelas</h3>
            <p>Sistem mengenali tiga tingkat roasting biji kopi:</p>
            <ul>
                <li><b>Light Roast</b></li>
                <li><b>Medium Roast</b></li>
                <li><b>Dark Roast</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Alur Kerja Aplikasi</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    steps = [
        ("1", "Upload Citra", "Pengguna mengunggah gambar biji kopi berformat JPG, JPEG, atau PNG."),
        ("2", "Preprocessing", "Gambar diubah ke RGB, resize 224×224, lalu dinormalisasi."),
        ("3", "Ekstraksi Pola", "ViT membaca pola warna, tekstur, dan karakter visual biji kopi."),
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

    st.markdown('<div class="section-title">Ringkasan dan Keunggulan</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("""
        <div class="dark-card">
            <h3>Ringkasan Sistem</h3>
            <p>
                Sistem ini dibuat untuk membantu identifikasi tingkat roasting biji kopi secara otomatis berdasarkan citra digital.
                Pada proses manual, penentuan tingkat roasting sering bergantung pada pengamatan visual sehingga hasilnya dapat berbeda antar pengamat.
            </p>
            <p>
                Melalui pendekatan deep learning, aplikasi ini memberikan prediksi yang lebih konsisten dengan menampilkan kelas roasting,
                nilai confidence, grafik probabilitas, serta pembahasan singkat dari hasil prediksi.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="dark-card">
            <h3>Kenapa Vision Transformer?</h3>
            <ul>
                <li>Mempelajari hubungan global antar area citra.</li>
                <li>Membaca gambar melalui pembagian patch.</li>
                <li>Cocok untuk klasifikasi berbasis pola visual.</li>
                <li>Membedakan warna dan tekstur roasting.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Karakteristik Visual Kelas Roasting</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    for col, key in zip([r1, r2, r3], ["Light", "Medium", "Dark"]):
        info = ROAST_INFO[key]
        notes_html = "".join([f"<li>{item}</li>" for item in info["notes"]])
        with col:
            st.markdown(f"""
            <div class="roast-card">
                <h3>{info['emoji']} {info['title']}</h3>
                <p>{info['desc']}</p>
                <ul>{notes_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.info("Masuk ke menu **Klasifikasi**, upload gambar biji kopi, lalu lihat hasil prediksi dan probabilitasnya. Gambar harus jelas, jangan foto gelap kayak masa depan kalau TA nggak kelar.")

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
    metric_cards = [
        ("99.83%", "Accuracy"),
        ("599", "Data Uji"),
        ("3", "Kelas Roasting"),
        ("ViT", "Arsitektur Model"),
    ]
    for col, (num, label) in zip([m1, m2, m3, m4], metric_cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-number">{num}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown("### Classification Report per Kelas")

    report_df = pd.DataFrame({
        "Kelas": ["Dark", "Light", "Medium", "Macro Avg", "Weighted Avg"],
        "Precision": [1.0000, 1.0000, 0.9944, 0.9981, 0.9983],
        "Recall": [0.9955, 1.0000, 1.0000, 0.9985, 0.9983],
        "F1-Score": [0.9977, 1.0000, 0.9972, 0.9983, 0.9983],
        "Support": [221, 199, 179, 599, 599]
    })

    st.dataframe(
        report_df.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Support": "{:.0f}"
        }),
        use_container_width=True
    )

    st.caption(
        "Nilai precision, recall, dan F1-score ditampilkan per kelas agar evaluasi tidak hanya bergantung pada nilai global."
    )

    st.markdown("### Contoh Citra Biji Kopi per Kelas")
    img1, img2, img3 = st.columns(3)
    for col, key in zip([img1, img2, img3], ["Light", "Medium", "Dark"]):
        with col:
            st.image(SAMPLE_IMAGES[key], caption=f"Contoh {key} Roast", use_container_width=True)
            st.markdown(f"**{ROAST_INFO[key]['title']}**")
            st.write(ROAST_INFO[key]["desc"])

    st.markdown("### Ringkasan Metrik Global")
    metrics = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision Weighted", "Recall Weighted", "F1-Score Weighted"],
        "Nilai": [0.9983, 0.9983, 0.9983, 0.9983]
    })
    st.dataframe(metrics, use_container_width=True)

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
