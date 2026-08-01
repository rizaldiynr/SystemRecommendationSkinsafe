import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. KONFIGURASI HALAMAN (BRANDING)
# ==========================================
st.set_page_config(
    page_title="GlowScout - Find Your Perfect Skincare Match",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. FUNGSI LOADER DATA
# ==========================================
@st.cache_resource
def load_data():
    file_path = 'fitur_skincare_lengkap.pkl'
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None


# ==========================================
# 3. FUNGSI GAMBAR ESTETIK
# ==========================================
def tampilkan_gambar_aman(url_gambar, lebar=150):
    placeholder_cantik = "https://cdn-icons-png.flaticon.com/512/2666/2666030.png"
   
    if isinstance(url_gambar, str) and len(url_gambar) > 10:
        try:
            st.image(url_gambar, width=lebar)
        except:
            st.image(placeholder_cantik, width=lebar, caption="Preview N/A")
    else:
        st.image(placeholder_cantik, width=lebar, caption="Preview N/A")


# ==========================================
# 4. LOAD DATA
# ==========================================
data_skripsi = load_data()

if data_skripsi is None:
    st.error("⚠️ Server Error: Database produk tidak ditemukan.")
    st.stop()

df = data_skripsi['dataframe']
fitur_gambar = data_skripsi['features_image']
fitur_teks = data_skripsi['features_text']


# ==========================================
# 5. SIDEBAR (TAMPILAN TIDAK DIUBAH)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3163/3163212.png", width=70)
    st.markdown("### **GlowScout**")
    st.caption("Teman pencari skincare kamu")
   
    st.markdown("---")
   
    st.write("**🔍 Metode Rekomendasi**")
    st.caption("Sistem menggunakan pendekatan Hybrid:")
   
    # ⚠️ disesuaikan dengan backend
    st.info("⚖️ **Bobot 20% Visual + 80% Teks (Hybrid Optimized)**")

    st.markdown("---")
    with st.expander("Tentang Aplikasi"):
        st.caption("Platform pencarian skincare berbasis analisis kandungan dan kemasan.")


# ==========================================
# 6. HALAMAN UTAMA
# ==========================================
st.title("Temukan Skincare yang Cocok untukmu ✨")

st.markdown("""
<style>
.big-font { font-size:18px !important; color: #555; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="big-font">Cari produk favoritmu, lalu temukan alternatif lain dengan kandungan yang mirip dan cocok untuk kebutuhan kulitmu.</p>',
    unsafe_allow_html=True
)

st.divider()


# --- BAGIAN 1: PENCARIAN ---
col_cari1, col_cari2 = st.columns([1, 2])

with col_cari1:
    st.subheader("1. Cari Brand / Produk")
    keyword = st.text_input("Ketik nama produk...", placeholder="Misal: The Ordinary, Laneige...")

produk_pilihan = None

if keyword:
    hasil_cari = df[df['product_name'].str.contains(keyword, case=False, na=False)]
   
    if len(hasil_cari) > 0:
        with col_cari2:
            st.subheader("2. Pilih Varian")
            nama_produk_pilihan = st.selectbox(
                f"Ditemukan {len(hasil_cari)} hasil pencarian:",
                hasil_cari['product_name'].tolist()
            )
            produk_pilihan = df[df['product_name'] == nama_produk_pilihan].iloc[0]
    else:
        st.info(f"Produk '{keyword}' belum tersedia di katalog kami. Coba kata kunci lain.")


# --- BAGIAN 2: PRODUK PILIHAN ---
if produk_pilihan is not None:
    st.divider()
   
    container = st.container(border=True)
    with container:
        c1, c2 = st.columns([1, 3])
       
        with c1:
            tampilkan_gambar_aman(produk_pilihan['image_url'], lebar=220)
           
        with c2:
            st.caption("PRODUK YANG ANDA CARI")
            st.markdown(f"## {produk_pilihan['product_name']}")
            st.markdown(f"**Brand:** {produk_pilihan['brand']} | **Kategori:** {produk_pilihan['category']}")
           
            url_target = produk_pilihan['product_url']
            if isinstance(url_target, str) and len(url_target) > 5:
                st.link_button("🛒 Beli / Cek Harga", url_target)
            else:
                st.button("🚫 Stok Habis", disabled=True)

            with st.expander("Lihat Detail Kandungan (Ingredients)"):
                st.write(produk_pilihan.get('ingredients', 'Informasi tidak tersedia'))


    # --- BAGIAN 3: TOMBOL ACTION ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        tombol_cari = st.button(
            "✨ TEMUKAN PRODUK SERUPA ✨",
            type="primary",
            use_container_width=True
        )


    # --- BAGIAN 4: HASIL REKOMENDASI ---
    if tombol_cari:
        st.divider()
        st.subheader("Rekomendasi Untuk Anda ❤️")
        st.markdown("Berikut beberapa produk dengan kemiripan kandungan dan karakteristik:")
       
        with st.spinner('Sedang mencocokkan kandungan & kemasan...'):
            try:
                idx = df.index.get_loc(produk_pilihan.name)
            except:
                temp_idx = df[df['product_name'] == produk_pilihan['product_name']].index[0]
                idx = df.index.get_loc(temp_idx)

            # ==========================================
            # 🔥 LOGIC BARU (SAMA DENGAN COLAB)
            # ==========================================
            sim_v = cosine_similarity([fitur_gambar[idx]], fitur_gambar).flatten()
            sim_t = cosine_similarity([fitur_teks[idx]], fitur_teks).flatten()

            # Normalisasi
            sim_v = (sim_v - sim_v.min()) / (sim_v.max() - sim_v.min() + 1e-9)
            sim_t = (sim_t - sim_t.min()) / (sim_t.max() - sim_t.min() + 1e-9)

            # Hybrid
            skor_final = (sim_v * 0.2) + (sim_t * 0.8)

            # Boost keyword
            keyword_boost = produk_pilihan['product_name'].split()[0].lower()
            bonus = df['product_name'].str.contains(keyword_boost, case=False, na=False).astype(int)
            skor_final = skor_final + (bonus * 0.05)

            # Filter kategori
            kategori_target = produk_pilihan['category']
            urutan = skor_final.argsort()[::-1]
            filtered = [i for i in urutan if df.iloc[i]['category'] == kategori_target]

            top_indices = filtered[1:6]

            # ==========================================

            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(5)
           
            for i, col in enumerate(cols):
                idx_rek = top_indices[i]
                row = df.iloc[idx_rek]
                score = skor_final[idx_rek]
               
                with col:
                    with st.container(border=True):
                        tampilkan_gambar_aman(row['image_url'], lebar=120)
                        st.markdown(f"**{score*100:.0f}% Match** 🔥")
                       
                        nama_full = row['product_name']
                        nama_pendek = nama_full[:35] + ".." if len(nama_full) > 35 else nama_full
                       
                        st.markdown(f"**{row['brand']}**")
                        st.caption(nama_pendek)
                       
                        if isinstance(row['product_url'], str) and len(row['product_url']) > 5:
                            st.link_button("🔗 Lihat Detail", row['product_url'], use_container_width=True)
                        else:
                            st.button("🚫 Stok Habis", disabled=True, key=f"btn_{i}", use_container_width=True)