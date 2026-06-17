import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sayfa ayarları
st.set_page_config(page_title="AI Personel Takip Sistemi", page_icon="🕵️‍♂️", layout="wide")

st.title("Personel Devam ve Anormallik Takip Sistemi")
st.markdown("""
Bu proje, personellerin günlük giriş-çıkış saatlerini analiz ederek **Machine Learning (Isolation Forest)** algoritması ile şüpheli/anormal hareketleri tespit etmektedir.
""")

# Veriyi yükleme fonksiyonu
@st.cache_data
def load_data():
    df = pd.read_csv("personel_verileri.csv")
    return df

df = load_data()

# Üst Kısım: Temel Metrikler (KPIs)
toplam_kayit = len(df)
anormal_kayit_sayisi = len(df[df['durum_analizi'] == 'ŞÜPHELİ/ANORMAL'])
normal_kayit_sayisi = toplam_kayit - anormal_kayit_sayisi

col1, col2, col3 = st.columns(3)
col1.metric(label="Toplam Giriş-Çıkış Logu", value=toplam_kayit)
col2.metric(label="Normal Tespit", value=normal_kayit_sayisi)
col3.metric(label="Şüpheli/Anormal Tespit", value=anormal_kayit_sayisi, delta="- Güvenlik Uyarısı", delta_color="inverse")

st.divider()

# Orta Kısım: Görselleştirme (Grafik)
st.subheader("Giriş Saatleri ve Çalışma Süresi Dağılımı")
st.write("Aşağıdaki grafikte makine öğrenmesi modelinin bulduğu şüpheli hareketler (gece mesaisi, hafta sonu veya çok kısa çalışmalar) kırmızı renkle izole edilmiştir.")

# Matplotlib ile basit bir Scatter Plot (Dağılım Grafiği)
fig, ax = plt.subplots(figsize=(10, 4))
normal_df = df[df['durum_analizi'] == 'Normal']
anormal_df = df[df['durum_analizi'] == 'ŞÜPHELİ/ANORMAL']

ax.scatter(normal_df['giris_saati_sayisal'], normal_df['calisma_suresi_dakika'], color='blue', label='Normal', alpha=0.6)
ax.scatter(anormal_df['giris_saati_sayisal'], anormal_df['calisma_suresi_dakika'], color='red', label='Anormal/Şüpheli', marker='x', s=100)

ax.set_xlabel('Sayısal Giriş Saati (Örn: 8.5 = 08:30)')
ax.set_ylabel('Çalışma Süresi (Dakika)')
ax.legend()
st.pyplot(fig)

st.divider()

# Alt Kısım: Veri Tabloları
colA, colB = st.columns(2)

with colA:
    st.subheader("Tespit Edilen Şüpheli Kayıtlar")
    st.dataframe(anormal_df[['personel_id', 'tarih', 'durum', 'durum_analizi']], use_container_width=True)

with colB:
    st.subheader("Normal Kayıtlar (Son 50)")
    st.dataframe(normal_df[['personel_id', 'tarih', 'durum', 'durum_analizi']].head(50), use_container_width=True)

st.caption("Geliştirici: Yusuf Alper Gülden | Python, Scikit-learn, Pandas, Streamlit kullanılarak hazırlanmıştır.")
