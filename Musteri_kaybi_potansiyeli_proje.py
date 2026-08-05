import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans

np.random.seed(42)

tarihler = pd.date_range(start='2026-01-01', end='2026-12-31', freq='1H')
df = pd.DataFrame(index=tarihler)

df["Gun_Ici_Girilen_Sefer"] = np.random.randint(0, 30, size=len(df))
df["Izlenilen_Film_Sayisi"] = np.random.randint(0, 10, size=len(df))
df["Yapilan_Sikayet_Sayisi"] = np.random.randint(0, 5, size=len(df))
df["Gun_Ici_Gecirilen_Saat"] = np.random.randint(0, 24, size=len(df))

musteri_kaybi_potansiyeli = (df["Gun_Ici_Girilen_Sefer"] * 0.1 + df["Izlenilen_Film_Sayisi"] * 0.2 + df["Yapilan_Sikayet_Sayisi"] * 2 + df["Gun_Ici_Gecirilen_Saat"] * 0.4)
df["Musteri_Kaybi_Potansiyeli"] = musteri_kaybi_potansiyeli + np.random.normal(0, 5, size=len(df))

#Anomali Tespit_Esigi
df.loc[100, "Musteri_Kaybi_Potansiyeli"] = 50  # Örnek bir anomali eklemek için]

dosya_adi = "musteri_kaybi_potansiyeli.csv"
df.to_csv(dosya_adi, index=True)
print(f"Veri başarıyla '{dosya_adi}' dosyasına kaydedildi.")

# Feature
df["Gun_Ici_Sefer_Carpim_Saat"] = df["Gun_Ici_Girilen_Sefer"] * df["Gun_Ici_Gecirilen_Saat"]

#Lag
df["Musteri_Kaybi_Potansiyeli_Lag1"] = df["Musteri_Kaybi_Potansiyeli"].shift(1)
df["Gun_Ici_Gecirilen_Saat_Lag24"] = df["Gun_Ici_Gecirilen_Saat"].shift(24) # 24 saatlik lag

#Rol
df["Musteri_Kaybi_Potansiyeli_Rolling_Mean_3"] = df["Musteri_Kaybi_Potansiyeli"].rolling(window=3).mean()

df = df.dropna()  # Lag ve rolling işlemlerinden dolayı oluşan NaN değerleri temizle

#Kumeleme
kmeans = KMeans(n_clusters=3, random_state=42)
kume_etiketleri = kmeans.fit_predict(
    df[["Gun_Ici_Girilen_Sefer", "Gun_Ici_Sefer_Carpim_Saat", "Yapilan_Sikayet_Sayisi", "Izlenilen_Film_Sayisi"]]
)
df["Musteri_Kaybi_Potansiyeli_Kume"] = kume_etiketleri.astype("int64")

feature_cols = [
    "Gun_Ici_Girilen_Sefer",
    "Izlenilen_Film_Sayisi",
    "Yapilan_Sikayet_Sayisi",
    "Gun_Ici_Gecirilen_Saat",
    "Musteri_Kaybi_Potansiyeli_Lag1",
    "Gun_Ici_Gecirilen_Saat_Lag24",
    "Musteri_Kaybi_Potansiyeli_Rolling_Mean_3",
    "Musteri_Kaybi_Potansiyeli_Kume",
    "Gun_Ici_Sefer_Carpim_Saat"
]

X = df[feature_cols]
y = df["Musteri_Kaybi_Potansiyeli"]

egitim_orani = int(0.8 * len(df))

X_train, X_test = X[:egitim_orani], X[egitim_orani:]
y_train, y_test = y[:egitim_orani], y[egitim_orani:]

import lightgbm as lgb

model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

hatalar = y_test - y_pred
esik_degeri = hatalar.mean() + (3 * hatalar.std())
anomaliler = hatalar > esik_degeri

sonuc_tablosu = df.iloc[egitim_orani:].copy()
sonuc_tablosu["Modelin_Beklentisi"] = np.around(y_pred, 1)
sonuc_tablosu["Anomali Mi"] = anomaliler

print("---Model Basari Karnesi---")
print(f"Ortalama Sapma(MAE): {mae}")
print(f"Basari Skoru (R^2): {r2:.4f}\n")

sonuc_tablosu['Rapor'] = np.where(
    sonuc_tablosu["Anomali Mi"] & (sonuc_tablosu["Musteri_Kaybi_Potansiyeli"] < 20),
    "Yüksek Müşteri Kaybı Potansiyeli",
    np.where(
        sonuc_tablosu["Anomali Mi"] & (sonuc_tablosu["Musteri_Kaybi_Potansiyeli"] >= 20),
        "Aşırı Müşteri Kaybı Potansiyeli",
        "Normal"
    )
)

yakalananlar = sonuc_tablosu[sonuc_tablosu["Anomali Mi"] == True]
print("Yakalanan Anomaliler:")
for index, row in yakalananlar.iterrows():
    print(f"Tarih: {index}, Müşteri Kaybı Potansiyeli: {row['Musteri_Kaybi_Potansiyeli']}, Modelin Beklentisi: {row['Modelin_Beklentisi']}, Rapor: {row['Rapor']}") 


izlenilen_film = row["Izlenilen_Film_Sayisi"]
yapilan_sikayet = row["Yapilan_Sikayet_Sayisi"]
gun_ici_gecirilen_saat = row["Gun_Ici_Gecirilen_Saat"]
musteri_kaybi_potansiyeli_lag1 = row["Musteri_Kaybi_Potansiyeli_Lag1"]
musteri_kaybi_potansiyeli_lag24 = row["Gun_Ici_Gecirilen_Saat_Lag24"]
musteri_kaybi_potansiyeli_rolling_mean_3 = row["Musteri_Kaybi_Potansiyeli_Rolling_Mean_3"]
mmusteri_kaybi_potansiyeli_kume = row["Musteri_Kaybi_Potansiyeli_Kume"]
musteri_kaybi_potansiyeli_carpim_saat = row["Gun_Ici_Sefer_Carpim_Saat"]

for gun_ici_sefer in range(0, 30):
    test_df = pd.DataFrame({
        "Gun_Ici_Girilen_Sefer": [gun_ici_sefer],
        "Izlenilen_Film_Sayisi": [izlenilen_film],
        "Yapilan_Sikayet_Sayisi": [yapilan_sikayet],
        "Gun_Ici_Gecirilen_Saat": [gun_ici_gecirilen_saat],
        "Musteri_Kaybi_Potansiyeli_Lag1": [musteri_kaybi_potansiyeli_lag1],
        "Gun_Ici_Gecirilen_Saat_Lag24": [musteri_kaybi_potansiyeli_lag24],
        "Musteri_Kaybi_Potansiyeli_Rolling_Mean_3": [musteri_kaybi_potansiyeli_rolling_mean_3],
        "Musteri_Kaybi_Potansiyeli_Kume": [mmusteri_kaybi_potansiyeli_kume],
        "Gun_Ici_Sefer_Carpim_Saat": [gun_ici_sefer * gun_ici_gecirilen_saat]
    })

    tahmin_edilen_potansiyel = model.predict(test_df)[0]
    print(f"Sefer Sayısı: {gun_ici_sefer}, Tahmin Edilen Müşteri Kaybı Potansiyeli: {tahmin_edilen_potansiyel:.2f}")

#Akilli Bildirim Sistemi

def potansiyel_bildirim_sistemi(row):
    if row["Anomali Mi"] and row["Musteri_Kaybi_Potansiyeli"] > 20:
        return "🚨 Anomali Tespit Edildi: Müşteri Kaybı Potansiyeli Beklentinin Üzerinde!"

    if row["Anomali Mi"] and row["Gun_Ici_Gecirilen_Saat"] < 2:
        return "⚠️ Anomali Tespit Edildi: Müşteri Kaybı Potansiyeli Beklentinin Altında, Ancak Geçirilen Saat Çok Düşük!"

    if not row["Anomali Mi"] and row["Gun_Ici_Girilen_Sefer"] <= 1:
        return "✅ Anomali Yok: Müşteri Kaybı Potansiyeli Beklentinin Altında, Ancak Geçirilen Sefer Çok Düşük!"

    return "✅ Normal Durum: Müşteri Kaybı Potansiyeli Beklentinin Altında veya Eşit."

sonuc_tablosu["potansiyel_bildirim"] = sonuc_tablosu.apply(potansiyel_bildirim_sistemi, axis=1)
df["potansiyel_bildirim"] = pd.Series([None] * len(df), dtype="object")
df.loc[sonuc_tablosu.index, "potansiyel_bildirim"] = sonuc_tablosu["potansiyel_bildirim"].astype("object").values

def akilli_aksiyon_ve_bildirim_sistemi(row):
  potansiyel = row["Musteri_Kaybi_Potansiyeli"]
  beklenti = row["Modelin_Beklentisi"]

  # Skoru yüzde formatına benzer bir risk dilimine katalım (Örnek mantık)
  if row["Anomali Mi"] and potansiyel > 25:
    return (
        f"🚨 KRİTİK RİSK (Skor: {potansiyel:.1f}): Müşteri kopmak üzere!"
        " 👉 **Aksiyon Önerisi:** Kullanıcıya özel %50 indirimli "
        "yenileme teklifi veya ücretsiz 1 ay abonelik kampanyası bildirimi"
        " gönder."
    )

  elif row["Anomali Mi"] and row["Yapilan_Sikayet_Sayisi"] > 2:
    return (
        f"⚠️ ŞİKAYET PATLAMASI (Skor: {potansiyel:.1f}): Müşteri "
        "mutsuzsuzluktan anomali yaratmış. 👉 **Aksiyon Önerisi:** Çağrı"
        " merkezinden doğrudan arama önceliği tanımla."
    )

  elif not row["Anomali Mi"] and potansiyel < 10:
    return (
        f"✅ DÜŞÜK RİSK (Skor: {potansiyel:.1f}): Müşteri sadık. 👉 **Aksiyon"
        " Önerisi:** Yeni çıkan film önerileri göndererek etkileşimi canlı"
        " tut."
    )

  else:
    return f"ℹ️ NORMAL DURUM (Skor: {potansiyel:.1f}): Standart takibe devam."


# Sonuç tablosuna bu akıllı aksiyonları ekleyelim:
sonuc_tablosu["Aksiyon_Raporu"] = sonuc_tablosu.apply(
    akilli_aksiyon_ve_bildirim_sistemi, axis=1
)

# Örnek bir çıktıyı ekrana basalım
for index, row in sonuc_tablosu[sonuc_tablosu["Anomali Mi"] == False].head(7).iterrows():
  print(f"Tarih: {index}\n{row['Aksiyon_Raporu']}\n" + "-" * 50)