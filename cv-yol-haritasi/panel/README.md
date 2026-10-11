# panel/ — Kontrol paneli nasıl çalışıyor?

```bash
python panel/app.py              # http://127.0.0.1:7860  (sadece senin bilgisayarından erişilir)
python panel/app.py --port 8000
```

## Dosya → görevi

| Dosya | Görevi |
|---|---|
| `app.py` | Gradio arayüzünü kurar: bölüm sekmeleri, proje alt sekmeleri, Olay Deposu, İşler. |
| `kayit.py` | **Tek liste**: panelde hangi proje dosyası/sınıfı hangi sırayla görünecek. |
| `otomatik_form.py` | `Ayarlar` dataclass'ı → slider/checkbox/dropdown. (`bilesen_tanimi` saf fonksiyon, test edilir.) |
| `isler.py` | Uzun işleri (eğitim) ayrı süreçte başlatır, log'u `veri/isler/<iş>/log.txt`'e yazar. |
| `test_panel.py` | Form eşlemesi, bütün projelerin hatasız yüklenmesi, panelin kurulabilmesi. |

## Bir butona bastığında ne oluyor?

**Kare işleyen projede ("İşle"):**

```
Tarayıcı (RGB görüntü + ayar değerleri)
  └▶ app.py: tek_kare()
       ├─ ayarlari_olustur(Ayarlar, değerler)      # ortak/arayuz.py — tipleri düzeltir
       ├─ proje = gr.State'teki örnek (yoksa yeni)  # takip izleri gibi durum burada yaşar
       ├─ proje.isle(rgb2bgr(görüntü), ayarlar, anlat)
       ├─ (isteğe bağlı) OlayDeposu().toplu_kaydet(sonuc.olaylar)
       └─ sonucu_panele_cevir(sonuc, anlat)        # 6 kutuya dağıtır
```

**Video sekmesi:** Aynı `isle()` her kare için çağrılır; işlenmiş video indirilebilir dosya olarak verilir.
Her 15 karede ara durum güncellenir. **Canlı sekmesi:** tarayıcı kamerasından gelen her kare aynı fonksiyondan geçer.

**Eylem (tek seferlik iş):** `UZUN_SURER=False` ise doğrudan `calistir()` çağrılır.
`UZUN_SURER=True` ise `isler.py` şu komutu ayrı süreçte çalıştırır ve log'u canlı gösterir:

```bash
python <proje_dosyası> <ayarlardan üretilmiş argümanlar> --cikti-klasor veri/isler/<iş> --anlat
```

Panelde gösterilen **"Eşdeğer komut"** budur — terminale yapıştırırsan aynı sonucu alırsın.

## Yeni proje eklemek

1. Bir dosya yaz: `Ayarlar` dataclass'ı (`ayar(...)` alanlarıyla) + `KareIsleyici` veya `Eylem` alt sınıfı.
   Sınıfta `ISIM`, `BOLUM` ("03" gibi), `DOSYA` (köke göre yol), `ACIKLAMA` doldur.
2. `kayit.py` → `PROJELER` listesine `("klasor/dosya.py", "SinifAdi")` ekle.
3. `pytest panel` — yeni projenin yüklendiğini ve `DOSYA` alanının doğru olduğunu kontrol eder.

## Sık sorunlar

- **Sekmede "⚠ yüklenemedi"**: dosyada sözdizimi/import hatası var; traceback sekmede yazar.
- **"Bu adım için 'X' paketi gerekli"**: o bölümün `requirements.txt`'ini kur.
- **Canlı webcam çalışmıyor**: tarayıcıya kamera izni ver; `127.0.0.1` dışında bir adresten açıyorsan tarayıcılar HTTPS ister.
