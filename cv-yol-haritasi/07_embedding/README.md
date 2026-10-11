# 07 · Embedding — Görsel hafıza ve "hangi one-shot yöntemi ne zaman?"

**Embedding** bir görüntüyü (CLIP'te metni de) bir vektöre çevirir: benzer şeyler uzayda yakın düşer. Bu bölümde:
1. **Kişisel fotoğraf hafızası**: Galerini indeksle, "a dog on the beach" yazıp bul. Hiç etiketleme yok.
2. **One-shot yöntem yarışı**: v1 one-shot fikrini bilimsel olarak değerlendir. Renk histogramı, ImageNet
   özellikleri, DINOv2, CLIP, YOLO-World ve kendi YOLO modelin aynı testte **ROC/AUC** ve hızla karşılaştırılır.

## Kavramlar

| Gömücü | Boyut | "Benzer" ne demek? | Metinle arama |
|---|---|---|---|
| `renk` | 512 | aynı renk dağılımı | ✗ |
| `yolo_cls` | 256 | aynı tür nesne (ImageNet sınıf bilgisi) | ✗ |
| `dinov2` | 384 | aynı nesne / görsel ayrıntı (kendi kendine denetimli) | ✗ |
| `clip` | 512 | aynı **anlam** (görüntü + metin ortak uzayı) | ✓ |

- **Kosinüs benzerliği**: Normalize vektörlerde iç çarpım. Arama = `V @ q` → en büyük k değer (`vektor_indeksi.py`).
- **FAISS**: Aynı matematik, milyonlarca vektörde hızlı. `faiss_kullan=True` ile sonuçların aynı olduğunu görebilirsin.
- **ROC / AUC**: Eşikten bağımsız ayırma gücü. AUC 1 = mükemmel, 0.5 = yazı tura (`oneshot_karsilastir.roc_auc`).

## Veri akışı

```
foto klasörü ─▶ indeksle.py ─▶ gomucu.goruntu() ─▶ VektorIndeksi ─▶ veri/indeksler/<isim>_<gömücü>.npz
                                                                         │
"a dog on the beach" ─▶ ara.py ─▶ gomucu.metin() ─▶ q ─────────────▶ V @ q ─▶ en benzer k fotoğraf
örnek fotoğraf ─────────▶ ara.py ─▶ gomucu.goruntu() ─▶ q ────────────┘

referans (6 foto) + test (pozitif/negatif) ─▶ oneshot_karsilastir.py ─▶ her yöntem: skorlar ─▶ ROC, AUC, doğruluk, ms
```

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `gomucu.py` | kütüphane | 4 gömücü tek arayüzde (`goruntu()`, CLIP'te `metin()`) |
| `vektor_indeksi.py` | kütüphane | numpy (ve opsiyonel FAISS) ile k-en-yakın arama; disk kaydı |
| `foto_hafiza/indeksle.py` | Eylem (uzun) | Artımlı fotoğraf indeksleme |
| `foto_hafiza/ara.py` | Eylem | Metin ya da örnek görüntüyle arama; `fotograf_ara()` (09 agent kullanır) |
| `oneshot_karsilastir.py` | Eylem (uzun) | Yöntem yarışı: ROC eğrileri, AUC, en iyi eşik, hız; atlanan yöntemlerin nedeni |

## Çalıştır

```bash
pip install -r 07_embedding/requirements.txt
python 07_embedding/foto_hafiza/indeksle.py --klasor ~/Pictures --gomucu clip --en-fazla 500 --anlat
python 07_embedding/foto_hafiza/ara.py --sorgu "a birthday cake with candles" --indeks galerim_clip
python 07_embedding/foto_hafiza/ara.py --ornek-goruntu ~/Pictures/kedi.jpg --indeks galerim_clip
python 07_embedding/oneshot_karsilastir.py --yontemler renk,yolo_cls,dinov2,clip --anlat           # sentetik örnekle
python 07_embedding/oneshot_karsilastir.py --klasor veri/oneshot/kupam --yontemler renk,dinov2,yolo_world,yolo_model \
       --istem "red mug" --yolo-model veri/egitimler/esyalar_cp/weights/best.pt --sinif kupam
```

## Neye bakmalısın?

- **Arama sonuçlarının puanları**: CLIP'te 0.2–0.35 arası normaldir. Önemli olan sıralama ve 1. ile 10. arasındaki fark.
- **"ort. skor (poz / neg)" sütunu**: İki ortalama ne kadar ayrık? `yolo_cls`'de ikisi de ~0.97: özellikler "kupa"yı
  tanıyor ama **senin** kupanı başka kupalardan ayırmıyor. DINOv2'de fark daha büyük olmalı.
- **ROC eğrileri**: Sol üst köşeye yakın eğri iyi. Zor negatiflerde (aynı renkte başka nesne) `renk` neden çöküyor?
- **ms/görüntü**: Doğruluk/hız dengesi. 7/24 kamerada hangisini çalıştırırdın?

## Deneyler

1. Kendi nesnenle `veri/oneshot/<nesne>/` kur: 6 referans + 30 pozitif (farklı oda/ışık/açı) + 30 negatif (benzer nesneler!). Bütün yöntemleri yarıştır, tabloyu kaydet.
2. Galerinde Türkçe vs İngilizce sorgu: "deniz kenarında köpek" vs "a dog by the sea". Sonuçlar nasıl değişiyor?
3. Aynı galeriyi `clip` ve `dinov2` ile indeksle; aynı örnek fotoğrafla ara. Hangisi "aynı yer/nesne", hangisi "aynı konsept" buluyor?
4. `vektor_indeksi.ara(..., faiss_kullan=True)` ile numpy sonuçlarının aynı olduğunu doğrula; 100k sahte vektörde süreleri karşılaştır.
5. İleri: 03'teki SORT'a DINOv2 görünüm benzerliği ekle → kendi mini-DeepSORT'un (kapanmadan sonra ID korunuyor mu?).

> Bu ortamda (bulut konteyneri) HuggingFace ve bazı model sunucuları erişime kapalıydı: `renk` ve `yolo_cls` uçtan uca
> test edildi; `dinov2`, `clip` ve `yolo_world` standart API'lerle yazıldı ama burada indirilip çalıştırılamadı.
> Kendi bilgisayarında ilk çalıştırmada ağırlıkları otomatik indirirler.
