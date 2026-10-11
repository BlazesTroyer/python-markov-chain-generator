# Görüntü İşleme Yol Haritası — Teori ve Proje Fikirleri

> Bu dosya "neyi, neden öğreniyorum?" kısmıdır. Kodun nasıl kullanılacağı için: [README.md](README.md)

## 0. Önce kapatılması gereken kavram boşlukları (1 hafta)

| Kavram | Kısa tanım | Neden önemli |
|---|---|---|
| **Augmentation (veri çoğaltma)** | Eğitim görüntülerini rastgele döndürme, kırpma, parlaklık/kontrast değiştirme, bulanıklaştırma, yatay çevirme, mosaic vb. ile *yapay olarak çeşitlendirmek*. Etiketler de (bbox, keypoint) birlikte dönüştürülür. | Az veriyle ezberlemeyi (overfitting) önler, modeli farklı ışık/açı/kameraya dayanıklı yapar. 6 fotoğraflık one-shot sisteminin zayıf noktası tam olarak budur. |
| **Fine-tuning / Transfer learning** | COCO'da eğitilmiş ağırlıklarla başlayıp kendi sınıflarınla eğitmeye devam etmek. | 100–300 etiketli görüntüyle işe yarar bir dedektör çıkarırsın. |
| **Train / Val / Test ayrımı** | Modeli eğittiğin, ayar yaptığın ve en son dürüstçe ölçtüğün veriler ayrı olmalı. | Aksi halde "%99 doğruluk" yanıltıcıdır. |
| **Metrikler** | IoU, Precision/Recall, F1, **mAP@50, mAP@50-95**; takip için **MOTA, IDF1, HOTA**. | Değişikliklerin gerçekten iyileştirip iyileştirmediğini sayıyla bilmek. |
| **Tespit vs. Takip** | Tespit: *her karede* "nerede ne var?". Takip: *kareler arasında* "bu, az önceki aynı nesne mi?" → her nesneye kalıcı ID. | Sayma, süre ölçme, davranış analizi takip olmadan yapılamaz. |

Mini alıştırma: `albumentations` ile bir görüntüye 10 farklı augmentation uygula, bbox'ların doğru dönüştüğünü çizerek kontrol et.

```python
import albumentations as A
tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.MotionBlur(p=0.2),
], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]))
out = tf(image=img, bboxes=boxes, labels=labels)
```

---

## 1. Veri Toplama, Etiketleme ve Augmentation

**Öğrenilecekler:** Etiketleme araçları (CVAT, Label Studio, Roboflow), YOLO etiket formatı, sınıf dengesizliği, otomatik ön-etiketleme (auto-labeling), augmentation politikası seçmek.

**Proje — "Kişisel Nesne Dataset Fabrikası"**
- Telefonla kendi eşyalarının (anahtar, cüzdan, kulaklık, kupa, ilaç kutusu, şarj aleti) kısa videolarını çek → `ffmpeg` ile kare çıkar.
- **Grounding DINO / YOLO-World / SAM 2** ile otomatik ön-etiketle, CVAT'ta düzelt.
- Aynı veriyle 3 versiyon üret: augmentation yok / hafif / ağır. Bölüm 2'de üçünü karşılaştır.
- Agent faydası: Bu dataset, aşağıdaki "Eşyalarım nerede?" agent aracının temeli olur.

---

## 2. Nesne Tespiti ve YOLO Fine-Tuning

**Öğrenilecekler:** Ultralytics YOLO (v8/11) ile `train`, `val`, `predict`, `export`; hiperparametreler (imgsz, epochs, batch, lr0, freeze), augmentation parametreleri (`mosaic`, `mixup`, `hsv_*`, `degrees`), `results.png` / confusion matrix okumak, overfitting'i loss eğrisinden görmek.

```bash
yolo detect train data=my_items.yaml model=yolo11n.pt epochs=100 imgsz=640
yolo detect val   model=runs/detect/train/weights/best.pt
yolo export       model=best.pt format=onnx
```

**Proje — "Masamdaki Eşya Dedektörü" + deney raporu**
- Bölüm 1 datasetiyle fine-tune et; augmentation açık/kapalı ve `freeze=10` vs. tam eğitim farkını mAP ile tablola.
- Zor örnekleri (hard negatives: elde tutulan nesne, yarı kapalı nesne) ekleyip ikinci tur eğit.
- **Phone call detector'ını yeniden yap:** Genel "cell phone" sınıfı yerine kendi verinle `phone_at_ear`, `phone_in_hand`, `no_phone` sınıflarını eğit; mevcut kural tabanlı versiyonla karşılaştır.
- Agent faydası: `where_is("anahtar")` aracı — nesne son nerede/ne zaman görüldü kaydı.

---

## 3. Takip (Tracking) Algoritmaları

**Öğrenilecekler (sırayla):**
1. **Centroid / IoU tracker** — en basit: ardışık karelerde en çok örtüşen kutuları eşleştir.
2. **SORT** — Kalman filtresi (hareket tahmini) + Macar algoritması (eşleştirme).
3. **DeepSORT** — SORT + görünüm (ReID) embedding'i; nesne kaybolup dönünce ID'yi korur.
4. **ByteTrack** — düşük güvenli tespitleri de ikinci turda eşleştirir; hızlı ve güçlü.
5. **BoT-SORT / OC-SORT** — kamera hareketi telafisi, oklüzyon dayanıklılığı.
6. Metrikler: MOTA, IDF1, HOTA (`TrackEval` ile).

**Proje A — "SORT'u sıfırdan yaz"** (Kalman filtresi numpy ile elle + `scipy.optimize.linear_sum_assignment`; bkz. `03_takip/`). Sonra Ultralytics'in `model.track(tracker="bytetrack.yaml")` çıktısıyla karşılaştır. Algoritmayı gerçekten anlamanın en hızlı yolu budur.

**Proje B — "Odak / Ekran Süresi Takipçisi"**
- Kendini ve telefonu takip et; "telefon elde" durumunun **ID bazlı süresini** ölç, günlük rapor çıkar.
- Bölge (zone) ve çizgi geçişi (line crossing) — `supervision` kütüphanesi ile.
- Agent faydası: `screen_time_today()` aracı; "bugün telefonda kaç dk geçirdim?" sorusuna cevap.

---

## 4. Yüz Analizi: Tanıma ve Duygu

**Öğrenilecekler:** Yüz tespiti (RetinaFace/SCRFD) → hizalama → embedding (**ArcFace / InsightFace**) → kosinüs benzerliği ile tanıma; eşik belirleme (ROC); duygu tanımada **zamansal yumuşatma** ve sınıf dengesizliği.

**Proje A — "Emotion Checker v2"**
- Kendi datasetinle küçük bir CNN'i (ör. MobileNetV3 / EfficientNet-B0, `timm`) **augmentation ile** fine-tune et; FER2013 / AffectNet ile ön-eğitip kendi verinle ince ayar yap.
- Kare bazlı tahmini EMA veya 2 sn'lik pencere çoğunluk oyu ile yumuşat.
- DeepFace sonucuyla aynı test setinde karşılaştır (confusion matrix).

**Proje B — "Tanıdık Yüz Karşılayıcı"**
- Ev/oda kamerası: kayıtlı kişiler (sen, aile) embedding veritabanında; tanınan kişi → agent'a olay (`person_arrived: "Uğur"`), tanınmayan → bildirim.
- Gizlilik: görüntü değil sadece embedding sakla, lokal çalıştır.

---

## 5. El, Poz ve Jest (MediaPipe'ı bir üst seviyeye taşımak)

**Öğrenilecekler:** Landmark'ları normalize etme (bilek merkezli, ölçek bağımsız), **statik** jest için MLP/RandomForest, **dinamik** jest (sallama, kaydırma) için zaman serisi modelleri (1D-CNN, GRU/LSTM), keypoint augmentation (gürültü, döndürme, ölçek).

**Proje A — "Jestle Agent Kontrolü"**
- Hand controller'ını kural tabanlıdan **öğrenilen** modele geçir: 30 karelik landmark dizilerini kaydet, 6–8 dinamik jest için GRU eğit.
- Jest → agent komutu: sağa kaydır = sonraki şarkı, avuç = dur, başparmak = "son mesajı oku".

**Proje B — "Ergonomi / Egzersiz Koçu"**
- MediaPipe Pose / YOLO-pose ile eğik oturuş tespiti ve şınav/squat tekrar sayacı (açı hesaplama + durum makinesi).
- Agent faydası: "30 dk'dır kambur oturuyorsun" hatırlatması.

---

## 6. Segmentasyon

**Öğrenilecekler:** Semantik vs. instance segmentasyon, **SAM 2** (prompt ile maske, videoda maske takibi), YOLO-seg fine-tuning, maske metrikleri (mask mAP, Dice).

**Proje — "One-Shot Tarayıcı v2"**
- Şu anki "eli görmezden gel" mantığını **SAM 2 + el maskesi çıkarımı** ile değiştir: nesneyi tıkla/kutu ver → SAM 2 videoda maskeyi takip eder → elden arındırılmış temiz nesne kırpıntıları otomatik toplanır.
- Bu kırpıntılarla otomatik augmentation + copy-paste augmentation (nesneyi farklı arka planlara yapıştırma) yaparak Bölüm 2 dedektörüne veri üret. → *6 fotoğraftan gerçek bir YOLO modeline* giden boru hattı.

---

## 7. Embedding, Few-Shot ve Açık Kelime Dağarcıklı (Open-Vocabulary) Görü

**Öğrenilecekler:** **CLIP / SigLIP** (görüntü–metin ortak uzayı), **DINOv2** (güçlü görsel özellikler), vektör arama (FAISS, Chroma), open-vocab dedektörler (**YOLO-World, Grounding DINO, OWLv2**) — eğitim yapmadan metinle nesne tespiti.

**Proje A — "Kişisel Görsel Hafıza / Fotoğraf Arama"**
- Telefon galerini SigLIP ile embed et → "geçen yaz denizde çektiğim köpek fotoğrafı" gibi doğal dil aramaları.
- Agent faydası: `search_photos(query)` aracı.

**Proje B — One-shot karşılaştırması**
- Mevcut one-shot yöntemin vs. DINOv2 embedding + kNN vs. YOLO-World prompt vs. fine-tuned YOLO: aynı test videosunda doğruluk/FPS tablosu. Hangi yöntemin ne zaman kullanılacağını öğretir.

---

## 8. OCR ve Doküman Anlama

**Öğrenilecekler:** Metin tespiti + tanıma (PaddleOCR, docTR, EasyOCR), perspektif düzeltme (homografi), yapı çıkarımı (tablo/alan).

**Proje — "Fiş/Fatura → Gider Takip Agent'ı"**
- Kamera/telefonla fiş çek → belge kenarı bul, düzleştir → OCR → LLM ile `{mağaza, tarih, tutar, kategori}` JSON → SQLite/Sheets.
- Agent faydası: "Bu ay markete ne harcadım?"

---

## 9. Vision-Language Modeller ve Agent Entegrasyonu (Ana eksen)

**Öğrenilecekler:** VLM'ler (Qwen2.5-VL, Florence-2, Moondream, Claude/GPT vision API'leri), **hızlı lokal model tetikler → pahalı VLM yorumlar** mimarisi, olay (event) şeması, tool/function calling, **MCP server** yazmak.

**Mimari:**
```
Kamera → YOLO + Tracker (lokal, gerçek zamanlı)
       → Olay üretici (person_arrived, object_moved, phone_pickup, gesture_x)
       → Olay hafızası (SQLite + embedding)
       → VLM (sadece önemli anlarda kırpılmış kare ile)
       → LLM Agent araçları: where_is(), who_was_here(), screen_time_today(),
         describe_now(), search_photos(), expenses()
```

**Capstone Proje — "Gözleri Olan Kişisel Asistan"**
- Önceki bölümlerdeki her modülü bir **MCP server** olarak sun; Claude Code / Claude Desktop veya kendi agent'ın bu araçları çağırsın.
- Örnek diyalog: "Anahtarlarımı en son nerede gördün?" → `where_is("anahtar")` → "18:42'de masanın sol köşesinde, kare ekte."

---

## 10. Dağıtım ve Optimizasyon

**Öğrenilecekler:** ONNX / ONNX Runtime, TensorRT (NVIDIA), OpenVINO (Intel), INT8/FP16 quantization, çoklu thread ile kamera okuma, kare atlama, batch işleme, edge cihazlar (Raspberry Pi 5 + Hailo, Jetson Orin Nano).

**Proje — "7/24 Çalışan Ev Kamerası"**
- Bölüm 2 modelini PyTorch → ONNX → (TensorRT/OpenVINO) olarak export et; FPS, gecikme, doğruluk kaybını tabloya yaz.
- Docker ile paketle, düşük güçlü cihazda sürekli çalıştır.

---

## 11. (Paralel, opsiyonel) Klasik Görüntü İşleme Temeli

Derin öğrenme öncesi temeller birçok sorunu çok daha ucuza çözer: renk uzayları (HSV), eşikleme, morfoloji, kontur, kenar (Canny), homografi, **optical flow** (Lucas–Kanade, Farneback), kamera kalibrasyonu.
- Mini proje: Optical flow ile "hareket var mı?" ön filtresi → hareket yoksa YOLO'yu çalıştırma (Bölüm 10'daki enerji tasarrufuna doğrudan katkı).

---

## Önerilen Sıra ve Zaman Çizelgesi (~14 hafta, haftada 8–10 saat)

| Hafta | Bölüm | Çıktı |
|---|---|---|
| 1 | 0 + 11 (temel) | Augmentation görselleştirici + metrik oyun alanı |
| 2–3 | 1 + 2 | İlk fine-tuned YOLO + augmentation deney tablosu |
| 4–5 | 3 | Sıfırdan SORT + ekran süresi takipçisi |
| 6 | 5 | Dinamik jest modeli |
| 7 | 4 | Emotion v2 + yüz tanıma |
| 8 | 6 | SAM 2 ile one-shot v2 → otomatik dataset |
| 9 | 7 | Görsel hafıza / fotoğraf arama |
| 10 | 8 | Fiş okuyucu |
| 11–13 | 9 | MCP tabanlı kişisel görsel asistan (capstone) |
| 14 | 10 | Export, optimizasyon, 7/24 dağıtım |

**Her projede uyulacak kural:** Bir `README` + metrik tablosu + "ne denedim, ne işe yaradı" notu. Bu, hem öğrenmeyi kalıcı yapar hem portföy oluşturur.

## Kaynaklar
- Ultralytics Docs (train/track/export), Albumentations Docs, Roboflow Blog (fine-tune rehberleri)
- `supervision` (Roboflow) — takip, bölge sayımı, görselleştirme
- Papers: SORT (Bewley 2016), DeepSORT (Wojke 2017), ByteTrack (Zhang 2022), SAM 2 (2024), DINOv2 (2023)
- Kurs: Stanford CS231n (temel), "Dive into Deep Learning" CV bölümleri

