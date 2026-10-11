# CV Yol Haritası — Kişisel / AI Agent Odaklı Görüntü İşleme Projeleri

Bu klasör, görüntü işlemede bir üst seviyeye geçmek için **bölüm bölüm, çalıştırılabilir ve tamamen
şeffaf** projeler içerir. Teori ve "neden bu sıra?" için → **[YOL_HARITASI.md](YOL_HARITASI.md)**

Hepsini tek yerden kullanmak için bir **kontrol paneli** var (aşağıda).

---

## 1. Kurulum

```bash
cd cv-yol-haritasi
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-temel.txt                  # opencv, numpy, gradio, albumentations, pytest...
# Her bölümün ek paketleri kendi klasöründeki requirements.txt'te (ör. 02 için ultralytics)
pip install -r 02_yolo_finetune/requirements.txt
```

Testleri çalıştır (webcam/GPU gerekmez):

```bash
pytest            # bütün testler
pytest 03_takip   # sadece bir bölüm
```

## 2. Kontrol paneli

```bash
python panel/app.py          # tarayıcıda http://127.0.0.1:7860
```

| Panelde | Ne işe yarar |
|---|---|
| **Bölüm sekmeleri (00…10)** | Her projenin alt sekmesi. Ayarlar + girdi + çıktılar. |
| **⚙️ Ayarlar** | Proje dosyasının tepesindeki `Ayarlar` sınıfından **otomatik** üretilir. Alt satırdaki açıklama ve "→ etki" notu da koddan gelir. |
| **🖼️ Görüntü / 🎞️ Video / 📷 Canlı** | Kare işleyen projelerde girdi seçimi. Canlı modda tarayıcı kamerası kullanılır. |
| **Çıktı / Ara adımlar** | Sonuç görüntüsü ve öğretici ara görseller (maske, Kalman tahmini, eşik…). |
| **Ne oluyor?** | Adım adım log (`ortak/anlat.py`). Her satırın sonunda `← dosya.py:fonksiyon`. |
| **Kod nerede?** | Bu çalıştırmada hangi dosyanın hangi fonksiyonu çalıştı, sırasıyla. |
| **Eşdeğer komut** | Panelde yaptığının terminaldeki birebir karşılığı. |
| **🗄️ Olaylar** | Agent hafızası: projelerin ürettiği olaylar (SQLite tablo). |
| **🧵 İşler** | Arka planda çalışan eğitim/indeksleme işleri ve log'ları. |

Panel nasıl çalışıyor / yeni proje nasıl eklenir → [panel/README.md](panel/README.md)

## 3. Komut satırı (panelsiz)

Her proje dosyası doğrudan da çalışır; argümanlar yine `Ayarlar` sınıfından otomatik gelir:

```bash
python 03_takip/sort_sifirdan.py --help                     # bütün ayarlar + açıklamaları
python 03_takip/sort_sifirdan.py --kaynak video.mp4 --anlat --goster
python 00_temeller/augmentation_gorsellestir.py --kaynak foto.jpg --debug-klasor dbg/
```

Ortak bayraklar: `--kaynak 0|video.mp4|foto.jpg|klasor/`, `--anlat` (adımları yaz), `--goster` (pencere),
`--cikti sonuc.mp4`, `--debug-klasor dbg/` (ara görselleri kaydet), `--olaylari-kaydet` (SQLite'a yaz).

## 4. Klasörler — nerede ne var?

| Klasör | Konu | İçindekiler |
|---|---|---|
| [`ortak/`](ortak/) | Altyapı | sözleşme (`arayuz.py`), log (`anlat.py`), kaynak, çizim, olay deposu, otomatik CLI |
| [`panel/`](panel/) | Kontrol paneli | Gradio uygulaması, otomatik form, iş yöneticisi |
| [`00_temeller/`](00_temeller/) | Augmentation, metrikler, klasik CV | sıfırdan augmentation, IoU/mAP, hareket kapısı |
| [`01_veri_fabrikasi/`](01_veri_fabrikasi/) | Dataset hazırlama | kare çıkarma, otomatik etiketleme, bölme, çoğaltma |
| [`02_yolo_finetune/`](02_yolo_finetune/) | YOLO eğitimi | eğitim, deney karşılaştırma, telefon dedektörü v2, eşya hafızası |
| [`03_takip/`](03_takip/) | Takip | IoU takipçi, Kalman, SORT (sıfırdan), ByteTrack, ekran süresi |
| [`04_yuz/`](04_yuz/) | Yüz | emotion v2 (eğitim + EMA), yüz tanıma (embedding) |
| [`05_el_poz/`](05_el_poz/) | El, poz, jest | dinamik jest (GRU), ergonomi koçu, tekrar sayacı |
| [`06_segmentasyon/`](06_segmentasyon/) | Segmentasyon | SAM 2 ile elden arındırılmış nesne toplama, copy-paste |
| [`07_embedding/`](07_embedding/) | Embedding | fotoğraf hafızası (CLIP), one-shot yöntem karşılaştırması |
| [`08_ocr/`](08_ocr/) | OCR | belge düzleştirme, fiş okuyucu → gider kaydı |
| [`09_agent/`](09_agent/) | Agent | olay üretici, VLM yorumlama, MCP server |
| [`10_dagitim/`](10_dagitim/) | Dağıtım | ONNX export, benchmark, Docker |

Her klasörün kendi README'si: kavram → veri akışı → "dosya → görevi" tablosu → çalıştırma → "neye bakmalısın?" → deneyler.

## 5. Kodu okuma rehberi (şeffaflık kuralları)

Bütün dosyalar aynı kalıpta yazıldı, böylece birini okuyan hepsini okuyabilir:

1. **Dosyanın başı**: Türkçe açıklama — ne yapar, girdi/çıktı, adım adım akış, kim kullanır.
2. **`Ayarlar` sınıfı**: Değiştirebileceğin her şey burada. Her alanın yanında açıklama ve `etki=` notu.
3. **`isle()` / `calistir()`**: Ana fonksiyon. İçindeki `anlat.adim(..., kod="dosya:fonksiyon")` satırları,
   log'da gördüğün satırların kaynağıdır — log'dan koda, koddan log'a geçiş birebir.
4. **"Sıfırdan" dosyaları**: Öğretici kısımlar (IoU, mAP, augmentation, Kalman, SORT, EMA, durum makineleri)
   kütüphane kullanmadan yazıldı. Kütüphane sürümü yanında **karşılaştırma** olarak durur.
5. **Ağır bağımlılıklar** (YOLO, MediaPipe, SAM 2, InsightFace…) fonksiyon içinde `gerekli(...)` ile istenir:
   eksikse panel çökmez, o sekme "şunu kur" der.

Veri akışı (bütün projelerde aynı):

```
kaynak.py ──kare──▶ Proje.isle(kare, ayarlar, anlat) ──Sonuc──▶ panel / CLI penceresi / video
                                 │                                   (görüntü, ara görseller, özet, veri)
                                 └──olaylar──▶ olay_deposu.py (SQLite) ──▶ 09_agent/mcp_server.py ──▶ LLM agent
```

## 6. Önerilen sıra

00 → 01 → 02 → 03 → 05 → 04 → 06 → 07 → 08 → 09 → 10 (ayrıntı ve süre tahmini: [YOL_HARITASI.md](YOL_HARITASI.md)).
Her projeden sonra kendi README'ndeki **"Deneyler"** bölümünü yap ve sonuçları not al — öğrenmenin asıl kısmı orası.

## 7. Gizlilik

Her şey **yerelde** çalışır: panel sadece `127.0.0.1`'de açılır, olaylar `veri/olaylar.db`'de durur, yüz tanıma
görüntü değil sadece embedding saklar. Dışarıya veri gönderen tek yerler, açıkça Claude API kullanan
**08 fiş okuyucu (LLM modu)** ve **09 VLM yorumlama**dır; ikisi de `ANTHROPIC_API_KEY` verilmedikçe çalışmaz.

## 8. Test durumu — neyi doğruladım, neyi doğrulayamadım?

Bu klasör bir bulut konteynerinde (GPU yok, webcam yok, bazı model sunucuları erişime kapalı) yazıldı ve test edildi.

```bash
pytest                       # 92 hızlı test (model indirmez, webcam/GPU gerekmez)
CV_AGIR_TEST=1 pytest        # + gerçek model testleri (YOLO, SAM, YuNet/SFace, MediaPipe poz, EasyOCR, ONNX export)
```

| Durum | Ne |
|---|---|
| ✅ Gerçek modellerle uçtan uca çalıştı | YOLO eğitimi (sentetik veri, CPU), tahmin + sıfırdan NMS (Ultralytics ile aynı sonuç), SORT/ByteTrack/BoT-SORT, YuNet+SFace yüz tanıma, duygu modeli eğitimi (timm, ön-eğitimsiz), MediaPipe poz (açı/ergonomi), SAM 2.1 + YOLO-seg, copy-paste dataset, YOLO-cls embedding + one-shot yarışı, EasyOCR fiş okuma, MCP sunucusu (gerçek MCP istemcisiyle), olay üretici, ONNX export + benchmark, 7/24 çalıştırıcı, kontrol paneli (tarayıcıda Playwright ile) |
| ⚠️ Kod yazıldı, mantığı testli, ama burada gerçek modelle çalıştırılamadı | **Claude API** (fiş okuyucu `claude` modu, VLM, asistan — API anahtarı yoktu; sahte istemciyle test edildi), **CLIP / DINOv2 / YOLO-World** (HuggingFace ve CLIP ağırlık sunucuları kapalıydı), **timm ön-eğitimli ağırlıkları**, **MediaPipe el** (örnek görüntülerde el yoktu), **webcam / canlı akış**, **Docker imajı** (derlenmedi), **TensorRT / OpenVINO** export |

Kendi bilgisayarında ilk çalıştırmada bunlar ağırlıklarını otomatik indirir. Bir sorun çıkarsa log'daki
`← dosya.py:fonksiyon` satırı sana nereye bakacağını gösterir.
