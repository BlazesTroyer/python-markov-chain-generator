# 01 · Veri Fabrikası — Kendi dataset'ini kurmak

İyi bir model, iyi bir dataset'ten çıkar. Bu bölüm, telefonla çektiğin videodan **eğitime hazır, sızıntısız,
kontrol edilmiş** bir YOLO dataset'ine giden hattı adım adım kurar.

## İş akışı

```
 (telefon videosu)                                          (ya da: ornek_dataset.py — sentetik, 1 dk)
       │
       ▼
 kare_cikar.py ──▶ veri/ham/<isim>/images/*.jpg           bulanık (Laplacian) + tekrar (dHash) elenir
       │
       ▼
 otomatik_etiketle.py ──▶ veri/ham/<isim>/labels/*.txt    YOLO-World ön-etiket ("anahtar=keys")
       │                    veri/ham/<isim>/onizleme/      göz kontrolü için çizilmiş kopyalar
       ▼
 (CVAT / Label Studio'da düzelt — "YOLO 1.1" formatı)
       │
       ▼
 etiket_kontrol.py ──▶ sağlık raporu                      format, sınıf dengesi, küçük kutu, negatif oranı
       │
       ▼
 dataset_bol.py ──▶ veri/datasetler/<isim>/{images,labels}/{train,val,test} + data.yaml   (GRUPLU bölme)
       │
       ▼
 augment_uret.py ──▶ veri/datasetler/<isim>_<politika>x<kopya>/    sadece train çoğaltılır
       │
       ▼
 02_yolo_finetune/egit.py --data veri/datasetler/<isim>/data.yaml
```

## Dosya → görevi

| Dosya | Görevi | Öğrettiği kavram |
|---|---|---|
| `yolo_format.py` | YOLO `.txt` okuma/yazma, piksel↔normalize, `data.yaml` | etiket formatı |
| `ornek_dataset.py` | Sentetik şekil dataset'i (daire/kare/üçgen) | hattı kamerasız denemek |
| `kare_cikar.py` | Videodan seçici kare çıkarma | keskinlik ölçümü, algısal hash |
| `otomatik_etiketle.py` | YOLO-World ile ön-etiket | open-vocabulary tespit, istem mühendisliği |
| `etiket_kontrol.py` | Dataset sağlık raporu + örnek ızgara | veri kalitesi, sınıf dengesizliği |
| `dataset_bol.py` | Gruplu train/val/test bölme | **veri sızıntısı** |
| `augment_uret.py` | Train'i çevrimdışı çoğaltma | online vs offline augmentation |

## Çalıştır (hızlı deneme — kamerasız)

```bash
python 01_veri_fabrikasi/ornek_dataset.py --goruntu-sayisi 300
python 01_veri_fabrikasi/etiket_kontrol.py --klasor veri/ham/sekiller --cikti-klasor dbg/kontrol
python 01_veri_fabrikasi/dataset_bol.py --kaynak veri/ham/sekiller --isim sekiller
python 01_veri_fabrikasi/augment_uret.py --dataset veri/datasetler/sekiller --politika hafif --kopya 2
```

## Kendi eşyalarınla

1. Her eşya için 3–5 kısa video çek (farklı oda, ışık, açı, mesafe; **elinde tutarken de**).
   Dosya adları farklı olsun: `anahtar_salon.mp4`, `anahtar_mutfak.mp4` … (dosya adı = grup).
2. `kare_cikar.py --video videolar/ --isim esyalarim`
3. `otomatik_etiketle.py --klasor veri/ham/esyalarim --siniflar "anahtar=keys, cuzdan=wallet, kupa=mug"`
4. `onizleme/` klasörüne bak. Hatalıları CVAT'ta düzelt (YOLO formatında içe aktar → düzelt → dışa aktar → `labels/`'ın üzerine kopyala).
5. `etiket_kontrol.py` → `dataset_bol.py` → 02.

## Neye bakmalısın?

- **Sınıf dengesi**: Bir sınıf diğerinin 5 katından fazlaysa model ona kayar. Az olandan daha çok çek.
- **Negatif oranı**: Hiç nesne olmayan kareler (%5–15) "boş masada hayalet anahtar görme" hatasını azaltır.
- **Küçük kutular**: Görüntünün %1'inden küçük nesneler 640px'te birkaç piksele düşer → `imgsz` artır ya da daha yakından çek.

## Deneyler

1. **Sızıntı deneyi**: `dataset_bol.py`'yi `--no-grupla` ile ikinci bir isimle çalıştır. 02'de ikisini eğit:
   gruplanmamış olanın val mAP'i daha yüksek ama **test** (ya da yeni video) sonucu daha kötü çıkacak. Nedenini açıkla.
2. **İstem deneyi**: `otomatik_etiketle.py`'de "keys" yerine "keychain", "car key" dene; hangisi senin anahtarını daha iyi buluyor?
3. `kare_cikar.py`'de `benzerlik_esigi`'ni 0 ve 15 yap; kaç kare kaydediliyor, etiketleme süren nasıl değişiyor?
