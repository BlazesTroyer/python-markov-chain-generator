# 02 · YOLO Fine-Tune — Kendi modelini eğitmek, ölçmek, kullanmak

İlk kez bir YOLO'yu kendi verinle eğiteceksin. Bu bölümde eğitimin **ne yaptığını**, **iyi gidip gitmediğini nasıl
anlayacağını** ve sonucu **kişisel bir işe (eşya hafızası, telefon dedektörü) nasıl bağlayacağını** görürsün.

## Kavramlar

- **Fine-tuning**: COCO'da eğitilmiş ağırlıklardan başlayıp sadece senin sınıflarını öğretmek. 100–300 görüntü yeter.
- **backbone / neck / head**: Özellik çıkarıcı / ölçek birleştirici / kutu+sınıf tahmincisi. `freeze=10` backbone'u dondurur.
- **epoch, batch, lr0, imgsz, patience**: Her birinin etkisi `egit.py`'deki `Ayarlar`'da `etki=` notunda.
- **Çevrimiçi augmentation**: Ultralytics eğitimde her görüntüyü anında değiştirir. `train_batch0.jpg` modelin
  gerçekte ne gördüğünü gösterir (mosaic: 4 görüntü tek karede).
- **NMS**: Aynı nesneye ait üst üste binen adaylardan en güvenlisini bırakma. `tahmin.py`'de sıfırdan yazıldı ve
  Ultralytics'le aynı sonucu verdiği test edildi.
- **Overfitting**: Train loss düşerken val loss yükselir. `egitim_yorumla()` bunu `results.csv`'den otomatik söyler.

## Veri akışı

```
01 → veri/datasetler/<isim>/data.yaml
        │
        ▼
egit.py ──model.train(**argümanlar)──▶ veri/egitimler/<isim>/
        │    (bütün argümanlar log'da)     ├─ weights/best.pt  ← en iyi val mAP'li epoch
        │                                  ├─ results.csv ──▶ egitim_yorumla() ──▶ "overfitting var mı?"
        │                                  ├─ train_batch0.jpg, results.png, confusion_matrix…
        │                                  └─ test/ (test bölmesinde dürüst ölçüm)
        ▼
tahmin.py / telefon_dedektoru_v2.py / esya_hafizasi.py  --model veri/egitimler/<isim>/weights/best.pt
                                                    └──olaylar──▶ veri/olaylar.db ──▶ NeredeSorgu / 09 agent
```

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `yolo_yardimci.py` | kütüphane | Model önbelleği, sonuç ayıklama, augmentation profilleri (kapali/varsayilan/agir) |
| `egit.py` | Eylem (uzun) | Eğitim + yorum + test ölçümü + öğretici görseller |
| `deney_karsilastir.py` | Eylem (uzun) | Tek değişkenli deney: augmentation / freeze / model / veri |
| `tahmin.py` | Kare işleyici | Tespit; `nms_goster` ile ham adaylar → sıfırdan NMS → Ultralytics |
| `telefon_dedektoru_v2.py` | Kare işleyici | Telefon kulakta/elde: kural (poz+mesafe) ya da model; durum makinesi; olay |
| `esya_hafizasi.py` | Kare işleyici + Eylem | Eşyaları seyrelterek kaydet (konum adı + kırpıntı); `nerede()` sorgusu |
| `data_ornek.yaml` | şablon | Elle dataset tanımı yazmak istersen |

## Çalıştır

```bash
# 1) Hızlı tam tur (sentetik veri, CPU'da ~1 dk):
python 01_veri_fabrikasi/ornek_dataset.py
python 01_veri_fabrikasi/dataset_bol.py
python 02_yolo_finetune/egit.py --epochs 10 --imgsz 256 --isim sekil_deneme --anlat

# 2) Deney: augmentation gerçekten işe yarıyor mu?
python 02_yolo_finetune/deney_karsilastir.py --deney augmentation --epochs 15 --imgsz 256

# 3) Kullan
python 02_yolo_finetune/tahmin.py --kaynak 0 --model veri/egitimler/sekil_deneme/weights/best.pt --goster
python 02_yolo_finetune/tahmin.py --kaynak foto.jpg --nms-goster --guven 0.1 --debug-klasor dbg/
python 02_yolo_finetune/telefon_dedektoru_v2.py --kaynak 0 --goster --anlat
python 02_yolo_finetune/esya_hafizasi.py --kaynak 0 --olaylari-kaydet --goster
```

## Neye bakmalısın?

- **`train_batch0.jpg`**: Augmentation'ın canlı hali. Etiketler hâlâ doğru yerde mi?
- **`results.png`**: `train/box_loss` ve `val/box_loss` birlikte iniyor mu? Ayrılıyorlarsa overfitting.
- **Confusion matrix**: "background" satırı/sütunu = kaçırma ve hayalet tespit. Hangi sınıflar karışıyor?
- **val mAP vs test mAP**: Test belirgin düşükse val'e fazla ayar yapmış (ya da sızıntı yaşamış) olabilirsin.
- **Telefon dedektörü log'u**: `kulak_mesafe` / `bilek_mesafe` sayılarını izle → eşikleri kendi kameranla ayarla.

## Deneyler

1. `deney_karsilastir.py --deney augmentation`: kapali / varsayilan / agir. Sonuçları tabloya yaz.
2. `--deney freeze`: Az veride freeze=10 daha mı iyi? Eğitim süresi ne kadar kısalıyor?
3. **Sızıntı**: 01'de `--no-grupla` ile ikinci dataset üret, `--deney veri --data2 ...` ile karşılaştır.
4. **Telefon v1 vs v2**: Kendi 2 dakikalık videonda elle "kulakta" anlarını not al; `kural` ve (eğittiysen) `model`
   modlarının yakaladığı geçişleri say. Hangisi daha az yanlış alarm veriyor? `kararlilik`'in etkisi ne?
5. **Kendi telefon modelin**: 01 hattıyla `phone_at_ear / phone_in_hand` sınıflı küçük bir dataset kur, eğit, `--mod model` ile dene.
