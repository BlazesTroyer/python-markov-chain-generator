# 06 · Segmentasyon — One-Shot Tarayıcı v2: 6 fotoğraftan gerçek bir YOLO modeline

Senin one-shot projen "eli tara, görmezden gel, nesneyi öğren" mantığıyla çalışıyordu. Burada aynı fikri **piksel
düzeyinde maskelerle** yapıp sonucu **etiketli bir dataset'e** çeviriyoruz. Böylece 02'deki gibi gerçek bir YOLO eğitebilirsin.

## Kavramlar

| Tür | Ne verir | Örnek |
|---|---|---|
| Semantik | her pikselin sınıfı | "bu pikseller yol" |
| Instance | her nesnenin ayrı maskesi + sınıfı | YOLO-seg: insan#1, insan#2 |
| İstemli (promptable) | "şu kutudaki/noktadaki şey"in maskesi; sınıf bilmez | SAM / SAM 2 |

- **Maske metrikleri**: IoU = kesişim/birleşim, Dice = 2·kesişim/(toplam). `maske_ortak.py`'de sıfırdan.
- **Copy-paste augmentation**: Maskeli nesneyi başka arka planlara yapıştır. Etiket kutusu maskeden hesaplandığı için
  **elle etiketleme yok ve kutu piksel kesin**.

## Hat

```
webcam (nesne elinde)                                         arka plan fotoğrafları (nesnesiz oda)
   │                                                                   │
   ▼                                                                   ▼
nesne_topla_sam.py                                             copy_paste_aug.py
   istem kutusu (ilk kare: ekran ortası; sonra önceki maske)      rastgele kesit + ölçek/döndürme/çevirme/parlaklık
   → SAM maskesi                                                  → alfa ile yapıştır (yeni nesne eskileri kapatır)
   → − el maskesi (MediaPipe dışbükey örtü)                      → kutu = maskenin kapsayan kutusu
   → en büyük parça                                               → veri/ham/<isim>/images|labels  (YOLO formatı)
   → veri/nesneler/<isim>/*.png  (SAYDAM kesitler) ───────────────┘
                                                                      ▼
                                                  01/etiket_kontrol → 01/dataset_bol → 02/egit
```

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `maske_ortak.py` | kütüphane | SAM istemli maske, el maskesi, kutu/en büyük parça, kesit (RGBA), IoU/Dice |
| `nesne_topla_sam.py` | Kare işleyici | One-shot v2: istem → SAM → el çıkarma → saydam kesit kaydı |
| `copy_paste_aug.py` | Eylem (uzun) | Kesitlerden etiketli YOLO dataset'i sentezleme |
| `seg_karsilastir.py` | Kare işleyici | YOLO-seg vs SAM maskeleri, IoU/Dice tablosu |

## Çalıştır

```bash
python 06_segmentasyon/nesne_topla_sam.py --kaynak 0 --isim kupam --goster --anlat    # kupayı ortada tut, yavaşça çevir
python 06_segmentasyon/nesne_topla_sam.py --kaynak 0 --isim anahtarim --goster
python 06_segmentasyon/copy_paste_aug.py --arka-plan-klasoru odam/ --goruntu-sayisi 500 --isim esyalar_cp
python 01_veri_fabrikasi/etiket_kontrol.py --klasor veri/ham/esyalar_cp
python 01_veri_fabrikasi/dataset_bol.py --kaynak veri/ham/esyalar_cp --isim esyalar_cp
python 02_yolo_finetune/egit.py --data veri/datasetler/esyalar_cp/data.yaml --epochs 50
python 06_segmentasyon/seg_karsilastir.py --kaynak foto.jpg --debug-klasor dbg/seg
```

## Neye bakmalısın?

- **"1_istem + SAM maskesi"**: SAM elini de nesneye dahil ediyor mu? **"2_el maskesi"** onu doğru kesiyor mu?
- **"4_kesit (saydam)"**: Kenarlarda arka plan kalıntısı var mı? (Varsa copy-paste'te "hale" oluşur, model onu öğrenebilir.)
- **Sentez görüntüler**: Ölçek/açı gerçekçi mi? Nesne havada mı süzülüyor? (Gerçek arka plan + makul ölçek önemli.)
- **seg_karsilastir**: IoU 0.9 altına nerede düşüyor? İnce parçalar (kulp, parmak) hangi modelde daha iyi?

## Deneyler

1. **Asıl deney — "6 fotoğraf vs SAM+copy-paste"**: Aynı nesne için
   (a) v1 yöntemin (6 fotoğraf), (b) bu hattın ürettiği YOLO modeli. Aynı **gerçek** test videosunda başarıyı karşılaştır.
2. `el_genislet` 0 / 25 / 60: Kesitlerde el kalıntısı ile nesneden kesilen pay arasındaki denge.
3. Copy-paste: sentetik arka plan vs kendi oda fotoğrafların → 02'de test mAP farkı.
4. `--takip` kapalıyken nesneyi ekranda gezdir: neden maske kayboluyor?
5. İleri: SAM 2'nin gerçek video belleği için Ultralytics `SAM2VideoPredictor`'ı dene; basit kutu takibiyle karşılaştır.
