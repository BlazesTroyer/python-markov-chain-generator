# 00 · Temeller — Augmentation, Metrikler, Klasik Görüntü İşleme

Bu bölüm, sonraki bütün bölümlerde kullanacağın üç temeli kapatır:
**augmentation nedir**, **"model iyi mi?" nasıl ölçülür**, ve **modelsiz ne yapılabilir**.

## Kavramlar

**Augmentation (veri çoğaltma):** Eğitim görüntülerini gerçekçi biçimde rastgele değiştirmek (ışık, açı, çevirme,
bulanıklık, kısmi kapanma). Model aynı nesnenin farklı hallerini görür → ezberlemez, gerçek hayata dayanıklı olur.
**Altın kural:** görüntü geometrik değişiyorsa (çevirme/döndürme/kırpma) **etiket kutuları da aynı şekilde değişmeli.**

| Tür | Örnek | Kutu ne olur? |
|---|---|---|
| Fotometrik | parlaklık, HSV, bulanıklık, gürültü | değişmez |
| Geometrik | çevirme, döndürme, ölçekleme | aynı dönüşümle taşınır |
| Kapatma | cutout | aynı kalır, nesnenin bir kısmı gizlenir |

**Metrikler:** IoU → eşleştirme (TP/FP/FN) → Precision/Recall → PR eğrisi → AP → mAP@50 ve mAP@50-95.
Hepsi `metrikler.py`'de sıfırdan, satır satır yorumlu.

## Veri akışı

```
augmentation_gorsellestir.py
   görüntü + kutu ──▶ augment_sifirdan.py (her dönüşüm: img, kutu → yeni img, yeni kutu, açıklama)
                       └─ ya da albumentations.ReplayCompose (hangi dönüşüm uygulandı kaydıyla)
                  ──▶ ızgara (yeşil kutular = dönüştürülmüş etiketler) + log

metrikler.py (Metrik oyun alanı)
   sentetik GT + "kusurlu model" tahminleri ──▶ eslestir() ──▶ pr_egrisi() ──▶ ortalama_hassasiyet() ──▶ map_hesapla()

hareket_kapisi.py
   önceki kare + şimdiki kare ──▶ fark/eşik/genişletme  YA DA  Farneback akış ──▶ konturlar ──▶ "hareket var/yok" olayı
```

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `augment_sifirdan.py` | kütüphane | 8 dönüşüm + politika (yok/hafif/agir), kutu dönüşüm matematiği. 01 de kullanır. |
| `augmentation_gorsellestir.py` | Kare işleyici | Dönüşümleri tek tek ya da politika olarak görselleştirir. |
| `metrikler.py` | kütüphane + Eylem | IoU, IoU matrisi, eşleştirme, PR, AP, mAP; "Metrik oyun alanı". 02, 03, 07 kullanır. |
| `hareket_kapisi.py` | Kare işleyici | Kare farkı / optik akış ile ucuz hareket algılama. 10'daki 7/24 kamera kullanır. |
| `klasik_islemler.py` | Kare işleyici | HSV renk eşiği, Canny, Otsu+kontur, adaptif eşik — ara adımlarıyla. |
| `test_00_temeller.py` | test | Örn. "çevrilen kutunun merkezi hâlâ kırmızı kareye mi düşüyor?" |

## Çalıştır

```bash
# Panel: 00 · Temeller sekmesi
python 00_temeller/augmentation_gorsellestir.py --kaynak foto.jpg --mod tek_tek --anlat --debug-klasor dbg/
python 00_temeller/augmentation_gorsellestir.py --kaynak foto.jpg --mod politika --yontem albumentations --anlat
python 00_temeller/metrikler.py --konum-gurultusu 0.15 --anlat
python 00_temeller/hareket_kapisi.py --kaynak 0 --yontem akis --goster
python 00_temeller/klasik_islemler.py --kaynak 0 --islem hsv_renk --h-min 100 --h-max 130 --goster   # mavi nesneler
```

## Neye bakmalısın?

- **Döndürmede** kutu biraz şişer: dönmüş dikdörtgeni saran eksen hizalı kutu daha büyüktür (`_kutulari_matrisle_donustur`).
- **`agir` politika** gerçekçi mi? Gerçekte hiç görmeyeceğin bozulmalar modele zarar verir.
- **Metrik oyun alanında** `konum_gurultusu`'nu artır: mAP50 az, mAP50-95 çok düşer. Neden? (İpucu: 0.95 IoU eşiği.)
- **Hareket kapısında** `fark` ışık titremesine, `akis` kamera gürültüsüne nasıl tepki veriyor?

## Deneyler (kendin yap)

1. Kendi nesnenin fotoğrafını çek, `--kutu` ile yerini yaz; hangi dönüşüm nesneni tanınmaz hale getiriyor?
2. `augment_sifirdan.py`'ye yeni bir dönüşüm ekle (ör. `dikey_cevir` — kutu formülünü kendin çıkar) ve teste ekle.
3. Metrik oyun alanında `yanlis_pozitif=3` yap: PR eğrisinin hangi kısmı bozuluyor? Güven eşiğini nereden seçerdin?
4. Hareket kapısının `min_alan` değerini odanda perde/ağaç hareketini yok sayacak kadar ayarla.
