# 04 · Yüz — Tanıma (kim?) ve Duygu (nasıl?)

İki proje: **Tanıdık yüz karşılayıcı** (embedding + eşik) ve **Emotion Checker v2** (kendi verinle fine-tune +
zamansal yumuşatma). İkisi de aynı temelden geçer: yüzü bul → kırp/hizala → modele ver.

## Kavramlar

**Yüz tanıma = sınıflandırma DEĞİL, benzerlik.** Yeni bir kişi eklemek için modeli yeniden eğitmezsin:
```
yüz ──YuNet──▶ kutu + 5 nokta ──hizala──▶ 112×112 ──SFace──▶ 128 sayılık vektör ──kosinüs──▶ kayıtlı vektörler
                                                                                        > eşik ? "Uğur" : "tanınmadı"
```
- **Embedding**: Aynı kişinin fotoğrafları uzayda yakın, farklı kişilerinki uzak olacak şekilde eğitilmiş vektör.
- **Eşik seçimi**: Yanlış kabul (FAR, güvenlik) ile yanlış ret (FRR, rahatsızlık) arasında denge.
  `veritabani.py` kendi fotoğraflarınla iki dağılımı çizer ve eşik önerir.

**Duygu tanıma = sınıflandırma.** Küçük bir CNN'in son katmanı senin sınıflarına göre eğitilir.
- **Sınıf dengesizliği** → kayıpta sınıf ağırlığı.
- **Augmentation** → ışık/açı/yakınlık çeşitliliği. Ama dikkat: sınıfı belirleyen özelliği bozan augmentation
  zarar verir (testte renkle ayrılan sahte sınıflarda ColorJitter öğrenmeyi tamamen durdurdu!).
- **Zamansal yumuşatma** (EMA / pencere oyu) + durum makinesi → titremeyen kararlar.

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `yuz_ortak.py` | kütüphane | YuNet tespit, 5 noktayla hizalama, SFace embedding, kosinüs, kırpma |
| `yuz_tanima/kaydet.py` | Kare işleyici | (1/3) Tek yüz varken hizalanmış fotoğrafı `veri/yuzler/<isim>/`'e kaydet |
| `yuz_tanima/veritabani.py` | Eylem | (2/3) Embedding veritabanı `veri/yuz_db.npz` + **eşik analizi** (FAR/FRR, histogram) |
| `yuz_tanima/tani.py` | Kare işleyici | (3/3) Tanıma; kişi başına benzerlik çubukları; `kisi_goruldu` / `tanimadik_yuz` olayları |
| `emotion_v2/duygu_topla.py` | Kare işleyici | Duygu dataset'i topla (`veri/duygu/<etiket>/<oturum>_*.jpg`) |
| `emotion_v2/egit.py` | Eylem (uzun) | timm ile fine-tune; açık eğitim döngüsü; eğriler, karışıklık matrisi, augment önizleme |
| `emotion_v2/canli.py` | Kare işleyici | Canlı tahmin + EMA/pencere yumuşatma + `duygu_degisti` olayı |

Modeller (`ortak/indir.py`): YuNet (~230 KB) ve SFace (~37 MB) ilk kullanımda `veri/modeller/` altına iner.

## Çalıştır

```bash
# Yüz tanıma
python 04_yuz/yuz_tanima/kaydet.py --kaynak 0 --isim ugur --goster          # kafanı yavaşça çevir
python 04_yuz/yuz_tanima/kaydet.py --kaynak 0 --isim kardesim --goster
python 04_yuz/yuz_tanima/veritabani.py --anlat --cikti-klasor dbg/yuz         # eşik önerisini oku
python 04_yuz/yuz_tanima/tani.py --kaynak 0 --esik 0.40 --olaylari-kaydet --goster

# Emotion v2
python 04_yuz/emotion_v2/duygu_topla.py --kaynak 0 --etiket mutlu --oturum sabah1 --goster
python 04_yuz/emotion_v2/egit.py --epochs 15 --anlat          # ön-eğitimli ağırlıklar ilk seferde indirilir
python 04_yuz/emotion_v2/canli.py --kaynak 0 --yumusatma ema --alfa 0.3 --goster
```

Kendi eski dataset'in varsa `veri/duygu/<sinif>/` düzenine kopyala; dosya adlarına `<oturum>_` öneki koy.

## Neye bakmalısın?

- **Eşik histogramı**: Yeşil (aynı kişi) ve kırmızı (farklı kişi) tepeler ne kadar ayrık? Gözlüklü/gözlüksüz,
  karanlık fotoğraflar eklediğinde yeşil tepe nasıl sola kayıyor?
- **tani.py benzerlik çubukları**: Eşiğe kıl payı yakın kararlar → o koşulda (ışık, açı) daha çok kayıt fotoğrafı ekle.
- **augmentation_onizleme**: Modelin gördüğü yüzler hâlâ o duyguyu taşıyor mu?
- **Karışıklık matrisi**: Hangi duygular karışıyor? Bu bir veri sorunu mu, gerçekten benzer ifadeler mi?
- **canli.py olasılık çubukları**: Gri (ham) titrerken mavi (yumuşatılmış) ne kadar sakin?

## Deneyler

1. Eşik deneyi: Bir arkadaşının fotoğrafıyla `tani.py`'yi çalıştır; `esik`'i 0.25 → 0.50 arası değiştir. Seni ne zaman tanımamaya başlıyor, onu ne zaman "sen" sanıyor?
2. `egit.py`: `--no-onegitimli` vs varsayılan; `--no-augmentation` vs varsayılan; `--dondur`. Val doğruluklarını tabloya yaz.
3. Sınıf ağırlığını kapat (`--no-sinif-agirligi`) ve dengesiz bir dataset'le eğit: az olan sınıfın doğruluğu ne oluyor?
4. `canli.py`: `alfa` 0.1 / 0.3 / 0.8; tepki gecikmesi ve titreme arasındaki dengeyi hisset.
5. DeepFace karşılaştırması: Aynı test fotoğraflarında `DeepFace.analyze(img, actions=["emotion"])` sonucu ile v2'yi kıyasla (karışıklık matrisi).
6. Daha güçlü tanıma: `pip install insightface onnxruntime` → `insightface.app.FaceAnalysis(name="buffalo_l")` ile
   `yuz_ortak.embedding`'i değiştir (ArcFace, 512 boyut). Eşik analizini tekrar yap.

## Gizlilik

Yüz verisi hassastır: Her şey yerelde; veritabanında sadece embedding + isim var. Başkalarının yüzünü kaydetmeden önce izin al.
