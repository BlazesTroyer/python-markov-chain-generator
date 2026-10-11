# 03 · Takip (Tracking) — Tespitten "kim, ne kadar süre, nereye?" sorusuna

**Tespit** her karede "nerede ne var?" der. **Takip** kareler arasında "bu, az önceki AYNI nesne mi?" der ve her
nesneye kalıcı bir **ID** verir. Sayma, süre ölçme, çizgi geçişi, davranış analizi takip olmadan yapılamaz.

## Algoritmalar (basitten gelişmişe)

| Algoritma | Fikir | Dosya | Zayıf yanı |
|---|---|---|---|
| **IoU takipçi** | Önceki kutuyla en çok örtüşen = aynı nesne (Macar algoritmasıyla bire-bir) | `iou_takipci.py` | Hızlı harekette ve kaçırmada ID kopar |
| **SORT** | IoU takipçi + **Kalman** tahmini (konum + hız) | `kalman.py`, `sort_sifirdan.py` | Sadece konum; uzun kapanmada ID karışabilir; ilk hız 0 |
| **DeepSORT** | SORT + görünüm embedding'i (ReID) | — (07'deki embedding fikri) | Daha yavaş |
| **ByteTrack** | Düşük güvenli tespitleri de 2. turda eşleştirir | `karsilastir_bytetrack.py` (Ultralytics) | — |
| **BoT-SORT** | ByteTrack + kamera hareketi telafisi (+ ReID) | `karsilastir_bytetrack.py` (Ultralytics) | — |

## SORT'un bir karesi (sort_sifirdan.py → `Sort.guncelle`)

```
tespitler (N,6) ─┐
                 ▼
 1. TAHMİN      her iz: kalman.tahmin()          → tahmin kutuları (kesikli sarı, ara görselde)
 2. MALİYET     IoU(tespitler × tahminler)       → matris (ara görselde ısı haritası)
 3. EŞLEŞTİR    Macar algoritması, IoU ≥ eşik     → eşleşmeler (yeşil çerçeveli hücreler)
 4. GÜNCELLE    eşleşen iz: kalman.guncelle(tespit)
 5. DOĞUM/ÖLÜM  eşleşmeyen tespit → yeni iz; eşleşmeyen iz kayip++ ; kayip > max_yas → sil
 6. ÇIKTI       vurus ≥ min_vurus ve bu karede görülen izler
```

Kalman filtresinin matrisleri ve denklemleri `kalman.py`'nin başında tek tek açıklandı.

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `kalman.py` | kütüphane | 7 boyutlu Kalman (cx, cy, alan, oran + hızlar), numpy ile |
| `iou_takipci.py` | kütüphane + Kare işleyici | `macar_eslestir()`, `IouTakipci`, demo |
| `sort_sifirdan.py` | kütüphane + Kare işleyici | `Sort`, demo (Kalman tahmini vs tespit, maliyet matrisi) |
| `takip_ortak.py` | kütüphane | Tespit kaynağı (yolo / **renk** — modelsiz), iz çizimi, matris görseli |
| `sentetik_test.py` | Eylem | Gerçek ID'leri bilinen sahnede IoU vs SORT: **MOTA, ID switch, parçalanma** |
| `karsilastir_bytetrack.py` | Kare işleyici | Ultralytics ByteTrack/BoT-SORT vs bizim SORT, yan yana |
| `ekran_suresi.py` | Kare işleyici + Eylem | Kişi ID'si başına telefon süresi; günlük rapor (`gunluk_ozet`) |
| `bolge_sayaci.py` | Kare işleyici | Çizgi geçişi (çapraz çarpım ile taraf testi), giriş/çıkış sayacı |

## Çalıştır

```bash
python 03_takip/sentetik_test.py --hiz 15 --anlat            # modelsiz, sayılarla karşılaştırma
python 03_takip/sort_sifirdan.py --kaynak 0 --dedektor renk --goster     # mavi bir nesneyle (modelsiz)
python 03_takip/sort_sifirdan.py --kaynak video.mp4 --goster --anlat      # YOLO ile kişiler
python 03_takip/karsilastir_bytetrack.py --kaynak video.mp4 --cikti karsilastirma.mp4
python 03_takip/ekran_suresi.py --kaynak 0 --olaylari-kaydet --goster
python 03_takip/bolge_sayaci.py --kaynak kapi.mp4 --cizgi 0.5,0,0.5,1 --goster
```

## Neye bakmalısın?

- **SortDemo ara görseli**: Kalman tahmini (kesikli) tespite ne kadar yakın? Nesne dönünce/durunca ne oluyor?
- **Maliyet matrisi**: Her satırda en fazla bir yeşil hücre. İki kişi yan yana geçerken değerler nasıl değişiyor?
- **"Hayalet" izler** (kesikli, "kayip N"): Kişi bir şeyin arkasından geçince ID'si `max_yas` boyunca bekliyor.
- **Sentetik yarış**: Hız arttıkça IoU takipçinin ID switch sayısı patlıyor; SORT çok daha dayanıklı.

## Deneyler

1. `sentetik_test.py`: hiz=5/15/25 ve kacirma=0/0.3 için tabloyu doldur. Hangi koşulda SORT da çöküyor? Neden? (İpucu: `sort_sifirdan.py` → "İLK HIZ" notu.)
2. `max_yas`'ı 0 ve 30 yap: kaçırmalı sahnede ID switch ve parçalanma nasıl değişiyor?
3. Kendi videonda `karsilastir_bytetrack.py`: Birinin önünden biri geçtiği anı bul; hangi takipçi ID'yi koruyor?
4. `kalman.py`'de `olcum_gurultusu`'nu 10 yap: kutular yumuşuyor mu, gecikiyor mu? (Kalman kazancı sezgisi.)
5. İleri seviye: SORT'a basit bir görünüm benzerliği ekle (07'deki embedding ile) → kendi mini-DeepSORT'un.
   Daha kapsamlı metrikler için: [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, IDF1).
