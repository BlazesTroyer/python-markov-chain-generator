# 05 · El, Poz ve Jest — MediaPipe'ı bir üst seviyeye taşımak

Hand controller'ını kurallardan **öğrenilen dinamik jestlere** taşıyoruz. Ayrıca vücut noktalarıyla iki kişisel
koç yapıyoruz: **duruş (ergonomi)** ve **egzersiz tekrar sayacı**.

> **MediaPipe notu:** Yeni sürümlerde eski `mp.solutions` API'si kaldırıldı. Burada her sürümde çalışan **Tasks API**
> (`HandLandmarker`, `PoseLandmarker`) kullanılır; model dosyaları ilk çalıştırmada `veri/modeller/`'e iner.
> Linux sunucuda `libEGL.so.1` hatası alırsan: `sudo apt install libegl1 libgles2` (masaüstünde genelde gerekmez).

## Kavramlar

- **Normalize etme** (`el_normalize`): Bileği orijine taşı → el boyuna böl → sol eli aynala. Aynı jest, ekranın
  neresinde ve hangi elle yapılırsa yapılsın aynı vektörü verir.
- **Statik vs dinamik jest**: Tek kare (avuç) vs zaman dizisi (kaydırma). Dinamik için **(T, 65)** dizi + **GRU**.
- **Nokta augmentation**: Görüntü değil NOKTA uzayında: döndürme, ölçek, gürültü, zaman kaydırma.
- **Eğitim/çıkarım tutarlılığı**: `dizi_hazirla()` hem eğitimde hem canlıda aynı fonksiyon — ayrı yazılırsa model sessizce bozulur.
- **Açı + histerezis**: Tekrar sayımında iki eşik, sınırdaki titremeyi sayma hatasına dönüştürmez.

## Veri akışı (jest)

```
webcam ─▶ HandLandmarker (21 nokta) ─▶ kare_ozelligi: el_normalize(63) + bilek(2)
            │
            ├─ jest_kaydet.py: HAZIRLAN → KAYIT(T kare) ─▶ veri/jestler/<etiket>/<oturum>_*.npy  (T, 65)
            │
            ├─ jest_egit.py: dizi_hazirla → nokta_augment → GRU/MLP ─▶ veri/modeller/jest_<isim>.pt
            │
            └─ jest_canli.py: son T kare ─▶ dizi_hazirla ─▶ model ─▶ olasılık ≥ eşik ─▶ olay {jest, komut}
                             (ya da kural modu: bilek Δx, parmak uç/PIP geometrisi)
```

## Dosya → görevi

| Dosya | Tür | Görevi |
|---|---|---|
| `noktalar.py` | kütüphane | MediaPipe Tasks sarmalayıcıları; `el_normalize`, `nokta_augment`, `aci`, `dikeyle_aci`, iskelet çizimi |
| `jest_kaydet.py` | Kare işleyici | Durum makineli dizi kaydı (`veri/jestler/`) |
| `jest_egit.py` | Eylem (uzun) | GRU vs MLP; oturum bazlı bölme; açık eğitim döngüsü; karışıklık matrisi |
| `jest_canli.py` | Kare işleyici | Kural ya da model ile canlı jest → `jest` olayı (komut eşlemeli, bekleme süreli) |
| `tekrar_sayaci.py` | Kare işleyici | Squat / şınav / biceps; eklem açısı + histerezis; açı grafiği |
| `ergonomi_kocu.py` | Kare işleyici | Boyun/gövde eğimi; kararlı kötü duruş süresi; `kotu_durus_uyari` olayı |

## Çalıştır

```bash
python 05_el_poz/jest_canli.py --kaynak 0 --mod kural --goster --anlat           # hemen dene (modelsiz)
python 05_el_poz/jest_kaydet.py --kaynak 0 --etiket sag_kaydir --oturum ev1 --goster
python 05_el_poz/jest_kaydet.py --kaynak 0 --etiket yok --oturum ev1 --goster     # "jest değil" örnekleri!
python 05_el_poz/jest_egit.py --model gru --anlat
python 05_el_poz/jest_canli.py --kaynak 0 --mod model --goster --olaylari-kaydet
python 05_el_poz/tekrar_sayaci.py --kaynak 0 --egzersiz squat --goster
python 05_el_poz/ergonomi_kocu.py --kaynak 0 --uyari-sn 10 --goster              # deneme için 10 sn
```

## Neye bakmalısın?

- **jest_canli log'u (kural)**: `bilek Δx` değerleri — gerçek kaydırmada kaç, el sallarken kaç? Eşiği buna göre ayarla.
- **Model modunda olasılıklar**: Jest yokken "yok" sınıfı yüksek mi? Değilse "yok" verisi topla.
- **Tekrar sayacının açı grafiği**: Açı eğrisi alt ve üst eşik çizgilerinin ikisini de geçiyor mu? Geçmiyorsa eşikleri kendine göre ayarla.
- **Ergonomi**: Kamerayı yana koy; kulak–omuz çizgisi öne eğildikçe boyun açısı nasıl büyüyor?

## Deneyler

1. Aynı veriyle `--model gru` ve `--model mlp` eğit; sonra jest hızını değiştirerek (yavaş/hızlı kaydırma) canlıda dene. Hangisi daha dayanıklı?
2. `--no-augmentation` ile eğit; özellikle elini farklı açıyla tuttuğunda fark var mı?
3. `jest_canli` kural vs model: 2 dakikalık normal masa başı çalışmasında kaç **yanlış tetikleme** oldu? (`🗄️ Olay Deposu`'nda say.)
4. Komutu gerçekten çalıştır: `pip install pyautogui`, olay deposunu dinleyen küçük bir döngüde
   `sonraki_sarki` → `pyautogui.press("nexttrack")`. (09'daki agent bunu daha düzenli yapar.)
5. `tekrar_sayaci.py`'ye yeni egzersiz ekle (ör. omuz pres: kalça-omuz-dirsek açısı): `EGZERSIZLER` sözlüğüne bir satır.
