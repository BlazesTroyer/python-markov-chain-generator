# ortak/ — Bütün projelerin paylaştığı altyapı

| Dosya | Görevi | Kim kullanır |
|---|---|---|
| `arayuz.py` | **Sözleşme**: `ayar(...)` alanı, `Sonuc`, `KareIsleyici`, `Eylem` | her proje, panel, CLI |
| `anlat.py` | "Ne oluyor?" log'u: `adim()`, `bilgi()`, `uyari()`, `kod_yerleri()` | her proje |
| `cli.py` | `Ayarlar`'dan otomatik argparse; kare döngüsü; `ayarlari_argv()` (panel → komut) | her proje dosyasının `__main__`'i, panel |
| `kaynak.py` | webcam / video / görüntü / klasör → `(no, kare)` üreteci; `VideoYazici` | CLI, panel video sekmesi |
| `cizim.py` | kutu, yazı, maske, ızgara; BGR↔RGB | her proje |
| `olay_deposu.py` | SQLite olay kaydı (`veri/olaylar.db`) | 00, 02, 03, 04, 05, 08, 09 |
| `yukle.py` | rakamla başlayan klasörlerden modül yükleme; `gerekli()` ile opsiyonel paket | her proje |

## Bir projenin iskeleti

```python
@dataclass
class Ayarlar:
    esik: float = ayar(0.5, "Güven eşiği", min=0, max=1, adim=0.05,
                       etki="Artırırsan daha az ama daha emin tespit")

class BenimProjem(KareIsleyici):
    ISIM, BOLUM, DOSYA = "Benim projem", "02", "02_yolo_finetune/benim.py"
    Ayarlar = Ayarlar

    def isle(self, kare, ayarlar, anlat):
        anlat.adim("Tespit yapıldı", kod=f"{DOSYA}:BenimProjem.isle", veri={"adet": 3})
        return Sonuc(goruntu=kare, ara_gorseller={"maske": ...}, olaylar=[{"tur": "...", ...}])

if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(BenimProjem)
```

Bu kadarı yeterli: komut satırı (`--esik 0.7 --anlat`) ve panel sekmesi otomatik oluşur.

## Koordinat ve renk kuralları

- Kutular hep `x1, y1, x2, y2` piksel (YOLO'nun `cx cy w h` normalize formatına dönüşüm 01'de).
- OpenCV **BGR**, panel/PIL **RGB**. Dönüşüm yalnızca panel sınırında yapılır (`cizim.bgr2rgb/rgb2bgr`).
- Takipte aynı ID her zaman aynı rengi alır (`cizim.id_rengi`) → ID değişimi gözle fark edilir.
