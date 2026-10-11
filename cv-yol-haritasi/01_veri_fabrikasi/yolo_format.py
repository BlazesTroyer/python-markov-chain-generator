"""
yolo_format.py — YOLO etiket formatı: okuma, yazma, dönüştürme. (Bölüm 01 ve 02'nin ortak dili)

YOLO ETİKET FORMATI NEDİR?
    Her görüntü için AYNI İSİMLİ bir .txt dosyası vardır:
        images/train/kupa_001.jpg   <->   labels/train/kupa_001.txt
    .txt içinde her satır bir nesnedir:
        <sinif_no> <cx> <cy> <w> <h>
    - sinif_no : 0'dan başlayan tam sayı (data.yaml'daki names listesindeki sırası)
    - cx, cy   : kutu MERKEZİ, görüntü genişliğine/yüksekliğine bölünmüş (0..1)
    - w, h     : kutu genişliği/yüksekliği, yine 0..1
    Örnek: "2 0.5 0.5 0.25 0.4" -> 2 numaralı sınıf, tam ortada, görüntünün %25 eni, %40 boyu.
    Nesne olmayan görüntü = BOŞ .txt (ya da hiç .txt yok). Bunlar "negatif örnek"tir ve faydalıdır!

    Neden normalize? Görüntü yeniden boyutlandırılınca (640x640'a) etiketler değişmesin diye.

KLASÖR DÜZENİ (Ultralytics'in beklediği):
    <dataset>/
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt
      images/test/*.jpg    labels/test/*.txt
      data.yaml            -> path, train, val, test, names

Bu projede iç formatımız piksel kutu (x1,y1,x2,y2). Dönüşümler burada, tek yerde.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def piksel_to_yolo(kutu, g: int, y: int) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2) piksel -> (cx,cy,w,h) normalize."""
    x1, y1, x2, y2 = kutu
    return ((x1 + x2) / 2 / g, (y1 + y2) / 2 / y, (x2 - x1) / g, (y2 - y1) / y)


def yolo_to_piksel(cx, cy, w, h, g: int, y: int) -> tuple[float, float, float, float]:
    """(cx,cy,w,h) normalize -> (x1,y1,x2,y2) piksel."""
    return ((cx - w / 2) * g, (cy - h / 2) * y, (cx + w / 2) * g, (cy + h / 2) * y)


def etiket_oku(txt_yolu: Path, g: int, y: int) -> tuple[np.ndarray, list[int], list[str]]:
    """Bir .txt etiket dosyasını oku.

    Döndürür: (kutular (N,4) piksel, sinif_nolari, sorunlar)
    'sorunlar' bozuk satırları açıklar (etiket_kontrol.py raporunda görünür).
    """
    kutular, siniflar, sorunlar = [], [], []
    if not Path(txt_yolu).exists():
        return np.zeros((0, 4), np.float32), [], []
    for i, satir in enumerate(Path(txt_yolu).read_text().splitlines(), 1):
        parca = satir.split()
        if not parca:
            continue
        if len(parca) != 5:
            sorunlar.append(f"{Path(txt_yolu).name}:{i} 5 değer bekleniyordu, {len(parca)} var")
            continue
        s, cx, cy, w, h = int(float(parca[0])), *map(float, parca[1:])
        if not all(0 <= v <= 1 for v in (cx, cy, w, h)):
            sorunlar.append(f"{Path(txt_yolu).name}:{i} değerler 0..1 dışında ({cx:.2f},{cy:.2f},{w:.2f},{h:.2f})")
        if w <= 0 or h <= 0:
            sorunlar.append(f"{Path(txt_yolu).name}:{i} sıfır/negatif boyut")
            continue
        kutular.append(yolo_to_piksel(cx, cy, w, h, g, y))
        siniflar.append(s)
    return np.array(kutular, np.float32).reshape(-1, 4), siniflar, sorunlar


def etiket_yaz(txt_yolu: Path, kutular, siniflar, g: int, y: int) -> None:
    """Piksel kutuları YOLO .txt olarak yaz (boş liste -> boş dosya = negatif örnek)."""
    Path(txt_yolu).parent.mkdir(parents=True, exist_ok=True)
    satirlar = []
    for k, s in zip(kutular, siniflar):
        cx, cy, w, h = piksel_to_yolo(k, g, y)
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        satirlar.append(f"{int(s)} {cx:.6f} {cy:.6f} {min(w, 1):.6f} {min(h, 1):.6f}")
    Path(txt_yolu).write_text("\n".join(satirlar) + ("\n" if satirlar else ""))


def etiket_yolu(goruntu_yolu: Path) -> Path:
    """images/train/a.jpg -> labels/train/a.txt  (Ultralytics kuralı: yoldaki 'images' -> 'labels')."""
    p = Path(goruntu_yolu)
    parcalar = list(p.parts)
    for i in range(len(parcalar) - 1, -1, -1):
        if parcalar[i] == "images":
            parcalar[i] = "labels"
            break
    return Path(*parcalar).with_suffix(".txt")


def goruntuleri_listele(klasor: Path) -> list[Path]:
    return sorted(p for p in Path(klasor).rglob("*") if p.suffix.lower() in GORUNTU_UZANTILARI)


def data_yaml_yaz(dataset_klasoru: Path, sinif_isimleri: list[str], test_var: bool = True) -> Path:
    """Ultralytics'in okuduğu data.yaml'ı yaz (PyYAML gerektirmeden, elle — format açıkça görünsün)."""
    dataset_klasoru = Path(dataset_klasoru).resolve()
    satirlar = [
        "# Ultralytics dataset tanımı (01_veri_fabrikasi/yolo_format.py:data_yaml_yaz üretti)",
        f"path: {dataset_klasoru}",
        "train: images/train",
        "val: images/val",
    ]
    if test_var:
        satirlar.append("test: images/test")
    satirlar.append("names:")
    satirlar += [f"  {i}: {isim}" for i, isim in enumerate(sinif_isimleri)]
    yol = dataset_klasoru / "data.yaml"
    yol.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    return yol


def data_yaml_oku(yol: Path) -> dict:
    """data.yaml'dan names ve path'i oku (basit ayrıştırıcı; PyYAML varsa onu kullanır)."""
    try:
        import yaml
        return yaml.safe_load(Path(yol).read_text(encoding="utf-8"))
    except ImportError:
        veri, names, names_blogu = {}, {}, False
        for satir in Path(yol).read_text(encoding="utf-8").splitlines():
            if satir.startswith("#") or not satir.strip():
                continue
            if satir.startswith("names:"):
                names_blogu = True
                continue
            if names_blogu and satir.startswith("  "):
                k, v = satir.strip().split(":", 1)
                names[int(k)] = v.strip()
            else:
                names_blogu = False
                k, v = satir.split(":", 1)
                veri[k.strip()] = v.strip()
        veri["names"] = names
        return veri
