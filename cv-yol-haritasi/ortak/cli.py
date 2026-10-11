"""
cli.py — Bir projenin Ayarlar dataclass'ından OTOMATİK komut satırı üretir ve çalıştırır.

NEDEN VAR?
    Panel ve komut satırı aynı kodu, aynı ayarlarla çalıştırsın; ikisi arasında gizli fark
    olmasın. Her proje dosyasının sonunda sadece şu vardır:

        if __name__ == "__main__":
            from ortak.cli import calistir
            calistir(SortSifirdan)

    Bu fonksiyon:
      1. Ayarlar alanlarından argparse argümanları üretir (--esik 0.5, --no-kalman ...)
      2. Başta "hangi proje, hangi dosya, hangi ayarlar" bilgisini basar (şeffaflık)
      3. KareIsleyici ise kaynaktaki her kareyi isle()'den geçirir; Eylem ise calistir()'ı çağırır
      4. Sonucu ekranda gösterir / videoya yazar / ara görselleri klasöre kaydeder

ORTAK BAYRAKLAR (KareIsleyici):
    --kaynak 0|video.mp4|klasor/   görüntü kaynağı
    --anlat                        her adımı konsola yaz
    --goster                       OpenCV penceresinde göster (q ile çık)
    --cikti sonuc.mp4              işlenmiş videoyu kaydet
    --debug-klasor dbg/            ara görselleri (maske, Kalman...) kare kare kaydet
    --olaylari-kaydet              Sonuc.olaylar'ı olay deposuna (SQLite) yaz
    --en-fazla 300                 en fazla kaç kare
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .anlat import Anlatici
from .arayuz import Eylem, KareIsleyici, ayar_alanlari, ayarlari_olustur


def _parser_olustur(proje_sinifi) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"[{proje_sinifi.BOLUM}] {proje_sinifi.ISIM}\n\n{proje_sinifi.ACIKLAMA}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("Proje ayarları (dosyanın tepesindeki Ayarlar sınıfından otomatik)")
    for a in ayar_alanlari(proje_sinifi.Ayarlar):
        bayrak = "--" + a.isim.replace("_", "-")
        if a.tip is bool:
            g.add_argument(bayrak, dest=a.isim, action=argparse.BooleanOptionalAction,
                           default=a.varsayilan, help=a.yardim_metni())
        else:
            g.add_argument(bayrak, dest=a.isim, type=a.tip, default=a.varsayilan,
                           choices=a.secenekler, help=a.yardim_metni() + f" (varsayılan: {a.varsayilan})")

    o = p.add_argument_group("Ortak bayraklar")
    o.add_argument("--anlat", action="store_true", help="Her adımı konsola yaz")
    if issubclass(proje_sinifi, KareIsleyici):
        o.add_argument("--kaynak", default="0", help="0 (webcam) | video.mp4 | görüntü | klasör/")
        o.add_argument("--goster", action="store_true", help="OpenCV penceresinde göster (q: çık)")
        o.add_argument("--cikti", default="", help="İşlenmiş videoyu bu yola kaydet (.mp4)")
        o.add_argument("--debug-klasor", default="", help="Ara görselleri bu klasöre kaydet")
        o.add_argument("--olaylari-kaydet", action="store_true", help="Olayları SQLite olay deposuna yaz")
        o.add_argument("--en-fazla", type=int, default=None, help="En fazla kaç kare işlensin")
    else:
        o.add_argument("--cikti-klasor", default="", help="Sonuç görsellerini bu klasöre kaydet")
    return p


def ayarlari_argv(ayar_sinifi, ayarlar) -> list[str]:
    """Ayarlar nesnesini komut satırı argümanlarına geri çevir.

    Panel uzun işleri (eğitim gibi) ayrı süreçte çalıştırırken bunu kullanır ve
    kullanıcıya "Eşdeğer komut: python ... --epochs 50" satırını gösterir.
    """
    argv = []
    for a in ayar_alanlari(ayar_sinifi):
        deger = getattr(ayarlar, a.isim)
        bayrak = "--" + a.isim.replace("_", "-")
        if a.tip is bool:
            argv.append(bayrak if deger else "--no-" + a.isim.replace("_", "-"))
        else:
            argv += [bayrak, str(deger)]
    return argv


def calistir(proje_sinifi, argv: list[str] | None = None):
    """Proje sınıfını komut satırından çalıştır. Testler argv vererek de çağırabilir."""
    args = _parser_olustur(proje_sinifi).parse_args(argv)
    ayarlar = ayarlari_olustur(proje_sinifi.Ayarlar, vars(args))

    # --- Şeffaflık başlığı: ne çalışıyor, hangi ayarlarla?
    print("=" * 78)
    print(f"Proje   : [{proje_sinifi.BOLUM}] {proje_sinifi.ISIM}")
    print(f"Dosya   : {proje_sinifi.DOSYA}")
    print(f"Ayarlar : {json.dumps(vars(ayarlar), ensure_ascii=False)}")
    print("=" * 78)

    anlat = Anlatici(konsola_bas=args.anlat)
    proje = proje_sinifi()

    if issubclass(proje_sinifi, Eylem):
        sonuc = proje.calistir(ayarlar, anlat)
        print(sonuc.metin)
        if sonuc.veri:
            print(json.dumps(sonuc.veri, ensure_ascii=False, indent=2, default=str))
        if args.cikti_klasor:
            _gorselleri_kaydet(sonuc, Path(args.cikti_klasor), "sonuc")
        return sonuc

    return _kare_dongusu(proje, ayarlar, anlat, args)


def _kare_dongusu(proje: KareIsleyici, ayarlar, anlat: Anlatici, args):
    from .kaynak import VideoYazici, kareleri_oku, video_fps

    yazici = VideoYazici(args.cikti, video_fps(args.kaynak)) if args.cikti else None
    depo = None
    if args.olaylari_kaydet:
        from .olay_deposu import OlayDeposu
        depo = OlayDeposu()
        print(f"Olaylar şuraya yazılacak: {depo.yol}")

    son = None
    t0, n = time.perf_counter(), 0
    for no, kare in kareleri_oku(args.kaynak, en_fazla=args.en_fazla):
        anlat.sifirla()
        if args.anlat:
            print(f"--- kare {no} ---")
        son = proje.isle(kare, ayarlar, anlat)
        n += 1
        if depo and son.olaylar:
            depo.toplu_kaydet(son.olaylar)
        cikti = son.goruntu if son.goruntu is not None else kare
        if yazici:
            yazici.yaz(cikti)
        if args.debug_klasor:
            _gorselleri_kaydet(son, Path(args.debug_klasor), f"{no:05d}")
        if args.goster:
            cv2.imshow(proje.ISIM, cikti)
            for isim, g in son.ara_gorseller.items():
                cv2.imshow(f"{proje.ISIM} | {isim}", g)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if yazici:
        yazici.kapat()
        print(f"Video kaydedildi: {args.cikti}")
    if args.goster:
        cv2.destroyAllWindows()
    sure = time.perf_counter() - t0
    print(f"{n} kare işlendi, ortalama {n / sure if sure else 0:.1f} FPS")
    if son is not None and son.metin:
        print("Son karenin özeti:\n" + son.metin)
    return son


def _gorselleri_kaydet(sonuc, klasor: Path, on_ek: str):
    from .cizim import dosya_adi

    klasor.mkdir(parents=True, exist_ok=True)
    if sonuc.goruntu is not None:
        cv2.imwrite(str(klasor / f"{on_ek}_cikti.jpg"), sonuc.goruntu)
    for isim, g in sonuc.ara_gorseller.items():
        if g is not None:
            cv2.imwrite(str(klasor / f"{on_ek}_{dosya_adi(isim)}.jpg"), g)
