"""
kamera_7x24.py — Olay üreticiyi (09) GÜNLERCE kesintisiz çalıştırmak için dayanıklı çalıştırıcı.

Panel ve "--goster"lı denemeler öğrenmek içindir. 7/24 çalışan bir sistemde başka sorunlar çıkar; bu dosya
her birini açıkça çözer:

    SORUN                                   ÇÖZÜM (bu dosyada)
    ─────────────────────────────────────── ─────────────────────────────────────────────────────────────
    İşleme yavaşsa kareler kuyrukta birikir  SonKareOkuyucu: ayrı bir thread kameradan SÜREKLİ okur, sadece
    ve sistem gerçek zamandan geri kalır     EN SON kareyi tutar. İşleyici hazır olunca en tazesini alır;
                                             aradakiler bilerek atlanır ("atlanan kare" sayılır).
    Kamera/RTSP bağlantısı kopar             Okuyucu kopunca bekleyip yeniden bağlanır (yeniden_baglan_sn).
    Bir karede beklenmeyen hata              Hata log'lanır, döngü DEVAM eder (tek kare sistemi çökertmez).
    Docker/systemd durdurmak ister (SIGTERM)  Sinyal yakalanır, kaynaklar düzgün kapatılır.
    "Çalışıyor mu?" bilinmez                 Her nabiz_sn'de bir NABIZ satırı: FPS, atlanan kare, olay sayısı.
    Enerji                                   09'daki hareket kapısı durgun sahnede tespiti atlar.

ÇALIŞTIR:
    python 10_dagitim/kamera_7x24.py --kaynak 0 --model yolo11n.onnx
    python 10_dagitim/kamera_7x24.py --kaynak rtsp://kullanici:sifre@192.168.1.20/stream --model best.onnx
    (Bütün 09 OlayUreteci ayarları da kullanılabilir: --vlm, --guven 0.5, --esya-esleme "..." vb.)
    Olaylar her zaman olay deposuna yazılır (veri/olaylar.db) -> 09 asistan / MCP sunucusu okur.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from ortak.anlat import Anlatici  # noqa: E402
from ortak.arayuz import ayarlari_olustur  # noqa: E402
from ortak.cli import _parser_olustur  # noqa: E402
from ortak.olay_deposu import OlayDeposu  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

ou = modul_yukle("09_agent/olay_ureteci.py")
log = logging.getLogger("kamera_7x24")


class SonKareOkuyucu:
    """Kameradan ayrı thread'de okur; sadece EN SON kareyi saklar (eskiler bilerek atılır)."""

    def __init__(self, kaynak: str, yeniden_baglan_sn: float = 5.0, gercek_zaman: bool = True):
        self.kaynak, self.yeniden_baglan_sn, self.gercek_zaman = kaynak, yeniden_baglan_sn, gercek_zaman
        self._kare, self._no, self._kilit = None, 0, threading.Lock()
        self._dur = threading.Event()
        self.okunan = 0
        self.bitti = False                     # video dosyası sonuna gelindi mi
        self._thread = threading.Thread(target=self._dongu, daemon=True)

    def baslat(self):
        self._thread.start()
        return self

    def _ac(self):
        return cv2.VideoCapture(int(self.kaynak) if self.kaynak.isdigit() else self.kaynak)

    def _dongu(self):
        dosya_mi = Path(self.kaynak).exists()
        while not self._dur.is_set():
            cap = self._ac()
            if not cap.isOpened():
                log.warning("Kaynak açılamadı: %s — %.0f sn sonra tekrar", self.kaynak, self.yeniden_baglan_sn)
                self._dur.wait(self.yeniden_baglan_sn)
                continue
            log.info("Kaynak açıldı: %s", self.kaynak)
            while not self._dur.is_set():
                ok, kare = cap.read()
                if not ok:
                    break
                with self._kilit:
                    self._kare, self._no = kare, self._no + 1
                self.okunan += 1
                if dosya_mi and not self.gercek_zaman:
                    while not self._dur.is_set() and self._kare is not None:   # dosya testinde her kareyi işlet
                        time.sleep(0.001)
            cap.release()
            if dosya_mi:
                self.bitti = True
                return
            log.warning("Akış koptu — %.0f sn sonra yeniden bağlanılacak", self.yeniden_baglan_sn)
            self._dur.wait(self.yeniden_baglan_sn)

    def al(self):
        """(kare_no, kare) — yeni kare yoksa (no, None). Alınan kare tüketilir."""
        with self._kilit:
            kare, no = self._kare, self._no
            self._kare = None
        return no, kare

    def durdur(self):
        self._dur.set()
        self._thread.join(timeout=5)


def calistir(argv=None):
    p = _parser_olustur(ou.OlayUreteci)            # 09'un bütün ayarları + ortak bayraklar (--kaynak ...)
    g = p.add_argument_group("7/24 çalıştırıcı")
    g.add_argument("--sure-sn", type=float, default=0, help="Bu kadar saniye sonra dur (0 = sonsuza dek)")
    g.add_argument("--nabiz-sn", type=float, default=60, help="Kaç saniyede bir NABIZ log satırı")
    g.add_argument("--yeniden-baglan-sn", type=float, default=5, help="Kopunca bekleme süresi")
    g.add_argument("--log-dosyasi", default="", help="Log'u ayrıca bu dosyaya yaz")
    g.add_argument("--her-kare", action="store_true", help="(video dosyası testi) kare atlamadan her kareyi işle")
    args = p.parse_args(argv)
    isleyiciler = [logging.StreamHandler()] + ([logging.FileHandler(args.log_dosyasi, encoding="utf-8")] if args.log_dosyasi else [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=isleyiciler, force=True)

    ayarlar = ayarlari_olustur(ou.OlayUreteci.Ayarlar, vars(args))
    log.info("Başlıyor: kaynak=%s ayarlar=%s", args.kaynak, vars(ayarlar))
    isleyici, depo = ou.OlayUreteci(), OlayDeposu()
    okuyucu = SonKareOkuyucu(args.kaynak, args.yeniden_baglan_sn, gercek_zaman=not args.her_kare).baslat()

    durdur = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (log.info("Durdurma sinyali alındı"), durdur.set()))

    t0 = son_nabiz = time.time()
    islenen = hatalar = toplam_olay = 0
    son_no = 0
    while not durdur.is_set():
        if args.sure_sn and time.time() - t0 >= args.sure_sn:
            break
        no, kare = okuyucu.al()
        if kare is None:
            if okuyucu.bitti:
                break
            time.sleep(0.005)
            continue
        son_no = no
        try:
            sonuc = isleyici.isle(kare, ayarlar, Anlatici())
            if sonuc.olaylar:
                depo.toplu_kaydet(sonuc.olaylar)
                toplam_olay += len(sonuc.olaylar)
                log.info("Olaylar: %s", ", ".join(o["tur"] for o in sonuc.olaylar))
            islenen += 1
        except Exception:  # noqa: BLE001 — tek kare sistemi çökertmesin
            hatalar += 1
            log.exception("Kare %d işlenirken hata (devam ediliyor)", no)
        if time.time() - son_nabiz >= args.nabiz_sn:
            gecen = time.time() - t0
            log.info("NABIZ: %.0f sn | işlenen %d (%.1f FPS) | kameradan okunan %d | atlanan %d | olay %d | hata %d",
                     gecen, islenen, islenen / gecen, okuyucu.okunan, max(0, son_no - islenen), toplam_olay, hatalar)
            son_nabiz = time.time()
    okuyucu.durdur()
    ozet = {"islenen": islenen, "okunan": okuyucu.okunan, "olay": toplam_olay, "hata": hatalar, "sure_sn": round(time.time() - t0, 1)}
    log.info("Durdu: %s", ozet)
    return ozet


if __name__ == "__main__":
    calistir()
