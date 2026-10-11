"""
olay_deposu.py — Kişisel agent'ın "hafızası": görülen olayları SQLite'a yazar/okur.

NEDEN VAR?
    Görüntü modelleri saniyede onlarca kare işler ama agent'ın ihtiyacı "ham kare" değil
    ANLAMLI OLAY'dır:
        - "18:42'de anahtar masada görüldü"         (02 esya_hafizasi)
        - "Telefon elime alındı / bırakıldı"          (03 ekran_suresi)
        - "Uğur odaya girdi"                          (04 yuz_tanima)
        - "Sağa kaydırma jesti yapıldı"               (05 jest_canli)
    Bunları tek bir tabloya yazarız; 09_agent/mcp_server.py bu tabloyu sorgulayarak
    where_is(), screen_time_today() gibi araçları cevaplar.

TABLO: olaylar
    id        INTEGER  otomatik
    zaman     REAL     unix zamanı (time.time())
    tur       TEXT     "nesne_goruldu" | "telefon_alindi" | "kisi_geldi" | "jest" | ...
    kaynak    TEXT     hangi proje yazdı ("02_yolo_finetune/esya_hafizasi.py")
    nesne     TEXT     ilgili nesne/kişi adı ("anahtar", "Uğur")
    kimlik    TEXT     takip ID'si (varsa)
    kutu      TEXT     JSON [x1,y1,x2,y2] (varsa)
    guven     REAL     model güveni (varsa)
    ek        TEXT     JSON — projeye özel ek bilgi
    kare_yolu TEXT     o anın kırpılmış görüntüsü kaydedildiyse yolu

VERİTABANI YERİ: cv-yol-haritasi/veri/olaylar.db  (CV_OLAY_DB ortam değişkeniyle değişir)
GİZLİLİK: Her şey yerel diskte; hiçbir yere gönderilmez.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .yukle import VERI  # noqa: E402

VARSAYILAN_DB = VERI / "olaylar.db"

_SEMA = """
CREATE TABLE IF NOT EXISTS olaylar (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    zaman     REAL NOT NULL,
    tur       TEXT NOT NULL,
    kaynak    TEXT,
    nesne     TEXT,
    kimlik    TEXT,
    kutu      TEXT,
    guven     REAL,
    ek        TEXT,
    kare_yolu TEXT
);
CREATE INDEX IF NOT EXISTS ix_tur_zaman   ON olaylar(tur, zaman);
CREATE INDEX IF NOT EXISTS ix_nesne_zaman ON olaylar(nesne, zaman);
"""


class OlayDeposu:
    """Küçük, okunur bir SQLite sarmalayıcı. Bağlantı her çağrıda açılıp kapanır (thread-güvenli)."""

    def __init__(self, yol: str | Path | None = None):
        self.yol = Path(yol or os.environ.get("CV_OLAY_DB", VARSAYILAN_DB))
        self.yol.parent.mkdir(parents=True, exist_ok=True)
        with self._baglan() as db:
            db.executescript(_SEMA)

    def _baglan(self):
        db = sqlite3.connect(self.yol)
        db.row_factory = sqlite3.Row  # satırları sözlük gibi okuyabilmek için
        return db

    # ------------------------------------------------------------------ yazma
    def kaydet(self, tur: str, *, kaynak: str = "", nesne: str = "", kimlik=None, kutu=None,
               guven: float | None = None, ek: dict | None = None, kare_yolu: str = "",
               zaman: float | None = None) -> int:
        """Bir olay ekle, yeni satırın id'sini döndür."""
        with self._baglan() as db:
            imlec = db.execute(
                "INSERT INTO olaylar (zaman, tur, kaynak, nesne, kimlik, kutu, guven, ek, kare_yolu)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    zaman or time.time(), tur, kaynak, nesne,
                    None if kimlik is None else str(kimlik),
                    None if kutu is None else json.dumps([round(float(v), 1) for v in kutu]),
                    guven, json.dumps(ek or {}, ensure_ascii=False), kare_yolu,
                ),
            )
            return imlec.lastrowid

    def toplu_kaydet(self, olaylar: list[dict]) -> int:
        """Sonuc.olaylar listesini (sözlükler) olduğu gibi yaz. Her sözlükte en az 'tur' olmalı."""
        for o in olaylar:
            o = dict(o)
            self.kaydet(o.pop("tur"), **o)
        return len(olaylar)

    # ------------------------------------------------------------------ okuma
    def sorgula(self, tur: str | None = None, nesne: str | None = None, baslangic: float | None = None,
                bitis: float | None = None, limit: int = 200) -> list[dict]:
        """Filtreli olay listesi (yeniden eskiye)."""
        kosul, param = [], []
        if tur:
            kosul.append("tur = ?"); param.append(tur)
        if nesne:
            kosul.append("LOWER(nesne) = LOWER(?)"); param.append(nesne)
        if baslangic is not None:
            kosul.append("zaman >= ?"); param.append(baslangic)
        if bitis is not None:
            kosul.append("zaman <= ?"); param.append(bitis)
        sql = "SELECT * FROM olaylar"
        if kosul:
            sql += " WHERE " + " AND ".join(kosul)
        sql += " ORDER BY zaman DESC LIMIT ?"
        param.append(limit)
        with self._baglan() as db:
            return [self._satir(r) for r in db.execute(sql, param)]

    def son_gorulme(self, nesne: str) -> dict | None:
        """Bir nesnenin/kişinin en son kaydı (where_is aracı bunu kullanır)."""
        sonuc = self.sorgula(nesne=nesne, limit=1)
        return sonuc[0] if sonuc else None

    def turler(self) -> list[tuple[str, int]]:
        """Hangi olay türünden kaç tane var? (panelde özet)"""
        with self._baglan() as db:
            return [(r[0], r[1]) for r in db.execute("SELECT tur, COUNT(*) FROM olaylar GROUP BY tur ORDER BY 2 DESC")]

    def temizle(self) -> None:
        """Bütün olayları sil (deneme yaparken)."""
        with self._baglan() as db:
            db.execute("DELETE FROM olaylar")

    @staticmethod
    def _satir(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["kutu"] = json.loads(d["kutu"]) if d["kutu"] else None
        d["ek"] = json.loads(d["ek"]) if d["ek"] else {}
        d["saat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["zaman"]))
        return d
