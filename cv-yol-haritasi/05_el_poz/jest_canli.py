"""
jest_canli.py — Jestle agent kontrolü: canlı jest tanıma -> komut olayı. Kural tabanlı ve öğrenilmiş, yan yana.

İKİ MOD:
    kural (senin hand controller'ına benzer taban çizgisi):
        sag_kaydir / sol_kaydir : son `pencere` karede bilek x'i > kaydirma_esigi kadar değişti mi?
        avuc                    : 4 parmağın ucu, kendi orta ekleminden (PIP) bileğe daha UZAK mı? (hepsi açık)
        yumruk                  : 4 parmak ucu, PIP ekleminden bileğe daha YAKIN mı? (hepsi kapalı)
    model (jest_egit.py çıktısı):
        Son T karenin özellik dizisi -> dizi_hazirla() (eğitimdekiyle AYNI) -> GRU/MLP -> olasılıklar.
        En olası sınıf 'yok' değilse ve olasılık >= guven_esigi ise tetiklenir.

TETİKLEME KURALLARI (yanlış alarmı azaltmak için):
    - bekleme_sn: Bir jest tetiklendikten sonra bu süre boyunca yeni jest kabul edilmez.
    - Tetiklenince pencere TEMİZLENİR (aynı hareket ikinci kez sayılmasın).

ÇIKTI OLAYI: {"tur": "jest", "nesne": "sag_kaydir", "ek": {"komut": "sonraki_sarki", "mod": "model"}}
    Komutu GERÇEKTEN çalıştırmak (müzik değiştirmek vb.) 09_agent'ın ya da senin işin: bu dosya sadece
    "ne istendi"yi söyler. README'de pyautogui ile medya tuşlarına bağlama örneği var.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402
from ortak.yukle import gerekli, modul_yukle, yol_coz  # noqa: E402

nk = modul_yukle("05_el_poz/noktalar.py")
jk = modul_yukle("05_el_poz/jest_kaydet.py")
je = modul_yukle("05_el_poz/jest_egit.py")
DOSYA = "05_el_poz/jest_canli.py"
PARMAK_UC, PARMAK_PIP = (8, 12, 16, 20), (6, 10, 14, 18)


def statik_el_durumu(n: np.ndarray) -> str:
    """'avuc' | 'yumruk' | 'diger' — parmak ucu ile PIP ekleminin bileğe uzaklığını karşılaştır."""
    bilek = n[0, :2]
    acik = [np.linalg.norm(n[u, :2] - bilek) > np.linalg.norm(n[p, :2] - bilek) for u, p in zip(PARMAK_UC, PARMAK_PIP)]
    return "avuc" if all(acik) else "yumruk" if not any(acik) else "diger"


def kaydirma(bilek_x: list[float], esik: float) -> str | None:
    """Pencere boyunca bilek x değişimi esikten büyükse yön. (Görüntü aynalanmamışsa: x artışı = görüntüde sağa.)"""
    if len(bilek_x) < 2:
        return None
    fark = bilek_x[-1] - bilek_x[0]
    return "sag_kaydir" if fark > esik else "sol_kaydir" if fark < -esik else None


def esleme_coz(metin: str) -> dict[str, str]:
    return {a.strip(): b.strip() for a, b in (p.split("=", 1) for p in metin.split(",") if "=" in p)}


@dataclass
class Ayarlar:
    mod: str = ayar("kural", "kural: el geometrisi | model: eğittiğin GRU/MLP", secenekler=["kural", "model"])
    model: str = ayar("veri/modeller/jest_v1.pt", "model modu: jest_egit.py çıktısı", tur="dosya")
    esleme: str = ayar("sag_kaydir=sonraki_sarki, sol_kaydir=onceki_sarki, avuc=dur, yumruk=sessize_al",
                       "jest=komut eşlemesi", tur="uzun_metin")
    guven_esigi: float = ayar(0.8, "model modu: tetikleme için en düşük olasılık", min=0.3, max=0.99, adim=0.01,
                              etki="Düşürürsen daha duyarlı ama daha çok yanlış tetikleme")
    kaydirma_esigi: float = ayar(0.25, "kural modu: bilek x değişimi (görüntü genişliğine oran)", min=0.05, max=0.8, adim=0.05)
    pencere: int = ayar(12, "kural modu: kaydırma için bakılan kare sayısı", min=3, max=60)
    statik_kare: int = ayar(8, "kural modu: avuç/yumruk kaç kare sürmeli", min=1, max=60)
    bekleme_sn: float = ayar(1.0, "İki tetikleme arası en az süre", min=0.0, max=10.0, adim=0.1)


class JestCanli(KareIsleyici):
    ISIM = "Jestle agent kontrolü (kural vs model)"
    BOLUM = "05"
    DOSYA = DOSYA
    ACIKLAMA = ("Canlı jest tanıma. `kural`: bilek hareketi + parmak geometrisi; `model`: kendi eğittiğin GRU/MLP. Tetiklenen jest "
                "eşlenen komutla `jest` olayı olarak yayınlanır.")
    GEREKSINIMLER = ("mediapipe", "torch (model modu)")
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()
        self._model = None

    def sifirla(self):
        self.ozellikler, self.bilek_x, self.statik = deque(maxlen=90), deque(maxlen=60), deque(maxlen=60)
        self.son_tetik = 0.0

    def _model_yukle(self, yol):
        if self._model is None or self._model[0] != yol:
            torch = gerekli("torch")
            k = torch.load(yol_coz(yol), map_location="cpu", weights_only=False)
            m = je.model_kur(k["tur"], k["T"], k["F"], len(k["siniflar"]), k["gizli"])
            m.load_state_dict(k["durum"])
            m.eval()
            self._model = (yol, m, k["siniflar"], k["T"])
        return self._model[1:]

    def isle(self, kare, a: Ayarlar, anlat):
        eller = nk.el_noktalari(kare, 1)
        cikti, olaylar, ara_veri = kare.copy(), [], {}
        if not eller:
            # El kaybolunca geçmişi temizle: el geri geldiğinde eski konumla "kaydırma" sanılmasın
            self.ozellikler.clear(); self.bilek_x.clear(); self.statik.clear()
            yazi_yaz(cikti, "el yok")
            return Sonuc(goruntu=cikti, metin="El yok.")
        el = eller[0]
        nk.iskelet_ciz(cikti, el["noktalar"], nk.EL_BAGLANTILARI, vurgula=PARMAK_UC)
        self.ozellikler.append(jk.kare_ozelligi(el))
        self.bilek_x.append(float(el["noktalar"][0, 0]))
        self.statik.append(statik_el_durumu(el["noktalar"]))

        jest = None
        if a.mod == "kural":
            son_x = list(self.bilek_x)[-a.pencere:]
            jest = kaydirma(son_x, a.kaydirma_esigi)
            son_statik = list(self.statik)[-a.statik_kare:]
            if jest is None and len(son_statik) == a.statik_kare and len(set(son_statik)) == 1 and son_statik[0] != "diger":
                jest = son_statik[0]
            ara_veri = {"bilek_x_degisim": round(son_x[-1] - son_x[0], 3), "el_durumu": self.statik[-1]}
            anlat.adim(f"Kural: bilek Δx={ara_veri['bilek_x_degisim']} (eşik ±{a.kaydirma_esigi}), el={self.statik[-1]} → {jest}",
                       kod=f"{DOSYA}:kaydirma / statik_el_durumu")
        else:
            torch = gerekli("torch")
            model, siniflar, T = self._model_yukle(a.model)
            if len(self.ozellikler) >= T:
                x = je.dizi_hazirla(np.stack(list(self.ozellikler)[-T:]))
                with torch.no_grad():
                    p = torch.softmax(model(torch.from_numpy(x)[None]), 1)[0].numpy()
                en = int(p.argmax())
                ara_veri = {s: round(float(v), 2) for s, v in zip(siniflar, p)}
                anlat.adim(f"Model: {siniflar[en]} ({p[en]:.2f})", kod="05_el_poz/jest_egit.py:dizi_hazirla → model", veri=ara_veri)
                if siniflar[en] != "yok" and p[en] >= a.guven_esigi:
                    jest = siniflar[en]
            else:
                anlat.bilgi(f"Pencere doluyor: {len(self.ozellikler)}/{T}")

        simdi = time.time()
        if jest and simdi - self.son_tetik >= a.bekleme_sn:
            komut = esleme_coz(a.esleme).get(jest, "")
            olaylar.append({"tur": "jest", "kaynak": DOSYA, "nesne": jest, "ek": {"komut": komut, "mod": a.mod}})
            anlat.adim(f"TETİKLENDİ: {jest} → komut '{komut}'")
            self.son_tetik = simdi
            self.ozellikler.clear(); self.bilek_x.clear(); self.statik.clear()   # aynı hareket iki kez sayılmasın
        yazi_yaz(cikti, [f"mod: {a.mod}", f"son jest: {olaylar[0]['nesne'] if olaylar else '-'}"])
        return Sonuc(goruntu=cikti, olaylar=olaylar, veri=ara_veri,
                     metin=f"Tetiklenen: **{olaylar[0]['nesne']} → {olaylar[0]['ek']['komut']}**" if olaylar else "Jest yok.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(JestCanli)
