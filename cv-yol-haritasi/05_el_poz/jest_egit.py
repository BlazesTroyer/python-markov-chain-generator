"""
jest_egit.py — Dinamik jest sınıflandırıcı eğit: GRU (zaman serisi) vs MLP (düzleştirilmiş), açık eğitim döngüsü.

GİRDİ: veri/jestler/<etiket>/*.npy, her biri (T, 65). (jest_kaydet.py üretir)

ÖN İŞLEME — dizi_hazirla() (canli tahminde de AYNEN kullanılır; eğitim/çıkarım tutarsızlığı en sık hatadır):
    bilek (x, y) sütunları dizinin İLK karesine göre görelileştirilir -> "nerede başladın" değil,
    "ne kadar ve hangi yöne hareket ettin" öğrenilir.

İKİ MODEL:
    gru : Kareleri sırayla okur, gizli durumunu (hafızasını) günceller; son gizli durum -> sınıf.
          Zamandaki SIRAYI doğal olarak anlar (sağa mı kaydı, sola mı?).
    mlp : Bütün diziyi (T*65) tek vektör yapar. Basit ve hızlı ama zaman kaymalarına hassas.
    Aynı veride ikisini karşılaştır: hangisi daha iyi genelliyor?

AUGMENTATION (sadece train, her epoch yeniden): noktalar.nokta_augment
    döndürme ±15°, ölçek ±10%, gürültü, zaman kaydırma ±3 kare.

ÇIKTI: veri/modeller/jest_<isim>.pt  (+ eğriler, karışıklık matrisi)
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import egri_ciz, matris_gorsel  # noqa: E402
from ortak.yukle import VERI, gerekli, modul_yukle, yol_coz  # noqa: E402

nk = modul_yukle("05_el_poz/noktalar.py")
eg4 = modul_yukle("04_yuz/emotion_v2/egit.py")   # oturuma_gore_bol'u yeniden kullan (aynı sızıntı kuralı)
DOSYA = "05_el_poz/jest_egit.py"


def dizi_hazirla(d: np.ndarray) -> np.ndarray:
    """(T, 65) -> (T, 65): bilek konumunu ilk kareye göre görelileştir."""
    d = d.astype(np.float32).copy()
    d[:, 63:65] -= d[0, 63:65]
    return d


def model_kur(tur: str, T: int, F: int, sinif: int, gizli: int):
    torch = gerekli("torch")
    nn = torch.nn

    class GRUJest(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(F, gizli, batch_first=True)    # (B, T, F) -> (B, T, gizli)
            self.cikis = nn.Linear(gizli, sinif)

        def forward(self, x):
            _, h = self.gru(x)                                 # h: (1, B, gizli) = son gizli durum
            return self.cikis(h[-1])

    class MLPJest(nn.Module):
        def __init__(self):
            super().__init__()
            self.ag = nn.Sequential(nn.Flatten(), nn.Linear(T * F, gizli), nn.ReLU(), nn.Dropout(0.2), nn.Linear(gizli, sinif))

        def forward(self, x):
            return self.ag(x)

    return GRUJest() if tur == "gru" else MLPJest()


@dataclass
class Ayarlar:
    klasor: str = ayar("veri/jestler", "Jest klasörü (alt klasör = etiket)", tur="klasor")
    isim: str = ayar("v1", "Model adı -> veri/modeller/jest_<isim>.pt")
    model: str = ayar("gru", "Model türü", secenekler=["gru", "mlp"], etki="İkisini de eğitip val doğruluğunu karşılaştır")
    gizli: int = ayar(64, "Gizli katman boyutu", min=8, max=512)
    augmentation: bool = ayar(True, "Nokta augmentation (döndürme/ölçek/gürültü/zaman kaydırma)")
    epochs: int = ayar(60, "Epoch", min=1, max=500)
    lr: float = ayar(0.003, "Öğrenme oranı", min=0.0001, max=0.05, adim=0.0001)
    batch: int = ayar(32, "Batch", min=2, max=256)
    val_orani: float = ayar(0.25, "Doğrulama oranı (oturum grubu bazında)", min=0.05, max=0.5, adim=0.05)
    tohum: int = ayar(0, "Tohum", min=0, max=9999)


class JestEgit(Eylem):
    ISIM = "Jest modeli eğit (GRU / MLP)"
    BOLUM = "05"
    DOSYA = DOSYA
    ACIKLAMA = ("Kaydedilen jest dizileriyle GRU ya da MLP eğitir; nokta augmentation, oturum bazlı bölme, açık eğitim döngüsü, "
                "eğriler ve karışıklık matrisi.")
    GEREKSINIMLER = ("torch",)
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        torch = gerekli("torch")
        torch.manual_seed(a.tohum)
        rng = np.random.default_rng(a.tohum)
        kok = yol_coz(a.klasor)
        siniflar = sorted(p.name for p in kok.iterdir() if p.is_dir())
        ornekler = []
        for no, s in enumerate(siniflar):
            for f in sorted((kok / s).glob("*.npy")):
                ornekler.append((f, no, f.stem.rsplit("_", 1)[0]))
        diziler = {str(f): dizi_hazirla(np.load(f)) for f, _, _ in ornekler}
        T, F = next(iter(diziler.values())).shape
        if any(d.shape != (T, F) for d in diziler.values()):
            raise ValueError("Bütün diziler aynı uzunlukta olmalı (jest_kaydet 'uzunluk' ayarını değiştirmeyin)")
        train, val, gruplu = eg4.oturuma_gore_bol(ornekler, a.val_orani, rng)
        anlat.adim(f"{len(siniflar)} jest {siniflar}, dizi şekli (T={T}, F={F}); train {len(train)}, val {len(val)}",
                   kod=f"{DOSYA}:dizi_hazirla", veri={siniflar[k]: v for k, v in sorted(Counter(e for _, e, _ in train).items())})
        if not gruplu:
            anlat.uyari("Tek oturum: rastgele bölündü (val iyimser olabilir).")

        def tensorlar(liste, augment):
            X = np.stack([nk.nokta_augment(diziler[str(f)], rng) if augment else diziler[str(f)] for f, _, _ in liste])
            return torch.from_numpy(X), torch.tensor([e for _, e, _ in liste])

        model = model_kur(a.model, T, F, len(siniflar), a.gizli)
        anlat.adim(f"Model: {model}", kod=f"{DOSYA}:model_kur")
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        kayip_fn = torch.nn.CrossEntropyLoss()
        Xv, yv = tensorlar(val, False)
        gecmis = {"train_kayip": [], "val_kayip": [], "val_dogruluk": []}
        en_iyi, en_iyi_tahmin = -1.0, None
        yol = VERI / "modeller" / f"jest_{a.isim}.pt"
        yol.parent.mkdir(parents=True, exist_ok=True)
        for ep in range(a.epochs):
            model.train()
            Xt, yt = tensorlar(train, a.augmentation)          # her epoch YENİ augmentation
            sira = torch.randperm(len(yt))
            toplam = 0.0
            for i in range(0, len(sira), a.batch):
                idx = sira[i:i + a.batch]
                opt.zero_grad()
                kayip = kayip_fn(model(Xt[idx]), yt[idx])
                kayip.backward()
                opt.step()
                toplam += kayip.item() * len(idx)
            model.eval()
            with torch.no_grad():
                cik = model(Xv)
                vk = kayip_fn(cik, yv).item()
                tahmin = cik.argmax(1)
                dogruluk = (tahmin == yv).float().mean().item()
            gecmis["train_kayip"].append(toplam / len(yt)); gecmis["val_kayip"].append(vk); gecmis["val_dogruluk"].append(dogruluk)
            if (ep + 1) % max(1, a.epochs // 10) == 0 or ep == 0:
                anlat.adim(f"epoch {ep + 1}: train_kayıp={gecmis['train_kayip'][-1]:.3f} val_kayıp={vk:.3f} val_doğruluk={dogruluk:.3f}",
                           kod=f"{DOSYA}:JestEgit.calistir (döngü)")
            if dogruluk > en_iyi:
                en_iyi, en_iyi_tahmin = dogruluk, tahmin.tolist()
                torch.save({"durum": model.state_dict(), "tur": a.model, "siniflar": siniflar, "T": T, "F": F, "gizli": a.gizli}, yol)
        cm = np.zeros((len(siniflar),) * 2, int)
        for g, t in zip(yv.tolist(), en_iyi_tahmin):
            cm[g, t] += 1
        egri = egri_ciz({"train_kayip": gecmis["train_kayip"], "val_kayip": gecmis["val_kayip"]}, "kayip")
        return Sonuc(goruntu=egri, ara_gorseller={"kayıp": egri, "val doğruluğu": egri_ciz({"val_dogruluk": gecmis["val_dogruluk"]}, "dogruluk"),
                                                  "karışıklık matrisi": matris_gorsel(cm, siniflar)},
                     veri={"val_dogruluk": en_iyi, "model": str(yol), "siniflar": siniflar},
                     metin=f"En iyi val doğruluğu **{en_iyi:.3f}** ({a.model}). Model: `{yol}` → jest_canli.py")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(JestEgit)
