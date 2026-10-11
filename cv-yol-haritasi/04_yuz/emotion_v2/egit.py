"""
egit.py — Emotion Checker v2: küçük bir CNN'i kendi duygu dataset'inle fine-tune et.
          Eğitim döngüsü (forward → loss → backward → step) AÇIKÇA yazılı; hiçbir "fit()" sihri yok.

NE ÖĞRETİR?
    1. Transfer learning (sınıflandırma): ImageNet'te eğitilmiş bir ağın son katmanını
       (classifier/head) senin duygu sınıflarına göre değiştirip eğitmek.
    2. AUGMENTATION'ın sınıflandırmadaki hali (torchvision transforms):
         RandomResizedCrop (yakınlaşma/kayma), HorizontalFlip (yüz simetrik — güvenli),
         ColorJitter (ışık), RandomRotation (kafa eğimi). Val'e ASLA uygulanmaz.
       "augmentation_onizleme" ara görseli: modelin GERÇEKTE gördüğü eğitim görüntüleri.
    3. SINIF DENGESİZLİĞİ: "nötr"den 500, "şaşırmış"tan 50 örnek varsa model hep "nötr" der ve
       yine de %80 doğruluk alır! Çözüm: kayıp fonksiyonunda az olan sınıfa daha büyük AĞIRLIK:
           agirlik[c] = toplam / (sinif_sayisi * adet[c])
    4. dondur=True: Sadece son katman eğitilir (hızlı, az veride kararlı). False: bütün ağ.
    5. Karışıklık matrisi: hangi duygu hangisiyle karışıyor? (ör. "üzgün" ↔ "yorgun")

VERİ: veri/duygu/<sinif>/<oturum>_<zaman>.jpg  (duygu_topla.py üretir; kendi dataset'in de bu düzende olabilir)
      Bölme OTURUMA göre yapılır (sızıntı önlemi). Tek oturum varsa rastgele bölünür ve UYARI verilir.
ÇIKTI: veri/modeller/duygu_<isim>/model.pt  (+ egriler.png, karisiklik.png)
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import egri_ciz, izgara, matris_gorsel  # noqa: E402
from ortak.yukle import VERI, gerekli, yol_coz  # noqa: E402

DOSYA = "04_yuz/emotion_v2/egit.py"
IMAGENET_ORT = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Ayarlar:
    klasor: str = ayar("veri/duygu", "Sınıf klasörlerinin olduğu yer", tur="klasor")
    isim: str = ayar("v2", "Model adı -> veri/modeller/duygu_<isim>/")
    model: str = ayar("mobilenetv3_small_100", "timm modeli", secenekler=["mobilenetv3_small_100", "efficientnet_b0", "resnet18"],
                      etki="mobilenet: CPU'da gerçek zamanlı; efficientnet: daha doğru, daha yavaş")
    onegitimli: bool = ayar(True, "ImageNet ön-eğitimli ağırlıklarla başla (ilk seferde indirilir)",
                            etki="Kapatırsan sıfırdan eğitim: az veride çok daha kötü sonuç — deneyip gör")
    dondur: bool = ayar(False, "Sadece son katmanı eğit (gövdeyi dondur)")
    augmentation: bool = ayar(True, "Eğitimde augmentation uygula", etki="Kapatıp val doğruluğunu karşılaştır")
    sinif_agirligi: bool = ayar(True, "Az örnekli sınıflara kayıpta daha büyük ağırlık ver")
    epochs: int = ayar(15, "Epoch sayısı", min=1, max=200)
    batch: int = ayar(32, "Batch boyutu", min=2, max=256)
    lr: float = ayar(0.0003, "Öğrenme oranı (AdamW)", min=0.00001, max=0.01, adim=0.00001)
    boyut: int = ayar(160, "Girdi boyutu (kare)", min=64, max=320, adim=16)
    val_orani: float = ayar(0.2, "Doğrulama oranı (oturum grubu bazında)", min=0.05, max=0.5, adim=0.05)
    tohum: int = ayar(0, "Tohum", min=0, max=9999)


def ornekleri_listele(kok: Path) -> tuple[list[tuple[Path, int, str]], list[str]]:
    """(yol, sınıf_no, grup) listesi ve sınıf adları. Grup = dosya adının son '_' öncesi (oturum)."""
    siniflar = sorted(p.name for p in kok.iterdir() if p.is_dir())
    ornekler = []
    for no, s in enumerate(siniflar):
        for f in sorted((kok / s).glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                grup = f.stem.rsplit("_", 1)[0] if "_" in f.stem else "tek"
                ornekler.append((f, no, grup))
    return ornekler, siniflar


def oturuma_gore_bol(ornekler, val_orani, rng):
    """Grupları (oturumları) val/train'e ayır. Tek grup varsa örnek bazında rastgele böl."""
    gruplar = sorted({g for _, _, g in ornekler})
    if len(gruplar) < 2:
        idx = rng.permutation(len(ornekler))
        n_val = max(1, int(len(ornekler) * val_orani))
        return [ornekler[i] for i in idx[n_val:]], [ornekler[i] for i in idx[:n_val]], False
    rng.shuffle(gruplar)
    n_val = max(1, int(round(len(gruplar) * val_orani)))
    val_g = set(gruplar[:n_val])
    return [o for o in ornekler if o[2] not in val_g], [o for o in ornekler if o[2] in val_g], True


def donusumler(boyut: int, augmentation: bool, egitim: bool):
    """torchvision v2 dönüşüm hattı. Sıra önemlidir: önce geometrik/renk, en son normalize."""
    T = gerekli("torchvision.transforms.v2", "pip install torchvision")
    torch = gerekli("torch")
    adimlar = [T.ToImage()]
    if egitim and augmentation:
        adimlar += [T.RandomResizedCrop(boyut, scale=(0.75, 1.0), antialias=True),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(12),
                    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25)]
    else:
        adimlar += [T.Resize((boyut, boyut), antialias=True)]
    adimlar += [T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_ORT.tolist(), IMAGENET_STD.tolist())]
    return T.Compose(adimlar)


def tensor_to_bgr(t) -> np.ndarray:
    """Normalize edilmiş tensörü (3,H,W) tekrar gösterilebilir BGR görüntüye çevir (önizleme için)."""
    img = t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_ORT
    return cv2.cvtColor((img.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _veri_kumesi(ornekler, tf):
    torch = gerekli("torch")

    class GoruntuKumesi(torch.utils.data.Dataset):
        def __len__(self):
            return len(ornekler)

        def __getitem__(self, i):
            yol, etiket, _ = ornekler[i]
            rgb = cv2.cvtColor(cv2.imread(str(yol)), cv2.COLOR_BGR2RGB)
            return tf(rgb), etiket
    return GoruntuKumesi()


class DuyguEgit(Eylem):
    ISIM = "Emotion v2 eğit (timm + augmentation)"
    BOLUM = "04"
    DOSYA = DOSYA
    ACIKLAMA = ("Kendi duygu dataset'inle küçük bir CNN'i fine-tune eder. Eğitim döngüsü açıkça yazılı; sınıf ağırlığı, augmentation, "
                "dondurma seçenekleri; çıktıda eğriler, karışıklık matrisi ve **modelin gördüğü augment'li görüntüler**.")
    GEREKSINIMLER = ("torch", "torchvision", "timm")
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        torch, timm = gerekli("torch"), gerekli("timm", "pip install timm")
        torch.manual_seed(a.tohum)
        rng = np.random.default_rng(a.tohum)
        ornekler, siniflar = ornekleri_listele(yol_coz(a.klasor))
        if len(siniflar) < 2:
            raise ValueError("En az 2 sınıf klasörü gerekli")
        train, val, gruplu = oturuma_gore_bol(ornekler, a.val_orani, rng)
        adet = Counter(e for _, e, _ in train)
        anlat.adim(f"{len(siniflar)} sınıf {siniflar}; train {len(train)}, val {len(val)}", kod=f"{DOSYA}:oturuma_gore_bol",
                   veri={siniflar[k]: v for k, v in sorted(adet.items())})
        if not gruplu:
            anlat.uyari("Tek oturum var: rastgele bölündü. Neredeyse aynı kareler train ve val'e düşebilir -> val doğruluğu iyimser!")

        tr_tf, val_tf = donusumler(a.boyut, a.augmentation, True), donusumler(a.boyut, a.augmentation, False)
        tr_dl = torch.utils.data.DataLoader(_veri_kumesi(train, tr_tf), batch_size=a.batch, shuffle=True)
        val_dl = torch.utils.data.DataLoader(_veri_kumesi(val, val_tf), batch_size=a.batch)
        onizleme = [tensor_to_bgr(tr_tf(cv2.cvtColor(cv2.imread(str(train[i % len(train)][0])), cv2.COLOR_BGR2RGB))) for i in range(8)]

        model = timm.create_model(a.model, pretrained=a.onegitimli, num_classes=len(siniflar))
        if a.dondur:
            bas = set(id(p) for p in model.get_classifier().parameters())
            for p in model.parameters():
                p.requires_grad = id(p) in bas
        egitilen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        anlat.adim(f"Model {a.model} (ön-eğitimli={a.onegitimli}); eğitilen parametre: {egitilen:,} / {sum(p.numel() for p in model.parameters()):,}",
                   kod="timm.create_model")

        agirlik = None
        if a.sinif_agirligi:
            agirlik = torch.tensor([len(train) / (len(siniflar) * max(adet.get(c, 0), 1)) for c in range(len(siniflar))], dtype=torch.float32)
            anlat.bilgi("Sınıf ağırlıkları: " + ", ".join(f"{s}={w:.2f}" for s, w in zip(siniflar, agirlik.tolist())))
        kayip_fn = torch.nn.CrossEntropyLoss(weight=agirlik)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)

        gecmis = {"train_kayip": [], "val_kayip": [], "val_dogruluk": []}
        en_iyi, en_iyi_durum, cikti = -1.0, None, VERI / "modeller" / f"duygu_{a.isim}"
        cikti.mkdir(parents=True, exist_ok=True)
        for ep in range(a.epochs):
            # ---- EĞİTİM: forward -> loss -> backward -> step
            model.train()
            toplam = 0.0
            for x, y in tr_dl:
                opt.zero_grad()
                kayip = kayip_fn(model(x), y)
                kayip.backward()
                opt.step()
                toplam += kayip.item() * len(y)
            # ---- DOĞRULAMA: gradyan yok, augmentation yok
            model.eval()
            vk, dogru, tahminler, gercekler = 0.0, 0, [], []
            with torch.no_grad():
                for x, y in val_dl:
                    cik = model(x)
                    vk += kayip_fn(cik, y).item() * len(y)
                    t = cik.argmax(1)
                    dogru += (t == y).sum().item()
                    tahminler += t.tolist(); gercekler += y.tolist()
            gecmis["train_kayip"].append(toplam / len(train))
            gecmis["val_kayip"].append(vk / max(len(val), 1))
            gecmis["val_dogruluk"].append(dogru / max(len(val), 1))
            anlat.adim(f"epoch {ep + 1}/{a.epochs}: train_kayıp={gecmis['train_kayip'][-1]:.3f} val_kayıp={gecmis['val_kayip'][-1]:.3f} "
                       f"val_doğruluk={gecmis['val_dogruluk'][-1]:.3f}", kod=f"{DOSYA}:DuyguEgit.calistir (döngü)")
            if gecmis["val_dogruluk"][-1] > en_iyi:
                en_iyi = gecmis["val_dogruluk"][-1]
                en_iyi_durum = (ep + 1, list(tahminler), list(gercekler))
                torch.save({"durum": model.state_dict(), "model": a.model, "siniflar": siniflar, "boyut": a.boyut}, cikti / "model.pt")

        ep_en_iyi, tah, ger = en_iyi_durum
        cm = np.zeros((len(siniflar), len(siniflar)), int)
        for g, t in zip(ger, tah):
            cm[g, t] += 1
        sinif_dogruluk = {s: float(cm[i, i] / max(cm[i].sum(), 1)) for i, s in enumerate(siniflar)}
        egriler = egri_ciz({"train_kayip": gecmis["train_kayip"], "val_kayip": gecmis["val_kayip"]}, "kayip")
        dogruluk = egri_ciz({"val_dogruluk": gecmis["val_dogruluk"]}, "val dogrulugu")
        matris = matris_gorsel(cm, siniflar)
        cv2.imwrite(str(cikti / "egriler.png"), egriler)
        cv2.imwrite(str(cikti / "karisiklik.png"), matris)
        anlat.adim(f"En iyi epoch {ep_en_iyi}: val doğruluğu {en_iyi:.3f} → {cikti / 'model.pt'}", veri=sinif_dogruluk)
        return Sonuc(goruntu=egriler,
                     ara_gorseller={"kayıp eğrileri": egriler, "val doğruluğu": dogruluk, "karışıklık matrisi (en iyi epoch)": matris,
                                    "augmentation_onizleme (modelin gördüğü)": izgara(onizleme, hucre=140)},
                     veri={"en_iyi_epoch": ep_en_iyi, "val_dogruluk": en_iyi, "sinif_dogruluk": sinif_dogruluk, "model": str(cikti / "model.pt")},
                     metin=f"En iyi val doğruluğu **{en_iyi:.3f}** (epoch {ep_en_iyi}). Sınıf bazında: " +
                           ", ".join(f"{s} {v:.2f}" for s, v in sinif_dogruluk.items()) + f"\n\nModel: `{cikti / 'model.pt'}` → canli.py")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(DuyguEgit)
