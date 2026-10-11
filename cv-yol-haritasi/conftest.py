"""
pytest ortak ayarı: Testler gerçek veri/ klasörünü KİRLETMESİN.

Her test oturumu başında CV_VERI_KOK geçici bir klasöre yönlendirilir. ortak/yukle.py içindeki
VERI bu değişkeni okuduğu için, testlerde üretilen datasetler/olaylar orada kalır ve silinir.
(Bu dosya, ortak/ modülleri İÇE AKTARILMADAN önce çalışır — bu yüzden ortam değişkeni burada ayarlanır.)
"""
import os
import tempfile

os.environ.setdefault("CV_VERI_KOK", tempfile.mkdtemp(prefix="cv_test_veri_"))
