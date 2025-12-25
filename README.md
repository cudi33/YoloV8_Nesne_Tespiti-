## 👤 Öğrenci Bilgileri
- *Ad Soyad:* Cudi Şami
- *Öğrenci Numarası:* 2012721308
---
# YOLOv8 ile Araba ve Motor Nesne Tespiti

## Makine Öğrenmesi Proje Ödevi

## Proje Tanımı

Bu projede, YOLOv8 derin öğrenme modeli kullanılarak görseller, videolar ve canlı kamera görüntüleri üzerinde **araba** ve **motor** nesnelerinin otomatik olarak tespit edilmesi amaçlanmıştır.

Model, proje kapsamında hazırlanan veri seti kullanılarak eğitilmiş ve elde edilen en iyi model ağırlıkları PyQt5 tabanlı bir masaüstü grafik arayüzü ile entegre edilmiştir.

Proje, uçtan uca bir nesne tespit sürecini kapsamaktadır:

- Veri seti hazırlanması
- YOLO formatında etiketleme
- YOLOv8 model eğitimi
- Performans metriklerinin değerlendirilmesi
- PyQt5 grafik arayüz geliştirilmesi

---

## Veri Seti Bilgileri

- **Sınıf Sayısı:** 2
  - `0 → Motor`
  - `1 → Araba`
- **Etiketleme Formatı:** YOLO (.txt)
- **Görüntü Formatı:** .jpg / .png
- **Veri Ayrımı:**
  - Eğitim (train)
  - Doğrulama (val)

Veri seti, YOLOv8 formatına uygun şekilde düzenlenmiş ve model eğitimi için kullanılmıştır.

---

## Model Eğitimi

- **Model:** YOLOv8
- **Framework:** Ultralytics YOLOv8
- **Eğitim Ortamı:** Google Colab
- **Model Çıkışı:** `best.pt`

Eğitim süreci boyunca model performansı **loss**, **precision**, **recall** ve **mAP** metrikleri üzerinden takip edilmiştir.  
Eğitim sonunda en iyi sonucu veren model ağırlıkları **best.pt** dosyası olarak kaydedilmiştir.

### Başlıca Başarı Metrikleri

- **Precision:** ≈ 0.83
- **Recall:** ≈ 0.83
- **mAP@50:** ≈ 0.87
- **mAP@50–95:** ≈ 0.56

---

## Grafik Arayüz (PyQt5)

Proje kapsamında geliştirilen PyQt5 tabanlı masaüstü uygulama ile kullanıcı:

- Bilgisayarından bir görüntü seçebilir
- Görüntü üzerinde YOLOv8 ile nesne tespiti yapabilir
- Tespit edilen araba ve motorları bounding box ile görebilir
- Tespit edilen nesne sayısını ekranda görüntüleyebilir
- Etiketlenmiş görüntüyü bilgisayarına kaydedebilir
- Video dosyaları üzerinde nesne tespiti yapabilir
- Canlı kamera üzerinden gerçek zamanlı nesne tespiti gerçekleştirebilir

Arayüz, kullanıcı dostu ve sade bir tasarım ile geliştirilmiştir.

---

## Kurulum ve Çalıştırma

### Sanal Ortam Oluşturma

```bash
python -m venv venv
```
### Sanal Ortamı Aktif Etme (Windows)
```
venv\Scripts\activate
```

### Gerekli Kütüphanelerin Kurulumu
```
pip install -r requirements.txt
```
### Uygulamayı Çalıştırma
```
python gui/main.py
```
---

### Kullanılan Teknolojiler

Python

YOLOv8 (Ultralytics)

PyTorch (CPU)

PyQt5

OpenCV

NumPy

Google Colab

---

### Proje Dosya Yapısı
```
yolo_gui_project/
│
├── gui/
│   └── main.py          # PyQt5 GUI uygulaması
│
├── model/
│   └── best.pt          # Eğitilmiş YOLOv8 modeli
│
├── yolo_training.ipynb  # Model eğitim süreci
├── requirements.txt    # Gerekli kütüphaneler
└── README.md            # Proje dokümantasyonu
```
---

### Akademik Dürüstlük
Bu projede kullanılan veri seti, etiketleme işlemleri ve yazılım kodları tamamen proje sahibine aittir. 

---






