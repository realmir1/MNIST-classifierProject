

# 🧠 El Yazısı Rakam Sınıflandırma Modeli (CNN)

Bu proje, **MNIST Extended Handwritten Digits** veri setini kullanarak **Convolutional Neural Network (CNN)** tabanlı bir görüntü sınıflandırma modeli oluşturur. Amaç, el yazısı rakamları (0-9) doğru bir şekilde tanımlayabilen bir yapay zeka modeli geliştirmektir.

---

## 🚀 Proje Özeti

Bu proje, TensorFlow ve Keras kütüphaneleri kullanılarak geliştirilmiş bir **derin öğrenme görüntü sınıflandırıcısıdır**. Model, Kaggle üzerinde bulunan **400k Augmented MNIST Extended Handwritten Digits** veri seti ile eğitilmektedir.

Model, 150x150 boyutundaki renkli görüntüleri (3 kanal) giriş olarak alır ve her görüntüyü ilgili sınıfa (örneğin "0", "1", "2" … "9") atar.

---

## 📂 Proje Yapısı

```
├── image_classifier.h5             # Eğitilmiş model dosyası
├── main.py                         # Ana Python kodu
├── README.md                       # Proje açıklaması
└── /MNIST Validation Set (4k)/     # Görsel veri seti (Kaggle'dan alınır)
```

---

## 🧩 Kullanılan Teknolojiler

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib**
* **ImageDataGenerator (veri artırma ve yükleme için)**

---

## ⚙️ Model Mimarisi

Model aşağıdaki katmanlardan oluşur:

1. **Conv2D (32 filtre, 3x3)** → ReLU aktivasyon
2. **MaxPooling2D (2x2)**
3. **Conv2D (64 filtre, 3x3)** → ReLU aktivasyon
4. **MaxPooling2D (2x2)**
5. **Conv2D (128 filtre, 3x3)** → ReLU aktivasyon
6. **MaxPooling2D (2x2)**
7. **Flatten**
8. **Dense (512 nöron, ReLU aktivasyon)**
9. **Dense (Çıkış katmanı, Softmax aktivasyon)**

---

## 🧠 Model Eğitimi

Model, `ImageDataGenerator` kullanılarak eğitim ve doğrulama verilerine bölünür:

```python
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_generator = train_datagen.flow_from_directory(veriyolu, subset='training')
validation_generator = train_datagen.flow_from_directory(veriyolu, subset='validation')
```

Eğitim:

```python
model.fit(train_generator, validation_data=validation_generator, epochs=10)
```

---

## 💾 Model Kaydetme

Eğitim tamamlandıktan sonra model şu şekilde kaydedilir:

```python
model.save("image_classifier.h5")
```

---

## 🔍 Görsel Tahmin Fonksiyonu

Aşağıdaki fonksiyon, tek bir görüntüyü modele gönderip tahmin edilen sınıfı görselleştirir:

```python
def gercek_deger(image_path, model, class_indices):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class]
    
    plt.imshow(img)
    plt.title(f"Tahmin: {predicted_label}")
    plt.axis("off")
    plt.show()
```

---

## 🧪 Örnek Kullanım

```python
from tensorflow.keras.models import load_model

model = load_model("image_classifier.h5")
gercek_deger("/path/to/test_image.jpg", model, train_generator.class_indices)
```

---

## 📊 Sonuç

Eğitim tamamlandıktan sonra model, el yazısı rakamları yüksek doğrulukla sınıflandırabilir. Performans metrikleri (`loss`, `accuracy`) `model.fit()` çıktısında görüntülenir.

