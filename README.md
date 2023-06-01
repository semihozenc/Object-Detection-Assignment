## YAPAY ZEKA İLE NESNE TESPİTİ  VE ALGORİTMALARIN KARŞILAŞTIRILMASI
## Uygulama Adımları
  1. Veri Setinin elde edilmesi
  
      Yapacağımız proje için hazırlanmış bir veri seti olmadığı için verileri kendimiz elde ettik. Bunun için 12 farklı ilaca ait, ilacın tanesinde 60 resim olmak üzere toplam       720 farklı resim kullandık.
      
  2. Veri Setinin Etiketlenmesi
    
      Yolo Algoritmasında kullanmak için 720 tane resim sınıflarına göre makesense.ai internet sitesinde Yolo formatında tek tek etiketlendi ve txt formatında bir çıktı             alındı.
    
      <img width="630" alt="image" src="https://github.com/semihozenc/Object-Detection-Assignment/assets/100075605/dd521dbf-ace2-4935-bbbd-ac37afee5e61">
    
      Faster R-CNN Algoritmasında kullanmak için 720 tane resim sınıflarına göre labelme uygulamasında 12 class olacak biçimde etiketlendi.
      Etiketlenen fotoğrafların .json uzantılı dosyaları oluşturuldu.
   
      <img width="641" alt="image" src="https://github.com/semihozenc/Object-Detection-Assignment/assets/100075605/d5fde396-3b1e-4818-9a55-c5a01e77fe93">
      
  3. Algoritmaların Elde Edilmesi  
  
      Github’dan Darknet dosyalarını elde ettik. İçinde YoloV4 algoritması var. 
      
      Aynı şekilde Github'dan Detectron2 dosyalarını elde ettik. İçinde Faster R-CNN algoritması var. 
      
      Modeller için gerekli düzenlemeleri yaptık ve eğitim işlemine geçtik.
      
  4. Eğitimin Gerçekleştirilmesi

  5. Modellerin Denenmesi




[Veri Seti, Test Görüntüleri ve Ağırlık dosyası içeren Drive Linki](https://drive.google.com/drive/folders/1OQ-FE_LdO4oLulo_yghf8AeDAdx6UCMW?usp=sharing)
