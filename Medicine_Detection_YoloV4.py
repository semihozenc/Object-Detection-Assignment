import cv2
import numpy as np

                                   #test edilecek resmin konumu
img = cv2.imread(r"/Users/semihozenc/Desktop/Medicine-Detection/source/testImages/a.png")  
#print(img)



#%% 2. Bölüm
img_width = img.shape[1]
img_height = img.shape[0]

        # görüntüyü blob formata çeviriyoruz :
        #(değişken, sabit değer, yolo416 blob ölçeği, bgr-rgb değişimi, crop)
img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)
#img_blob = cv2.dnn.blobFromImage(img, 1/510, (832,832), swapRB=True, crop=False)

        # modelin tanıyacağı labelları giriyoruz :
labels = ["ibucold","Majezik","Amoklavin","Aksef","Parol","Pinix","Dexforte","Augmentin","Dolorex","Apirex","Evigen","Metpamid"]

       
        
        # kutucuk renkleri ayarlıyoruz :
colors = ["255, 0, 0","0, 0, 0","151, 255, 255","139, 101, 139","139, 69, 19","132, 112, 255","65, 105, 225","0, 0, 238","0, 255, 0","255, 255, 0","255, 130, 71","255, 20, 147","255, 130, 71"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(30,1))

        
#%% 3. Bölüm

#Modelin config dosyası ile ağırlık dosyasının yolunu göstererek modeli tanımlıyoruz.
model = cv2.dnn.readNetFromDarknet(r"/Users/semihozenc/Desktop/Medicine-Detection/source/yolov4.cfg",r"/Users/semihozenc/Desktop/Medicine-Detection/source/medicine_yolov4_last.weights")

layers = model.getLayerNames()
output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)

        # ================================ non-maximum supression : operation 1 ================================
        # fazladan kutuları yok etmek için kullandık  . Sonuçta en yüksek oranlı kutuyu gösterecek :
        
ids_list = []
boxes_list = []
confidences_list = []

        # ================================ non-maximum supression : operation 1 end =============================

        # tespit işlemine başlıyoruz. 
for detection_layer in detection_layers:
    for object_detection in detection_layer:

                scores = object_detection[5:] # puan tutuyoruz, 5 değer aldık
                predicted_id = np.argmax(scores) # en yüksek değerli indeksi çekiyoruz
                confidence = scores[predicted_id] # en güvenilir skoru alıyoruyz ve tutuyoruz

                #if confidence > 0.35: # güven skoru bu değerlerden büyükler için işlem devam edecek
                if confidence > 0.8:
                    
                    # kutuyu çizerken sol alt köşeden başlayıp sağ üst köşeye gideceğiz
                    label = labels[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                    (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                    # x ve y noktalarının özel koordinatı
                    start_x = int(box_center_x - (box_width / 2))
                    start_y = int(box_center_y - (box_height / 2))

                    #==================== non-maximum supression : operation 2 ====================
                    # döngü içindeki değerleri listeliyoruz ve alta iletiyoruz
                    # for içinde yukarda oluşturulan kutuları dolduruyoruz :
                    ids_list.append(predicted_id)
                    confidences_list.append(float(confidence))
                    boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

        # cv2.dnn.NMSBoxes() maximum confidenceları liste yapar
        # 0.5 ve 0.4 trashold değerleri yani standart
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

for max_id in max_ids: # liste içindeki değeri çekeceğiz
            max_class_id = max_id # max_class_id aslında nesnenin tutulduğu id olacak
            box = boxes_list[max_class_id] # box en iyi değeri tutacak

            start_x = box[0] # box'un başlangıç noktası indis değeri
            start_y = box[1]
            box_width = box[2]
            box_height = box[3] # box eni ve boyu

            predicted_id = ids_list[max_class_id] 
            label = labels[predicted_id] # ilgili labelı yukardan uygun şekilde çektik
            confidence = confidences_list[max_class_id] # confidence oranı sağlam olanı aldık



            end_x = start_x + box_width
            end_y = start_y + box_height

            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
            label = "'{}' - Dogruluk Orani : {:.2f}%".format(label, confidence * 100)
            print("Tahmin Edilen Ilac {}".format(label))

           # kutuyla ilgili tüm parametreler hazır. çizim başlıyor :
            cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,2)
            cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)

#bize resim dosyası olarak çıktı verecek
cv2.imwrite(r"/Users/semihozenc/Desktop/Medicine-Detection/source/predictImage/predictedImage.jpg", img) 


