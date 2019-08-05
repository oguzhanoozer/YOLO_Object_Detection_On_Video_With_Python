# gerekli kütüphaneleri import ediyoruz.

import cv2
import imageio
from darkflow.net.build import TFNet

# yoloyu kullanmak için gerekli dosya yollarını veriyoruz.
# Bu eklentilere ek olarak birde gpu değeri vardı
# fakat benim bilgisayarım gpu destekli olmadığı için yazmadım.

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15
}

tfnet = TFNet(option)

def detect(frame):
    # result değeri her frame üzerinde nesne tanıma yapıp bize bazı sonuçlar döndürüyor
    # bottomright değeri hem x, hem y değeri için
        # x değeri tespit edilen nesnenin bulunduğu karenin yatay uzunluk değeri
        # y değeri tespit edilen nesnenin bulunduğu karenin dikey uzunluk değeri
    # topleft değeri hem x , hem y değeri için
        # x değeri,  tespit edilen nesnenin bulunduğu karenin başladığı üst soldaki x koordinatı
        # y değeri,  tespit edilen nesnenin bulunduğu karenin başladığı üst soldaki y koordinatı
    # confidence değeri, nesne tespitinin olasılık değeri
    # label değeri, tespit edilen nesnenin hangi label sınıfına girdiği değeri(car, bus person vb)
    results = tfnet.return_predict(frame)

    for result in results:
        # topleft değerlerini alıyoruz
        tl = (result['topleft']['x'], result['topleft']['y'])
        # bottom right değerlerini alıyoruz
        br = (result['bottomright']['x'], result['bottomright']['y'])
        # nesnenin hangi label a ait olduğunu buluyoruz.
        label = result['label']
        # aldığımız değerlerle opencv kutuphanesini kullanarak dikdörtgen çizdiriyoruz
        # frame değeri o framein üstüne çizdiğimizi belirtiyoruz.
        # t1 değerleri x, y koordinatları
        # br değeri width ve height uzunluk değeri

        if label == 'slm':

            # color_Rectangle,  buda çizilen dikdörtgenin reng değeri
            color_Rectangle = (0, 255, 0)

            # lineThick değeri dikdörtgenin
            lineThick = 2

            frames = cv2.rectangle(frame, tl, br, color_Rectangle, lineThick)

            # diğer işlemimiz ise tespit edilen nesne üzerine o nesnenin label değerini yazdırmak

            # label degeri yukarıda aldğımız label değeri
            # t1 değeri texti nereye yazacağımızın koordinatı
            # cv2.FONT_HERSHEY_COMPLEX ise text stili

            # textSize, textin büyüklüğünü verir
            textSize = 1

            #textColor, textin rengini veren değişken
            textColor = (0,0,0)

            #textThickness, textin kalınlığını verir
            textThickness = 2

            texting = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, textSize, textColor, textThickness)

    # daha sonra her frame için işlemler gerçekleştirilıp return ile
    # yazdırılmak için döndürülür

    return frame

# programın başlangıç kısmı burasıdır.

# reader değişkeni, imageio ile videoyu okumamızı sağlıyor.
reader = imageio.get_reader('trafik.mp4')

# get_meta_data ile videonun fps'ini alıyoruz.
# fps değerini almamızın nedeni, videoyu yazdırken fps değerlerin eşit olması için.
fps = reader.get_meta_data()['fps']

# writer değişkeni, imageio ile  bir output videosu oluşturmayı sağlıyor.
# fps değerini de aldığımız fps değerine eşitledik.
writer = imageio.get_writer('trafikOutput.mp4', fps=fps)

# for döngüsü ile her framede  detect fonksiyonunu çağırıyoruz
# i değeri sadece hangi framede olduğumuzu gösteriyor.
# frame ise bizim kullanacağımız değer
for i, frame in enumerate(reader):
    # frame değerini detect fonksiyonundan gelen değere eşitliyoruz
    frame = detect(frame)
    # detect fonksiyonu ile nesne tanımayı yapılım gelen frame değerini
    # writer değişkenine yani yani  output değerine yazıyoruz
    writer.append_data(frame)
    print(i)

# yazılan videoyu kapatıyoruz
writer.close()