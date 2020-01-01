import numpy as np
 
def perceptron():
    # girdiler, x[0] kolonu bias vektörüdür
    # neden bias vektörünü ekledik:
    # tahmin fazla veya az yapıldığı zaman , bir sonraki hesaplamada ağırlıklar buna göre güncellenmelidir
    # ve bias değeri toplam fonksiyonuna sabit bir çarpan ile katıldığından ( bias * weight[0] )
    #    bir sonraki iterasyonda tahminin daha doğru yapılabilmesini sağlar[2]
    # bias vektörü olmadan, eğer tüm girdiler 0 olsa idi,makine sürekli 0 çıktısını üretecekti ki örneğin NAND
    #    kapısında 0-0 girişi için 1 değeri makine tarafından bulunamiyacaktı[2]
    # bu kodun çıktısında bias değeri ile çarpılacak ağırlık değerinin (weights [0])her yanlış tahminden sonra 
    #    nasıl artıp azaldığı gözlemlenebilir
    x = [[1, 0., 0.],
         [1, 0., 1.],
         [1, 1., 0.],
         [1, 1., 1.]]
    # çıktılar
    y = [1.,
         1.,
         1.,
         0.]
          
    learning_rate = 0.1 # öğrenme oranı
    treshold = 0.0      # aktivasyon fonksiyonu eşik değeri
    numEpoch = 1        # tur sayısı,tüm verisetini dolaşınca 1 artar
    numIter = 1         # iterasyon sayısı,verisetindeki her bir satırdan sonra 1 artar 
    weights = np.zeros(len(x[0]))     # ağırlık vektörü
    predict_vector = np.ones(len(x))  # tahmin vektörü
    errors = np.ones(len(x))          # hata vektörü
    loss_function_results = []        # kayıp fonksiyonu değerleri
 
    while 1:
        for i in range(len(x)):  # input matrisindeki satır sayısı kadar dönecek değişken
            # input vektörü ile ağırlık vektörü iç çarpımı
            # bias vektörünü input matrisine eklemeyip bu toplamı
            # f = np.dot(x[i],weights) + bias
            # şeklinde de yapabiliriz
            f = np.dot(x[i], weights)
            # aktivasyon fonksiyonu,basit olarak toplam değerini bir eşik değeri ile karşılaştırıp karar verir
            if f >= treshold:
                predict = 1.
            else:
                predict = 0.
            # tahminimizi tahmin vektörüne kayıt ediyoruz
            predict_vector[i] = predict
            print('iterasyon : ',numIter,' ağırlıklar : ',weights ,'tahmin : ',predict, ' gerçek değer : ',y[i])
            numIter+=1
            # eğer tahminimiz doğru çıkmaz ise ağırlıkları güncelliyoruz
            # amaç optimum ağırlık vektörünün bulunması
            if predict != y[i]:
                for j in range(len(x[0])):
                    weights[j] = weights[j] + learning_rate * (y[i] - predict) * x[i][j]
        # bir iterasyon sonucunda tahmin yapıldıktan sonra sıra hata hesaplamasına gelir
        # hata : gerçek sonuç ile tahmin ettiğimiz sonucun farkının karesi olarak hesaplanır
        # eğer doğru tahmin edersek hatamız 0 olur
        for i in range(len(x)):
            errors[i] = (y[i] - predict_vector[i]) ** 2
        # kayıp fonksiyonu : iterasyonlar sonucu hesaplanan hata değerlerinin toplamının yarısı
        # makine öğrenmesi algoritmalarındaki amaç bu kayıp fonksiyonunu minimize etmektir
        # bu algoritma kayıp fonksiyonu 0 olana kadar ( yani tüm çıkışlar doğru tahmin edilene kadar ) çalışır 
        # kayıp fonksiyonu tüm iterasyonlar bittikten sonra,yani bir epoch sonucu,hesaplanır
        loss_function_results.append(0.5 * np.sum(errors))
        if (0.5 * np.sum(errors)) == 0:
            break
        numEpoch += 1
    print('epoch sayısı : ',numEpoch)
    print('her epoch sonucu hesaplanan kayıp fonksiyonu değerleri : ', loss_function_results)
    print('optimum ağırlık vektörü : ',weights)
 
if __name__ == "__main__":
    perceptron()
