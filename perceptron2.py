def predict(row,weights,bias):
    def activation(result):
        return 1.0 if result >= 0.0 else 0.0
    # veriler doğrusal ayrılabilir ise(linear separable) h(x) doğrusunun verisetini ayırmasını bekleriz
    # ağırlıklar (W) üzerinde değişiklik yapmak doğruyu sadece orijin(0,0) üzerinde değiştirir
    # doğrunun eksenler üzerinde hareket etmesi için sabit bir değer ( bias ) ekliyoruz
    # h(x) = X.W + b
    sum = 0.0
    for i in range(len(row)):
        sum += row[i]*weights[i]
    sum+=bias
    return activation(sum)
 
def cost_function(row_count,square_of_errors):
    # shocastic gradient descent algoritması için toplam karesel hata fonksiyonu
    sum = 0
    for i in range(len(square_of_errors)):
        sum += 0.5 * square_of_errors[i]
    return (1/row_count)*sum
 
 
# setups
# bu giriş-çıkış matrisleri https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# sitesinden alıntıdır
x = [[2.7810836,2.550537003],
    [1.465489372,2.362125076],
    [3.396561688,4.400293529],
    [1.38807019,1.850220317],
    [3.06407232,3.005305973],
    [7.627531214,2.759262235],
    [5.332441248,2.088626775],
    [6.922596716,1.77106367],
    [8.675418651,-0.242068655],
    [7.673756466,3.508563011]]
y = [0,0,0,0,0,1,1,1,1,1]
weights = [0,0]
bias = 0
learning_rate = 0.1
# bu vektör verisetinin satır sayısı kadar olmalıdır,şimdilik manuel giriyoruz
square_of_errors =  [0,0,0,0,0,0,0,0,0,0]
epoch = 0
# shocastic gradient descent optimization algorithm
while 1:
    for j in range(len(x)):
        # algoritmanın yaptığı tahmin sonucu
        prediction = predict(x[j], weights, bias)
        error = prediction - y[j]
        # hatanın karesini alıp vektöre kayıt ediyoruz,daha sonra bu vektörü backpropagation için kullanacağız
        # backpropagation : önceki yapılan hataların karelerinin toplamı
        square_of_errors[j] = error * error
        # eğer doğru tahmin yapılır ise ağırlıkları güncellemeye gerek yok
        if error != 0:
            # ağırlıkları ve bias değerini güncelle
            for i in range(len(weights)):
                weights[i] = weights[i] - learning_rate * error * x[j][i]
            bias = bias - learning_rate * error
    # tüm satırlar için tahmin yapıldıktan sonra cost fonksiyonu hesaplanır
    # toplam karesel hata fonksiyonu hesabı,backpropagation
    result = cost_function(len(x),square_of_errors)
    epoch+=1
    print('epoch : ',epoch,' error : ',result)
    # hatalar kareleri toplamı 0 çıkar ise algoritma optimum ağırlıkları bulmuştur
    # algoritma maaliyet fonksiyonu belirli bir eşik değerine ulaşınca(bu örnekte 0) veya 
    # kullanıcı tarafından belirlenen epoch sayısı kadar çalışınca sonlandırılabilir
    if result == 0: break
print('weights : ',weights,'\nbias : ',bias)
