import numpy as np
import matplotlib.pyplot as plt
 
# giriş veriseti ve etiketleri,feature_set 5x3 , labels 5x1  [1]
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([1,0,0,1,1]).reshape(5,1)
MSE_Array = []
epoch_array = []
# hiperparametreler
np.random.seed(42)
W1 = np.random.rand(2,3)
W2 = np.random.rand(2,2)
W3 = np.random.rand(1,2)
b2 = np.random.rand(1)
b3 = np.random.rand(1)
b4 = np.random.rand(1)
lr = 0.05
 
# her katmanın aaktivasyon fonksiyonu sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))
def derivate_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
 
for epoch in range(10000):
    epoch_array.append(epoch)
    # input 5x3
    # batch gradien descent algoritması kullanılacağından tüm verisetini alıyoruz,sonra maaliyet fonksiyonu hesaplanacak
    input = feature_set
 
    # forward propagation 1 , 1. katmandan 2. katmana geçiş
    # Z2 = W1*A1 + b2 -> A1 = input
    # Z2 -> 2x5 = (2x3).(5x3)T + b2
    Z2 = np.dot(W1,input.transpose()) + b2
    A2 = sigmoid(Z2)
    # forward propagaition 2 , 2. katmandan 3. katmana geçiş
    # Z3 = W2*A2 + b3
    # Z3 -> 2x5 = (2x2).(2x5) + b3
    Z3 = np.dot(W2,A2) + b3
    A3 = sigmoid(Z3)
    # forward propagation 3 , 3. katmandan 4. katmana (çıkış katmanı) geçiş
    # Z4 = W3*A3 + b4
    # Z4 -> 1x5 = (1x2).(2x5) + b4
    # tahmin = A4 = f(Z4)
    Z4 = np.dot(W3,A3) + b4
    tahmin = A4 = sigmoid(Z4)
 
    # maaliyet hesaplaması
    # error = 1x5
    error = tahmin - labels.transpose()
    MSE = (1/len(input)) * np.square(error).sum()
    MSE_Array.append(MSE)
    print(MSE)
    # maaliyeti hesapladıktan sonra parametreler güncellenmelidir
 
    # maaliyet fonksiyonunun türevleri
    # W3,b4
    derivate_MSE_by_W3 = (2/len(input)) * np.dot((error*derivate_sigmoid(Z4)),A3.transpose())
    derivate_MSE_by_b4 = (2/len(input)) * (error*derivate_sigmoid(Z4))
 
    # W2,b3
    derivate_MSE_by_W2 = (2/len(input)) * np.dot(np.dot((error * derivate_sigmoid(Z4)).transpose(),W3).transpose()*derivate_sigmoid(Z3),A2.transpose())
    derivate_MSE_by_b3 = (2/len(input)) * np.dot((error * derivate_sigmoid(Z4)).transpose(),W3).transpose()*derivate_sigmoid(Z3)
 
    # W1,b2
    derivate_MSE_by_W1 = (2/len(input)) * np.dot((error * derivate_sigmoid(Z4)),np.dot(W3,derivate_sigmoid(Z3)).transpose()) * np.dot(np.dot(W2,derivate_sigmoid(Z2)),input)
    derivate_MSE_by_b2 = (2/len(input)) * np.dot((error * derivate_sigmoid(Z4)),np.dot(W3,derivate_sigmoid(Z3)).transpose()) * np.dot(W2,derivate_sigmoid(Z2))
 
    # güncellemeler
    W1 = W1 - lr*derivate_MSE_by_W1
    W2 = W2 - lr*derivate_MSE_by_W2
    W3 = W3 - lr*derivate_MSE_by_W3
    b2 = b2 - lr*derivate_MSE_by_b2
    b3 = b3 - lr*derivate_MSE_by_b3
    b4 = b4 - lr*derivate_MSE_by_b4
# bu kod parçası [2] den alıntı
plt.plot(epoch_array,MSE_Array)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()
