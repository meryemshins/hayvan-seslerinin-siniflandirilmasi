import os   
#dosya ve klasör yapılarıyla çalışmak için gerekli kütüphane
import numpy as np  
#matris ve sayı dizileri için gerekli kütüphane
from scipy import signal  
#Scipy, karmaşık matematiksel problemlerin hesaplanmasına yardımcı kütüphane. Numpy ile uyumlu
from scipy.io import wavfile 
#wav ses dosyalarını okumak/yazmak için gerekli kütüphane
import tensorflow as tf  
#derin sinir ağlarının eğitimi ve çıkarımı için gerekli kütüphane
from tensorflow import keras 
from tensorflow import layers, models, backend, optimizers  
#derin öğrenme için kullanılan kütüphane
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten    # katman işlemleri.
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization   # katman işlemleri.
from keras.regularizers import l2 


dirname = './data/train/dog'  # köpek train dosya yolu.
train_dog = []   # boş liste.
for filename in os.listdir(dirname):
    sample_rate, samples = wavfile.read(dirname+'/'+filename)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    train_dog.append(spectrogram)
    
dirname = './data/train/cat'  # kedi train dosya yolu.
train_cat = []   # boş liste.
for filename in os.listdir(dirname):       
#os.listdir(): dizin içindeki dosya ve klasörleri listeler.
    sample_rate, samples = wavfile.read(dirname+'/'+filename)        
    # her bir wav dosyasını okur. sample_rate = sesin derinliği, samples = sesin değeri.
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate) 
    # her bir ses için spektrogram(ses frekanslarının zamana göre görsel gösterimi) oluşturur. 
    train_cat.append(spectrogram)     # boş listeye spektrogramları ekler.

dirname = './data/test/dogs'
test_dog = []
for filename in os.listdir(dirname):
    sample_rate, samples = wavfile.read(dirname+ '/' +filename)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    test_dog.append(spectrogram)
    
dirname = './data/test/cats'
test_cat = []
for filename in os.listdir(dirname):
    sample_rate, samples = wavfile.read(dirname+ '/' +filename)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    test_cat.append(spectrogram)

# Kedi ve Köpek eğitim ve test verileri için 4 ayrı liste oluşturdu. 
# Listelerin içine her sesin spektrogramlarını ekledi.

max_length = max([s.shape[1] for s in train_dog]+[s.shape[1] for s in train_cat]+
                 [s.shape[1] for s in test_dog]+[s.shape[1] for s in test_cat])
# Her bir ses dosyasını tek kolonda(spektrogram kolonu) sıralayarak toplam .wav sayısını bulur. 
# Tüm elemanlar.


def pad_spec(spectrogram, max_length):
    return np.pad(spectrogram, ((0,0),(0,max_length-spectrogram.shape[1])), mode='mean')
# her bir ses için max uzunluğa eşitleme yapar. 0 dolgusunu (padding) ekler.

Y_train = np.array([1]*len(train_dog)+[0]*len(train_cat))  
# train dog ve cat verilerini dizide toplama.
X_train = np.array([pad_spec(s, max_length) for s in train_dog]+[pad_spec(s, max_length) for s in train_cat])

# x_train, dizisini padding sonucu oluşturdu.

Y_test = np.array([1]*len(test_dog)+[0]*len(test_cat))    # test dog ve cat verilerini dizide toplama. 
X_test = np.array([pad_spec(s, max_length) for s in test_dog]+[pad_spec(s, max_length) for s in test_cat])

# x_test, dizisini padding sonucu oluşturdu.

p = np.random.permutation(X_train.shape[0])   # train verilerini rastgele sıralar.
X_train = X_train[p]    # x_train ve y_train dizilerine rastgele sıralanmış train verilerini atadı
Y_train = Y_train[p]

p = np.random.permutation(X_test.shape[0])   # test verilerini rastgele sıralar.
X_test = X_test[p]      # x_test ve y_test dizilerine rastgele sıralanmış test verilerini atadı
Y_test = Y_test[p]

backend.clear_session()     # önceki modellerden kalan tüm düğümleri kaldırarak belleği boşaltır ve yavaşlamayı önler.

model = Sequential()  # sequential(), model oluşturma için.
model.add(Conv1D(8, (3), activation='relu', padding='same', input_shape=(129,1283)))   
# ilk katman. 8, nöron sayısı. (3) filtre , same padding , veri etrafına sıfır değerlerinden oluşan bir çerçeve ekler. 
# input shape, giriş boyutu. Conv1D, ses ve metin analizlerinde kullanılır. 
model.add(MaxPooling1D((2)))      # verilere maksimum ortaklama işlemi yapacak. ortaklama boyutu = 2.
model.add(BatchNormalization())   # katmanlar eş zamanlı olarak eğitilir. Eğitimin hızlanmasını sağlar.
model.add(Flatten())              # Veriyi tek boyuta çevirir.
model.add(Dropout(rate=0.7))      # Bazı nöronlar kaldırılır. Eğitim performansı artar. 0-1 arası değer alır. 
                                  #0.7 iken %70 nöron sıfırlanır.
model.add(Dense(8, kernel_regularizer=l2(l=0.01)))    # 2.katman 8 nöronlu. l2, ağırlıkların düzenlenmesi.
model.add(Dense(1, activation='sigmoid'))             # 3.katman 1 nöronlu çıkış katmanı.

model.summary()               # modelin yapılandırması.

model.compile(optimizer='adam',                  # ağın derlenmesi.
              loss='binary_crossentropy',        # adam, optimizasyon yöntemi. metrics, doğruluk oranı gösterimi.
              metrics=['accuracy'])              # loss, kayıp fonksiyonu.

model.fit(X_train, Y_train, epochs=50, batch_size=16,   # ağın eğitilmesi.
                    validation_data=(X_test, Y_test))   # epoch, devir sayısı. 
                                                        #batch, her eğitimde değerlendirilecek örnek sayısı grupları.

# x_tarin ve y_train ağın girdi verileri. validation_data, girdilere göre doğrulama verileri.