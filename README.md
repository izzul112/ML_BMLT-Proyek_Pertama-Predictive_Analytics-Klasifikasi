# Laporan Proyek Machine Learning - Habib Azizul Haq


## Domain Proyek

**Klasifikasi Harga Handphone (HP).**

Saat ini hp sudah sangat mudah untuk dimiliki oleh hampir semua orang, mulai dari yg harganya ratusan ribu hingga puluhan juta, tentu tergantung pada kebutuhan masing-masing orang, ada yang menganggap harga puluhan juta itu wajar dan ada yang mengangggapnya itu kemahalan.

- Hal tersebut tentu membuat kita kadang tidak jadi membeli suatu hp, karena kita khawatir apakah nanti semua fitur tersebut Gimmik atau bukan.
- Kita kadang juga merasa khawatir apakah semua fitur di suatu hp tersebut secara harga sesuai apa tidak, tentu kita tidak ingin terkena *genjutsu* sales-sales hp di toko yang mengatakah "Beli hp ini aja kak, PUBG rata kanan semua!", namun setelah di rumah kita baru sadar kok beli hp yang ini? Padahal tadi maunya bukan yang ini.
- Sebagai orang yang jarang tau berita tentang teknologi, dan sedang mencari hp dengan kelas harga tertentu kita kadang bingung dan kesusahan dalam menentukan pilihan

Solusi dari masalah tersebut, kita kembangkan sebuah model Machine Learning (ML) untuk membantu kita yang kesulitan menentukan sebuah rentang kelas harga dari hp, apakah hp dengan spesifikasi sekian-sekian yang berada di kelas harga menengah, apakah sesuai dengan kelas harganya, atau kita di tipu oleh sales-sales hp, karena mereka ingin prodak tersebut segera habis.

- Menurut (Aqmal Maulana, [Spek HP berdasar harga](https://www.droidlime.com/fitur/ini-bedanya-hp-entry-level-mainstream-high-end-dan-flagship/))
- Menurut (Bang Dedy Irvan, [Membedakan Hape Pakai Gimmick Marketing](https://youtu.be/cvp-YcCgHpU))
- Menurut (Bang Tira, [Genjutsu Marketing](https://youtu.be/pjKR2xfV83Y))

## Business Understanding

### Problem Statements Menjelaskan pernyataan masalah latar belakang:

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap kelas harga suatu hp?
- Berapa kelas harga pasar hp dengan karakteristik atau fitur tertentu?

### Goals Menjelaskan tujuan dari pernyataan masalah:

- Mengetahui fitur yang paling berkorelasi dengan kelas harga hp.
- Membuat model machine learning yang dapat memprediksi kelas harga hp seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements: 

- Untuk menghasilkan model yang optimal namun tetap sederhana kita akan menggunakan 3 algoritma yaitu KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier.
- Dari 3 algoritma di atas kita akan menggunakan metrik `accuracy_score` dan `classification_report` yang memang cocok untuk kasus klasifikasi, untuk melihat pada ketiga algoritma diatas mana yang paling powerfull.

## Data Understanding

Kita akan menggunkan dataset yang berisi 21 variabel yang biasanya sering ditanyakan ketika membeli hp atau mungkin menjadi suatu standar yang digunakan masyarakat dalam menggolongkan kelas harga dari suatu hp.

Sumber dataset yang akan kita gunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?datasetId=11167&select=train.csv). Yang di publish oleh ABHISHEK SHARMA.

### Variabel-variabel pada Mobile Price Classification adalah sebagai berikut:

- battery_power : total kapasitas baterai dalam mAh
- blue : apakah memiliki bluethoot atau tidak
- clock_speed : kecepatan dari prosesor
- dual_sim : apakah mendukung dua sim card
- fc : kamera depan dalam mega pixel
- four_g : sudah 4G atau belum
- int_memory : memory internal
- m_dep : kedalaman hp dalam satuan cm
- mobile_wt : berat dari hp
- n_cores : jumlah dari core prosessor
- pc : kamera utama dalam mega pixel
- px_height : tinggi resolusi dalam satuan pixel
- px_width : lebar resolusi dalam satuan pixel
- ram : jumlah ram yang dipakai dalam satuan Megabytes
- sc_h : tinggi layar hp dalam satuan cm
- sc_w : lebar layar hp dalam satuan cm
- talk_time : lama penggunaan setelah satu kali pengisian daya
- three_g : sudah 3G atau belum
- touch_screen : sudah layar sentuh apa belum
- wifi : punya wifi atau tidak
- price_range : rentang harga

Dari 21 variabel diatas variabel price_range adalah sasaran kita, kita akan membuat model berdasar rentang harga yang ada di dalam variabel price_range, dengan nilai:
- 0 = low cost (Entry-level)
- 1 = medium cost (Mid-range)
- 2 = high cost (High-end)
- 3 = very high cost (Flagship / hp para Sultan)

Kita akan melakukan sedikit analisis untuk mengethui beberapa hal tentang variabel diatas diantaranya :
1.  Kita akan melihat ada berapa jumlah baris dari data dalam tampilan tabel.
2.  Kita akan melihat tipe data dari 21 kolom di dataset.
3.  Kita akan mengecek deskripsi statistik data.
4.  Kita akan melihat apakah ada data yang kosong / tidak ada isinya.
5.  Kita akan melihat apakah data kita memiliki outliers.
6.  Kita akan melihat variabel apa saja yang memiliki hubungan yang kuat atas klasifikasi harga suatu hp.

Langsung saja kita mulai dari yang pertama yaitu:

1. Kita akan melihat ada berapa jumlah baris dari data dalam tampilan tabel.
   
   Kita akan menggunakan menggunakan kode `data_train`, `data_train` adalah nama variabel yang kita ganakan saat meload dataset kita.
   
   ![Screenshot_4](https://user-images.githubusercontent.com/43197282/181404706-66ca4953-b7ff-48c5-9ed8-5b2f2e076c71.jpg)
   
   Dari kode `data_train` kita mendapat informasi:
   - Ada 2000 baris dalam dataset
   - Dan seperti yang dijelaskan dalam deskipsi variabel ada 21 kolom
   
2. Kita akan melihat tipe data dari 21 kolom di dataset.  
   
   Kita akan menggunakan `data_train.info()`, `.info` akan menampilkan informasi tipe data 21 kolom dataset yang kita gunakan.
   
   ![Screenshot_5](https://user-images.githubusercontent.com/43197282/181405084-a9c9f4b6-3896-4bfd-bbb6-708f0f36c36b.jpg)
   
   Dari output terlihat bahwa:   
   - Terdapat 2 kolom numerik dengan tipe data float64 yaitu : clock_speed dan m_dept.
   - Terdapat 19 kolom numerik dengan tipe data int64 yaitu : battery_power, blue, dual_sim, fc, four_g, int_memory, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_schreen, wifi dan price_range.

3. Kita akan mengecek deskripsi statistik data.
   
   Kita akan mengeceknya dengan fungsi `describe()`.
   
   ![Screenshot_6](https://user-images.githubusercontent.com/43197282/181405401-23f53058-0be9-490d-81eb-b69ae42ab5b2.jpg)
   
   Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:
   - Count adalah jumlah sampel pada data.
   - Mean adalah nilai rata-rata.
   - Std adalah standar deviasi.
   - Min yaitu nilai minimum setiap kolom.
   - 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
   - 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
   - 75% adalah kuartil ketiga.
   - Max adalah nilai maksimum.

4. Kita akan melihat apakah ada data yang kosong / tidak ada isinya.
   
   Kita akan mengeceknya dengan fungsi `.isnull().sum()`. Fungsi `.isnull()` akan mengecek apakah ada data kosng pada setiap baris pada semua kolom di dataset kita, kemudian kita gunakan juga fungsi `.sum()` untuk menjumlahnya sehingga hasilnya akan seperti berikut.
   
   ![image](https://user-images.githubusercontent.com/43197282/181405542-3567e895-3f3d-41a2-b0f2-73fbd57ba549.png)
   
   Dari hasil outputnya data kita tidak memiliki data yang kosong.
   
5. Kita akan melihat apakah data kita memiliki outliers.
   
   Kita akan mengeceknya dengan fungsi `sns.boxplot(x=data_train['px_height'])` [1]. Variabel `sns` adalah tempat kita menampung library **Seaborn** yang akan kita gunakan untuk memvisualisaikan dataset kita. Fungsi `.boxplot()` akan menampilkan visualisasi dari dataset kita dengan visualisasi seperti gambar berikut:
   
   ![gambar boxplot](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:3be38c69ec4f1ee07ce8c24e42ce23fb20210910131731.png)
   
   Dalam hal ini kita menggunakan sampel `px_heght` dari gambar bisa kita lihat adanya outliers.
   
   ![Screenshot_2](https://user-images.githubusercontent.com/43197282/180590253-ec55f05c-352f-483b-b467-00d22b634432.jpg)
   
   Tenang tidak perlu panik kita akan mengatasinya dengan teknik IQR dimana data yang berada di luar Q1 dan Q3 adalah outlier, dimana kita akan menentukan nilai batas atas dan bawah, dengan persamaan berikut:
   - Batas bawah = Q1 - 1.5 * IQR
   - Batas atas = Q3 + 1.5 * IQR
   
   Langsung saja kita set variabel Q1, Q2 dan IQR-nya
   - `Q1 = data_train.quantile(0.25)`
   - `Q3 = data_train.quantile(0.75)`
   - `IQR=Q3-Q1`
   
   Setelah itu kita hitung dan hapus sekalian data yang tidak sesuai dengan spesifikasi yang kita buat dengan kode berikut
   - `data_train=data_train[~((data_train<(Q1-1.5*IQR))|(data_train>(Q3+1.5*IQR))).any(axis=1)]`
   - `data_train.shape` kemudian kita periksa juga sisa dataset kita, yaitu tersisa 1506 baris dan 21 kolom
   
6. Kita akan melihat variabel apa saja yang memiliki hubungan yang kuat atas klasifikasi harga suatu hp.
   
   Pertama kita akan melihat 21 kolom yang ada dalam dataset dalam bentuk tabel supaya kita bisa lebih mengetahui tentang isi dari nilai setiap kolom dengan kode berikut `data_train.hist(bins=50, figsize=(20,15))` fungsi `.hist()` akan menampilkan data dalam bentul diagram dengan variabel `bins=50` adalah jumlah menara pada data yaitu 50 sedang `figsize=(20,15)` adalah ukuran dari diagram kita. setelah itu kita gunkan kode `plt.show()` untuk menampilkan diagramnya. 
   
   ![image](https://user-images.githubusercontent.com/43197282/181405856-80d471d1-f29e-429f-bd67-f2c4cfe01c31.png)
   ![image](https://user-images.githubusercontent.com/43197282/181405967-e26feb2b-7656-4495-93cb-f1a54bcb40f8.png)

   Dari tampilan diagram dapat kita simpulkan bahwa ada beberapa variabel yang bertipe biner atau yang isinya hanya 0 dan 1, dan ada yang isinya angka lebih dari itu. Kita lihat list dibawah untuk lebi jelas mana yang memiliki fitur biner dan bukan.
   - binary_features = blue, dual_sim, four_g, three_g, touch_screen, wifi
   - non_binary_features = battery_power, clock_speed, fc, int_memory, m_dep, mobile_wt, n_core, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, price_range
   
   Selanjutnya kita lanjut melihat setiap korelasi setiap fitur dengan dengan price_range, dengan koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah.
   
   Kita akan melihat untuk binary_features terlebih dulu.
   
   ![image](https://user-images.githubusercontent.com/43197282/181406382-d6f9a958-025c-4382-a45c-5da3f41c34d2.png)
   ![image](https://user-images.githubusercontent.com/43197282/181406470-b2795dff-f344-472b-99ef-fe31fe2db0fa.png)

   - Di binary_features, fitur three_g tidak memiliki gambar atau gambarnya putih bersih, karena nilai hanya 1 saja atau kebanyakan hp pada dataset sudah masuk kategori three_g semua, jadi kita akan menghapus fitur ini, fitur four_g dan dual_sim juga karena memiliki korelasi paling kecil yaitu 0.
   
   Selanjutnya kita akan lihat non_binary_features.
   
   ![image](https://user-images.githubusercontent.com/43197282/181407544-5ac646cb-a9a0-42c0-b82b-41bdcba2db87.png)
   ![image](https://user-images.githubusercontent.com/43197282/181407805-21fbc560-3e0a-49c6-a60e-a3f0ff1af4e9.png)

   - Di non_binary_features, fitur **ram** merupakan fitur dengan tingkat korelasi tertinggi yaitu: 0.92. Sedang n_core dan talk_time memiliki korelasi paling kecil yaitu 0, maka kita akan menghapus 2 fitur ini.
   
   Kita cek lagi data kita dan hasilnya adalah:
   
   ![image](https://user-images.githubusercontent.com/43197282/181408010-c1b18257-4c02-4f77-8b3c-859bfcd7e982.png)

   Dengan ini total kolom yang kita miliki ada 16, menyusut 5 dari total di awal kita memiliki 21 kolom.
   

7. Kita akan melihat perbandingan jumlah sampel dari variabel price_range
   
   Kita akan melihatnya dalam bentuk tabel.
   
   ![image](https://user-images.githubusercontent.com/43197282/181406238-04db00eb-e9ef-4f5b-9255-4c0b543aba05.png)
   
   Dari hasil visualisasi diatas dapat kita simpulkan bahawa dataset kita memiliki jumlah sampel yang seimbang atau tidak berat sebelah.
   
## Data Preparation

Di proses ini kita akan melakukan 2 proses yaitu:
1. Reduksi dimensi dengan Principal Component Analysis (PCA).
2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.

Baiklah mari kita lakukan hal diatas satu-persatu:

1. Reduksi dimensi dengan Principal Component Analysis (PCA).
   Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang paling populer adalah Principal Component Analysis atau disingkat menjadi PCA. Teknik PCA digunakan untuk mereduksi variabel asli menjadi sejumlah kecil variabel baru yang tidak berkorelasi linier, disebut komponen utama (PC). Komponen utama ini dapat menangkap sebagian besar varians dalam variabel asli. Sehingga, saat teknik PCA diterapkan pada data, ia hanya akan menggunakan komponen utama dan mengabaikan sisanya.
   
   Berikut penjelasan untuk masing-masing komponen utama (PC):
   - PC pertama mewakili arah varians maksimum dalam data. Ia paling banyak menangkap informasi dari semua fitur dalam data. 
   - PC kedua menangkap sebagian besar informasi yang tersisa setelah PC pertama. 
   - PC ketiga menangkap sebagian besar informasi yang tersisa setelah PC pertama, PC kedua, dst.
   
   Selanjutnya kita import dulu kelas PCA nya dengan kode `from sklearn.decomposition import PCA`, setelah itu kita set variabel dari kelas PCA kita, untuk `n_components=2` dan `n_components=3` kita set = 2 dan 3 karena ada 5 fitur yang akan kita proses secara sendiri-sendiri, sedang `random_state=123)` kita set dengan = 123, `random_state` ini bebas kita isi dengan angka berapapun asal itu masih bilangan integer.
   
   Setelah itu kita print hasilnya yang `n_components=2` dulu hasilnya adalah `array([0.75, 0.25])` artinya 75% dari kedua fitur ada di PC pertama dan 25% di PC kedua, Sedang untuk `n_components=3` hasilnya adalah `array([0.757, 0.241, 0.002])` yang artinya 75% dari ke 3 fitur ada di PC pertama, 24% ada di PC kedua dan 0,002 atau sisanya ada di PC3.
   
   Selanjutnya Kita reduksi kelima fitur tersebut menjadi 2 yaitu resolution_px dan dimension_hp, resolution_px adalah dimensi pexel sedang dimension_hp adalah dimensi dari hp tersebut. Kita akan melakukan perubahan sebagai berikut
   - `n_components=1` karena hanya ada 1 komponen saja.
   - Fit model dengan data masukan.
   - Tambahkan fitur baru ke dataset dengan nama 'resolution_px' dan dimension_hp, kemudian lakukan proses transformasi.
   - Drop kolom 'px_height', 'px_width' untuk dimension_px, sedang 'sc_h','sc_w', 'm_dep' untuk dimension_hp.
   
   Dan hasilnya seperti gambar di bawah:
   
   ![image](https://user-images.githubusercontent.com/43197282/181408326-4db730c3-c4c8-4427-8daa-d5fe9907051d.png)
   
   Kita sudah memiliki 2 fitur baru yaitu 'resolution_px' dan 'dimension_hp'.
   
2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.
   Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik klasifikasi model terhadap data baru. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. Proporsi pembagian bisanya 80:20 jika data kita hanya 1000-an.
   
   Pertama kita import dulu kelas train_test_split dengan kode `from sklearn.model_selection import train_test_split`, setelah itu kita hapus fitur price_range pada variabel `X` dengan kode `X = data_train.drop(["price_range"],axis =1)` untuk variabel `y` kita set isinya price_range dengan kode `y = data_train["price_range"]`, setelah itu kita bagi dataset kita dengan proporsi data latih 80% dan data tes 20% dengan kode `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)`.
   
   Hasil nya adalah: 
   
   ![image](https://user-images.githubusercontent.com/43197282/181408549-38678b2d-a7fa-45f5-bd29-734e4c8deb3b.png)
    
   Setelah itu kita print dan lihat total dari pembagiannya, dengan data latih berjumlah 1204 dan data tes berjumlah 302.

## Modeling

Di tahap ini kita akan menggunakan 3 jenis algoritma yaitu:
- K-Nearest Neighbor (KNN) dengan KNeighborsClassifier
- Random Forest dengan RandomForestClassifier
- Boosting Algorithm dengan GradientBoostingClassifier

Baiklah mari kita jabarkan ketiga poin diatas satu-persatu:

1. K-Nearest Neighbor (KNN) dengan KNeighborsClassifier [2]

   Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.
   
   KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi.

   Kelebihan pada Algoritma KNN adalah:
   - Algoritma K-NN kuat dalam mentraining data yang noisy.
   - Algoritma K-NN sangat efektif jika datanya besar.
   - Mudah diimplementasikan.
   
   Kekurangan pada Algoritma KNN adalah:
   - Algoritma K-NN perlu menentukan nilai parameter K.
   - Sensitif pada data pencilan / titik data yang terpaut jauh dari titik data lainnya.
   - Rentan pada variabel yang non-informatif.
   

2. Random Forest dengan RandomForestClassifier [3]

   Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.
   
   Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir.
   
   Kelebihan pada Algoritma Random Forest adalah:
   - Dapat mengatasi noise dan missing value.
   - Dapat mengatasi data dalam jumlah yang besar.
   
   Kekurangan pada Algoritma Random Forest adalah:
   - Interpretasi yang sulit.
   - Membutuhkan tuning model yang tepat untuk data.
   
   
3. Boosting Algorithm dengan GradientBoostingClassifier [4]

   Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.
   
   Seperti namanya, boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa.
   
   Kelebihan pada Algoritma Boosting adalah:
   - Hasil pemodelan yang lebih akurat.
   - Model yang stabil dan lebih kuat (robust).
   
   Kekurangan pada Algoritma Boosting adalah:
   - Pengurangan kemampuan interpretasi model.
   - Tingkat kesulitan yang tinggi.
   
## Evaluation

Ditahap ini kita akan mengevaluasi setiap model dengan menggunakan metrik `accuracy_score` dan `classification_report`. Dan diakhir kita akan memilih model yang paling baik dari evaluasi menggunakan kedua mertik tersebut.

Baiklah tanpa berlama-lama lagi kita lihat cara cara kerja dari kedua metrik diatas:

1. Cara kerja metrik `accuracy_score` [5]

   Metrik `accuracy_score` bekerja dengan rumus, Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) [6], dimana:
   - TP = True Positive 
   - TN = True Negative
   - FP = False Positive
   - FN = False Negatif
   Dalam klasifikasi multilabel, fungsi mengembalikan akurasi subset. Jika seluruh rangkaian label yang diprediksi untuk sampel secara akurat cocok dengan rangkaian label yang sebenarnya. Maka akurasi subset adalah 1,0 jika tidak, akurasinya hampir 0,0.
   
   Hasil keluaran dari metriknya adalah seperti berikut:
   
   ![image](https://user-images.githubusercontent.com/43197282/181408987-795ac4f8-a338-4c3f-a436-0f26984f1a5c.png)

   Mari kita viasualisasikan hasil diatas ke dalam tampilan yang lebih menarik:
   
   ![Screenshot_3](https://user-images.githubusercontent.com/43197282/180794358-a739ae28-2ae1-4d56-b81a-c7266ae63cc4.jpg)
   
   Dari hasil diatas **Model KNN** bisa dibilang yang terbaik menurut metrik `accuracy_score`.
   
2. Cara kerja metrik `classification_report` [7]
   
   Metrik `classification_report` bekerja dengan mengembalikan 4 nilai yaitu 'precision', 'recall', 'f1-score', 'support'. Dengan acuan seperti berikut:
   - Jika dataset memiliki jumlah data False Negatif dan False Positif yang seimbang (Symmetric), maka bisa gunakan **Accuracy**, tetapi jika tidak seimbang, maka sebaiknya menggunakan **F1-Score**.
   - Dalam suatu problem, jika lebih memilih False Positif lebih baik terjadi daripada False Negatif, misalnya: Dalam kasus Fraud/Scam, kecenderungan model mendeteksi transaksi sebagai fraud walaupun kenyataannya bukan, dianggap lebih baik, daripada transaksi tersebut tidak terdeteksi sebagai fraud tetapi ternyata fraud. Untuk problem ini sebaiknya menggunakan **Recall**.
   - Sebaliknya, jika lebih menginginkan terjadinya True Negatif dan sangat tidak menginginkan terjadinya False Positif, sebaiknya menggunakan **Precision**. [8]
   
   Hasil printnya sebagai berikut:
   
   ![image](https://user-images.githubusercontent.com/43197282/181409539-68111655-b3d7-458a-8050-a42157deedda.png)
   
   ![image](https://user-images.githubusercontent.com/43197282/181409694-bee35df1-6066-4147-85aa-05fc2309c089.png)
   
   ![image](https://user-images.githubusercontent.com/43197282/181409862-5d568ad2-bb25-4ff3-aebb-85721637200a.png)

   Dengan hasil:
   
      - KNN = 92%
      - Random Forest = 78%
      - Boosting = 90%
   
   Bisa di simpulkan **Model KNN** masih memimpin soal hasil, kita melihatnya dari Test nya bukan Trainnya. Kita lihat nilai accuracy karena jumlah data kita yang seimbang.
   
   
## Conclusion

- Setelah Kita melihat hasil visual dari metrik evaluasi bisa kita simpulkan bahwa **Model KNN** memang yang paling powerfull, kita melihat nya dari hasil pengujian menggunakankan data tes.
- Kita juga akhirnya tau bahawa fitur yang paling berpengaruh menentukan kelas harga dari suatu hp adalah **ram**.
- Kita juga berhasil membuat 3 model model machine learning yang dapat memprediksi kelas harga hp berdasar fitur-fitur yang ada, dan kita mendapatkan dari 3 model yang ada Model KNN-lah yang paling powerfull.

## Reference

[1] HHM, "Mengenal Box-Plot (Box and Whisker Plots)", in Articles Binus University. Tersedia [tautan](https://accounting.binus.ac.id/2020/12/19/mengenal-box-plot-box-and-whisker-plots/). Diakses pada: Juli 2022.

[2] Scikit-learn K-NN Algorithm Documentation. Tersedia [tautan](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). Diakses pada: Juli 2022.

[3] Scikit-learn Random Forest Algorithm Documentation. Tersedia [tautan](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Diakses pada: Juli 2022.

[4] Scikit-learn Boosting Algorithm Documentation. Tersedia [tautan](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html). Diakses pada: Juli 2022.

[5] Scikit-learn accuracy_score Metrics Documentation. Tersedia [tautan](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). Diakses pada: Juli 2022.

[6] Kumar, Bijay. "Scikit learn accuracy_score", in Python Guides. Tersedia [tautan](https://pythonguides.com/scikit-learn-accuracy-score/). Diakses pada: Juli 2022.

[7] Scikit-learn classification_report Metrics Documentation. Tersedia [tautan](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report). Diakses pada: Juli 2022.

[8] Hendra, "Terminologi Machine Learning", in Blog. Tersedia [tautan](https://www.hendra.my.id/2020/09/terminologi-machine-learning.html). Diakses pada: Juli 2022.
