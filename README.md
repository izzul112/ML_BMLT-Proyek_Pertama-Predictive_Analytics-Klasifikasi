# Laporan Proyek Machine Learning - Habib Azizul Haq


## Domain Proyek

**Klasifikasi Harga Handphone (HP).**

Saat ini hp sudah sangat mudah untuk dimiliki oleh hampir semua orang, mulai dari yg harganya ratusan ribu hingga puluhan juta, tentu tergantung pada kebutuhan masing-masing orang, ada yang menganggap harga puluhan juta itu wajar dan ada yang mengangggapnya itu kemahalan.

**Rubrik/Kriteria Tambahan (Opsional):**

- Hal tersebut tentu membuat kita kadang tidak jadi membeli suatu hp, karena kita khawatir apakah nanti semua fitur tersebut Gimmik atau bukan.
- Kita kadang juga merasa khawatir apakah semua fitur di suatu hp tersebut secara harga sesuai apa tidak, tentu kita tidak ingin terkena *genjutsu* sales-sales hp di toko yang mengatakah "Beli hp ini aja kak, PUBG rata kanan semua!", namun setelah di rumah kita baru sadar kok beli hp yang ini? Padahal tadi maunya bukan yang ini.
- Sebagai orang yang jarang tau berita tentang teknologi, dan sedang mencari hp dengan kelas harga tertentu kita kadang bingung dan kesusahan dalam menentukan pilihan

Solusi dari masalah tersebut, kita kembangkan sebuah model Machine Learning (ML) untuk membantu kita yang kesulitan menentukan sebuah rentang kelas harga dari hp, apakah hp dengan spesifikasi sekian-sekian yang berada di kelas harga menengah, apakah sesuai dengan kelas harganya, atau kita di tipu oleh sales-sales hp, karena mereka ingin prodak tersebut segera habis.

- referensi
- referensi

## Business Understanding

### Problem Statements Menjelaskan pernyataan masalah latar belakang:

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap kelas harga suatu hp?
- Berapa kelas harga pasar hp dengan karakteristik atau fitur tertentu?

### Goals Menjelaskan tujuan dari pernyataan masalah:

- Mengetahui fitur yang paling berkorelasi dengan kelas harga hp.
- Membuat model machine learning yang dapat memprediksi kelas harga hp seakurat mungkin berdasarkan fitur-fitur yang ada.

**Rubrik/Kriteria Tambahan (Opsional):**

### Solution statements: 

- Untuk menghasilkan model yang optimal namun tetap sederhana kita akan menggunakan 3 algoritma yaitu KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier.
- Dari 3 algoritma di atas kita akan menggunakan metrik Mean Absolute Error yang memang cocok untuk kasus klasifikasi, untuk melihat pada ketiga algoritma diatas mana yang paling powerfull.

## Data Understanding

Kita akan menggunkan dataset yang berisi 21 variabel yang biasanya sering ditanyakan ketika membeli hp atau mungkin menjadi suatu standar yang digunakan masyarakat dalam menggolongkan kelas harga dari suatu hp. Sumber dataset yang akan kita gunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?datasetId=11167&select=train.csv)

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

Dari 21 variabel diatas variabel price_range adalah sasaran kita, kita akan membuat model berdasar rentang harga yang ada di dalam variabel price_range, dengan nilai 0 = low cost (murah), 1 = medium cost (standar), 2 = high cost (mahal / flagship), 3 = very high cost (hp para sultan).

**Rubrik/Kriteria Tambahan (Opsional):**
1. Kita akan melihat ada berapa jumlah baris dari data dalam tampilan tabel.
   Kita akan menggunakan menggunakan `data_train.head()`, `data_train` adalah nama variabel yang kita ganakan saat meload dataset kita, sedang fungsi `.head()` akan menampilkan 5 baris data teratas dari keseluruhan dataset kita. Dari kode `data_train.head()` kita mendapat informasi:
   - Ada 2000 baris dalam dataset
   - Dan seperti yang dijelaskan dalam deskipsi variabel ada 21 kolom
   
2. Kita akan melihat tipe data dari 21 kolom di dataset.  
   Kita akan menggunakan `data_train.info()`, `.info` akan menampilkan informasi tipe data 21 kolom dataset yang kita gunakan. Dari output terlihat bahwa:   
   - Terdapat 2 kolom numerik dengan tipe data float64 yaitu : clock_speed dan m_dept
   - Terdapat 19 kolom numerik dengan tipe data int64 yaitu : battery_power, blue, dual_sim, fc, four_g, int_memory, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_schreen, wifi dan price_range

3. Kita akan mengecek deskripsi statistik data.
   Kita akan mengeceknya dengan fungsi `describe()`. Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:
   - Count adalah jumlah sampel pada data.
   - Mean adalah nilai rata-rata.
   - Std adalah standar deviasi.
   - Min yaitu nilai minimum setiap kolom.
   - 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
   - 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
   - 75% adalah kuartil ketiga.
   - Max adalah nilai maksimum.

4. Kita akan melihat apakah ada data yang kosong / tidak ada isinya.
   Kita akan mengeceknya dengan fungsi `.isnull().sum()`. Fungsi `.isnull()` akan mengecek apakah ada data kosng pada setiap baris pada semua kolom di dataset kita, kemudian kita gunakan juga fungsi `.sum()` untuk menjumlahnya sehingga hasilnya akan seperti berikut. Dari hasil outputnya data kita tidak memiliki data yang kosong.
   
5. Kita akan melihat apakah data kita memiliki outliers
   Kita akan mengeceknya dengan fungsi `sns.boxplot(x=data_train['px_height'])`. Variabel `sns` adalah tempat kita menampung library **Seaborn** yang akan kita gunakan untuk memvisualisaikan dataset kita. Fungsi `.boxplot()` akan menampilkan visualisasi dari dataset kita dengan visualisasi seperti gambar berikut:
   ![gambar boxplot](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:3be38c69ec4f1ee07ce8c24e42ce23fb20210910131731.png)
   
   Dalam hal ini kita menggunakan sampel `px_heght` dari gambar bisa kita lihat adanya outliers.
   
   ![Screenshot_2](https://user-images.githubusercontent.com/43197282/180590253-ec55f05c-352f-483b-b467-00d22b634432.jpg)
   
   Tenang tidak perlu panik kita akan mengatasinya dengan teknik IQR dimana data yang berada di luar Q1 dan Q3 adalah outlier, dimana kita akan menentukan nilai batas atas dan bawah, dengan persamaan berikut:
   - Batas bawah = Q1 - 1.5 * IQR
   - Batas atas = Q3 + 1.5 * IQR

5. Kita akan melihat variabel apa saja yang memiliki hubungan yang kuat atas klasifikasi harga suatu hp
