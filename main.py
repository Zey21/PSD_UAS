import  streamlit as st
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Mengimpor model dari file eksternal
with open('D:\projectPSD\model_knn.pkl', 'rb') as file:
    knn = pickle.load(file)

st.title("Aplikasi PBA-NLP")

# inisialisasi data
tab1, tab2 = st.tabs(["Description data", "Processing"])

with tab1:
    st.subheader("Deskripsi")
    st.write(
        "Analisis Sentimen Terhadap Bakal Calon Presiden 2024 dengan Algoritma Naïve Bayes")
    st.caption("""Presiden di Indonesia dipilih melalui masyarakat dengan melalui proses demokrasi yaitu
pemilihan presiden (pilpres) yang dilaksanakan setiap 5 tahun sekali. Menjadi seorang presiden memiliki beberapa
persyaratan yang dimana persyaratan tersebut adalah seseorang tidak diperbolehkan menjadi presiden apabila orang
tersebut sebelumnya telah menjadi presiden selama 2 periode secara berturut – turut, yang dalam hal ini presiden
Indonesia saat ini sudah tidak bisa mencalonkan kembali menjadi Presiden pada pilpres selanjutnya yang akan terlaksana
pada tahun 2024. Berdasarkan tersebut banyak bermunculan survei elektabilitas terhadap beberapa tokoh publik yang memiliki
elektabilitas baik yang menjadikan tokoh ini bisa dijadikan bakal calon presiden Indonesia di pilpres pada tahun 2024. Berdasarkan penjelasan tersebut penelitian ini akan dilakukan sebuah analisa sentimen terhadap bakal calon
presiden 2024 dengan menggunakan algoritme naïve bayes yang nantinya hasil penelitian ini dapat dimanfaat masyarakat
sebagai bahan referensi dalam memilih pemimpinnya di kemudian hari pada tahun 2024.""")

with tab2:
    st.subheader("Processing Data")
    input1 = st.number_input("Masukan umur : ")
    input2 = st.number_input("Edukasi Ayah rate(1-4) : ")
    input3 = st.number_input("Edukasi Ibu rate(1-4) : ")
    input4 = st.number_input("Waktu belajar rate(1-10) : ")
    input5 = st.number_input("Kedekatan keluarga rate(1-5) : ")
    input6 = st.number_input("Waktu luang rate(1-5) : ")
    input7 = st.number_input("Waktu bermain bersama teman rate(1-5) : ")
    input8 = st.number_input("Seberapa sering mengonsumsi alcohol ketika bekerja rate(1-5) : ")
    input9 = st.number_input("Seberapa sering mengonsumsi alcohol ketika liburan rate(1-5) : ")
    input10 = st.text_input("Jenis kelamin (L/P) : ").lower()
    input11 = st.text_input("Apakah masih tinggal bersama orang tua (Y/T) : ").lower()
    #Biner
    if input10 == "l":
        L = 1
        P = 0
    else :
        L = 0
        P = 1
    
    if input11 == "y":
        T = 1
        A = 0
    else:
        T = 0
        A = 1
    
    #Proses
    X_test = np.array([input1,input2,input3,input4,input5,input6,input7,input8,input9,L,P,A,T])
    X_test = X_test.reshape(-1,1)
    scaler = MinMaxScaler()

    # Melakukan penskalaan fitur
    X = scaler.fit_transform(X_test)
    X = X.reshape(-1)
    
        
    if st.button("Check Result"): 
        X = X.reshape(1,-1)
        y_pred = knn.predict(X)
        if y_pred[0] == 1 :
            ket = "Sangat Sehat"
        elif y_pred[0] == 2:
            ket = "Sehat"
        elif y_pred[0] == 3:
            ket = "Normal"
        elif y_pred[0] == 4:
            ket = "Buruk"
        elif y_pred[0] == 5:
            ket = "Sangat Buruk"
            
        st.write("Prediksi :",y_pred[0],", Keterangan :",ket)
        
    else:
        st.write("Output :")
        pass
