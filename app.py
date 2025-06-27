from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np
from io import BytesIO


app = Flask(__name__)
CORS(app)

# Load model CNN
model = load_model("COBA7model_buncis_inceptionv3.h5")


# Label dan rekomendasi
labels = ['Hawar Daun', 'Karat Daun', 'Daun Sehat', 'Thrips']
recommendations = {
    'Hawar Daun': "Sanitasi lahan, Pengaturan jarak tanam agar sirkulasi udara baik dan kelembaban tidak terlalu tinggi., Penyemprotan ekstrak tanaman antimikroba seperti ekstrak bawang putih atau serai. Untuk penanganan secara kimiawi bisa menggunakan fungisida berbahan aktif mancozeb, klorotalonil, atau tembaga hidroksida, serta disinfeksi benih sebelum tanam menggunakan fungisida sistemik seperti carbendazim.",
    'Karat Daun': "Penggunaan mulsa organik untuk menekan percikan spora dari tanah, rotasi tanaman dengan tanaman non-inang selama 1–2 musim tanam, penyemprotan larutan kompos (PGPR) atau biofungisida berbasis Trichoderma. Untuk penanganan secara kimiawi bisa menggunakan fungisida berbahan aktif azoksistrobin, propikonazol, atau difenokonazol (Semprot secara berkala (7–10 hari sekali) terutama saat kondisi lembab).",
    'Daun Sehat': "Tanaman sehat. Lanjutkan perawatan rutin.",
    'Thrips': "Penggunaan perangkap kuning (yellow sticky trap) untuk monitoring dan pengendalian awal, pemanfaatan musuh alami seperti Orius spp. (serangga predator thrips), penyemprotan ekstrak daun nimba (Azadirachta indica) yang bersifat insektisida nabati, pengelolaan kelembaban lahan karena thrips lebih aktif di lingkungan kering. Untuk penanganan secara kimiawi bisa menggunakan insektisida berbahan aktif seperti abamectin, imidacloprid, atau spinetoram (Semprot saat populasi mulai meningkat, fokus pada permukaan bawah daun)"
}

@app.route('/predict', methods=['POST'])
def predict():
    print("[INFO] Menerima request /predict")

    if 'file' not in request.files:
        print("[ERROR] Tidak ada file dalam request")
        return jsonify({'error': 'File tidak ditemukan'}), 400

    file = request.files['file']
    print(f"[INFO] File diterima: {file.filename}")

    try:
         # Preprocessing yang sesuai dengan InceptionV3
        img = image.load_img(BytesIO(file.read()), target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)  # nilai input [-1, 1]
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        pred = model.predict(img_array)
        class_index = np.argmax(pred[0])
        result = labels[class_index]
        recommendation = recommendations[result]

        print(f"[INFO] Prediksi: {result}")
        return jsonify({'prediction': result, 'recommendation': recommendation})

    except Exception as e:
        print(f"[ERROR] Gagal memproses gambar: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
