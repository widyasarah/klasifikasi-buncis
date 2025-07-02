from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load model .tflite
interpreter = tflite.Interpreter(model_path="new_model.tflite")
interpreter.allocate_tensors()

# Ambil detail input/output model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label dan rekomendasi
labels = ['Hawar Daun', 'Karat Daun', 'Daun Sehat', 'Thrips']
recommendations = {
    'Hawar Daun': "Sanitasi lahan, Pengaturan jarak tanam agar sirkulasi udara baik dan kelembaban tidak terlalu tinggi., Penyemprotan ekstrak tanaman antimikroba seperti ekstrak bawang putih atau serai. Untuk penanganan secara kimiawi bisa menggunakan fungisida berbahan aktif mancozeb, klorotalonil, atau tembaga hidroksida, serta disinfeksi benih sebelum tanam menggunakan fungisida sistemik seperti carbendazim.",
    'Karat Daun': "Penggunaan mulsa organik untuk menekan percikan spora dari tanah, rotasi tanaman dengan tanaman non-inang selama 1–2 musim tanam, penyemprotan larutan kompos (PGPR) atau biofungisida berbasis Trichoderma. Untuk penanganan secara kimiawi bisa menggunakan fungisida berbahan aktif azoksistrobin, propikonazol, atau difenokonazol (Semprot secara berkala (7–10 hari sekali) terutama saat kondisi lembab).",
    'Daun Sehat': "Tanaman sehat. Lanjutkan perawatan rutin.",
    'Thrips': "Penggunaan perangkap kuning (yellow sticky trap) untuk monitoring dan pengendalian awal, pemanfaatan musuh alami seperti Orius spp. (serangga predator thrips), penyemprotan ekstrak daun nimba (Azadirachta indica) yang bersifat insektisida nabati, pengelolaan kelembaban lahan karena thrips lebih aktif di lingkungan kering. Untuk penanganan secara kimiawi bisa menggunakan insektisida berbahan aktif seperti abamectin, imidacloprid, atau spinetoram(Semprot saat populasi mulai meningkat, fokus pada permukaan bawah daun)"
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan'}), 400

    file = request.files['file']

    try:
        # Baca dan proses gambar
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = img.resize((299, 299))
        img_array = np.asarray(img).astype(np.float32)
        img_array = (img_array / 127.5) - 1.0  # Normalisasi untuk InceptionV3
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(output_data))
        result = labels[class_index]
        recommendation = recommendations[result]

        return jsonify({'prediction': result, 'recommendation': recommendation})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
