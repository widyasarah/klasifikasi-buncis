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
    'Hawar Daun': "Sanitasi lahan, pengaturan jarak tanam, dan penyemprotan fungisida seperti mancozeb.",
    'Karat Daun': "Gunakan mulsa, rotasi tanaman, dan fungisida seperti propikonazol.",
    'Daun Sehat': "Tanaman sehat. Lanjutkan perawatan rutin.",
    'Thrips': "Gunakan perangkap kuning, semprot ekstrak daun nimba, atau insektisida seperti abamectin."
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
