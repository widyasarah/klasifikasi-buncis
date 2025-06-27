function uploadImage() {
  const input = document.getElementById('imageInput');
  const file = input.files[0];
  const resultDiv = document.getElementById('result');
  const previewImage = document.getElementById('previewImage');

  if (!file) {
    resultDiv.innerHTML = "Silakan pilih gambar daun.";
    previewImage.style.display = "none";
    return;
  }

  // Tampilkan preview gambar
  const reader = new FileReader();
  reader.onload = function (e) {
    previewImage.src = e.target.result;
    previewImage.style.display = "block";
  };
  reader.readAsDataURL(file); // <-- Ini WAJIB

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error("Server error: " + errorText);
      }
      return response.json();
    })
    .then((data) => {
      resultDiv.innerHTML = `
        <strong>Hasil Klasifikasi:</strong> ${data.prediction} <br>
        <strong>Rekomendasi Penanganan:</strong> ${data.recommendation}
      `;
    })
    .catch((error) => {
      console.error("[ERROR]", error);
      resultDiv.innerHTML = "Terjadi kesalahan saat memproses gambar:<br>" + error.message;
    });
}
