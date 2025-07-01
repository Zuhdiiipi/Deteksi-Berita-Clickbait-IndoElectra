# predict.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model dan tokenizer
model_path = "model-deteksi-clickbait"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Pastikan ke CPU atau GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fungsi prediksi
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class



# Contoh penggunaan
judul = "Dolar AS Pagi Ini Menguat ke Level Rp 16.210"
hasil = predict(judul)
print("Judul Berita : ",judul)
print(f"Hasil prediksi: {'Clickbait' if hasil == 1 else 'Non-clickbait'}")
