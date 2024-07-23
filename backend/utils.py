from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from g4f.client import Client
# from openai import OpenAI

client = Client()
model = load_model('models/model.h5')
output_class = ["battery", "glass", "metal","organic", "paper", "plastic"]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def preprocessing_input(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img) # ResNet50 preprocess_input
    return img

async def predict(new_image_path):
    try:
        test_image = preprocessing_input(new_image_path)
        predicted_array = model.predict(test_image)
        predicted_value = output_class[np.argmax(predicted_array)]
        predicted_accuracy = round(np.max(predicted_array) * 100, 2)

        return predicted_value, predicted_accuracy
    except Exception as e:
        return f"Error processing image: {str(e)}", 0
    

def input_trash(input):
    messages = [
        {"role": "system", "content": "Anda adalah seorang propagandis perlindungan lingkungan. Tugas Anda adalah memberi saya informasi paling penting tentang jenis sampah yang saya berikan dalam daftar hanya 2 baris (ubah nama jenis sampah menjadi bahasa Indonesia sebelum menjawab), dengan fokus pada proses penguraian dan cara melakukannya itu. menangani limbah jenis ini. Hindari detail atau penjelasan yang tidak perlu."},
    ]
    messages.append(
        {"role": "user", "content": f"{input}"},
    )
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content
    return reply
