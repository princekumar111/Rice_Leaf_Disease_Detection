import requests

url = "https://rice-leaf-disease-detection-waco.onrender.com/predict"

files = {
    "image": open("dataset\Brown_spot\Brown_spot (12).jpg", "rb")
}

response = requests.post(url, files=files)

print(response.json())