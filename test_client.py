import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "waveform_signal": [0.02, 0.05, 0.12, 0.08, -0.01, -0.25, 0.45, 0.10, 0.03, 0.01],
    "rr_intervals": [0.82, 0.79, 0.81, 0.84]
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Prediction Result:", response.json())
except Exception as e:
    print("Error connecting to server:", e)