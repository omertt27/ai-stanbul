import requests

API_TOKEN = 'your_apify_api_token'
STORE_ID = 'ceS7j5KCjXdSNiRe8'
RECORD_KEY = 'INPUT'

url = f'https://api.apify.com/v2/key-value-stores/{STORE_ID}/records/{RECORD_KEY}?token={API_TOKEN}'

response = requests.get(url)

if response.status_code == 200:
    input_data = response.json()
    print("Fetched INPUT record:", input_data)
else:
    print(f"Failed to fetch record: {response.status_code} - {response.text}")
