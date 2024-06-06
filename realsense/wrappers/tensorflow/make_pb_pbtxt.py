import requests
import json

def fetch_volume_and_text_data():
    url = "https://api.example.com/get_data"
    headers = {
        "Authorization": "Bearer YOUR_ACCESS_TOKEN",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        
        # 부피 데이터와 글자 데이터를 추출
        volume_data = [[item['volume'], item['id']] for item in data['volumes']]
        text_data = [[item['text'], item['id']] for item in data['texts']]
        
        return volume_data, text_data
    else:
        print("Failed to fetch data:", response.status_code)
        return [], []
# 테스트 함수 실행
fetch_volume_and_text_data()
