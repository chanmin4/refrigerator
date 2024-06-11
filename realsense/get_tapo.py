import requests

API_KEY = '92332b61-f209-4c4d-a9a5-f186d5661207'
DEVICE_ID = '047f9cb9-214d-4c19-8e66-7110bea501bd'

def get_cctv_stream_url():
    url = f'https://api.smartthings.com/v1/devices/{DEVICE_ID}/status'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    response_data = response.json()
    print(response_data)
    # CCTV 스트림 URL을 추출 (예시: "streamUrl" 키에 저장된 URL)
    stream_url = response_data['components']['main']['videoStream']['stream']['value']['InHomeURL']
    
    return stream_url

# CCTV 스트림 URL 가져오기
stream_url = get_cctv_stream_url()
print(f'CCTV Stream URL: {stream_url}')
