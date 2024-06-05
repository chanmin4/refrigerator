import requests
import time
import json
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag

# SmartThings API를 위한 설정
api_token = '9c5d5a1a-ed3f-4fdc-8ae2-7b61baa01146'
device_id = '8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1'
headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json'
}

# EdgeAgent 설정
edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac')
dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/', credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl')
edgeAgentOptions.DCCS = dccsOptions
edgeAgent = EdgeAgent(edgeAgentOptions)
edgeAgent.connect()

def send_power_usage_to_datahub(power_usage):
    edgeData = EdgeData()
    tag = EdgeTag(deviceId="volume_camera", tagName="power_usage", value=power_usage)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Sent power usage to DataHub: {power_usage}")

def fetch_and_send_power_usage(device_id):
    try:
        url = f'https://api.smartthings.com/v1/devices/8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        device_status = response.json()
        power_usage = device_status['components']['main']['powerMeter']['power']['value']
        print(f"Device Power Usage: {power_usage} Watts")
        
        # DataHub로 데이터 전송
        send_power_usage_to_datahub(power_usage)

    except Exception as e:
        print(f"Error: {e}")

# 5분마다 전력 사용량을 조회하고 보내기
while True:
    fetch_and_send_power_usage(device_id)
    time.sleep(5)  # 300초 = 5분
