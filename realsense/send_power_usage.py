import requests
import json
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, constant
import time
# EdgeAgent 설정
edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac')
edgeAgentOptions.connectType = constant.ConnectType['DCCS']
dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/', credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl')
edgeAgentOptions.DCCS = dccsOptions
edgeAgent = EdgeAgent(edgeAgentOptions)
edgeAgent.connect()

def fetch_power_usage(device_id, access_token):
    #smartthings id
    url = f"https://api.smartthings.com/v1/devices/8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1/status"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        power_usage = data['components']['main']['powerMeter']['power']['value']
        send_data_to_datahub(power_usage)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Success at {current_time}!")
    else:
        print("Failed to fetch data:", response.status_code)

def send_data_to_datahub(power_usage):
    edgeData = EdgeData()
    tag = EdgeTag(deviceId="volume_camera", tagName="power_usage", value=str(power_usage))
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Sent power usage to DataHub: {power_usage}")
def fetch_and_send_power_usage_periodically(device_id, access_token):
    while True:
        fetch_power_usage(device_id, access_token)
        time.sleep(5)  # 1800초는 30분을 의미합니다.
# Example usage
#smartthing id&token
device_id = '8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1'
access_token = '927f5961-87fe-4cb0-89f4-675520460baf'
fetch_and_send_power_usage_periodically(device_id, access_token)
