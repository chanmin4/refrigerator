import time
import adafruit_dht
import board
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, constant
# EdgeAgent 설정
edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac')
edgeAgentOptions.connectType = constant.ConnectType['DCCS']
dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/', credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl')
edgeAgentOptions.DCCS = dccsOptions
edgeAgent = EdgeAgent(edgeAgentOptions)
edgeAgent.connect()
def send_tem_to_datahub(total_volume_l):
    edgeData = EdgeData()
    deviceId = 'volume_camera'
    tagName = 'temperature'
    tag = EdgeTag(deviceId, tagName, total_volume_l)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Total volume {total_volume_l} L sent to DataHub")

def send_hu_to_datahub(total_volume_l):
    edgeData = EdgeData()
    deviceId = 'volume_camera'
    tagName = 'humidity'
    tag = EdgeTag(deviceId, tagName, total_volume_l)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Total volume {total_volume_l} L sent to DataHub")

dht_device=adafruit_dht.DHT22(board.D4)
while(True):
    try:
        temperature = dht_device.temperature
        humidity=dht_device.humidity
        print(f"Temp: {temperature:.1f} C Humidity:{humidity:.1f}%")
        send_hu_to_datahub(humidity)
        send_hu_to_datahub(temperature)
    except RuntimeError as error:
        print(error.args[0])
        
    time.sleep(30.0)
    
    
    
    
