from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, EdgeConfig, DeviceConfig, TextTagConfig, AnalogTagConfig

def create_edge_agent():
    edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac')
    edgeAgentOptions.connectType = constant.ConnectType['DCCS']
    dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/', credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl')
    edgeAgentOptions.DCCS = dccsOptions
    edgeAgent = EdgeAgent(edgeAgentOptions)
    return edgeAgent

def on_connected(edgeAgent, isConnected):
    print("Connected to DataHub!")
    config = generate_config()
    edgeAgent.uploadConfig(action=constant.ActionType['Create'], edgeConfig=config)

def on_disconnected(edgeAgent, isDisconnected):
    print("Disconnected from DataHub.")

def generate_config():
    config = EdgeConfig()
    deviceConfig = DeviceConfig(id='volume_camera', name='volume_cal', description='', deviceType='camera', retentionPolicyName='')
    text = TextTagConfig(name='capture_image', description='Captured Image', readOnly=False, arraySize=1)
    analog = AnalogTagConfig(name='volume', description='Volume in L', readOnly=False, arraySize=0, spanHigh=10000, spanLow=0, engineerUnit='L', integerDisplayFormat=4, fractionDisplayFormat=2)
    deviceConfig.textTagList.append(text)
    deviceConfig.analogTagList.append(analog)
    config.node.deviceList.append(deviceConfig)
    return config

edgeAgent = create_edge_agent()
edgeAgent.on_connected = on_connected
edgeAgent.on_disconnected = on_disconnected
edgeAgent.connect()
