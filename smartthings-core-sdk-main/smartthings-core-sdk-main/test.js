const { SmartThingsClient, BearerTokenAuthenticator } = require('@smartthings/core-sdk');
const axios = require('axios');  // HTTP 요청을 위한 라이브러리

// SmartThings API를 위한 설정
const smartThingsClient = new SmartThingsClient(new BearerTokenAuthenticator('9c5d5a1a-ed3f-4fdc-8ae2-7b61baa01146'));

async function fetchAndSendPowerUsage(deviceId) {
    try {
        const deviceStatus = await smartThingsClient.devices.getStatus(deviceId);
        const powerUsage = deviceStatus.components.main.powerMeter.power.value;

        console.log(`Device Power Usage: ${powerUsage} Watts`);

        // DataHub로 데이터를 보내기
        const data = {
            description: "Current power usage of the device",
            readOnly: false,
            arraySize: 0,
            engUnit: "Watts",
            fraDspFmt: 0,
            spanHigh: 10000,  // 예상 최대 전력 사용량
            spanLow: 0,       // 최소 전력 사용량
            state0: "Normal",
            state1: "High",
            formula: "None",
            interval: "1s",   // 업데이트 간격, 필요시 조정
            startAt: new Date().toISOString(),
            enable: true,
            tags: [{
                nodeId: "3607ae3d-5e1e-4171-8706-b6a111fa05ac",
                deviceId: "volume_camera",
                tagName: 'power_usage',
                alias: 'Power Usage'
            }],
            value: powerUsage
        };

        await axios.put('https://portal-datahub-trainingapps-eks004.sa.wise-paas.com/api/v1/Tags/3607ae3d-5e1e-4171-8706-b6a111fa05ac/volume_camera/power_usage', data);
    } catch (error) {
        if (error.response) {
            console.error('Error response:', error.response.data);
            console.error('Status:', error.response.status);
            console.error('Headers:', error.response.headers);
        } else if (error.request) {
            console.error('No response received:', error.request);
        } else {
            console.error('Error setting up request:', error.message);
        }
        console.error('Request configuration:', error.config);
    }
}

// 5분마다 전력 사용량을 조회하고 보내기
setInterval(() => fetchAndSendPowerUsage('8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1'), 500);  // 300000ms = 5 minutes
