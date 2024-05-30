const { SmartThingsClient, BearerTokenAuthenticator } = require('@smartthings/core-sdk');
const axios = require('axios');  // HTTP 요청을 위한 라이브러리

// SmartThings API를 위한 설정
const smartThingsClient = new SmartThingsClient(new BearerTokenAuthenticator('여기에-PAT-토큰-입력'));

async function fetchAndSendPowerUsage(deviceId) {
    try {
        const deviceStatus = await smartThingsClient.devices.getStatus(deviceId);
        const powerUsage = deviceStatus.components.main.powerMeter.power.value;

        console.log(`Device Power Usage: ${powerUsage} Watts`);

        // DataHub로 데이터를 보내기
        const data = {
            value: powerUsage,
            timestamp: new Date().toISOString()
        };

        await axios.post('http://your-datahub-url/api/data', data); // DataHub의 실제 URL과 엔드포인트를 사용하세요
    } catch (error) {
        console.error('Error fetching or sending device power usage:', error);
    }
}

// 5분마다 전력 사용량을 조회하고 보내기
setInterval(() => fetchAndSendPowerUsage('8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1'), 300000); // 300000ms = 5분
