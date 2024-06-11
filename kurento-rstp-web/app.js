const WebSocket = require('ws');
const http = require('http');
const ffmpeg = require('fluent-ffmpeg');

const server = http.createServer();
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    const ffmpegProcess = ffmpeg('rtsp://chanmin4:location1957@211.112.65.101:554/stream1')
        .addOptions([
            '-f mpegts',
            '-codec:v mpeg1video',
            '-s 640x480',
            '-b:v 800k',
            '-r 30'
        ])
        .on('error', (err) => {
            console.error('FFmpeg error:', err);
            ws.close();
        });

    ffmpegProcess.pipe().on('data', (data) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(data);
        }
    });

    ws.on('close', () => {
        ffmpegProcess.kill();
    });
});

const port = 8082; // 포트를 8082로 변경
server.listen(port, () => {
    console.log(`WebSocket server is running on ws://0.0.0.0:${port}`);
});
