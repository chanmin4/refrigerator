const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3001;

app.use('/proxy', createProxyMiddleware({
    target: 'http://211.112.65.101:8082', // 원래 스트림 URL
    changeOrigin: true,
    pathRewrite: {
        '^/proxy': '', // 경로 재작성
    },
    onProxyReq: (proxyReq, req, res) => {
        proxyReq.setHeader('Access-Control-Allow-Origin', '*');
    }
}));

app.listen(PORT, () => {
    console.log(`Proxy server is running at http://localhost:${PORT}`);
});
