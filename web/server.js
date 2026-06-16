/**
 * NYC Taksi Talep Tahmini - Web Sunucusu
 * ========================================
 * Node.js + Express ile:
 * 1. public/ klasöründen statik dosya sunumu
 * 2. /api/* isteklerini FastAPI servisine proxy
 *
 * Ankara Üniversitesi - 3522 Bulut Bilişim Dersi
 */

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const API_TARGET = process.env.API_TARGET || 'http://localhost:8000';

// CORS
app.use(cors());

// JSON body parser
app.use(express.json());

// API Proxy: /api/* → FastAPI (localhost:8000)
app.use('/api', createProxyMiddleware({
    target: API_TARGET,
    changeOrigin: true,
    pathRewrite: { '^/api': '' },
    onProxyReq: (proxyReq, req, res) => {
        // POST body'yi proxy'e ilet
        if (req.body && Object.keys(req.body).length > 0) {
            const bodyData = JSON.stringify(req.body);
            proxyReq.setHeader('Content-Type', 'application/json');
            proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
            proxyReq.write(bodyData);
        }
    },
    onError: (err, req, res) => {
        console.error('Proxy hatası:', err.message);
        res.status(502).json({
            error: 'API servisine bağlanılamıyor',
            detail: `FastAPI servisi ${API_TARGET} adresinde çalışıyor mu?`,
            hint: 'cd api && uvicorn main:app --port 8000 --reload'
        });
    }
}));

// Statik dosya sunumu (public/ klasörü)
app.use(express.static(path.join(__dirname, 'public')));

// SPA fallback
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Sunucu başlat
app.listen(PORT, () => {
    console.log('');
    console.log('🚕 NYC Taksi Talep Tahmini - Web Sunucusu');
    console.log('==========================================');
    console.log(`   Web:  http://localhost:${PORT}`);
    console.log(`   API:  ${API_TARGET} (proxy: /api/*)`);
    console.log('');
});
