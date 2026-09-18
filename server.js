const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = __dirname;

const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff2': 'font/woff2',
    '.woff': 'font/woff',
    '.ttf': 'font/ttf'
};

let prevCpus = os.cpus();

function getCpuUsage() {
    const currentCpus = os.cpus();
    const perCore = currentCpus.map((cpu, i) => {
        const prev = prevCpus[i] || cpu;
        const prevTotal = Object.values(prev.times).reduce((a, b) => a + b, 0);
        const currTotal = Object.values(cpu.times).reduce((a, b) => a + b, 0);
        const prevIdle = prev.times.idle;
        const currIdle = cpu.times.idle;
        const totalDiff = currTotal - prevTotal;
        const idleDiff = currIdle - prevIdle;
        const usage = totalDiff > 0 ? Math.round(((totalDiff - idleDiff) / totalDiff) * 100) : 0;
        return {
            core: i + 1,
            usage: Math.min(100, Math.max(0, usage))
        };
    });
    prevCpus = currentCpus;

    const totalUsage = Math.round(perCore.reduce((acc, c) => acc + c.usage, 0) / perCore.length);
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;

    return {
        timestamp: Date.now(),
        model: currentCpus[0]?.model || 'Host CPU',
        speed: currentCpus[0]?.speed || 0,
        coreCount: currentCpus.length,
        overallUsage: totalUsage,
        perCore,
        memory: {
            totalMB: Math.round(totalMem / (1024 * 1024)),
            freeMB: Math.round(freeMem / (1024 * 1024)),
            usedMB: Math.round(usedMem / (1024 * 1024)),
            usedPercent: Math.round((usedMem / totalMem) * 100)
        },
        uptime: Math.round(os.uptime()),
        platform: os.platform()
    };
}

const server = http.createServer((req, res) => {
    // API endpoint for real system metrics
    if (req.url === '/api/system-metrics' || req.url === '/api/cpu') {
        const data = getCpuUsage();
        res.writeHead(200, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'no-cache, no-store, must-revalidate'
        });
        return res.end(JSON.stringify(data));
    }

    // Serve static files
    let reqPath = req.url.split('?')[0];
    if (reqPath === '/') reqPath = '/index.html';

    const safePath = path.normalize(path.join(PUBLIC_DIR, reqPath));
    if (!safePath.startsWith(PUBLIC_DIR)) {
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        return res.end('Forbidden');
    }

    fs.stat(safePath, (err, stats) => {
        if (err || !stats.isFile()) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            return res.end('404 Not Found');
        }

        const ext = path.extname(safePath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'application/octet-stream';

        res.writeHead(200, { 'Content-Type': contentType });
        fs.createReadStream(safePath).pipe(res);
    });
});

server.listen(PORT, () => {
    console.log(`Autonomous Drone Simulation server running at http://localhost:${PORT}`);
});
