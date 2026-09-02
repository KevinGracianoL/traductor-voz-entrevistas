# Deploy demo — micro VM fuerzafiel

**No es el pipeline real.** El teleprompter real corre en `localhost` en tu laptop (ADR-005). Este deploy es solo demo de portafolio con feed estático, sin audio, para mostrar el UI con dominio/TLS.

- **VM:** 2 vCPU / 947 MB RAM — no alcanza para ASR, solo para FastAPI+WS demo.
- **Proxy:** Caddy (TLS auto, ~20 MB) vs nginx+certbot (~100 MB + cron).
- **Licencia:** VB-CABLE donationware — ojo si es máquina de empresa, requiere reinicio.

```bash
./deploy/deploy.sh
# o manual: rsync + pip install fastapi/uvicorn + Caddyfile + systemd
```

Feed demo: `POST /api/transcripcion` con `{"en":"hello","es":"hola"}` y WS broadcastea. Sin micrófono, sin RealtimeSTT.
