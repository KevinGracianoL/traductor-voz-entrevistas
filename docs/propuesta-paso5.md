# Propuesta Paso 5 — Teleprompter en vivo (para Hal)

**Objetivo:** UI mínima que muestre en tiempo real lo que dice el entrevistador (EN) y su traducción (ES) para que Kevin lea y verifique, desplegada en micro VM con dominio/TLS/nginx.

## Alcance mínimo (entregable en 1 PR)

- `src/traductor/ui/app.py` — FastAPI:
  - `GET /` → `teleprompter.html` (dos columnas grandes, ES+EN, auto-scroll)
  - `WebSocket /ws` → recibe `{en, es, ts}` del pipeline local y lo broadcastea a browsers
  - `POST /api/transcripcion` alternativo para polling si WS no conviene
- `src/traductor/ui/templates/teleprompter.html` + `static/style.css` — sin JS framework, solo WS + DOM
- `deploy/` — `nginx.conf` (proxy 8000 → 443), `systemd` service, `deploy.sh` (rsync + systemctl), `Caddy` como alternativa a certbot si prefieres
- Tests: `tests/test_ui.py` sin levantar server (TestClient), 100% en esa capa

## Fuera de alcance (siguiente PR si hace falta)

- Autenticación / multi-usuario
- Historial persistente (por ahora solo memoria)
- TTS en el browser

## Preguntas para Hal

1. ¿FastAPI+WS te parece bien o prefieres Flask-SocketIO / Hugo estático con polling?
2. ¿La micro VM `fuerzafiel` (2 vCPU/947 MB) alcanza para servir FastAPI+WS o lo dejamos solo como reverse proxy y el pipeline sigue local?
3. ¿Dominio/TLS lo gestionamos con Caddy (auto TLS) o nginx+certbot como en el plan original?

Con tu OK arranco el `paso5-teleprompter` con el stack que elijas.

CC @aldoeliacim
