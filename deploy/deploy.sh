#!/usr/bin/env bash
set -e
# Deploy demo a fuerzafiel (micro VM 2 vCPU / 947 MB) — solo feed demo, no audio real
HOST="fuerzafiel"
DIR="/home/kevin/traductor-voz-entrevistas"

echo "→ rsync a $HOST"
rsync -avz --exclude venv --exclude .git --exclude __pycache__ ./ $HOST:$DIR/

echo "→ pip install (sin torch, solo demo deps)"
ssh $HOST "cd $DIR && python3 -m venv venv --system-site-packages || true && ./venv/bin/pip install -q fastapi==0.115.0 uvicorn==0.30.0"

echo "→ systemd"
ssh $HOST "sudo cp $DIR/deploy/traductor.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now traductor"

echo "→ Caddy"
ssh $HOST "sudo cp $DIR/deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy || sudo systemctl enable --now caddy"

echo "✓ https://traductor-demo.kevingraciano.dev"
