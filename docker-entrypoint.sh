#!/bin/sh
set -eu

requirements=/opt/pyshell/requirements.txt
if [ -e "$requirements" ]; then
    environment=$(mktemp -d /tmp/pyshell-env.XXXXXX)
    cp -a /opt/venv/. "$environment/"
    uv pip install --no-cache --python "$environment/bin/python" -r "$requirements"
    uv pip check --no-cache --python "$environment/bin/python"
    export PATH="$environment/bin:$PATH"
    exec "$environment/bin/python" -m pyshell "$@"
fi

exec /opt/venv/bin/python -m pyshell "$@"
