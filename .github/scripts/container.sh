set -euo pipefail

image=$1
docker run --rm --network none --entrypoint python "$image" -c '
import os
from pathlib import Path
assert os.getuid() != 0
assert Path.cwd() == Path("/workspace")
path = Path("smoke.txt")
path.write_text("ok")
assert path.read_text() == "ok"
'

container=$(docker run --detach --network none --read-only \
    --tmpfs /tmp:size=512m,mode=1777,exec "$image")
cleanup() {
    status=$?
    if (( status != 0 )); then
        docker logs "$container" || true
    fi
    docker rm --force "$container" || true
    exit "$status"
}
trap cleanup EXIT
docker exec -i "$container" python - <<'PY'
import json
import time
import urllib.request
from urllib.error import URLError

url = "http://127.0.0.1:8000/"
for attempt in range(30):
    try:
        with urllib.request.urlopen(url + "health", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}
        break
    except URLError:
        time.sleep(1)
else:
    raise RuntimeError("Execution API did not start")

def execute(code):
    request = urllib.request.Request(
        url + "execute/stream", data=json.dumps({"code": code}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        events = [json.loads(line) for line in response]
        assert events[-1] == {"type": "execution_finished"}
        assert events[-2]["type"] == "result"
        return events[-2]

assert execute("value = 42")["exception"] is None
assert execute("print(value)")["stdout"] == "42\n"
PY
