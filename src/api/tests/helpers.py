import json


def execute(client, code):
    response = client.post("/execute/stream", json={"code": code})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    result, finished = events
    assert finished == {"type": "execution_finished"}
    assert result["type"] == "result"
    return {key: value for key, value in result.items() if key != "type"}
