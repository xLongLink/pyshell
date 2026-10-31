import json


def execute(client, code):
    response = client.post("/execute/stream", json={"code": code})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {"type": "execution_finished"}
    result = events[-2]
    assert result["type"] == "result"
    ui = []
    for event in events[:-2]:
        assert event["type"] in {"a2ui", "widget"}
        ui.extend(event["messages"] if event["type"] == "a2ui" else [event])
    return {**{key: value for key, value in result.items() if key != "type"}, "ui": ui}
