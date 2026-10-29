from concurrent.futures import ThreadPoolExecutor


def test_health_without_model_credentials(api_client):
    assert api_client.get("/").json() == {"status": "ok"}


def test_execution_preserves_namespace_and_captures_output(api_client):
    assert api_client.post("/", json={"code": "import math\nvalue = math.sqrt(16)"}).status_code == 200
    response = api_client.post("/", json={"code": "print(value)\nimport sys\nprint('warning', file=sys.stderr)\ndone()"})
    assert response.json() == {"stdout": "4.0\n", "stderr": "warning\n", "exception": None, "done": True}


def test_reset_clears_namespace(api_client):
    api_client.post("/", json={"code": "value = 42"})
    assert api_client.post("/reset").status_code == 200
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]


def test_errors_preserve_prior_updates_and_do_not_kill_server(api_client):
    result = api_client.post("/", json={"code": "value = 42\nprint('before')\n1 / 0\nvalue = 99"}).json()
    assert result["stdout"] == "before\n"
    assert "ZeroDivisionError" in result["exception"]
    assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "42\n"
    for source in ("if :", "raise SystemExit(1)", "raise KeyboardInterrupt"):
        assert api_client.post("/", json={"code": source}).json()["exception"]
    assert api_client.get("/").status_code == 200


def test_output_and_done_are_per_request(api_client):
    api_client.post("/", json={"code": "print('old')\ndone()"})
    assert api_client.post("/", json={"code": "pass"}).json() == {
        "stdout": "", "stderr": "", "exception": None, "done": False,
    }


def test_invalid_request_is_rejected(api_client):
    assert api_client.post("/", json={"command": "print(42)"}).status_code == 422


def test_requests_are_serialized(api_client):
    api_client.post("/", json={"code": "value = 0\nimport time"})

    def increment(_):
        return api_client.post("/", json={"code": "previous = value\ntime.sleep(0.01)\nvalue = previous + 1"})

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(increment, range(8)))
    assert all(response.status_code == 200 for response in responses)
    assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "8\n"
