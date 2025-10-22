import json
import inspect
from fastapi.testclient import TestClient
from api.app import app
import api.dev as dev

client = TestClient(app)
openapi = client.get("/openapi.json").json()

def route_spec(path: str):
    spec = openapi.get("paths", {}).get(path)
    if not spec:
        return None
    operation = spec.get("get") or spec.get("post")
    if not operation:
        return None
    params = [
        {
            "name": p.get("name"),
            "in": p.get("in"),
            "required": p.get("required", False),
            "schema": p.get("schema", {}),
        }
        for p in operation.get("parameters", [])
    ]
    return {
        "method": "get" if "get" in spec else "post",
        "params": params,
    }

spec = route_spec("/dev/run")
signature = None
source_excerpt = None
doc = None

if hasattr(dev, "dev_run"):
    signature = str(inspect.signature(dev.dev_run))
    source = inspect.getsource(dev.dev_run)
    source_excerpt = source[:1200]
    doc = inspect.getdoc(dev.dev_run)

print(
    json.dumps(
        {
            "has_dev_run": hasattr(dev, "dev_run"),
            "signature": signature,
            "doc": doc,
            "openapi": spec,
            "source_excerpt": source_excerpt,
        },
        indent=2,
    )
)
