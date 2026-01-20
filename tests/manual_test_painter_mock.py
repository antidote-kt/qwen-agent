import base64
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import time

import requests

# Ensure repository root on path to import examples.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXAMPLES = os.path.join(ROOT, "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)

try:
    # Avoid console cp936 encoding errors on Windows when printing emojis
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from examples.t2i import BatchStoryboardPainter


class DummyResp:
    def __init__(self, data: dict, status_code: int = 200, text: str = ""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._data


def run_case(name: str, shots, ref_image_dir=None):
    payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        payloads.append(json)
        return DummyResp({"code": 200, "data": [{"task_id": "task_" + name}]})

    def fake_get(url, headers=None, timeout=None):
        return DummyResp(
            {
                "data": {
                    "status": "completed",
                    "progress": 100,
                    "result": {"images": [{"url": ["https://fake.com/img.png"]}]},
                }
            }
        )

    # Monkeypatch requests
    orig_post, orig_get = requests.post, requests.get
    orig_sleep = time.sleep
    requests.post, requests.get = fake_post, fake_get
    time.sleep = lambda *_: None  # speed up polling
    try:
        painter = BatchStoryboardPainter()
        params = {
            "json_content": shots,
            "resolution": "1280*720",
            "style_modifier": "",
            "save_images": False,
        }
        if ref_image_dir:
            params["ref_image_dir"] = ref_image_dir
        result = painter.call(json.dumps(params))
        try:
            parsed = json.loads(result)
        except Exception:
            parsed = {"_raw": result}
        return SimpleNamespace(result=parsed, payloads=payloads)
    finally:
        requests.post, requests.get = orig_post, orig_get
        time.sleep = orig_sleep


def create_dummy_png(dir_path, name):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, name)
    with open(path, "wb") as f:
        # minimal PNG header + IHDR chunk placeholder
        f.write(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y3lRc8AAAAASUVORK5CYII="
            )
        )
    return path


def main():
    tmpdir = tempfile.mkdtemp(prefix="painter_mock_")

    # Case 1: keyframe_url is file:// local image
    local_img = create_dummy_png(tmpdir, "01.png")
    shots_local = [
        {"shot_id": "01", "t2i_prompt": "a cat", "keyframe_url": f"file://{local_img}"}
    ]
    out1 = run_case("local", shots_local)
    print("Case local result:", out1.result)
    print("Case local payloads:", out1.payloads)

    # Case 2: keyframe_url is http image
    shots_http = [
        {"shot_id": "01", "t2i_prompt": "a dog", "keyframe_url": "https://img.test/dog.jpg"}
    ]
    out2 = run_case("http", shots_http)
    print("Case http result:", out2.result)
    print("Case http payloads:", out2.payloads)

    # Case 3: no keyframe_url, use ref_image_dir lookup
    shots_refdir = [{"shot_id": "01", "t2i_prompt": "a bird"}]
    out3 = run_case("refdir", shots_refdir, ref_image_dir=tmpdir)
    print("Case dir result:", out3.result)
    print("Case dir payloads:", out3.payloads)

    # Simple assertions for manual run
    assert out1.payloads and out1.payloads[0]["image_urls"][0].startswith("data:image"), "local file not converted to data URI"
    assert out2.payloads and out2.payloads[0]["image_urls"][0] == "https://img.test/dog.jpg", "http url not forwarded"
    assert out3.payloads and out3.payloads[0]["image_urls"][0].startswith("data:image"), "ref_image_dir not converted to data URI"
    print("All mock cases passed.")


if __name__ == "__main__":
    main()
