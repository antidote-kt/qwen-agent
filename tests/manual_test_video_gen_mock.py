import json
import os
import sys
from http import HTTPStatus
from types import SimpleNamespace

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXAMPLES = os.path.join(ROOT, "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)

# Monkeypatch targets must be imported after sys.path tweaks
import dashscope

import examples.video_generator as video_generator
from examples.assistant_video_gen import MyKeyframePlanner, MyVideoGen


def mock_dashscope_generation_call(model, messages, result_format=None, stream=False):
    # Return a dummy JSON shot list with two shots
    content = json.dumps(
        [
            {"id": 1, "description": "Shot 1: a cat jumps", "duration": 4},
            {"id": 2, "description": "Shot 2: the cat sleeps", "duration": 5},
        ]
    )

    class Choice:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class Output:
        def __init__(self, content):
            self.choices = [Choice(content)]

    return SimpleNamespace(status_code=HTTPStatus.OK, output=Output(content))


def mock_sample_async_call(prompt, input_img_url=None, duration=5):
    # Emulate successful video generation with deterministic URL
    return {"status": "success", "video_url": f"https://fake.com/video_{duration}s.mp4"}


def mock_score_video(video_url):
    # Emulate scoring result
    return {"overall_score": 80}


def run():
    # Patch external dependencies
    dashscope.Generation.call = mock_dashscope_generation_call
    video_generator.sample_async_call = mock_sample_async_call
    import examples.assistant_video_gen as av  # re-import to pick patched sample_async_call

    av.sample_async_call = mock_sample_async_call  # ensure within MyVideoGen scope
    av.score_video = mock_score_video

    planner = MyKeyframePlanner()
    shots_json = planner.call(json.dumps({"prompt": "Test video", "num_shots": 2}))
    print("Planner output:", shots_json)

    gen = MyVideoGen()
    result = gen.call(
        json.dumps(
            {
                "prompt": "Refined prompt",
                "shots": json.loads(shots_json)["shots"],
            }
        )
    )
    print("Video gen output:", result)


if __name__ == "__main__":
    run()
