import json

import pytest

from jarvis.ai.brain import Brain
from jarvis.ai.ollama_client import ChatResult, ToolCall
from jarvis.ai.router import CommandRouter
from jarvis.profile import Profile
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.registry import Tool, ToolArgs, ToolRegistry, ToolResult
from jarvis.turnlog import TurnLog, build_report, read_turns


def test_turns_and_later_annotations_merge(tmp_path):
    log = TurnLog(tmp_path / "t.jsonl", clock=iter([1.0, 2.0]).__next__)
    log.record("open   VS Code", "fast", ["open_app"], 0.0123)
    log.annotate(via="voice", stt_s=1.234, tts_first_s=0.5)
    log.record("tell me a joke", "model", [], 3.456)
    turns = read_turns(tmp_path / "t.jsonl")
    assert [t["text"] for t in turns] == ["open VS Code", "tell me a joke"]
    assert turns[0]["stt_s"] == 1.23 and turns[0]["via"] == "voice" and turns[1].get("stt_s") is None


def test_the_log_trims_itself_and_never_crashes(tmp_path):
    path = tmp_path / "t.jsonl"
    log = TurnLog(path, max_bytes=2000)
    for i in range(200):
        log.record(f"sentence number {i}", "model", [], 1.0)
    assert path.stat().st_size < 4000 and read_turns(path)[-1]["text"] == "sentence number 199"
    broken = TurnLog(tmp_path / "file.txt" / "x.jsonl")  # parent is not a folder
    (tmp_path / "file.txt").write_text("x")
    broken.record("hi", "model", [], 1.0)  # swallowed, logged


def test_read_ignores_garbage_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('not json\n{"id": 1, "text": "a", "route": "fast", "tools": [], "brain_s": 0.1, "ok": true}\n[1]\n')
    assert len(read_turns(path)) == 1 and read_turns(tmp_path / "missing.jsonl") == []


def test_report_lists_commands_the_model_had_to_work_out():
    turns = [
        {"id": 1, "text": "open the thing", "route": "fast", "tools": ["open_app"], "brain_s": 0.0, "ok": True},
        {"id": 2, "text": "Could you pull up Power BI?", "route": "model", "tools": ["open_app"], "brain_s": 3.0, "ok": True, "stt_s": 1.0, "tts_first_s": 0.4},
        {"id": 3, "text": "could you pull up power bi", "route": "model", "tools": ["open_app"], "brain_s": 5.0, "ok": True},
        {"id": 4, "text": "tell me a joke", "route": "model", "tools": [], "brain_s": 2.0, "ok": False},
    ]
    report = build_report(turns)
    assert "4 turns recorded: 1 handled instantly" in report and "3 needed the model" in report
    assert "2x  could you pull up power bi   (4.0 s; tools: open_app)" in report
    assert "Plain conversation" in report and "tell me a joke" in report
    assert "Speech recognition averages 1.0 s" in report and "1 turns ended in an error" in report
    assert "No turns recorded" in build_report([])


class Empty(ToolArgs):
    pass


class Client:
    def __init__(self, calls=()):
        self.calls = calls

    async def chat(self, messages):
        return "Sure thing."

    async def chat_with_tools(self, messages, tools):
        return ChatResult("", tuple(self.calls))


def tool_brain(tmp_path, client, router=None):
    registry = ToolRegistry()
    registry.register(Tool("open_app", "d", {"type": "object", "properties": {}}, Empty, lambda a: ToolResult.launched("Opening it.")))
    log = TurnLog(tmp_path / "t.jsonl")
    return Brain(client, "sys", 4, executor=ToolExecutor(registry), router=router, turn_log=log), tmp_path / "t.jsonl"


async def test_brain_records_fast_and_model_turns(tmp_path):
    brain, path = tool_brain(tmp_path, Client([ToolCall("open_app", {})]), CommandRouter(Profile(), {"open_app"}))
    await brain.respond("could you maybe open the app please")  # not a fast-path sentence
    turns = read_turns(path)
    assert turns[0]["route"] == "model" and turns[0]["tools"] == ["open_app"] and turns[0]["ok"] is True


async def test_brain_records_errors_too(tmp_path):
    class Boom(Client):
        async def chat_with_tools(self, messages, tools):
            raise RuntimeError("down")

    brain, path = tool_brain(tmp_path, Boom())
    with pytest.raises(RuntimeError):
        await brain.respond("hello")
    assert read_turns(path)[0]["ok"] is False


def test_voice_assistant_adds_stage_timings(tmp_path):
    from tests.test_orchestrator import build  # reuse the fakes

    seen = {}
    log = TurnLog(tmp_path / "t.jsonl")
    log.record("hello jarvis", "model", [], 1.0)
    assistant, *_ = build()
    assistant._annotate = lambda **kw: (seen.update(kw), log.annotate(**kw))
    assistant.run()
    assert seen["via"] == "voice" and {"stt_s", "tts_first_s", "audio_s"} <= set(seen)
    assert read_turns(tmp_path / "t.jsonl")[0]["via"] == "voice"
    json.dumps(seen)


async def test_tool_eval_scores_the_model(monkeypatch, capsys):
    from jarvis import diagnostics
    from jarvis.config import Settings
    from jarvis.evalset import CASES

    class Health:
        reachable, model_available, detail = True, True, ""

    class FakeOllama:
        async def health(self):
            return Health()

        async def chat_with_tools(self, messages, tools):
            text = messages[-1]["content"].lower()
            return ChatResult("", (ToolCall("current_time", {}),) if "what time" in text else ())

        async def chat(self, messages):
            return "ok"

        async def aclose(self):
            pass

    monkeypatch.setattr(diagnostics.OllamaClient, "from_settings", staticmethod(lambda s: FakeOllama()))
    lines = []
    assert await diagnostics._tool_eval(Settings(), "other:model", say=lines.append) == 0
    text = "\n".join(lines)
    assert "Model other:model" in text and "Correct:" in text and "wanted open_app, got just talk" in text
    assert len(CASES) >= 40


def test_ack_tick_plays_while_recognition_runs_and_can_be_off():
    import threading

    from jarvis.voice.factory import make_ack_sound
    from jarvis.voice.wakeword import make_ack
    from tests.test_orchestrator import build

    played = threading.Event()
    assistant, *_ = build()
    assistant._ack = played.set
    assistant.run()
    assert played.wait(2)
    clip = make_ack()
    assert 0.03 < clip.duration < 0.15 and float(abs(clip.samples).max()) <= 0.2
    assert make_ack_sound(object(), False) is None


async def test_cache_test_judges_by_time_not_token_count(capsys):
    from jarvis import diagnostics
    from jarvis.ai.ollama_client import LlmStats

    class Brain:
        def _system_text(self):
            return "sys"

    class Client:
        def __init__(self, seconds):
            self.seconds, self.last_stats = iter(seconds), None

        async def chat_with_tools(self, messages, tools):
            self.last_stats = LlmStats(prompt_tokens=2056, prompt_s=next(self.seconds))  # tokens always look big

    await diagnostics._cache_test(Client([1.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), Brain(), [])
    assert "The cache works for every case" in capsys.readouterr().out
    await diagnostics._cache_test(Client([1.8, 0.1, 0.2, 0.2, 0.2, 1.8, 1.9, 1.8]), Brain(), [])
    assert "begins with the same words" in capsys.readouterr().out
    await diagnostics._cache_test(Client([1.8] * 8), Brain(), [])
    assert "not reusing the stored prompt reliably" in capsys.readouterr().out
