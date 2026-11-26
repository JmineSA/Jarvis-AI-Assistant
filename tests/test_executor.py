import time

import pytest

from jarvis.tools.executor import ToolExecutor
from jarvis.tools.registry import Tool, ToolArgs, ToolRegistry, ToolResult, ToolStatus


class EchoArgs(ToolArgs):
    text: str


class Recorder:
    def __init__(self):
        self.runs = []

    def make(self, name="echo", risk="safe", sleep=0.0, crash=False):
        def run(args):
            self.runs.append(args)
            if crash:
                raise RuntimeError("secret internal detail")
            time.sleep(sleep)
            return ToolResult.done(f"echo {args.text}")

        return Tool(
            name=name,
            description="d",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            args_model=EchoArgs,
            run=run,
            risk=risk,
        )


def make_executor(*tools, **kwargs):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return ToolExecutor(registry, **kwargs)


async def approve(prompt):
    approve.prompts.append(prompt)
    return True


approve.prompts = []


async def refuse(prompt):
    return False


async def test_runs_a_valid_tool():
    rec = Recorder()
    result = await make_executor(rec.make()).run("echo", {"text": "hi"})
    assert result.status is ToolStatus.DONE and result.message == "echo hi"


async def test_unknown_tool_is_refused():
    result = await make_executor(Recorder().make()).run("format_disk", {})
    assert result.status is ToolStatus.FAILED
    assert "don't have a tool" in result.message


async def test_invalid_arguments_never_reach_the_tool():
    rec = Recorder()
    executor = make_executor(rec.make())
    assert (await executor.run("echo", {})).status is ToolStatus.FAILED
    assert (await executor.run("echo", {"text": ["not", "a", "string"]})).status is ToolStatus.FAILED
    assert rec.runs == []


async def test_arguments_given_as_json_text_are_accepted():
    result = await make_executor(Recorder().make()).run("echo", '{"text": "hi"}')
    assert result.message == "echo hi"


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", 42])
async def test_unusable_argument_shapes_fail_cleanly(bad):
    rec = Recorder()
    result = await make_executor(rec.make()).run("echo", bad)
    assert result.status is ToolStatus.FAILED and rec.runs == []


async def test_extra_arguments_from_the_model_are_ignored():
    result = await make_executor(Recorder().make()).run("echo", {"text": "hi", "sudo": True})
    assert result.status is ToolStatus.DONE


async def test_missing_arguments_for_a_tool_without_parameters():
    rec = Recorder()
    result = await make_executor(rec.make()).run("echo", None)
    assert result.status is ToolStatus.FAILED  # text is required


async def test_risky_tool_without_a_way_to_confirm_is_denied():
    rec = Recorder()
    result = await make_executor(rec.make(risk="confirm")).run("echo", {"text": "x"})
    assert result.status is ToolStatus.DENIED and rec.runs == []


async def test_risky_tool_refused_by_user_does_not_run():
    rec = Recorder()
    result = await make_executor(rec.make(risk="confirm")).run("echo", {"text": "x"}, refuse)
    assert result.status is ToolStatus.DENIED and rec.runs == []


async def test_risky_tool_approved_by_user_runs_and_prompt_names_the_action():
    rec = Recorder()
    approve.prompts.clear()
    result = await make_executor(rec.make(risk="confirm")).run("echo", {"text": "x"}, approve)
    assert result.status is ToolStatus.DONE and len(rec.runs) == 1
    assert "echo" in approve.prompts[0] and "text=x" in approve.prompts[0]


async def test_always_confirm_applies_to_safe_tools_too():
    rec = Recorder()
    executor = make_executor(rec.make(), always_confirm=["echo"])
    assert (await executor.run("echo", {"text": "x"})).status is ToolStatus.DENIED
    assert (await executor.run("echo", {"text": "x"}, approve)).status is ToolStatus.DONE


async def test_a_crashing_confirmation_prompt_counts_as_no():
    async def broken(prompt):
        raise RuntimeError("no keyboard")

    rec = Recorder()
    result = await make_executor(rec.make(risk="confirm")).run("echo", {"text": "x"}, broken)
    assert result.status is ToolStatus.DENIED and rec.runs == []


async def test_crashing_tool_returns_a_plain_failure_without_leaking_details():
    result = await make_executor(Recorder().make(crash=True)).run("echo", {"text": "x"})
    assert result.status is ToolStatus.FAILED
    assert "secret internal detail" not in result.message


async def test_slow_tool_times_out():
    executor = make_executor(Recorder().make(sleep=0.5), timeout=0.05)
    result = await executor.run("echo", {"text": "x"})
    assert result.status is ToolStatus.FAILED and "too long" in result.message


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(Recorder().make())
    with pytest.raises(ValueError):
        registry.register(Recorder().make())


def test_registry_schema_format_matches_what_ollama_expects():
    registry = ToolRegistry()
    registry.register(Recorder().make())
    [schema] = registry.schemas()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["text"]
    assert "echo" in registry and len(registry) == 1
