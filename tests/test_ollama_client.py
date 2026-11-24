import json

import httpx
import pytest

from jarvis.ai.ollama_client import (
    OllamaClient,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


def make_client(handler) -> OllamaClient:
    return OllamaClient(
        host="http://ollama.test",
        model="qwen2.5:3b-instruct",
        transport=httpx.MockTransport(handler),
    )


async def test_chat_returns_reply_and_sends_expected_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hi there. "}})

    client = make_client(handler)
    assert await client.chat(MESSAGES) == "Hi there."
    assert seen["body"]["model"] == "qwen2.5:3b-instruct"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == MESSAGES
    await client.aclose()


async def test_chat_unreachable_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with pytest.raises(OllamaUnavailableError):
        await client.chat(MESSAGES)
    await client.aclose()


async def test_chat_read_timeout_raises_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = make_client(handler)
    with pytest.raises(OllamaTimeoutError):
        await client.chat(MESSAGES)
    await client.aclose()


async def test_chat_missing_model_gives_pull_hint():
    client = make_client(lambda request: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(OllamaResponseError, match="ollama pull"):
        await client.chat(MESSAGES)
    await client.aclose()


async def test_chat_server_error_raises_response_error():
    client = make_client(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(OllamaResponseError, match="500"):
        await client.chat(MESSAGES)
    await client.aclose()


async def test_chat_bad_payload_raises_response_error():
    client = make_client(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(OllamaResponseError):
        await client.chat(MESSAGES)
    await client.aclose()


async def test_chat_empty_reply_raises_response_error():
    client = make_client(lambda request: httpx.Response(200, json={"message": {"content": "  "}}))
    with pytest.raises(OllamaResponseError):
        await client.chat(MESSAGES)
    await client.aclose()


async def test_chat_stream_yields_pieces():
    body = (
        b'{"message":{"content":"Hel"},"done":false}\n'
        b'{"message":{"content":"lo"},"done":false}\n'
        b'{"message":{"content":""},"done":true}\n'
    )
    client = make_client(lambda request: httpx.Response(200, content=body))
    pieces = [piece async for piece in client.chat_stream(MESSAGES)]
    assert pieces == ["Hel", "lo"]
    await client.aclose()


async def test_chat_stream_error_chunk_raises():
    body = b'{"error":"out of memory"}\n'
    client = make_client(lambda request: httpx.Response(200, content=body))
    with pytest.raises(OllamaResponseError, match="out of memory"):
        _ = [piece async for piece in client.chat_stream(MESSAGES)]
    await client.aclose()


async def test_health_ok_when_model_installed():
    payload = {"models": [{"name": "qwen2.5:3b-instruct"}]}
    client = make_client(lambda request: httpx.Response(200, json=payload))
    status = await client.health()
    assert status.reachable and status.model_available
    await client.aclose()


async def test_health_model_missing():
    payload = {"models": [{"name": "llama3.2:3b"}]}
    client = make_client(lambda request: httpx.Response(200, json=payload))
    status = await client.health()
    assert status.reachable and not status.model_available
    assert "ollama pull" in status.detail
    await client.aclose()


async def test_health_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    status = await client.health()
    assert not status.reachable and not status.model_available
    await client.aclose()


async def test_untagged_model_name_matches_latest():
    payload = {"models": [{"name": "phi3:latest"}]}
    client = OllamaClient(
        host="http://ollama.test",
        model="phi3",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    assert (await client.health()).model_available
    await client.aclose()


# ------------------------------------------------------------------ tool calling

TOOLS = [{"type": "function", "function": {"name": "current_time", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


async def test_chat_with_tools_sends_tools_and_parses_requests():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "open_app", "arguments": {"name": "vscode"}}}],
        }
        return httpx.Response(200, json={"message": message})

    client = make_client(handler)
    result = await client.chat_with_tools(MESSAGES, TOOLS)
    assert seen["body"]["tools"] == TOOLS
    assert seen["body"]["stream"] is False
    assert result.content == ""
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("open_app", {"name": "vscode"})]
    await client.aclose()


async def test_chat_with_tools_plain_answer_has_no_calls():
    client = make_client(lambda request: httpx.Response(200, json={"message": {"content": " Hello. "}}))
    result = await client.chat_with_tools(MESSAGES, TOOLS)
    assert result.content == "Hello." and result.tool_calls == ()
    await client.aclose()


async def test_chat_with_tools_http_error_becomes_response_error():
    client = make_client(lambda request: httpx.Response(400, text="does not support tools"))
    with pytest.raises(OllamaResponseError, match="400"):
        await client.chat_with_tools(MESSAGES, TOOLS)
    await client.aclose()


async def test_chat_with_tools_unreachable_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with pytest.raises(OllamaUnavailableError):
        await client.chat_with_tools(MESSAGES, TOOLS)
    await client.aclose()


def test_parse_tool_calls_is_forgiving():
    from jarvis.ai.ollama_client import parse_tool_calls

    raw = [
        {"function": {"name": "a", "arguments": '{"x": 1}'}},   # arguments as JSON text
        {"function": {"name": "b", "arguments": "not json"}},    # unreadable arguments -> empty
        {"function": {"name": "c"}},                              # no arguments
        {"function": {"arguments": {}}},                          # no name: dropped
        "nonsense",                                               # not an object: dropped
    ]
    calls = parse_tool_calls(raw)
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {}), ("c", {})]
    assert parse_tool_calls(None) == () and parse_tool_calls("x") == ()
