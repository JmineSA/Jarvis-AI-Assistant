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


# ------------------------------------------------------------------ speed features


def _client(handler, **kwargs):
    return OllamaClient("http://ollama.test", "qwen2.5:3b-instruct", transport=httpx.MockTransport(handler), **kwargs)


async def test_requests_keep_the_model_loaded_and_cap_the_reply():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hi"}})

    await _client(handler, keep_alive="30m", num_predict=200).chat([{"role": "user", "content": "x"}])
    assert seen[0]["keep_alive"] == "30m" and seen[0]["options"]["num_predict"] == 200


async def test_timing_is_read_from_the_reply():
    body = {
        "message": {"content": "hi"}, "total_duration": 2_500_000_000, "load_duration": 100_000_000,
        "prompt_eval_count": 1800, "prompt_eval_duration": 1_200_000_000,
        "eval_count": 20, "eval_duration": 900_000_000,
    }
    client = _client(lambda r: httpx.Response(200, json=body))
    await client.chat([{"role": "user", "content": "x"}])
    stats = client.last_stats
    assert stats.total_s == 2.5 and stats.prompt_tokens == 1800 and stats.output_tokens == 20
    assert "1800 tokens" in stats.summary()


async def test_missing_timing_fields_are_fine():
    client = _client(lambda r: httpx.Response(200, json={"message": {"content": "hi"}}))
    await client.chat([{"role": "user", "content": "x"}])
    assert client.last_stats.total_s == 0.0


async def test_warm_up_asks_for_one_token_and_never_raises():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": ""}})

    await _client(handler).warm_up([{"role": "user", "content": "Hello."}], [{"type": "function"}])
    assert seen[0]["options"]["num_predict"] == 1 and seen[0]["tools"]

    def down(request):
        raise httpx.ConnectError("no")

    await _client(down).warm_up([{"role": "user", "content": "Hello."}])  # must not raise


@pytest.mark.parametrize(
    ("size", "vram", "expected"),
    [(2000, 2000, "GPU 100%"), (2000, 1000, "GPU 50% / CPU 50%"), (2000, 0, "CPU")],
)
async def test_processor_says_where_the_model_runs(size, vram, expected):
    body = {"models": [{"name": "qwen2.5:3b-instruct", "size": size, "size_vram": vram}]}
    assert await _client(lambda r: httpx.Response(200, json=body)).processor() == expected


async def test_processor_is_none_when_not_loaded_or_unreachable():
    assert await _client(lambda r: httpx.Response(200, json={"models": []})).processor() is None

    def down(request):
        raise httpx.ConnectError("no")

    assert await _client(down).processor() is None
