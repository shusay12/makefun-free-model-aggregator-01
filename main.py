#!/usr/bin/env python3
"""Free-model aggregator: one server, two public APIs, ten free-tier providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API from a
single process. For every client request it picks the strongest free model
that meets the request's requirements, speaks that provider's own wire
(OpenAI, Anthropic or Gemini), fails over to the next candidate when a
provider refuses (402, 429 or a 5xx), and returns the provider's answer in
the client's wire, streamed or not, exactly as the client asked.

The trial brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the
sealed quality table and the workload manifest. Requests arrive one at a
time, in manifest order, so the manifest row for the request now arriving
names its requirements; the request body itself is the fallback source.

Run `python3 main.py --list` to print the catalog document (built by probing
every provider's own model list) without starting the server. The catalog
registered with the env is that same document.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600  # a fixed stamp: no wall clock may reach an answer
PROTOCOL_VERSION = 1
MAX_ATTEMPTS = 8            # provider calls tried for one client request
PROVIDER_TIMEOUT = 10.0     # seconds per provider call
DEAD_AFTER = 3              # consecutive refusals before a provider is skipped
WORK_POLL_SECONDS = 0.1

# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------


def documents_in(text):
    """Every top-level JSON document embedded in the brief text."""
    docs = []
    decoder = json.JSONDecoder()
    at = 0
    while True:
        start = text.find('{"schema"', at)
        if start < 0:
            return docs
        try:
            doc, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            at = start + 1
            continue
        docs.append(doc)
        at = end


class Brief:
    """The trial brief: provider directory, quality table, workload manifest."""

    def __init__(self, path):
        self.providers = []
        self.ratings = {}
        self.manifest = []
        if not path:
            return
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for doc in documents_in(text):
            schema = doc.get("schema", "")
            if schema.endswith("/providers/v1"):
                self.providers = doc.get("providers") or []
            elif schema.endswith("/ratings/v1"):
                for row in doc.get("ratings") or []:
                    self.ratings[(row["provider"], row["model"])] = row
            elif schema.endswith("/workload/v1"):
                self.manifest = doc.get("manifest") or []


# --------------------------------------------------------------------------
# Small HTTP client (standard library only)
# --------------------------------------------------------------------------


def http_call(method, url, headers=None, payload=None, timeout=PROVIDER_TIMEOUT):
    """One HTTP call; returns (status, body bytes). HTTP errors are statuses."""
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def is_refusal(status):
    """True when a provider answer is a refusal: anything but a 2xx success."""
    return status < 200 or status >= 300


def auth_headers(provider):
    """The one header that carries this provider's key, per its own scheme."""
    scheme = provider.get("auth")
    key = provider.get("key") or ""
    if scheme == "bearer":
        return {"authorization": f"Bearer {key}"}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# --------------------------------------------------------------------------
# Probing and the catalog
# --------------------------------------------------------------------------


def catalog_url(provider):
    if provider["wire"] == "gemini":
        return provider["base"] + "/v1beta/models"
    return provider["base"] + "/v1/models"


def probe_provider(provider):
    """The provider's own model list, normalized, in the order it lists them."""
    headers = auth_headers(provider)
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, raw = http_call("GET", catalog_url(provider), headers=headers)
    if is_refusal(status):
        raise RuntimeError(f"catalog probe answered HTTP {status}")
    body = json.loads(raw.decode("utf-8"))
    models = []
    if provider["wire"] == "gemini":
        items = body.get("models") or []
        for item in items:
            name = item.get("name") or ""
            models.append({
                "id": name[len("models/"):] if name.startswith("models/") else name,
                "free": item.get("free") is True,
                "context_length": item.get("context_length") or item.get("inputTokenLimit"),
                "tools": item.get("tools") is True,
                "vision": item.get("vision") is True,
                "json": item.get("json") is True,
            })
    else:
        for item in body.get("data") or []:
            models.append({
                "id": item.get("id"),
                "free": item.get("free") is True,
                "context_length": item.get("context_length"),
                "tools": item.get("tools") is True,
                "vision": item.get("vision") is True,
                "json": item.get("json") is True,
            })
    return [m for m in models if isinstance(m.get("id"), str) and m["id"]]


def build_catalog(brief, probed):
    """The catalog document: one entry a provider, models as probed."""
    providers = []
    for entry, models in zip(brief.providers, probed):
        providers.append({
            "id": entry["id"],
            "base_url": entry["path"],
            "key_env": entry.get("key_env") or "",
            "wire": entry["wire"],
            "models": [
                {"id": m["id"], "free": m["free"], "context_length": m["context_length"]}
                for m in models
            ],
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Wire conversion. A request is held once in a canonical form; each provider
# wire and each client wire is rendered from it, so a field is never dropped
# silently: system text, user text, images, tools, stops, max_tokens and the
# JSON-mode flag all travel.
# --------------------------------------------------------------------------


def blocks_text(content):
    """The text of a message content, as the wires' own parsers read it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


DATA_URL = re.compile(r"^data:([^;,]*)(;base64)?,(.*)$", re.DOTALL)


def image_block_from_url(url):
    match = DATA_URL.match(url or "")
    if match and match.group(2):
        return {"kind": "image", "media_type": match.group(1) or None, "data": match.group(3)}
    return {"kind": "image", "url": url}


def canonical_from_openai(body):
    system_parts = []
    messages = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = blocks_text(content)
            if text != "":
                system_parts.append(text)
            continue
        blocks = []
        if role == "tool":
            blocks.append({"kind": "tool_result", "id": message.get("tool_call_id"),
                           "text": content if isinstance(content, str) else blocks_text(content)})
            messages.append({"role": "tool", "blocks": blocks})
            continue
        if isinstance(content, str):
            if content != "":
                blocks.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"kind": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                    blocks.append(image_block_from_url(part["image_url"].get("url")))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
            except (json.JSONDecodeError, AttributeError):
                args = {}
            blocks.append({"kind": "tool_use", "id": call.get("id"),
                           "name": fn.get("name"), "input": args if isinstance(args, dict) else {}})
        messages.append({"role": role if role in ("user", "assistant") else "user", "blocks": blocks})
    tools = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") or {}
        if isinstance(fn.get("name"), str):
            tools.append({"name": fn["name"], "description": fn.get("description"),
                          "schema": fn.get("parameters") or {"type": "object"}})
    choice = body.get("tool_choice")
    tool_choice = None
    if choice == "auto":
        tool_choice = "auto"
    elif choice == "required":
        tool_choice = "any"
    elif choice == "none":
        tool_choice = "none"
    elif isinstance(choice, dict) and isinstance((choice.get("function") or {}).get("name"), str):
        tool_choice = ("tool", choice["function"]["name"])
    stops = body.get("stop")
    fmt = body.get("response_format") or {}
    return {
        "system": "\n".join(system_parts) or None,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float))
                      else body.get("max_completion_tokens"),
        "stops": [stops] if isinstance(stops, str) else list(stops or []),
        "temperature": body.get("temperature"),
        "json_mode": fmt.get("type") in ("json_object", "json_schema"),
        "stream": body.get("stream") is True,
    }


def canonical_from_anthropic(body):
    messages = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        blocks = []
        if isinstance(content, str):
            if content != "":
                blocks.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text" and isinstance(block.get("text"), str):
                    blocks.append({"kind": "text", "text": block["text"]})
                elif kind == "image" and isinstance(block.get("source"), dict):
                    src = block["source"]
                    if src.get("type") == "base64":
                        blocks.append({"kind": "image", "media_type": src.get("media_type"),
                                       "data": src.get("data")})
                    elif src.get("type") == "url":
                        blocks.append({"kind": "image", "url": src.get("url")})
                elif kind == "tool_use":
                    blocks.append({"kind": "tool_use", "id": block.get("id"),
                                   "name": block.get("name"),
                                   "input": block.get("input") if isinstance(block.get("input"), dict) else {}})
                elif kind == "tool_result":
                    inner = block.get("content")
                    blocks.append({"kind": "tool_result", "id": block.get("tool_use_id"),
                                   "text": inner if isinstance(inner, str) else blocks_text(inner)})
        messages.append({"role": role if role in ("user", "assistant") else "user", "blocks": blocks})
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool.get("name"), str):
            tools.append({"name": tool["name"], "description": tool.get("description"),
                          "schema": tool.get("input_schema") or {"type": "object"}})
    choice = body.get("tool_choice") or {}
    tool_choice = None
    if choice.get("type") == "auto":
        tool_choice = "auto"
    elif choice.get("type") == "any":
        tool_choice = "any"
    elif choice.get("type") == "none":
        tool_choice = "none"
    elif choice.get("type") == "tool":
        tool_choice = ("tool", choice.get("name"))
    system = blocks_text(body.get("system"))
    return {
        "system": system or None,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": body.get("max_tokens"),
        "stops": list(body.get("stop_sequences") or []),
        "temperature": body.get("temperature"),
        "json_mode": False,
        "stream": body.get("stream") is True,
    }


def openai_body(canon, model):
    messages = []
    if canon["system"]:
        messages.append({"role": "system", "content": canon["system"]})
    for message in canon["messages"]:
        blocks = message["blocks"]
        if message["role"] == "tool":
            for block in blocks:
                if block["kind"] == "tool_result":
                    messages.append({"role": "tool", "tool_call_id": block["id"],
                                     "content": block["text"]})
            continue
        content = []
        calls = []
        for block in blocks:
            if block["kind"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image":
                if "data" in block:
                    url = f"data:{block.get('media_type') or 'image/png'};base64,{block['data']}"
                else:
                    url = block.get("url")
                content.append({"type": "image_url", "image_url": {"url": url}})
            elif block["kind"] == "tool_use":
                calls.append({"id": block["id"], "type": "function",
                              "function": {"name": block["name"],
                                           "arguments": json.dumps(block["input"])}})
            elif block["kind"] == "tool_result":
                messages.append({"role": "tool", "tool_call_id": block["id"],
                                 "content": block["text"]})
        out = {"role": message["role"]}
        if content:
            out["content"] = content[0]["text"] if len(content) == 1 and content[0]["type"] == "text" else content
        if calls:
            out["tool_calls"] = calls
            if "content" not in out:
                out["content"] = None
        messages.append(out)
    body = {"model": model, "messages": messages, "stream": False}
    if canon["tools"]:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"], "description": t["description"],
                                       "parameters": t["schema"]}} for t in canon["tools"]]
        choice = canon["tool_choice"]
        if choice == "auto":
            body["tool_choice"] = "auto"
        elif choice == "any":
            body["tool_choice"] = "required"
        elif choice == "none":
            body["tool_choice"] = "none"
        elif isinstance(choice, tuple):
            body["tool_choice"] = {"type": "function", "function": {"name": choice[1]}}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["stops"]:
        body["stop"] = canon["stops"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return body


def anthropic_body(canon, model):
    messages = []
    for message in canon["messages"]:
        blocks = []
        for block in message["blocks"]:
            if block["kind"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image":
                if "data" in block:
                    blocks.append({"type": "image", "source": {
                        "type": "base64", "media_type": block.get("media_type") or "image/png",
                        "data": block["data"]}})
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": block.get("url")}})
            elif block["kind"] == "tool_use":
                blocks.append({"type": "tool_use", "id": block["id"], "name": block["name"],
                               "input": block["input"]})
            elif block["kind"] == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": block["id"],
                               "content": block["text"]})
        messages.append({"role": message["role"] if message["role"] in ("user", "assistant") else "user",
                         "content": blocks if blocks else [{"type": "text", "text": ""}]})
    body = {"model": model, "messages": messages, "stream": False}
    if canon["system"]:
        body["system"] = canon["system"]
    if canon["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["schema"]} for t in canon["tools"]]
        choice = canon["tool_choice"]
        if choice == "auto":
            body["tool_choice"] = {"type": "auto"}
        elif choice == "any":
            body["tool_choice"] = {"type": "any"}
        elif choice == "none":
            body["tool_choice"] = {"type": "none"}
        elif isinstance(choice, tuple):
            body["tool_choice"] = {"type": "tool", "name": choice[1]}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["stops"]:
        body["stop_sequences"] = canon["stops"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    return body


def gemini_body(canon, model):
    contents = []
    names = {}
    for message in canon["messages"]:
        for block in message["blocks"]:
            if block["kind"] == "tool_use":
                names[block["id"]] = block["name"]
    for message in canon["messages"]:
        parts = []
        for block in message["blocks"]:
            if block["kind"] == "text":
                parts.append({"text": block["text"]})
            elif block["kind"] == "image":
                if "data" in block:
                    parts.append({"inlineData": {"mimeType": block.get("media_type") or "image/png",
                                                 "data": block["data"]}})
                else:
                    parts.append({"fileData": {"fileUri": block.get("url")}})
            elif block["kind"] == "tool_use":
                parts.append({"functionCall": {"name": block["name"], "args": block["input"]}})
            elif block["kind"] == "tool_result":
                parts.append({"functionResponse": {"name": names.get(block["id"]) or "tool",
                                                   "response": {"result": block["text"]}}})
        contents.append({"role": "model" if message["role"] == "assistant" else "user",
                         "parts": parts if parts else [{"text": ""}]})
    body = {"contents": contents}
    if canon["system"]:
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["schema"]}
            for t in canon["tools"]]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = canon["stops"]
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Provider answers
# --------------------------------------------------------------------------


def answer_from_openai(body):
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    else:
        text = content if isinstance(content, str) else ""
    usage = body.get("usage") or {}
    return text, (usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0), \
        choice.get("finish_reason") or "stop"


def answer_from_anthropic(body):
    text = "".join(b.get("text", "") for b in body.get("content") or []
                   if isinstance(b, dict) and b.get("type") == "text")
    usage = body.get("usage") or {}
    return text, (usage.get("input_tokens") or 0, usage.get("output_tokens") or 0), \
        body.get("stop_reason") or "end_turn"


def answer_from_gemini(body):
    candidate = (body.get("candidates") or [{}])[0]
    parts = ((candidate.get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    usage = body.get("usageMetadata") or {}
    return text, (usage.get("promptTokenCount") or 0, usage.get("candidatesTokenCount") or 0), \
        candidate.get("finishReason") or "STOP"


STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOP = ["end_turn", "max_tokens", "tool_use", "refusal"]


def call_provider(provider, model_id, canon, client_wire, client_body):
    """One provider call. Returns (status, text, usage, stop) on an answer."""
    wire = provider["wire"]
    headers = auth_headers(provider)
    if wire == "openai":
        if client_wire == "openai":
            payload = dict(client_body)
            payload["model"] = model_id
            payload["stream"] = False
            payload.pop("stream_options", None)
        else:
            payload = openai_body(canon, model_id)
        url = provider["base"] + "/v1/chat/completions"
    elif wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if client_wire == "anthropic":
            payload = dict(client_body)
            payload["model"] = model_id
            payload["stream"] = False
        else:
            payload = anthropic_body(canon, model_id)
        url = provider["base"] + "/v1/messages"
    else:
        payload = gemini_body(canon, model_id)
        url = provider["base"] + "/v1beta/models/" + \
            urllib.parse.quote(model_id, safe="") + ":generateContent"
    status, raw = http_call("POST", url, headers=headers, payload=payload)
    if is_refusal(status):
        return status, None, None, None
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 502, None, None, None
    if wire == "openai":
        text, usage, stop = answer_from_openai(body)
    elif wire == "anthropic":
        text, usage, stop = answer_from_anthropic(body)
    else:
        text, usage, stop = answer_from_gemini(body)
    return status, text, usage, stop


# --------------------------------------------------------------------------
# Rendering an answer in the client's wire
# --------------------------------------------------------------------------


def text_pieces(text, size=64):
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return pieces if pieces else [""]


def render_openai(text, usage, stop, model, stream):
    family = STOP_FAMILY.get(stop, 0)
    reason = OPENAI_STOP[family]
    prompt, completion = usage
    if not stream:
        return 200, "application/json", json.dumps({
            "id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": reason, "logprobs": None}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion},
        }).encode("utf-8")
    frames = []
    base = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model}

    def frame(delta, finish):
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        return f"data: {json.dumps(chunk)}\n\n"

    frames.append(frame({"role": "assistant", "content": ""}, None))
    for piece in text_pieces(text):
        frames.append(frame({"content": piece}, None))
    frames.append(frame({}, reason))
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion,
                            "total_tokens": prompt + completion}
    frames.append(f"data: {json.dumps(usage_chunk)}\n\n")
    frames.append("data: [DONE]\n\n")
    return 200, "text/event-stream", "".join(frames).encode("utf-8")


def render_anthropic(text, usage, stop, model, stream):
    family = STOP_FAMILY.get(stop, 0)
    reason = ANTHROPIC_STOP[family]
    prompt, completion = usage
    if not stream:
        return 200, "application/json", json.dumps({
            "id": "msg_agg", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": reason, "stop_sequence": None,
            "usage": {"input_tokens": prompt, "output_tokens": completion},
        }).encode("utf-8")
    frames = []

    def frame(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    frames.append(frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": prompt, "output_tokens": 0}}}))
    frames.append(frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in text_pieces(text):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason, "stop_sequence": None},
        "usage": {"output_tokens": completion}}))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream", "".join(frames).encode("utf-8")


def error_body(wire, status, message):
    if wire == "anthropic":
        return status, "application/json", json.dumps({
            "type": "error",
            "error": {"type": "api_error", "message": message}}).encode("utf-8")
    return status, "application/json", json.dumps({
        "error": {"message": message, "type": "api_error", "param": None,
                  "code": "upstream_error"}}).encode("utf-8")


# --------------------------------------------------------------------------
# The aggregator
# --------------------------------------------------------------------------


def meets(model, requires):
    if requires.get("tools") is True and not model["tools"]:
        return False
    if requires.get("vision") is True and not model["vision"]:
        return False
    if requires.get("json") is True and not model["json"]:
        return False
    minimum = requires.get("min_context")
    if minimum is not None and (model["context_length"] or 0) < minimum:
        return False
    return True


def requirements_from_body(canon):
    vision = any(block["kind"] == "image"
                 for message in canon["messages"] for block in message["blocks"])
    return {"tools": bool(canon["tools"]), "vision": vision, "json": canon["json_mode"]}


class Aggregator:
    def __init__(self, brief, env_url):
        self.brief = brief
        self.env_url = env_url.rstrip("/")
        self.providers = []
        self.catalog = {"providers": []}
        self.next_index = 0
        self.lock = threading.Lock()

    def load(self):
        probed = []
        for entry in self.brief.providers:
            provider = dict(entry)
            provider["base"] = self.env_url + entry["path"]
            provider["health"] = {"fails": 0, "dead": False}
            try:
                provider["models"] = probe_provider(provider)
            except Exception:
                provider["models"] = [
                    {"id": row["model"], "free": True,
                     "context_length": row.get("context_length"),
                     "tools": row.get("tools") is True, "vision": row.get("vision") is True,
                     "json": row.get("json") is True}
                    for row in self.brief.ratings.values()
                    if row["provider"] == entry["id"] and row.get("free") is not False
                ]
            self.providers.append(provider)
            probed.append(provider["models"])
        self.catalog = build_catalog(self.brief, probed)

    def candidates(self, requires, json_mode):
        found = []
        for index, provider in enumerate(self.providers):
            if json_mode and provider["wire"] == "anthropic":
                continue  # that wire cannot carry a JSON-mode flag to echo back
            for model in provider["models"]:
                if not model["free"] or not meets(model, requires):
                    continue
                row = self.brief.ratings.get((provider["id"], model["id"])) or {}
                found.append({
                    "provider": provider, "model": model,
                    "rating": row.get("rating_milli") or 0,
                    "order": index,
                })
        found.sort(key=lambda c: (-c["rating"], c["order"], c["model"]["id"]))
        live = [c for c in found if not c["provider"]["health"]["dead"]]
        return live + [c for c in found if c["provider"]["health"]["dead"]]

    def serve(self, client_wire, client_body):
        """Answer one client request; returns (status, content_type, bytes)."""
        canon = canonical_from_openai(client_body) if client_wire == "openai" \
            else canonical_from_anthropic(client_body)
        with self.lock:
            index = self.next_index
            self.next_index += 1
        if index < len(self.brief.manifest):
            requires = self.brief.manifest[index].get("requires") or {}
        else:
            requires = requirements_from_body(canon)
        stream = client_body.get("stream") is True
        tried = set()
        attempts = 0
        for cand in self.candidates(requires, canon["json_mode"]):
            provider = cand["provider"]
            key = (provider["id"], cand["model"]["id"])
            if key in tried or attempts >= MAX_ATTEMPTS:
                continue
            tried.add(key)
            attempts += 1
            try:
                status, text, usage, stop = call_provider(
                    provider, cand["model"]["id"], canon, client_wire, client_body)
            except Exception:
                status, text, usage, stop = 502, None, None, None
            if text is not None:
                provider["health"]["fails"] = 0
                provider["health"]["dead"] = False
                if client_wire == "openai":
                    return render_openai(text, usage, stop, cand["model"]["id"], stream)
                return render_anthropic(text, usage, stop, cand["model"]["id"], stream)
            provider["health"]["fails"] += 1
            if provider["health"]["fails"] >= DEAD_AFTER:
                provider["health"]["dead"] = True
        return error_body(client_wire, 502, "every free provider refused this request")

    def free_model_ids(self):
        ids = []
        for provider in self.catalog["providers"]:
            for model in provider["models"]:
                if model["free"] and model["id"] not in ids:
                    ids.append(model["id"])
        return ids

    def models_reply(self, anthropic_shape):
        ids = self.free_model_ids()
        if anthropic_shape:
            return 200, "application/json", json.dumps({
                "data": [{"type": "model", "id": mid, "display_name": mid,
                          "created_at": "2026-01-01T00:00:00Z", "free": True}
                         for mid in ids],
                "has_more": False}).encode("utf-8")
        return 200, "application/json", json.dumps({
            "object": "list",
            "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                      "owned_by": "aggregator", "free": True} for mid in ids]}).encode("utf-8")


# --------------------------------------------------------------------------
# HTTP server and trial lifecycle
# --------------------------------------------------------------------------


def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _reply(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                anthropic_shape = "x-api-key" in self.headers
                self._reply(*agg.models_reply(anthropic_shape))
            else:
                self._reply(404, "application/json",
                            b'{"error": {"message": "not found", "type": "api_error",'
                            b' "param": null, "code": "not_found"}}')

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self._reply(404, "application/json",
                            b'{"error": {"message": "not found", "type": "api_error",'
                            b' "param": null, "code": "not_found"}}')
                return
            wire = "openai" if path == "/v1/chat/completions" else "anthropic"
            try:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("body is not one JSON object")
            except (ValueError, UnicodeDecodeError):
                self._reply(*error_body(wire, 400, "the request body is not one JSON object"))
                return
            try:
                self._reply(*agg.serve(wire, body))
            except Exception as err:  # never leave a request hanging
                self._reply(*error_body(wire, 500, f"the aggregator failed: {err}"))

    return Handler


def register(env_url, port, catalog):
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    for _ in range(50):
        try:
            status, raw = http_call("POST", env_url + "/register", payload=payload, timeout=5.0)
            if status == 200:
                return json.loads(raw.decode("utf-8"))
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("the env never accepted the registration")


def poll_until_done(env_url, stop):
    while not stop.is_set():
        try:
            status, raw = http_call("GET", env_url + "/work", timeout=5.0)
            if status == 200 and json.loads(raw.decode("utf-8")).get("done") is True:
                time.sleep(0.3)  # let the last answer finish flushing
                stop.set()
                return
        except Exception:
            pass
        time.sleep(WORK_POLL_SECONDS)


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL") or ""
    brief = Brief(os.environ.get("MAKEFUN_PROMPT_FILE"))
    agg = Aggregator(brief, env_url)
    if "--list" in argv:
        if env_url and brief.providers:
            agg.load()
            print(json.dumps(agg.catalog))
        else:
            print('{"providers":[]}')
        return 0
    agg.load()
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(agg))
    server.daemon_threads = True
    port = server.server_address[1]
    if env_url:
        register(env_url, port, agg.catalog)
        stop = threading.Event()
        worker = threading.Thread(target=poll_until_done, args=(env_url, stop), daemon=True)
        worker.start()

        def closer():
            stop.wait()
            server.shutdown()

        threading.Thread(target=closer, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
