#!/usr/bin/env python3
"""The best free AI model aggregator -- make.fun bounty free-model-aggregator-01.

One server, two client APIs (OpenAI- and Anthropic-compatible), over ten
simulated free-tier model providers.  It reads the trial brief at
$MAKEFUN_PROMPT_FILE (provider directory, sealed quality table, workload
manifest), probes each provider's catalog route in the provider's own wire,
registers the one catalog at $MAKEFUN_ENV_URL/register, serves the client's
twenty requests, polls $MAKEFUN_ENV_URL/work, and exits 0 when the env says
done (a process still running at the 180 s wall scores zero).

Routing: free models only (a paid model is never a legal pick), meeting the
request's manifest requirements (tools / vision / JSON / min_context, looked
up by arrival order -- the client sends the manifest in order, one request
at a time), ranked by the sealed rating_milli desc, provider id asc, model
id asc.  A provider that refuses (402 / 429 / 5xx / timeout) fails over to
the next candidate; each candidate is tried at most once per request.

Wires: the client speaks OpenAI or Anthropic; each provider speaks OpenAI,
Anthropic or Gemini.  Same-wire answers are forwarded byte-for-byte;
cross-wire answers are normalised (text, provider-recorded usage, provider
stop reason) and rebuilt in the client's shape.  Stops, max_tokens,
temperature, tools and JSON mode pass through exactly as the client sent
them, so the provider's echo of the request matches the client's.

Standard library only (Python 3.13 target; also runs on 3.12).  No pip
dependencies, no model weights, no provider API keys committed anywhere:
keys come from the brief's directory at run time.

Determinism (required by the bounty): no wall clock, no unseeded
randomness, no iteration over unordered collections.  Response ids and the
`created` field are fixed or derived from the request content (sha256),
never from the clock.  A Retry-After header is recorded but never slept on,
so a replay leaves the same state export.

Usage:
    python3 main.py                 # trial mode (needs MAKEFUN_PROMPT_FILE
                                    # and MAKEFUN_ENV_URL): probe, register,
                                    # serve, poll /work, exit 0 on done
    python3 main.py --list          # print the catalog document, byte for
                                    # byte the same document the server
                                    # registers at startup, then exit
    python3 main.py --port 9000     # manual mode (no MAKEFUN_ENV_URL):
                                    # serve on the given port, no register

Environment:
    MAKEFUN_PROMPT_FILE  path to the trial brief (three JSON documents)
    MAKEFUN_ENV_URL      base URL of the trial env holding the providers
    PORT / MAKEFUN_PORT  port for manual mode (default 8000)
"""

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL = 0.1          # seconds between GET /work polls
PROBE_TIMEOUT = 5            # seconds per provider catalog probe
PROVIDER_TIMEOUT = 10        # seconds per provider call; a timeout fails over
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 32 * 1024 * 1024

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def canonical_json(obj):
    """Canonical bytes for a document: sorted keys, 2-space indent."""
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def request_id(payload):
    """Deterministic id derived from the request content."""
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return digest[:24]


# ---------------------------------------------------------------------------
# The brief: three JSON documents inside the prompt file, found by `schema`.
# ---------------------------------------------------------------------------
def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _provider_of(v):
    if not isinstance(v, dict):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _str(v.get(key)) != "":
            base = _str(v.get(key))
            break
    return {"id": pid, "name": _str(v.get("name")) or pid, "wire": wire,
            "auth": auth, "base": base, "key_env": _str(v.get("key_env")),
            "key": _str(v.get("key"))}


def _rating_of(v):
    if not isinstance(v, dict):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {"model": model, "provider": provider,
            "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
            "context_length": _num_or_none(v.get("context_length")),
            "tools": v.get("tools") is True, "vision": v.get("vision") is True,
            "json": v.get("json") is True, "free": v.get("free") is True}


def _workload_request_of(v):
    if not isinstance(v, dict):
        return None
    rid = _str(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if isinstance(v.get("requires"), dict) else {}
    out = {"id": rid, "api": "anthropic" if v.get("api") == "anthropic" else "openai",
           "stream": v.get("stream") is True, "weight": 2 if v.get("weight") == 2 else 1,
           "requires": {"tools": requires.get("tools") is True,
                        "vision": requires.get("vision") is True,
                        "json": requires.get("json") is True}}
    if isinstance(requires.get("min_context"), (int, float)):
        out["requires"]["min_context"] = requires["min_context"]
    return out


def read_brief(path):
    """The brief at `path`: directory, ratings table, workload manifest."""
    brief = {"directory": {"schema": PROVIDERS_SCHEMA, "providers": []},
             "ratings": None, "workload": None, "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory_doc = doc
            break
    if directory_doc is None:
        brief["error"] = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    providers = []
    for v in directory_doc["providers"]:
        p = _provider_of(v)
        if p is not None:
            providers.append(p)
    brief["directory"]["providers"] = providers
    ratings_doc = None
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
            break
    if ratings_doc is not None:
        rows = []
        for v in ratings_doc["ratings"]:
            r = _rating_of(v)
            if r is not None:
                rows.append(r)
        brief["ratings"] = {"schema": RATINGS_SCHEMA, "ratings": rows}
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
            break
    if workload_doc is not None:
        manifest = []
        for v in workload_doc["manifest"]:
            w = _workload_request_of(v)
            if w is not None:
                manifest.append(w)
        brief["workload"] = {"schema": WORKLOAD_SCHEMA, "manifest": manifest}
    return brief


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def join_base(env_url, base):
    """A base path joined to the env's URL; an absolute base is used as-is."""
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if b == "":
        return prefix
    return "%s%s%s" % (prefix, "" if b.startswith("/") else "/", b.rstrip("/"))


def auth_headers(auth, key):
    """The headers a provider wants its key in."""
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# Catalog probe: each provider's catalog route, in its own wire.
# ---------------------------------------------------------------------------
def _model_in(wire, v):
    if not isinstance(v, dict):
        return None
    if wire == "gemini":
        mid = _str(v.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    ctx = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        ctx = _num_or_none(v.get(key))
        if ctx is not None:
            break
    display = _str(v.get("display_name")) or _str(v.get("displayName")) or mid
    return {"id": mid, "display_name": display,
            "free": _bool_or_none(v.get("free")),
            "context_length": ctx,
            "tools": _bool_or_none(v.get("tools")),
            "vision": _bool_or_none(v.get("vision")),
            "json": _bool_or_none(v.get("json"))}


def _models_in(wire, body):
    if not isinstance(body, dict):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    out = []
    for v in items:
        m = _model_in(wire, v)
        if m is not None:
            out.append(m)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT):
    """Probe one provider's catalog route, tolerating every failure."""
    base = join_base(env_url, provider["base"])
    url = "%s/v1beta/models" % base if provider["wire"] == "gemini" else "%s/v1/models" % base
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "status": resp.status,
                        "models": [], "error": "GET %s answered HTTP %s" % (url, resp.status)}
            body = json.loads(resp.read().decode("utf-8"))
            return {"provider": provider, "ok": True, "status": 200,
                    "models": _models_in(provider["wire"], body), "error": None}
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": 0, "models": [],
                "error": "GET %s failed: %s" % (url, exc)}


def model_free(model, row):
    """Whether a model is free: the quality table's row where it has one
    (the sealed truth), the listing's own word where it does not, and paid
    when neither says.  A paid model is never routed."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    """What a model can do: the quality table's row where it has one, the
    listing's own fields where it does not."""
    return {"context_length": (row["context_length"] if row is not None and row["context_length"] is not None
                               else model["context_length"]),
            "tools": (row["tools"] if row is not None else model["tools"]),
            "vision": (row["vision"] if row is not None else model["vision"]),
            "json": (row["json"] if row is not None else model["json"])}


def meets_requirements(requires, caps):
    """Whether a model's capabilities meet one request's requirements.
    An unknown capability never meets one."""
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    if requires.get("min_context") is not None and (
            caps["context_length"] is None or caps["context_length"] < requires["min_context"]):
        return False
    return True


def build_catalog(probes, ratings):
    """The catalog the probes and the quality table make: every provider
    that answered, every model it listed.  Its base_url is the directory's
    own path (/p/<id>), never the path joined to the env URL: the env's
    origin differs between a live pass and its replay, and the catalog is
    part of the state export a replay must reproduce byte for byte."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": ctx})
        providers.append({"id": probe["provider"]["id"],
                          "base_url": probe["provider"]["base"],
                          "key_env": probe["provider"]["key_env"],
                          "wire": probe["provider"]["wire"],
                          "models": models})
    return {"providers": providers}


def catalog_text(catalog):
    """The one catalog document, byte for byte, for --list and /register."""
    return (json.dumps(catalog) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Request analysis and routing
# ---------------------------------------------------------------------------
def requirements_from_body(body):
    """Requirements read off a request body, when the manifest has no row
    for it.  A guess only: the manifest is the declared source."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision,
            "json": fmt.get("type") in ("json_object", "json_schema")}


def rank_candidates(probes, ratings, requires):
    """All free models meeting the requirements, best first: sealed
    rating_milli desc, provider id asc, model id asc."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in probe["models"]:
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            if not meets_requirements(requires, capabilities_of(model, row)):
                continue
            rating = row["rating_milli"] if row is not None else 0
            out.append({"provider": probe["provider"], "model": model,
                        "rating": rating})
    out.sort(key=lambda c: (-c["rating"], c["provider"]["id"], c["model"]["id"]))
    return out


# ---------------------------------------------------------------------------
# Wire translation: OpenAI <-> Anthropic <-> Gemini
# ---------------------------------------------------------------------------
def _openai_text_of_content(content):
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "\n".join(parts)


def openai_messages_to_anthropic(messages):
    """OpenAI chat messages -> Anthropic messages + system string."""
    system_parts = []
    out = []
    for msg in messages or []:
        role = msg.get("role")
        if role == "system":
            system_parts.append(_openai_text_of_content(msg.get("content")))
            continue
        content = msg.get("content")
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        header, _, b64 = url.partition(",")
                        media = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
                        blocks.append({"type": "image",
                                       "source": {"type": "base64",
                                                  "media_type": media,
                                                  "data": b64}})
                    elif url:
                        blocks.append({"type": "image",
                                       "source": {"type": "url", "url": url}})
                elif ptype == "input_image":
                    blocks.append({"type": "image",
                                   "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": part.get("image_url", "")}})
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            blocks.append({"type": "tool_use", "id": call.get("id", ""),
                           "name": fn.get("name", ""), "input": args})
        if role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result",
                 "tool_use_id": msg.get("tool_call_id", ""),
                 "content": _openai_text_of_content(content)}]})
        else:
            out.append({"role": role if role in ("user", "assistant") else "user",
                        "content": blocks})
    return "\n".join(system_parts), out


def anthropic_messages_to_openai(messages, system):
    """Anthropic messages + system -> OpenAI chat messages.  Text and
    image parts of one message stay in the one message, in order."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        parts = []
        tool_calls = []
        tool_results = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image":
                    src = part.get("source") or {}
                    if src.get("type") == "base64":
                        parts.append({"type": "image_url",
                                      "image_url": {"url": "data:%s;base64,%s" % (
                                          src.get("media_type", "image/png"),
                                          src.get("data", ""))}})
                    elif src.get("type") == "url":
                        parts.append({"type": "image_url",
                                      "image_url": {"url": src.get("url", "")}})
                elif ptype == "tool_use":
                    tool_calls.append({
                        "id": part.get("id", ""),
                        "type": "function",
                        "function": {"name": part.get("name", ""),
                                     "arguments": json.dumps(
                                         part.get("input") or {},
                                         sort_keys=True)}})
                elif ptype == "tool_result":
                    text = part.get("content")
                    if isinstance(text, list):
                        text = _openai_text_of_content(text)
                    tool_results.append((part.get("tool_use_id", ""), str(text or "")))
        if tool_results:
            for tool_id, text in tool_results:
                out.append({"role": "tool", "tool_call_id": tool_id, "content": text})
            continue
        entry = {"role": role if role in ("user", "assistant", "system") else "user"}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        entry["content"] = parts if parts else ""
        out.append(entry)
    return out


def openai_tools_to_anthropic(tools):
    out = []
    for tool in tools or []:
        fn = tool.get("function") or tool
        out.append({"name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {
                        "type": "object", "properties": {}}})
    return out


def anthropic_tools_to_openai(tools):
    out = []
    for tool in tools or []:
        out.append({"type": "function",
                    "function": {"name": tool.get("name", ""),
                                 "description": tool.get("description", ""),
                                 "parameters": tool.get("input_schema") or {
                                     "type": "object", "properties": {}}}})
    return out


def _gemini_part_from_openai(part):
    """One OpenAI content part -> a Gemini part (text or inline image)."""
    if not isinstance(part, dict):
        return None
    if part.get("type") == "text":
        return {"text": part.get("text", "")}
    if part.get("type") == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            header, _, b64 = url.partition(",")
            media = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
            return {"inlineData": {"mimeType": media, "data": b64}}
        if url:
            return {"fileData": {"fileUri": url}}
    return None


def gemini_contents(messages_openai):
    """OpenAI messages -> (systemInstruction text, Gemini contents).

    System messages become systemInstruction (never a content); images
    become inlineData parts so the provider echoes the image sha; tool
    calls become functionCall parts."""
    system_parts = []
    contents = []
    for msg in messages_openai:
        role = msg.get("role")
        if role == "system":
            text = _openai_text_of_content(msg.get("content"))
            if text:
                system_parts.append(text)
            continue
        parts = []
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                gp = _gemini_part_from_openai(part)
                if gp is not None:
                    parts.append(gp)
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            parts.append({"functionCall": {"name": fn.get("name", ""),
                                           "args": args}})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user",
                             "parts": parts})
    return "\n".join(system_parts), contents


def _gemini_tools_openai(tools):
    return [{"functionDeclarations": [
        {"name": (t.get("function") or {}).get("name", ""),
         "description": (t.get("function") or {}).get("description", ""),
         "parameters": (t.get("function") or {}).get("parameters") or {}}
        for t in tools]}]


def _gemini_tools_anthropic(tools):
    return [{"functionDeclarations": [
        {"name": t.get("name", ""), "description": t.get("description", ""),
         "parameters": t.get("input_schema") or {}}
        for t in tools]}]


def build_provider_request(env_url, provider, model_id, wire_in, payload):
    """Build (url, headers, body_dict) for one provider call.

    Stops, max_tokens, temperature, tools and JSON mode pass through
    exactly as the client sent them -- max_tokens is never defaulted, so
    the provider's echo of the request matches the client's."""
    wire = provider["wire"]
    base = join_base(env_url, provider["base"])
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    stream = bool(payload.get("stream"))

    if wire_in == "openai":
        messages = payload.get("messages") or []
        tools = payload.get("tools")
        max_tokens = payload.get("max_tokens")
        response_format = payload.get("response_format") or {}
        needs_json = response_format.get("type") in ("json_object", "json_schema")
        stop = payload.get("stop")
        if wire == "openai":
            body = {"model": model_id, "messages": messages}
            if isinstance(max_tokens, (int, float)):
                body["max_tokens"] = max_tokens
            if payload.get("temperature") is not None:
                body["temperature"] = payload["temperature"]
            if payload.get("top_p") is not None:
                body["top_p"] = payload["top_p"]
            if tools:
                body["tools"] = tools
            if payload.get("tool_choice") is not None:
                body["tool_choice"] = payload["tool_choice"]
            if needs_json:
                body["response_format"] = response_format
            if stop is not None:
                body["stop"] = stop
            if stream:
                body["stream"] = True
                body["stream_options"] = {"include_usage": True}
            return "%s/v1/chat/completions" % base, headers, body
        if wire == "anthropic":
            system, anth_messages = openai_messages_to_anthropic(messages)
            body = {"model": model_id, "messages": anth_messages}
            if isinstance(max_tokens, (int, float)):
                body["max_tokens"] = max_tokens
            if system:
                body["system"] = system
            if payload.get("temperature") is not None:
                body["temperature"] = payload["temperature"]
            if payload.get("top_p") is not None:
                body["top_p"] = payload["top_p"]
            if tools:
                body["tools"] = openai_tools_to_anthropic(tools)
            if payload.get("tool_choice") is not None:
                body["tool_choice"] = payload["tool_choice"]
            if stop is not None:
                body["stop_sequences"] = stop if isinstance(stop, list) else [stop]
            if stream:
                body["stream"] = True
            return "%s/v1/messages" % base, headers, body
        # gemini
        system_text, contents = gemini_contents(messages)
        body = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        config = {}
        if isinstance(max_tokens, (int, float)):
            config["maxOutputTokens"] = max_tokens
        if payload.get("temperature") is not None:
            config["temperature"] = payload["temperature"]
        if payload.get("top_p") is not None:
            config["topP"] = payload["top_p"]
        if stop is not None:
            config["stopSequences"] = stop if isinstance(stop, list) else [stop]
        if needs_json:
            config["responseMimeType"] = "application/json"
        if tools:
            body["tools"] = _gemini_tools_openai(tools)
        if config:
            body["generationConfig"] = config
        action = "streamGenerateContent" if stream else "generateContent"
        return "%s/v1beta/models/%s:%s" % (base, model_id, action), headers, body

    # wire_in == "anthropic"
    messages = payload.get("messages") or []
    system = payload.get("system") or ""
    tools = payload.get("tools")
    max_tokens = payload.get("max_tokens")
    stop_sequences = payload.get("stop_sequences")
    if wire == "anthropic":
        body = {"model": model_id, "messages": messages}
        if isinstance(max_tokens, (int, float)):
            body["max_tokens"] = max_tokens
        if system:
            body["system"] = system
        if payload.get("temperature") is not None:
            body["temperature"] = payload["temperature"]
        if payload.get("top_p") is not None:
            body["top_p"] = payload["top_p"]
        if tools:
            body["tools"] = tools
        if payload.get("tool_choice") is not None:
            body["tool_choice"] = payload["tool_choice"]
        if stop_sequences is not None:
            body["stop_sequences"] = stop_sequences
        if stream:
            body["stream"] = True
        return "%s/v1/messages" % base, headers, body
    if wire == "openai":
        oai_messages = anthropic_messages_to_openai(messages, system)
        body = {"model": model_id, "messages": oai_messages}
        if isinstance(max_tokens, (int, float)):
            body["max_tokens"] = max_tokens
        if payload.get("temperature") is not None:
            body["temperature"] = payload["temperature"]
        if payload.get("top_p") is not None:
            body["top_p"] = payload["top_p"]
        if tools:
            body["tools"] = anthropic_tools_to_openai(tools)
        if payload.get("tool_choice") is not None:
            body["tool_choice"] = payload["tool_choice"]
        if stop_sequences is not None:
            body["stop"] = stop_sequences
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return "%s/v1/chat/completions" % base, headers, body
    # gemini
    oai_messages = anthropic_messages_to_openai(messages, system)
    system_text, contents = gemini_contents(oai_messages)
    body = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    config = {}
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if payload.get("temperature") is not None:
        config["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        config["topP"] = payload["top_p"]
    if stop_sequences is not None:
        config["stopSequences"] = stop_sequences
    if tools:
        body["tools"] = _gemini_tools_anthropic(tools)
    if config:
        body["generationConfig"] = config
    action = "streamGenerateContent" if stream else "generateContent"
    return "%s/v1beta/models/%s:%s" % (base, model_id, action), headers, body


# ---------------------------------------------------------------------------
# Provider answers -> normalised result {text, tool_calls, usage, finish}
# ---------------------------------------------------------------------------
def _parse_sse_frames(text):
    """An SSE stream -> [(event, data)] in order; [DONE] kept as data."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _answer_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _gemini_finish(finish):
    return {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
            "RECITATION": "content_filter"}.get(finish, "stop")


def normalise_provider_response(wire, raw_text, content_type):
    """Return {text, tool_calls, usage, finish} for any wire, or None when
    the provider's answer has no usable shape.  Usage is the provider's
    recorded usage, never an estimate; finish is the provider's stop
    reason in OpenAI vocabulary."""
    tool_calls = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    is_stream = "text/event-stream" in (content_type or "") or raw_text.lstrip().startswith("data:")
    if is_stream:
        text_parts = []
        finish = "stop"
        for _event, data in _parse_sse_frames(raw_text):
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if not isinstance(chunk, dict):
                continue
            if wire == "openai":
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        text_parts.append(delta["content"])
                    for call in delta.get("tool_calls") or []:
                        tool_calls.append(call)
                    if isinstance(choice.get("finish_reason"), str):
                        finish = choice["finish_reason"]
                if isinstance(chunk.get("usage"), dict):
                    usage["input_tokens"] = chunk["usage"].get("prompt_tokens") or 0
                    usage["output_tokens"] = chunk["usage"].get("completion_tokens") or 0
            elif wire == "anthropic":
                ctype = chunk.get("type")
                if ctype == "content_block_delta":
                    delta = chunk.get("delta") or {}
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        text_parts.append(delta["text"])
                    elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                        text_parts.append(delta["partial_json"])
                elif ctype == "message_delta":
                    delta = chunk.get("delta") or {}
                    if isinstance(delta.get("stop_reason"), str):
                        finish = {"end_turn": "stop", "max_tokens": "length",
                                  "tool_use": "tool_calls", "refusal": "content_filter"}.get(
                                      delta["stop_reason"], "stop")
                    if isinstance(chunk.get("usage"), dict):
                        usage["output_tokens"] = chunk["usage"].get("output_tokens") or 0
                elif ctype == "message_start":
                    msg = chunk.get("message") or {}
                    if isinstance(msg.get("usage"), dict):
                        usage["input_tokens"] = msg["usage"].get("input_tokens") or 0
            else:  # gemini
                for cand in chunk.get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        if isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
                        if isinstance(part.get("functionCall"), dict):
                            fc = part["functionCall"]
                            tool_calls.append({
                                "id": "call_" + hashlib.sha256(
                                    canonical_json(fc)).hexdigest()[:12],
                                "type": "function",
                                "function": {"name": fc.get("name", ""),
                                             "arguments": json.dumps(
                                                 fc.get("args") or {}, sort_keys=True)}})
                    if isinstance(cand.get("finishReason"), str):
                        finish = _gemini_finish(cand["finishReason"])
                if isinstance(chunk.get("usageMetadata"), dict):
                    usage["input_tokens"] = chunk["usageMetadata"].get("promptTokenCount") or 0
                    usage["output_tokens"] = chunk["usageMetadata"].get("candidatesTokenCount") or 0
        return {"text": "".join(text_parts), "tool_calls": tool_calls,
                "usage": usage, "finish": finish}
    try:
        data = json.loads(raw_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if wire == "openai":
        text_parts = []
        finish = "stop"
        for choice in data.get("choices") or []:
            msg = choice.get("message") or {}
            if msg.get("content"):
                text_parts.append(_answer_text(msg["content"]))
            for call in msg.get("tool_calls") or []:
                tool_calls.append(call)
            if isinstance(choice.get("finish_reason"), str):
                finish = choice["finish_reason"]
        if isinstance(data.get("usage"), dict):
            usage["input_tokens"] = data["usage"].get("prompt_tokens") or 0
            usage["output_tokens"] = data["usage"].get("completion_tokens") or 0
        return {"text": "".join(text_parts), "tool_calls": tool_calls,
                "usage": usage, "finish": finish}
    if wire == "anthropic":
        text_parts = []
        finish = "end_turn"
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""), "type": "function",
                    "function": {"name": block.get("name", ""),
                                 "arguments": json.dumps(block.get("input") or {},
                                                         sort_keys=True)}})
        if isinstance(data.get("usage"), dict):
            usage["input_tokens"] = data["usage"].get("input_tokens") or 0
            usage["output_tokens"] = data["usage"].get("output_tokens") or 0
        if isinstance(data.get("stop_reason"), str):
            finish = {"end_turn": "stop", "max_tokens": "length",
                      "tool_use": "tool_calls", "refusal": "content_filter"}.get(
                          data["stop_reason"], "stop")
        return {"text": "".join(text_parts), "tool_calls": tool_calls,
                "usage": usage, "finish": finish}
    # gemini
    text_parts = []
    finish = "stop"
    for cand in data.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            if isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            if isinstance(part.get("functionCall"), dict):
                fc = part["functionCall"]
                tool_calls.append({
                    "id": "call_" + hashlib.sha256(canonical_json(fc)).hexdigest()[:12],
                    "type": "function",
                    "function": {"name": fc.get("name", ""),
                                 "arguments": json.dumps(fc.get("args") or {},
                                                         sort_keys=True)}})
        if isinstance(cand.get("finishReason"), str):
            finish = _gemini_finish(cand["finishReason"])
    if isinstance(data.get("usageMetadata"), dict):
        usage["input_tokens"] = data["usageMetadata"].get("promptTokenCount") or 0
        usage["output_tokens"] = data["usageMetadata"].get("candidatesTokenCount") or 0
    return {"text": "".join(text_parts), "tool_calls": tool_calls,
            "usage": usage, "finish": finish}


def call_provider(env_url, provider, model_id, wire_in, payload):
    """One provider attempt.  Returns (status, raw_bytes, content_type) on
    a 2xx answer, or (None, error_dict, None) on any refusal: the caller
    fails over to the next candidate."""
    url, headers, body = build_provider_request(env_url, provider, model_id,
                                                wire_in, payload)
    req = urllib.request.Request(url, data=canonical_json(body),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT) as resp:
            raw = resp.read()
            return resp.status, raw, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = ""
        return None, {"status": exc.code, "error": detail,
                      "retry_after": exc.headers.get("Retry-After") if exc.headers else None}, None
    except Exception as exc:
        return None, {"error": str(exc)}, None


# ---------------------------------------------------------------------------
# Response shaping (OpenAI / Anthropic wire fidelity)
# ---------------------------------------------------------------------------
def openai_response(model_id, result, payload, stream):
    rid = "chatcmpl-" + request_id(payload)
    usage = result["usage"]
    finish = "tool_calls" if result["tool_calls"] else (result.get("finish") or "stop")
    if not stream:
        message = {"role": "assistant", "content": result["text"]}
        if result["tool_calls"]:
            message["tool_calls"] = result["tool_calls"]
        return {"id": rid, "object": "chat.completion", "created": 0,
                "model": model_id,
                "choices": [{"index": 0, "message": message,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": usage["input_tokens"],
                          "completion_tokens": usage["output_tokens"],
                          "total_tokens": usage["input_tokens"] + usage["output_tokens"]}}
    chunks = []
    base = {"id": rid, "object": "chat.completion.chunk", "created": 0,
            "model": model_id}
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}]
    chunks.append(first)
    text = result["text"]
    for i in range(0, len(text), 16):
        chunk = dict(base)
        chunk["choices"] = [{"index": 0,
                             "delta": {"content": text[i:i + 16]},
                             "finish_reason": None}]
        chunks.append(chunk)
    for call in result["tool_calls"]:
        chunk = dict(base)
        chunk["choices"] = [{"index": 0,
                             "delta": {"tool_calls": [call]},
                             "finish_reason": None}]
        chunks.append(chunk)
    last = dict(base)
    last["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
    last["usage"] = {"prompt_tokens": usage["input_tokens"],
                     "completion_tokens": usage["output_tokens"],
                     "total_tokens": usage["input_tokens"] + usage["output_tokens"]}
    chunks.append(last)
    lines = ["data: " + json.dumps(chunk, sort_keys=True) + "\n" for chunk in chunks]
    lines.append("data: [DONE]\n")
    return "\n".join(lines) + "\n"


def anthropic_response(model_id, result, payload, stream):
    rid = "msg_" + request_id(payload)
    usage = result["usage"]
    content = []
    if result["text"]:
        content.append({"type": "text", "text": result["text"]})
    for call in result["tool_calls"]:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        content.append({"type": "tool_use", "id": call.get("id", ""),
                        "name": fn.get("name", ""), "input": args})
    stop = "tool_use" if result["tool_calls"] else {
        "stop": "end_turn", "length": "max_tokens",
        "tool_calls": "tool_use", "content_filter": "refusal"}.get(
            result.get("finish") or "stop", "end_turn")
    if not stream:
        return {"id": rid, "type": "message", "role": "assistant",
                "content": content, "model": model_id,
                "stop_reason": stop,
                "usage": {"input_tokens": usage["input_tokens"],
                          "output_tokens": usage["output_tokens"]}}
    events = []
    events.append({"type": "message_start",
                   "message": {"id": rid, "type": "message",
                               "role": "assistant", "content": [],
                               "model": model_id, "stop_reason": None,
                               "usage": {"input_tokens": usage["input_tokens"],
                                         "output_tokens": 0}}})
    index = 0
    if result["text"]:
        events.append({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}})
        text = result["text"]
        for i in range(0, len(text), 16):
            events.append({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta",
                                     "text": text[i:i + 16]}})
        events.append({"type": "content_block_stop", "index": 0})
        index = 1
    for call in result["tool_calls"]:
        fn = call.get("function") or {}
        events.append({"type": "content_block_start", "index": index,
                       "content_block": {"type": "tool_use",
                                         "id": call.get("id", ""),
                                         "name": fn.get("name", ""),
                                         "input": {}}})
        args = fn.get("arguments") or "{}"
        for i in range(0, len(args), 16):
            events.append({"type": "content_block_delta", "index": index,
                           "delta": {"type": "input_json_delta",
                                     "partial_json": args[i:i + 16]}})
        events.append({"type": "content_block_stop", "index": index})
        index += 1
    events.append({"type": "message_delta",
                   "delta": {"stop_reason": stop},
                   "usage": {"output_tokens": usage["output_tokens"]}})
    events.append({"type": "message_stop"})
    lines = []
    for event in events:
        lines.append("event: " + event["type"])
        lines.append("data: " + json.dumps(event, sort_keys=True))
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The aggregator: routing + state
# ---------------------------------------------------------------------------
class Aggregator:
    def __init__(self, env_url, brief, probes, catalog):
        self.env_url = env_url
        self.brief = brief
        self.probes = probes
        self.ratings = brief["ratings"]
        self.catalog = catalog
        self.catalog_bytes = catalog_text(catalog)
        self.catalog_sha256 = hashlib.sha256(self.catalog_bytes).hexdigest()
        self.manifest_index = 0
        self.lock = threading.Lock()
        self.served_total = 0
        self.failures = {}    # provider id -> refused count
        self.attempts = []    # [{provider, model, outcome}] in order, capped
        self.retry_after = {}  # provider id -> last Retry-After seen

    def served_models(self):
        """The free models across the providers that answered, in the
        brief's provider order, sorted by id within a provider: what
        GET /v1/models lists."""
        out = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            for model in sorted(probe["models"], key=lambda m: m["id"]):
                row = rating_of_model(self.ratings, probe["provider"]["id"], model["id"])
                if not model_free(model, row):
                    continue
                out.append({"id": model["id"], "provider": probe["provider"]["id"],
                            "display_name": model["display_name"]})
        return out

    def requirements_for(self, body):
        """The requirements of the request now arriving: the manifest's row
        by arrival order, else the body's own."""
        manifest = (self.brief["workload"] or {}).get("manifest") or []
        entry = manifest[self.manifest_index] if self.manifest_index < len(manifest) else None
        self.manifest_index += 1
        if entry is not None:
            return entry["requires"]
        return requirements_from_body(body)

    def state_export(self):
        return {"attempts": list(self.attempts),
                "catalog_sha256": self.catalog_sha256,
                "failures": dict(sorted(self.failures.items())),
                "format": "free-model-aggregator-state/v1",
                "retry_after": dict(sorted(self.retry_after.items())),
                "served_total": self.served_total}

    def route(self, wire_in, payload):
        """Try candidates in rank order; return (candidate, status, raw,
        content_type) on the first success, or (None, error) when nothing
        can serve the request.  Each candidate is tried at most once, in a
        fixed order, so the outcome depends only on the request and the
        providers' answers -- never on timing."""
        requires = self.requirements_for(payload)
        candidates = rank_candidates(self.probes, self.ratings, requires)
        if not candidates:
            return None, {"error": "no free model in the catalog meets this request",
                          "code": "no_candidate"}
        with self.lock:
            for candidate in candidates:
                provider = candidate["provider"]
                model_id = candidate["model"]["id"]
                status, raw, content_type = call_provider(
                    self.env_url, provider, model_id, wire_in, payload)
                entry = {"model": model_id, "outcome": "ok" if status else "refused",
                         "provider": provider["id"]}
                self.attempts.append(entry)
                if len(self.attempts) > 200:
                    self.attempts.pop(0)
                if isinstance(raw, dict) and raw.get("retry_after") is not None:
                    self.retry_after[provider["id"]] = str(raw["retry_after"])
                if status:
                    self.served_total += 1
                    return candidate, status, raw, content_type
                self.failures[provider["id"]] = self.failures.get(provider["id"], 0) + 1
            return None, {"error": "every candidate provider refused the request",
                          "code": "all_refused",
                          "tried": [c["model"]["id"] for c in candidates]}


def openai_error(message, type_="invalid_request_error"):
    return {"error": {"message": message, "type": type_, "code": None}}


def anthropic_error(message, type_="api_error"):
    return {"type": "error", "error": {"type": type_, "message": message}}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
AGGREGATOR = None
SERIAL = threading.Lock()  # serialise requests: state stays deterministic


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep logs out of stdout; no timestamps
        pass

    def _send(self, status, body, content_type="application/json"):
        if isinstance(body, str):
            raw = body.encode("utf-8")
        elif isinstance(body, bytes):
            raw = body
        else:
            raw = canonical_json(body)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def do_GET(self):
        with SERIAL:
            if self.path in ("/health", "/"):
                self._send(200, {"status": "ok",
                                 "catalog_sha256": AGGREGATOR.catalog_sha256})
            elif self.path == "/catalog":
                self._send(200, AGGREGATOR.catalog_bytes)
            elif self.path == "/state":
                self._send(200, AGGREGATOR.state_export())
            elif self.path == "/v1/models":
                self._handle_models()
            else:
                self._send(404, {"error": "not found"})

    def _handle_models(self):
        """GET /v1/models, in the shape of the auth header it got:
        x-api-key means Anthropic, Authorization means OpenAI."""
        models = AGGREGATOR.served_models()
        if isinstance(self.headers.get("x-api-key"), str):
            self._send(200, {
                "data": [{"type": "model", "id": m["id"],
                          "display_name": m["display_name"],
                          "created_at": FIXED_ANTHROPIC_CREATED_AT}
                         for m in models],
                "has_more": False})
        else:
            self._send(200, {
                "object": "list",
                "data": [{"id": m["id"], "object": "model",
                          "created": FIXED_OPENAI_CREATED,
                          "owned_by": m["provider"]} for m in models]})

    def do_POST(self):
        with SERIAL:
            payload = self._read_json()
            if self.path == "/v1/chat/completions":
                if payload is None:
                    self._send(400, openai_error("the request body is not one JSON object"))
                    return
                self._handle_chat(payload)
            elif self.path == "/v1/messages":
                if payload is None:
                    self._send(400, anthropic_error("the request body is not one JSON object",
                                                    "invalid_request_error"))
                    return
                self._handle_messages(payload)
            else:
                self._send(404, {"error": "not found"})

    def _answer(self, wire_in, payload):
        """Route one request; return (status, body, content_type)."""
        candidate, status, raw, content_type = AGGREGATOR.route(wire_in, payload)
        if candidate is None:
            if wire_in == "anthropic":
                return 503, anthropic_error(status["error"], "overloaded_error"), "application/json"
            return 503, openai_error(status["error"], "no_free_model"), "application/json"
        provider = candidate["provider"]
        model_id = candidate["model"]["id"]
        stream = bool(payload.get("stream"))
        # Same wire as the client: forward the provider's answer as it is.
        if provider["wire"] == wire_in:
            return status, raw, content_type or "application/json"
        result = normalise_provider_response(provider["wire"],
                                             raw.decode("utf-8", "replace"),
                                             content_type)
        if result is None:
            if wire_in == "anthropic":
                return 502, anthropic_error("the provider answered in a shape that could not be read"), "application/json"
            return 502, openai_error("the provider answered in a shape that could not be read"), "application/json"
        if wire_in == "openai":
            body = openai_response(model_id, result, payload, stream)
        else:
            body = anthropic_response(model_id, result, payload, stream)
        ctype = "text/event-stream" if stream else "application/json"
        return 200, body, ctype

    def _handle_chat(self, payload):
        status, body, content_type = self._answer("openai", payload)
        self._send(status, body, content_type)

    def _handle_messages(self, payload):
        status, body, content_type = self._answer("anthropic", payload)
        self._send(status, body, content_type)


# ---------------------------------------------------------------------------
# Trial protocol: register, then poll /work until done and exit 0.
# ---------------------------------------------------------------------------
def _http_json(url, method="GET", body=None, timeout=5):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def register(env_url, port, catalog):
    reply = _http_json("%s/register" % env_url.rstrip("/"), method="POST",
                       body={"protocol_version": PROTOCOL_VERSION,
                             "port": port, "catalog": catalog}, timeout=5)
    if isinstance(reply, dict) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %d requests" % reply["requests"])
    return reply


def poll_until_done(server, env_url):
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            work = _http_json("%s/work" % env_url.rstrip("/"), timeout=2)
        except Exception:
            continue  # the env is not reachable yet: try again
        if isinstance(work, dict) and work.get("done") is True:
            server.shutdown()
            server.server_close()
            return 0


def load(env_url, prompt_file):
    """Read the brief, probe every provider, build the one catalog."""
    brief = read_brief(prompt_file)
    if brief["error"] is not None:
        log(brief["error"])
    probes = [probe_provider(env_url, provider)
              for provider in brief["directory"]["providers"]]
    for probe in probes:
        if not probe["ok"] and probe["error"] is not None:
            log(probe["error"])
    catalog = build_catalog(probes, brief["ratings"])
    return brief, probes, catalog


def main(argv):
    global AGGREGATOR
    port = int(os.environ.get("PORT") or os.environ.get("MAKEFUN_PORT") or DEFAULT_PORT)
    host = "0.0.0.0"
    show_list = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            sys.stdout.write(__doc__)
            return 0
        elif arg == "--list":
            show_list = True
        elif arg == "--port" and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                sys.stderr.write("aggregator: --port needs an integer\n")
                return 2
            i += 1
        elif arg == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 1
        else:
            sys.stderr.write("aggregator: unknown argument %r\n" % arg)
            sys.stderr.write("usage: python3 main.py [--list] [--port N] [--host HOST]\n")
            return 2
        i += 1
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief, probes, catalog = load(env_url, prompt_file)
    if show_list:
        # The same document the register body carries, byte for byte:
        # print the one built object and stop before any server starts.
        sys.stdout.buffer.write(catalog_text(catalog))
        sys.stdout.buffer.flush()
        return 0
    AGGREGATOR = Aggregator(env_url, brief, probes, catalog)
    if env_url == "":
        # Manual mode: no trial env, serve on the given port, no register.
        server = ThreadingHTTPServer((host, port), Handler)
        log("listening on %s:%d with %d providers, %d free models (manual mode)"
            % (host, port, len(catalog["providers"]),
               len(AGGREGATOR.served_models())))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    address = server.server_address
    port = address[1]
    # 8080 is the env's on one layout; the OS's ephemeral range never holds
    # it, and the guard says so.
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%d with %d providers, %d free models"
        % (port, len(catalog["providers"]), len(AGGREGATOR.served_models())))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(env_url, port, catalog)

    def _exit(signum=None, frame=None):
        server.shutdown()
        sys.exit(0)

    import signal
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    return poll_until_done(server, env_url)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
