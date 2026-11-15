#!/usr/bin/env python3
"""Tests for the aggregator against the REAL make.fun trial protocol.

Usage: python3 tests/test_aggregator.py
Checks (stdlib only):
  unit: brief parsing (directory / ratings / manifest from the kit's
    practice trial-0 prompt.txt), RULES catalog shape, rating-based
    free-only routing with manifest requirements, provider request
    building (stops, stream_options.include_usage, no max_tokens
    default, Gemini systemInstruction + inlineData images),
    provider-answer normalisation (usage taken from the provider,
    never estimated), and SSE framing (blank-line separated frames).
  integration: the kit's own env server + scorer (the same binaries
    grade-local runs) drive practice trial-0 end to end: the bot must
    register, serve 20/20 valid, exit 0, and score >= 95; --list must
    print byte-for-byte the catalog the bot registered.
Exits 0 on pass, 1 on any failure. Prints PASS/FAIL per check.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = os.path.join(ROOT, "main.py")
KIT = os.path.expanduser("~/workspace/makefun/kit/free-model-aggregator-01-kit")
RUN_TRIAL = os.path.expanduser("~/workspace/makefun/grade-local/run-trial.sh")
TRIAL0 = os.path.join(KIT, "practice/trial-0")
STATE0 = os.path.expanduser("~/workspace/makefun/grade-local/manual-t0")

FAILURES = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" :: " + str(detail)) if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def load_main():
    spec = importlib.util.spec_from_file_location("agg_main", MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load_main()


def fake_probe(pid, wire, models, key_env="KEY", base=None):
    return {"provider": {"id": pid, "name": pid, "wire": wire, "auth": "bearer",
                         "base": base or ("/p/" + pid), "key_env": key_env,
                         "key": "k-" + pid},
            "ok": True, "status": 200, "models": models, "error": None}


def fake_model(mid, free=True, ctx=131072, tools=None, vision=None, json_=None):
    return {"id": mid, "display_name": mid, "free": free, "context_length": ctx,
            "tools": tools, "vision": vision, "json": json_}


# --- unit: brief parsing -------------------------------------------------
brief = M.read_brief(os.path.join(TRIAL0, "input/prompt.txt"))
check("brief: no error", brief["error"] is None, brief["error"])
check("brief: 10 providers", len(brief["directory"]["providers"]) == 10,
      len(brief["directory"]["providers"]))
check("brief: ratings present", brief["ratings"] is not None
      and len(brief["ratings"]["ratings"]) > 20)
check("brief: manifest 20 requests", brief["workload"] is not None
      and len(brief["workload"]["manifest"]) == 20)
man = brief["workload"]["manifest"]
check("brief: r01 tools+vision", man[0]["requires"].get("tools") is True
      and man[0]["requires"].get("vision") is True, man[0])
check("brief: r10 min_context 200000", man[9]["requires"].get("min_context") == 200000,
      man[9]["requires"])
check("brief: r08 json, anthropic", man[7]["requires"].get("json") is True
      and man[7]["api"] == "anthropic", man[7])

# --- unit: catalog shape (RULES.txt) --------------------------------------
probes = [fake_probe("groq", "openai",
                     [fake_model("llama-3.1-8b-instant", True, 131072, True, False, True),
                      fake_model("paid-x", True, 8192)]),
          fake_probe("openrouter", "openai",
                     [fake_model("anthropic/claude-opus-5.5", False, 200000),
                      fake_model("z-ai/glm-5.3-flash:free", True, 262144, True, False, True)])]
ratings = {"schema": M.RATINGS_SCHEMA, "ratings": [
    {"model": "llama-3.1-8b-instant", "provider": "groq", "rating_milli": 1400,
     "context_length": 131072, "tools": True, "vision": False, "json": True, "free": True},
    {"model": "paid-x", "provider": "groq", "rating_milli": 9999,
     "context_length": 8192, "tools": True, "vision": True, "json": True, "free": False},
    {"model": "anthropic/claude-opus-5.5", "provider": "openrouter", "rating_milli": 1999,
     "context_length": 200000, "tools": True, "vision": True, "json": True, "free": False},
    {"model": "z-ai/glm-5.3-flash:free", "provider": "openrouter", "rating_milli": 1475,
     "context_length": 262144, "tools": True, "vision": False, "json": True, "free": True}]}
catalog = M.build_catalog(probes, ratings)
check("catalog: only 'providers' key", set(catalog.keys()) == {"providers"}, catalog.keys())
prov0 = catalog["providers"][0]
check("catalog: provider keys", set(prov0.keys()) == {"id", "base_url", "key_env", "wire", "models"},
      prov0.keys())
check("catalog: base_url is the path", prov0["base_url"] == "/p/groq", prov0["base_url"])
check("catalog: model keys", set(prov0["models"][0].keys()) == {"id", "free", "context_length"},
      prov0["models"][0])
paid = [m for p in catalog["providers"] for m in p["models"] if m["id"] in ("paid-x", "anthropic/claude-opus-5.5")]
check("catalog: paid models marked not free", all(m["free"] is False for m in paid), paid)

# --- unit: routing ---------------------------------------------------------
cands = M.rank_candidates(probes, ratings, {"tools": True})
check("routing: paid model never a candidate",
      all(c["model"]["id"] != "anthropic/claude-opus-5.5" for c in cands), [c["model"]["id"] for c in cands])
check("routing: best rating first", cands[0]["model"]["id"] == "z-ai/glm-5.3-flash:free",
      [c["model"]["id"] for c in cands])
cands_v = M.rank_candidates(probes, ratings, {"tools": True, "vision": True})
check("routing: vision filter excludes non-vision", cands_v == [], cands_v)
cands_ctx = M.rank_candidates(probes, ratings, {"min_context": 200000})
check("routing: min_context 200k keeps only 262144/200000-free",
      [c["model"]["id"] for c in cands_ctx] == ["z-ai/glm-5.3-flash:free"],
      [c["model"]["id"] for c in cands_ctx])

# --- unit: provider request building --------------------------------------
ENV = "http://127.0.0.1:8080"
prov_oai = {"id": "groq", "wire": "openai", "auth": "bearer", "base": "/p/groq", "key": "k"}
prov_ant = {"id": "zai", "wire": "anthropic", "auth": "x-api-key", "base": "/p/zai", "key": "k"}
prov_gem = {"id": "gemini", "wire": "gemini", "auth": "x-goog-api-key", "base": "/p/gemini", "key": "k"}
body_oai = {"model": "x", "messages": [{"role": "system", "content": "sys"},
                                       {"role": "user", "content": "hi"}],
            "max_tokens": 200, "stop": ["\n\n###"], "stream": True}
url, headers, body = M.build_provider_request(ENV, prov_oai, "llama", "openai", body_oai)
check("req: openai url", url == ENV + "/p/groq/v1/chat/completions", url)
check("req: openai stop passthrough", body.get("stop") == ["\n\n###"], body.get("stop"))
check("req: openai stream_options include_usage",
      body.get("stream_options") == {"include_usage": True}, body.get("stream_options"))
check("req: openai max_tokens exact", body.get("max_tokens") == 200, body.get("max_tokens"))
url, headers, body = M.build_provider_request(ENV, prov_oai, "llama", "openai",
                                              {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
check("req: no max_tokens default", "max_tokens" not in body, body)
img_url = "data:image/png;base64,iVBORw0KGgo="
body_gem = {"model": "x", "messages": [{"role": "system", "content": "sys"},
                                       {"role": "user", "content": [{"type": "text", "text": "look"},
                                                                    {"type": "image_url", "image_url": {"url": img_url}}]}],
            "max_tokens": 128, "stop": ["END"], "response_format": {"type": "json_object"}}
url, headers, body = M.build_provider_request(ENV, prov_gem, "gemini-3.8-flash", "openai", body_gem)
check("req: gemini url", url == ENV + "/p/gemini/v1beta/models/gemini-3.8-flash:generateContent", url)
check("req: gemini systemInstruction",
      body.get("systemInstruction") == {"parts": [{"text": "sys"}]}, body.get("systemInstruction"))
parts = body["contents"][0]["parts"]
check("req: gemini inlineData image",
      any(p.get("inlineData", {}).get("data") == "iVBORw0KGgo=" for p in parts), parts)
check("req: gemini stopSequences + json",
      body.get("generationConfig", {}).get("stopSequences") == ["END"]
      and body.get("generationConfig", {}).get("responseMimeType") == "application/json",
      body.get("generationConfig"))
url, headers, body = M.build_provider_request(
    ENV, prov_ant, "glm-4.7-flash", "anthropic",
    {"model": "x", "messages": [{"role": "user", "content": "hi"}],
     "max_tokens": 300, "stop_sequences": ["\n\nHuman:"], "stream": True})
check("req: anthropic url + stops", url == ENV + "/p/zai/v1/messages"
      and body.get("stop_sequences") == ["\n\nHuman:"], (url, body.get("stop_sequences")))
check("req: anthropic max_tokens exact, no default", body.get("max_tokens") == 300, body)

# --- unit: normalisation (provider usage, never estimated) -----------------
sse_oai = ('data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
           'data: {"choices":[{"delta":{"content":"ANSWER"}}]}\n\n'
           'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
           'data: {"choices":[],"usage":{"prompt_tokens":41,"completion_tokens":7,"total_tokens":48}}\n\n'
           'data: [DONE]\n\n')
norm = M.normalise_provider_response("openai", sse_oai, "text/event-stream")
check("norm: openai SSE text+usage+finish", norm["text"] == "ANSWER"
      and norm["usage"] == {"input_tokens": 41, "output_tokens": 7}
      and norm["finish"] == "stop", norm)
norm = M.normalise_provider_response(
    "gemini", '{"candidates":[{"content":{"parts":[{"text":"ANSWER"}],"role":"model"},'
    '"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,'
    '"candidatesTokenCount":3,"totalTokenCount":13}}', "application/json")
check("norm: gemini JSON usage+finish", norm["text"] == "ANSWER"
      and norm["usage"] == {"input_tokens": 10, "output_tokens": 3}
      and norm["finish"] == "stop", norm)
norm = M.normalise_provider_response(
    "anthropic", '{"type":"message","content":[{"type":"text","text":"ANSWER"}],'
    '"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":2}}',
    "application/json")
check("norm: anthropic JSON stop mapping", norm["finish"] == "stop"
      and norm["usage"] == {"input_tokens": 9, "output_tokens": 2}, norm)

# --- unit: SSE framing (blank-line separated frames) ----------------------
res = {"text": "ANSWER ECHO{}", "tool_calls": [], "usage": {"input_tokens": 5, "output_tokens": 2}, "finish": "stop"}
out = M.openai_response("m", res, {"messages": []}, True)
frames = [f for f in out.split("\n\n") if f.strip()]
check("sse: openai frames split cleanly", all(f.startswith("data: ") for f in frames)
      and frames[-1] == "data: [DONE]" and len(frames) >= 3, frames[:2])
out = M.anthropic_response("m", res, {"messages": []}, True)
frames = [f for f in out.split("\n\n") if f.strip()]
check("sse: anthropic frames have event+data", all(
    f.startswith("event: ") and "\ndata: " in f for f in frames)
    and frames[0].startswith("event: message_start")
    and frames[-1].startswith("event: message_stop"), frames[:1])

# --- integration: real kit env + scorer, practice trial-0 -----------------
if os.path.exists(RUN_TRIAL) and os.path.exists(os.path.join(KIT, "program/scorer/score.mjs")):
    proc = subprocess.run([RUN_TRIAL, "0"], capture_output=True, text=True, timeout=120)
    scores_path = os.path.join(STATE0, "scores/scores.json")
    ok = proc.returncode == 0 and os.path.exists(scores_path)
    scores = json.load(open(scores_path)) if ok else {}
    check("trial-0: bot exits 0", proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:])
    check("trial-0: 20/20 valid", scores.get("working") == 1 and scores.get("served") == 100, scores)
    check("trial-0: score >= 95", scores.get("score", 0) >= 95, scores)
    check("trial-0: catalog 100", scores.get("catalog") == 100, scores)
    # --list must print byte-for-byte the catalog the bot registered.
    state = json.load(open(os.path.join(STATE0, "export/state.json")))
    env_proc = None
    try:
        env = dict(os.environ,
                   MAKEFUN_ENV_DATA_DIR=os.path.join(TRIAL0, "input/env"),
                   MAKEFUN_STATE_DIR=os.path.expanduser("~/workspace/makefun/grade-local/list-check"),
                   MAKEFUN_ENV_PORT="8081")
        os.makedirs(env["MAKEFUN_STATE_DIR"], exist_ok=True)
        env_proc = subprocess.Popen(["node", os.path.join(KIT, "program/env/server.mjs")],
                                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if os.path.exists(os.path.join(env["MAKEFUN_STATE_DIR"], "ready")):
                break
            time.sleep(0.1)
        list_env = dict(os.environ, MAKEFUN_ENV_URL="http://127.0.0.1:8081",
                        MAKEFUN_PROMPT_FILE=os.path.join(TRIAL0, "input/prompt.txt"))
        out = subprocess.run([sys.executable, MAIN, "--list"], env=list_env,
                             capture_output=True, timeout=30)
        listed = json.loads(out.stdout.decode("utf-8"))
        expected = (json.dumps(state["catalog"]) + "\n").encode("utf-8")
        check("list: exit 0 + 10 providers", out.returncode == 0
              and len(listed.get("providers", [])) == 10, out.returncode)
        check("list: byte-for-byte the registered catalog", out.stdout == expected,
              "listed %d bytes vs registered %d bytes" % (len(out.stdout), len(expected)))
    finally:
        if env_proc is not None:
            env_proc.terminate()
else:
    check("trial-0: kit/scorer present", False, "kit or run-trial.sh missing")

print()
if FAILURES:
    print("FAILURES: %d: %s" % (len(FAILURES), "; ".join(FAILURES)))
    sys.exit(1)
print("ALL CHECKS PASSED")
