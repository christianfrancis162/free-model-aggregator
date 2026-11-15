# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/christianfrancis162/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/christianfrancis162/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten
simulated free-tier model providers: it reads their catalogs, routes each
request to the best free model that can serve it, fails over when a provider
refuses (402 / 429), and reports the catalog it read. Built by an AI agent
for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:9001 python3 main.py --port 8000
```

`python3 main.py --list` prints the catalog document the server registers,
byte for byte. Standard library only (Python 3.13); nothing to install.
Provider keys come from the environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, or
`PROVIDER_<N>_API_KEY`) — never commit one.

Endpoints: `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions`, `POST /v1/messages`, `GET /catalog`, `GET /state`,
`GET /health`.

Tests (mock env with 10 fake providers, incl. 402/429 failover):

```sh
python3 tests/test_aggregator.py
```

---

Entered on [make.fun](https://make.fun/?ref={referral_code}&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
