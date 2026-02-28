# Om's Den

**A terminal-based AI coding agent** — stream responses, call tools (read/write files, run shell, search, web), and get things done from the CLI. Built from scratch with Python, OpenAI-compatible APIs (e.g. [OpenRouter](https://openrouter.ai)), and a focus on clarity and control.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Interactive & one-shot modes** — Chat in the terminal or run a single prompt and exit.
- **Streaming** — See the model’s reply and tool calls as they happen.
- **Rich CLI (Om's Den)** — Clear prompts, panels, and tool output.
- **Tools** — `read_file`, `write_file`, `edit`, `list_dir`, `grep`, `glob`, `shell`, `web_search`, `web_fetch`, `todos`, `memory`, plus MCP and subagents.
- **Safety** — Approval policies (on-request, auto, never, yolo), dangerous-command blocking, path checks.
- **Context** — Compression when near token limits, tool-output pruning, usage tracking.
- **Sessions** — Save, resume, checkpoints; optional hooks before/after agent and tools.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/oms0401/claude-code-from-scratch.git
cd claude-code-from-scratch
uv sync
# or: pip install -e .
```

### 2. Set your API key

Create a `.env` in the project root:

```env
API_KEY=your-openrouter-or-openai-key
```

For **OpenRouter** keys (`sk-or-v1-...`), the app uses `https://openrouter.ai/api/v1` automatically. For OpenAI, leave `BASE_URL` unset. Optional override:

```env
BASE_URL=https://openrouter.ai/api/v1
```

### 3. Run

```bash
# Interactive (Om's Den)
uv run main.py
# or
python main.py

# One-shot
uv run main.py "Add a README section for installation"
```

Use `--cwd` to set the working directory (where the agent reads/writes files):

```bash
uv run main.py --cwd /path/to/your/project "Refactor the login module"
```

---

## Project layout

| Directory | Purpose |
|-----------|---------|
| `agent/` | Agent loop, session, events, persistence |
| `client/` | LLM client and stream/response types |
| `config/` | Config model and TOML/env loading |
| `context/` | Conversation state, compression, loop detection |
| `hooks/` | Before/after agent and tool hooks |
| `prompts/` | System prompt, compression, loop-breaker |
| `safety/` | Approval policy and confirmation |
| `tools/` | Tool base, registry, builtins, MCP, subagents |
| `ui/` | Terminal UI (Rich) |
| `utils/` | Errors, paths, token counting |
| `main.py` | CLI entry (Click) |

---

## Configuration

- **Config file** — Optional `config.toml` under `~/.config/ai-agent/` or project `.ai-agent/`.
- **Environment** — `API_KEY`, optional `BASE_URL`; loaded from `.env` via `python-dotenv`.
- **Defaults** — Model: `z-ai/glm-4.5-air:free` (OpenRouter); approval: on-request; cwd: current directory.

See [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) for a detailed code walkthrough.

---

## Commands (interactive)

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/exit`, `/quit` | Exit Om's Den |
| `/clear` | Clear conversation |
| `/config` | Show configuration |
| `/model <name>` | Change model |
| `/approval <mode>` | Change approval mode |
| `/stats` | Session statistics |
| `/tools` | List tools |
| `/mcp` | MCP server status |
| `/save` | Save session |
| `/checkpoint` | Create checkpoint |
| `/restore <id>` | Restore checkpoint |
| `/sessions` | List saved sessions |
| `/resume <id>` | Resume session |

---

## License

MIT.

---

## Acknowledgments

Inspired by Claude Code and similar agentic coding assistants. Built for developers who want a transparent, configurable agent they can run locally and extend.
