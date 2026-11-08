# Codex API setup for the experiment Agent

The launcher stages provider configuration and file credentials into a temporary
home for each Agent. It excludes the operator's projects, plugins, MCP servers,
memories and prior sessions. Configuration and credentials must remain outside
this repository.

## Select the provider home

The existing `EMBODIEDEVO_CODEX_HOME` deployment key is retained for compatibility.
Choose an external directory, or point it at an existing compatible API-key home:

```bash
export EMBODIEDEVO_CODEX_HOME="$HOME/.config/dynamicsadapt/codex"
install -d -m 700 "$EMBODIEDEVO_CODEX_HOME"
```

Copy one of `configs/codex.openai.example.toml` or
`configs/codex.custom.example.toml` to that directory as `config.toml`. Preserve
existing settings if the directory is already configured. For the gateway example,
set `base_url` to the provider's API base URL; the supplied template uses the
Responses wire protocol. Keep real endpoints and credentials in the external copy.

With the configured Codex CLI available on the host, save the key for the selected
provider using the template's file credential store. Disable shell tracing first:

```bash
read -r -s -p "API key: " dynamicsadapt_api_key
printf '\n'
printf '%s' "$dynamicsadapt_api_key" | \
  CODEX_HOME="$EMBODIEDEVO_CODEX_HOME" codex login --with-api-key
unset dynamicsadapt_api_key
chmod 600 "$EMBODIEDEVO_CODEX_HOME/auth.json"
chmod 600 "$EMBODIEDEVO_CODEX_HOME/config.toml"
```

The launcher copies `auth.json` and whitelisted provider settings. Merely exporting
`OPENAI_API_KEY` on the host does not configure the isolated container. Without an
override it uses the normal `CODEX_HOME` or `~/.codex`; a dedicated provider home
keeps measured runs independent of interactive work.

## Set the experiment model

Set the `agent` block in the experiment JSON, not an old RoboDojo task file:

```json
{
  "preset": "codex",
  "model": "gpt-6-astra",
  "reasoning_effort": "high",
  "service_tier": "fast",
  "image": "dynamicsadapt-agent:latest",
  "network": "filtered",
  "cpus": 4,
  "memory_mb": 8192
}
```

These values reproduce the example's settings, not a provider entitlement check.
The launcher explicitly passes model, effort and service tier to the CLI, overriding
home defaults. Record changes in a new run configuration.

`network: "filtered"` names the existing Docker bridge configuration; it does not
filter destination hosts. Configure a reachable endpoint or
`EMBODIEDEVO_AGENT_HTTPS_PROXY` when needed. `localhost` inside the container refers
to the container, not the host.

To inspect staging without sending a model request:

```bash
.venv/bin/python - <<'PY_CHECK'
import tempfile
from pathlib import Path
from orchestrator.launch_agent import stage_home

with tempfile.TemporaryDirectory(prefix="dynamicsadapt-auth-check-") as directory:
    staged = stage_home(Path(directory) / "agent-home") / ".codex"
    assert (staged / "config.toml").is_file()
    assert (staged / "auth.json").is_file()
    print("Provider files staged; no model request sent.")
PY_CHECK
```

This checks file staging only. Endpoint/model access and Docker connectivity need
separate checks with actual service resources. Preserve Agent logs and sessions
when diagnosing authentication, routing or model failures; never print API keys.
