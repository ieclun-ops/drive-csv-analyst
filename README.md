# drive-csv-analyst

A conversational data analyst agent built with the Mistral Agents SDK (`mistralai-agents`).
The user pastes a Google Drive link to a CSV file and asks questions in plain language. The
agent fetches the file through the user's Google Drive connection and answers with real
numbers, computed only by fixed tools — there is no code-execution tool of any kind.

## Tools (fixed set, all called directly by the model)

- `load_csv(drive_link)` — resolves the Drive file ID from a link, downloads it via the
  `google_drive_mcp` connector (Google Sheets are exported as CSV), parses it, and makes it
  the active dataset for the session.
- `describe_columns()` — per-column dtype, null/unique counts, numeric min/max/mean/median,
  top values for text columns.
- `filter_rows(filters, limit)` — AND-ed row filters (`==`, `!=`, `>`, `>=`, `<`, `<=`,
  `contains`, `not_contains`, `in`, `not_in`, `is_null`, `not_null`) with match count and a
  row preview.
- `group_by_aggregate(group_by, aggregations, filters, limit)` — group-by aggregates
  (`sum`, `mean`, `median`, `min`, `max`, `count`, `count_distinct`, `std`), optionally on a
  filtered subset; empty `group_by` gives one overall row.

## Safety properties (no arbitrary code execution)

- All four data tools are `model_access="direct"`: the model calls them as typed function
  calls, never from generated code.
- `builtins=Builtins(skills=False, subagents=False, background_processes=False)` and the
  default `NoSandbox`: no shell, no file tools, no subagents, no skills.
- The platform requires programmatic tool calling while a connector is attached, so the
  `run_typescript` program tool technically exists; it is blocked two ways: a
  `guardrails=[NoCodeExecution()]` hook skips every `run_typescript`/`self.*` call before it
  runs (the model reads the refusal as the tool result), and a `permission_policy` denies the
  `self.*` helpers. The model's only usable tools are the four fixed ones plus the read-only
  `google_drive_mcp` connector tools.
- The instructions tell the model to never write code and to ground every number in tool
  results.
- Datasets are held per session (`ctx.session.id`) in the worker; conversation history and
  sessions are stored by Mistral on a managed deployment, so follow-ups remember the file
  and everything discussed. A deployed session idles out after 4 h and ends after 24 h.

## Run locally

```bash
uv run --env-file .env agents chat          # interactive, agent runs in this process
export MISTRAL_API_KEY=...                   # or put it in .env
uv run python test_tools.py                 # scripted-model test, no API key needed
```

## Deploy (Mistral-managed on Mistral Cloud)

The deployment builds the pushed commit of the GitHub repository (the GitHub connector must
be connected in Mistral Studio so the platform can clone it):

```bash
git add -A && git commit -m "drive-csv-analyst agent" && git push
uv run --env-file .env agents deploy        # hardened automatically (on-behalf-of Drive connector)
```

After deploy, chat with it:

- in Vibe Code Web: `https://chat.mistral.ai/managed-agents/drive-csv-analyst?source=agents_api`
- from a terminal: `uv run --env-file .env agents chat drive-csv-analyst`
- from code: `agents.Agent.load_from(agent_name="drive-csv-analyst")`, then
  `session = await agent.start_session()` and `await session.run(message)` per message;
  resume later with `agent.session(session_id)`.
