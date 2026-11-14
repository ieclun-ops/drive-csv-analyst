# AGENTS.md

## What this is

A conversational data analyst agent (`drive-csv-analyst`) built on the Mistral Agents SDK. The user pastes a Google Drive link to a CSV and asks questions in plain language; the agent answers with numbers computed only by four fixed tools — there is no code-execution tool of any kind.

- `main.py` — the entire agent: the four tools (`load_csv`, `describe_columns`, `filter_rows`, `group_by_aggregate`), Drive helpers, the `INSTRUCTIONS` prompt, the `NoCodeExecution` guardrail, the `NO_CODE_EXECUTION_POLICY` permission policy, and the `agents.Agent` definition.
- `test_tools.py` — tests for the helpers, the tool results, and a scripted-model agent run (no API key needed).
- `Dockerfile` — worker image for Mistral-managed deployment (`agents run-worker`), runs as non-root.

## Commands

```bash
uv sync                                        # install dependencies (Python >=3.12, pandas, mistralai-agents)
uv run python test_tools.py                    # run all tests, no API key needed
uv run --env-file .env agents chat             # interactive local chat (needs MISTRAL_API_KEY in .env)
uv run --env-file .env agents deploy            # deploy to Mistral Cloud (push first; GitHub connector must be connected)
```

There is no separate lint/format/typecheck config; keep the existing style (positional-only `ctx` args in tools, `Annotated` + pydantic `Field` descriptions on every tool parameter, module-level constants for limits).

## Architecture invariants — do not break

- **No code execution.** All four data tools are `model_access="direct"`. `builtins=Builtins(skills=False, subagents=False, background_processes=False)`. The `run_typescript` program tool must stay blocked both by the `NoCodeExecution` hook (skips `run_typescript`/`self.*` calls) and by `NO_CODE_EXECUTION_POLICY` (denies `self.*`). Any new tool must follow the same pattern.
- **Only the four fixed tools plus the read-only `google_drive_mcp` connector.** The connector is called via `ToolCallClient("google_drive_mcp")` (`DRIVE.call_tool(...)`); Google Sheets are exported as `text/csv`.
- **Session-scoped datasets.** Parsed DataFrames live in the in-process `_DATASETS` dict keyed by `ctx.session.id`; `load_csv` replaces the previous dataset. Limits are the module constants (`MAX_TEXT_CHARS`, `MAX_ROWS`, `MAX_RETURN_ROWS`, `MAX_COLUMNS_REPORTED`, `MAX_SESSION_DATASETS`).
- **Every number in an answer comes from a tool result.** The `INSTRUCTIONS` prompt forbids invented or estimated figures; keep that language when editing it.
- User-facing tool errors are raised as `agents.ToolError` with actionable messages (they are shown to the model verbatim).

## Testing notes

- `test_tools.py` runs as a plain script (not pytest): `test_helpers()`, `test_tool_results()`, then the scripted-model run `test_agent_scripted()`, printing `ALL TESTS PASSED` at the end.
- The scripted-model test monkeypatches `main._fetch_drive_text` to avoid Drive calls, and asserts that `run_typescript` never runs and that the guardrail blocks it.
- After changing any tool schema, filter/aggregate logic, or the prompt, run `uv run python test_tools.py` and extend the tests to cover the change.
