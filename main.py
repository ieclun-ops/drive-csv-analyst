import asyncio
import base64
import csv
import io
import json
import re
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents
from mistralai.workflows.plugins.mistralai.connectors import ToolCallClient

DRIVE = ToolCallClient("google_drive_mcp")

MAX_TEXT_CHARS = 20_000_000
MAX_ROWS = 2_000_000
MAX_RETURN_ROWS = 100
MAX_COLUMNS_REPORTED = 200
MAX_SESSION_DATASETS = 64

GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"

_DRIVE_ID_PATTERNS = (
    re.compile(r"/d/([a-zA-Z0-9_-]{10,})"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})"),
)
_BARE_ID = re.compile(r"[a-zA-Z0-9_-]{10,}$")


class _Dataset:
    def __init__(self, title: str, source: str, frame: pd.DataFrame) -> None:
        self.title = title
        self.source = source
        self.frame = frame


_DATASETS: dict[str, _Dataset] = {}


def _extract_file_id(link: str) -> str:
    text = link.strip()
    for pattern in _DRIVE_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    if _BARE_ID.fullmatch(text):
        return text
    raise agents.ToolError(
        "I could not find a Google Drive file ID in that link. "
        "Paste a full Drive link (https://drive.google.com/file/d/<id>/view or "
        "https://docs.google.com/.../<id>/edit) or the bare file ID."
    )


def _unwrap_tool_result(result: Any) -> dict:
    if isinstance(result, dict):
        if "result" in result and isinstance(result["result"], dict):
            return result["result"]
        return result
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    parsed = json.loads(block["text"])
                except (ValueError, TypeError):
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return {"content": result}
    raise agents.ToolError(f"Unexpected response from the Google Drive connector: {type(result).__name__}")


async def _fetch_drive_text(file_id: str) -> tuple[str, str, str]:
    meta = _unwrap_tool_result(await DRIVE.call_tool("get_file_metadata", {"file_id": file_id}))
    title = str(meta.get("title") or file_id)
    mime = str(meta.get("mime_type") or "")
    size = meta.get("file_size")
    if size is not None:
        try:
            if int(size) > MAX_TEXT_CHARS:
                raise agents.ToolError(
                    f"'{title}' is about {int(size)} bytes, too large to load (limit {MAX_TEXT_CHARS}). "
                    "Export a smaller CSV from Drive and share that link."
                )
        except (TypeError, ValueError):
            pass
    if mime == GOOGLE_SHEET_MIME:
        arguments = {"file_id": file_id, "export_mime_type": "text/csv"}
    else:
        arguments = {"file_id": file_id}
    downloaded = _unwrap_tool_result(await DRIVE.call_tool("download_file_content", arguments))
    content = downloaded.get("content")
    if not content:
        raise agents.ToolError(
            f"Could not download '{title}' ({mime or 'unknown type'}). If it is a Google Doc or Sheet, "
            "export it as CSV in Drive first, or share the CSV file itself."
        )
    try:
        text = base64.b64decode(content).decode("utf-8-sig")
    except (ValueError, UnicodeDecodeError) as error:
        raise agents.ToolError(f"'{title}' is not a UTF-8 text/CSV file ({error}).") from error
    if len(text) > MAX_TEXT_CHARS:
        raise agents.ToolError(f"'{title}' is too large to load (limit {MAX_TEXT_CHARS} characters).")
    return title, mime, text


def _parse_csv(text: str) -> pd.DataFrame:
    sample = text[:20_000]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip")
    except Exception as error:
        raise agents.ToolError(f"Could not parse the file as CSV: {error}") from error
    if frame.empty or len(frame.columns) == 1:
        raise agents.ToolError(
            "The file parsed as a single column; it may not be a CSV (wrong delimiter or not CSV at all)."
        )
    if len(frame) > MAX_ROWS:
        raise agents.ToolError(f"The file has {len(frame)} rows; the limit is {MAX_ROWS}.")
    return frame


def _get_dataset(ctx: agents.ToolCallbackContext) -> _Dataset:
    dataset = _DATASETS.get(ctx.session.id)
    if dataset is None:
        raise agents.ToolError(
            "No CSV is loaded in this conversation yet. Share a Google Drive link to a CSV file first."
        )
    return dataset


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


Operator = Literal[
    "==", "!=", ">", ">=", "<", "<=",
    "contains", "not_contains", "in", "not_in", "is_null", "not_null",
]
AggFunction = Literal["sum", "mean", "median", "min", "max", "count", "count_distinct", "std"]


class FilterSpec(BaseModel):
    column: str = Field(description="Column to filter on. Must be one of the dataset's columns.")
    operator: Operator = Field(
        description=(
            "Comparison: ==, !=, >, >=, <, <= (numeric columns compare numerically, others as text), "
            "contains / not_contains (case-insensitive substring), in / not_in (value must be a list), "
            "is_null / not_null (value ignored)."
        )
    )
    value: Any = Field(default=None, description="Value to compare against. A list for 'in'/'not_in'. Ignored for is_null/not_null.")


class AggregationSpec(BaseModel):
    function: AggFunction = Field(description="Aggregate to compute.")
    column: str | None = Field(
        default=None,
        description="Column to aggregate. Required for all functions except 'count' (which counts rows).",
    )


def _apply_one_filter(frame: pd.DataFrame, spec: FilterSpec) -> pd.Series:
    if spec.column not in frame.columns:
        raise agents.ToolError(
            f"Unknown column '{spec.column}'. Available columns: {', '.join(map(str, frame.columns))}"
        )
    column = frame[spec.column]
    op = spec.operator
    if op == "is_null":
        return column.isna()
    if op == "not_null":
        return column.notna()
    if op in ("in", "not_in"):
        if not isinstance(spec.value, list):
            raise agents.ToolError(f"operator '{op}' needs 'value' to be a list, got {type(spec.value).__name__}.")
        numeric = pd.to_numeric(column, errors="coerce")
        if numeric.notna().sum() >= max(1, int(0.8 * len(column))):
            try:
                values = [float(v) for v in spec.value]
            except (TypeError, ValueError):
                mask = column.astype(str).isin([str(v) for v in spec.value])
            else:
                mask = numeric.isin(values)
        else:
            mask = column.astype(str).isin([str(v) for v in spec.value])
        return ~mask if op == "not_in" else mask
    if op in ("contains", "not_contains"):
        mask = column.astype(str).str.contains(str(spec.value), case=False, regex=False, na=False)
        return ~mask if op == "not_contains" else mask
    numeric = pd.to_numeric(column, errors="coerce")
    if numeric.notna().sum() >= max(1, int(0.8 * len(column))):
        try:
            operand: Any = float(spec.value)
        except (TypeError, ValueError):
            raise agents.ToolError(
                f"Column '{spec.column}' is numeric; '{op}' needs a number, got {spec.value!r}."
            ) from None
        series: pd.Series = numeric
    else:
        operand = str(spec.value)
        series = column.astype(str)
    if op == "==":
        return series == operand
    if op == "!=":
        return series != operand
    if op == ">":
        return series > operand
    if op == ">=":
        return series >= operand
    if op == "<":
        return series < operand
    return series <= operand


def _as_filter(spec: FilterSpec | dict[str, Any]) -> FilterSpec:
    return spec if isinstance(spec, FilterSpec) else FilterSpec.model_validate(spec)


def _as_aggregation(spec: AggregationSpec | dict[str, Any]) -> AggregationSpec:
    return spec if isinstance(spec, AggregationSpec) else AggregationSpec.model_validate(spec)


def _apply_filters(frame: pd.DataFrame, filters: list[FilterSpec | dict[str, Any]] | None) -> pd.DataFrame:
    if not filters:
        return frame
    mask = pd.Series(True, index=frame.index)
    for spec in filters:
        mask &= _apply_one_filter(frame, _as_filter(spec))
    return frame[mask]


@agents.tool(model_access="direct")
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    drive_link: Annotated[
        str,
        Field(description="Google Drive link (or bare file ID) of the CSV file or Google Sheet to analyze."),
    ],
) -> dict[str, Any]:
    """Load a CSV file from the user's Google Drive (via the google_drive_mcp connector) as this conversation's active dataset. Google Sheets are exported as CSV automatically. Replaces any previously loaded dataset in this session."""
    file_id = _extract_file_id(drive_link)
    title, mime, text = await _fetch_drive_text(file_id)
    frame = _parse_csv(text)
    if len(_DATASETS) >= MAX_SESSION_DATASETS:
        _DATASETS.pop(next(iter(_DATASETS)))
    _DATASETS[ctx.session.id] = _Dataset(title, drive_link, frame)
    return {
        "source": title,
        "file_id": file_id,
        "mime_type": mime,
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "preview": _records(frame.head(3)),
        "note": "Dataset is now active for this conversation; use describe_columns, filter_rows and group_by_aggregate on it.",
    }


@agents.tool(model_access="direct")
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Describe the active dataset: every column's type, null/unique counts, numeric min/max/mean/median, and the most common values for text columns."""
    dataset = _get_dataset(ctx)
    frame = dataset.frame
    columns: list[dict[str, Any]] = []
    truncated = False
    for name in frame.columns:
        if len(columns) >= MAX_COLUMNS_REPORTED:
            truncated = True
            break
        series = frame[name]
        info: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            info.update(
                min=float(series.min()),
                max=float(series.max()),
                mean=float(series.mean()),
                median=float(series.median()),
            )
        else:
            top = series.dropna().astype(str).value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        samples = series.dropna().head(3)
        info["samples"] = [str(v) for v in samples]
        columns.append(info)
    return {
        "source": dataset.title,
        "rows": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns_truncated": truncated,
        "columns": columns,
    }


@agents.tool(model_access="direct")
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    filters: Annotated[list[FilterSpec], Field(description="One or more conditions; rows must match all of them (AND).")],
    limit: Annotated[int, Field(description=f"Max rows to return in the preview (default 20, max {MAX_RETURN_ROWS}).")] = 20,
) -> dict[str, Any]:
    """Filter rows of the active dataset and return the match count plus a preview of the matching rows."""
    dataset = _get_dataset(ctx)
    matched = _apply_filters(dataset.frame, filters)
    limit = max(1, min(limit, MAX_RETURN_ROWS))
    return {
        "source": dataset.title,
        "matched": int(len(matched)),
        "of_total": int(len(dataset.frame)),
        "returned": int(min(limit, len(matched))),
        "truncated": int(len(matched)) > limit,
        "rows": _records(matched.head(limit)),
    }


@agents.tool(model_access="direct")
async def group_by_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[
        list[str],
        Field(description="Columns to group by. Pass an empty list for one overall total row across the whole dataset."),
    ],
    aggregations: Annotated[list[AggregationSpec], Field(description="Aggregates to compute per group.")],
    filters: Annotated[
        list[FilterSpec] | None,
        Field(default=None, description="Optional row filters applied before aggregating (e.g. only one region)."),
    ] = None,
    limit: Annotated[int, Field(description=f"Max groups to return (default 50, max {MAX_RETURN_ROWS}).")] = 50,
) -> dict[str, Any]:
    """Compute group-by aggregates (sum, mean, median, min, max, count, count_distinct, std) over the active dataset, optionally restricted by filters. Returns one row per group."""
    dataset = _get_dataset(ctx)
    frame = _apply_filters(dataset.frame, filters)
    for column in group_by:
        if column not in frame.columns:
            raise agents.ToolError(
                f"Unknown group-by column '{column}'. Available columns: {', '.join(map(str, frame.columns))}"
            )
    aggregations = [_as_aggregation(spec) for spec in aggregations]
    for spec in aggregations:
        if spec.column is None:
            if spec.function != "count":
                raise agents.ToolError("'column' is required for every function except 'count'.")
        elif spec.column not in frame.columns:
            raise agents.ToolError(
                f"Unknown column '{spec.column}'. Available columns: {', '.join(map(str, frame.columns))}"
            )

    pandas_fn = {
        "sum": "sum", "mean": "mean", "median": "median", "min": "min",
        "max": "max", "count": "count", "count_distinct": "nunique", "std": "std",
    }
    named: dict[str, Any] = {}
    for spec in aggregations:
        label = "count" if spec.column is None else f"{spec.function}({spec.column})"
        if spec.column is None:
            named[label] = pd.NamedAgg(column=group_by[0] if group_by else frame.columns[0], aggfunc="size")
        else:
            named[label] = pd.NamedAgg(column=spec.column, aggfunc=pandas_fn[spec.function])

    if group_by:
        result = frame.groupby(group_by, dropna=False, sort=True).agg(**named).reset_index()
    else:
        row = {label: frame[agg.column].agg(agg.aggfunc) for label, agg in named.items()}
        result = pd.DataFrame([row])

    first_agg = next(iter(named)) if named else None
    if first_agg is not None and pd.api.types.is_numeric_dtype(result[first_agg]):
        result = result.sort_values(first_agg, ascending=False, na_position="last")

    limit = max(1, min(limit, MAX_RETURN_ROWS))
    total_groups = int(len(result))
    return {
        "source": dataset.title,
        "rows_aggregated": int(len(frame)),
        "groups": total_groups,
        "returned": int(min(limit, total_groups)),
        "truncated": total_groups > limit,
        "rows": _records(result.head(limit)),
    }


INSTRUCTIONS = """You are a conversational data analyst. The user shares a link to a CSV file in their Google Drive and asks questions about it in plain language. Your job is to answer with real numbers computed from their data.

Workflow:
1. When the user shares a Google Drive link or file name for a CSV, call load_csv with it. If they only give a file name, use the google_drive_mcp.search_files connector tool to find its link or file ID first, then call load_csv.
2. Call describe_columns to learn the schema (columns, types, ranges) before answering quantitative questions.
3. Compute every number with the fixed tools: filter_rows to inspect a subset of rows, and group_by_aggregate for counts, sums, averages, min/max per group. When a question is about a subset (e.g. "only in France"), pass the matching filters to group_by_aggregate (or filter_rows first to see the rows).
4. Reply in plain language: lead with the answer, then the supporting numbers as a small Markdown table when there are several. Always state the exact filters and groupings you used. If a result was truncated, say so.

Rules:
- Never invent, estimate or round into existence any number: every figure in your answer must come from a tool result in this conversation.
- Your only data tools are load_csv, describe_columns, filter_rows and group_by_aggregate, plus the google_drive_mcp connector tools to find and read Drive files. You cannot and must not run code: never call run_typescript or write programs — it is denied. If a question needs something the tools cannot do, say so plainly and offer the closest thing you can compute.
- If no CSV is loaded yet, ask the user for the Drive link.
- This conversation keeps its full history and the active dataset: follow-ups like "and what about last month?" or "how many of those?" refer to what was already loaded and discussed, so do not ask the user to re-share the file. If the user shares a new link, call load_csv again; it replaces the active dataset.
- If a column or value does not exist, list the columns that do exist (from describe_columns or load_csv) and ask for clarification.
- Keep answers concise and factual.
"""

class NoCodeExecution(agents.Hook):
    async def pre_tool_call(self, ctx, call):
        if call.tool_name == "run_typescript" or call.tool_id.startswith("self."):
            return agents.skip(
                "Code execution is disabled for this agent: run_typescript and the program helpers are "
                "blocked. Use only the fixed tools load_csv, describe_columns, filter_rows, "
                "group_by_aggregate, and the google_drive_mcp connector tools to find Drive files."
            )


NO_CODE_EXECUTION_POLICY = agents.PermissionPolicy(
    default="allow",
    rules=[agents.PermissionRule(tools=("self.*",), decision="deny")],
)

agent = agents.Agent(
    model="mistral-large-4",
    name="drive-csv-analyst",
    description=(
        "Conversational data analyst for CSV files in Google Drive. Loads the file through the "
        "user's Google Drive connection and answers questions with fixed tools only "
        "(load_csv, describe_columns, filter_rows, group_by_aggregate) — no code execution."
    ),
    instructions=INSTRUCTIONS,
    tools=[load_csv, describe_columns, filter_rows, group_by_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
    builtins=agents.Builtins(skills=False, subagents=False, background_processes=False),
    permission_mode="auto",
    permission_policy=NO_CODE_EXECUTION_POLICY,
    guardrails=[NoCodeExecution()],
    max_iterations=30,
)


async def _demo() -> None:
    state = await agent.run(
        "Here is my CSV: https://drive.google.com/file/d/YOUR_FILE_ID/view — "
        "load it, describe the columns, then tell me the row count."
    )
    print(state.text)


if __name__ == "__main__":
    asyncio.run(_demo())
