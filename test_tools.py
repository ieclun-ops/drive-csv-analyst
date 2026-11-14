import asyncio
import base64

from mistralai import agents

import main

SAMPLE_CSV = """order_id,region,category,amount,quantity,date
1,France,books,12.50,2,2026-01-05
2,France,tech,199.99,1,2026-01-07
3,Germany,books,8.00,1,2026-01-09
4,Germany,tech,49.50,3,2026-01-12
5,Spain,food,23.10,4,2026-01-15
6,France,food,15.75,2,2026-02-02
7,Spain,tech,320.00,1,2026-02-04
8,Germany,food,9.99,1,2026-02-06
"""


async def fake_fetch(file_id: str) -> tuple[str, str, str]:
    return "sample.csv", "text/csv", SAMPLE_CSV


def test_helpers() -> None:
    frame = main._parse_csv(SAMPLE_CSV)
    assert list(frame.columns) == ["order_id", "region", "category", "amount", "quantity", "date"]
    assert len(frame) == 8
    assert main._extract_file_id("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing") == "1AbCdEfGhIjKlMnOp"
    assert main._extract_file_id("https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOp/edit#gid=0") == "1AbCdEfGhIjKlMnOp"
    assert main._extract_file_id("https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOp&export=download") == "1AbCdEfGhIjKlMnOp"

    spec = main.FilterSpec(column="region", operator="==", value="France")
    assert int(main._apply_filters(frame, [spec]).shape[0]) == 3
    spec = main.FilterSpec(column="amount", operator=">=", value=100)
    assert int(main._apply_filters(frame, [spec]).shape[0]) == 2
    spec = main.FilterSpec(column="category", operator="contains", value="TECH")
    assert int(main._apply_filters(frame, [spec]).shape[0]) == 3
    spec = main.FilterSpec(column="region", operator="in", value=["Spain", "Germany"])
    assert int(main._apply_filters(frame, [spec]).shape[0]) == 5
    spec = main.FilterSpec(column="amount", operator="<", value="10")
    assert int(main._apply_filters(frame, [spec]).shape[0]) == 2
    print("helpers OK")


async def test_agent_scripted() -> None:
    main._fetch_drive_text = fake_fetch  # type: ignore[assignment]
    scripted = agents.testing.ScriptedModel(
        [
            agents.testing.call("load_csv", drive_link="https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view"),
            agents.testing.call("describe_columns"),
            agents.testing.run_typescript("const r = await tools.client.group_by_aggregate({group_by: ['region'], aggregations: []});"),
            agents.testing.call(
                "group_by_aggregate",
                group_by=["region"],
                aggregations=[
                    {"function": "sum", "column": "amount"},
                    {"function": "count"},
                    {"function": "mean", "column": "amount"},
                ],
                filters=[{"column": "category", "operator": "!=", "value": "books"}],
            ),
            agents.testing.call(
                "filter_rows",
                filters=[{"column": "region", "operator": "==", "value": "France"}],
                limit=10,
            ),
            agents.testing.reply("Done. Loaded 8 rows; France has 2 non-book rows; totals per region computed."),
        ]
    )
    test_agent = agents.Agent(
        model="mistral-large-4",
        name="test-agent",
        instructions=main.INSTRUCTIONS,
        tools=[main.load_csv, main.describe_columns, main.filter_rows, main.group_by_aggregate],
        builtins=agents.Builtins(skills=False, subagents=False, background_processes=False),
        permission_mode="auto",
        permission_policy=main.NO_CODE_EXECUTION_POLICY,
        guardrails=[main.NoCodeExecution()],
    ).override(model=scripted)
    state = await test_agent.run("Load my CSV and summarize it per region.")
    assert isinstance(state, agents.Completed), state
    calls = [c.name for c in state.tool_calls]
    print("tool calls:", calls)
    assert calls == [
        "client.load_csv",
        "client.describe_columns",
        "client.group_by_aggregate",
        "client.filter_rows",
    ], calls
    assert "run_typescript" not in calls, "code-execution tool must never run"

    class _Call:
        tool_name = "run_typescript"
        tool_id = "self.run_typescript"
        arguments = {"code": "1+1"}

    decision = await main.NoCodeExecution().pre_tool_call(None, _Call())
    assert decision is not None and "disabled" in str(decision).lower()
    print("guardrail blocks run_typescript OK")
    print("scripted agent run OK:", state.text)


def test_tool_results() -> None:
    import pandas as pd

    frame = main._parse_csv(SAMPLE_CSV)
    ctx = agents.ToolCallbackContext.local()
    main._DATASETS[ctx.session.id] = main._Dataset("sample.csv", "link", frame)

    async def run() -> None:
        described = await main.describe_columns(ctx)
        assert described["rows"] == 8 and described["column_count"] == 6
        amount = next(c for c in described["columns"] if c["name"] == "amount")
        assert amount["min"] == 8.0 and amount["max"] == 320.0

        filtered = await main.filter_rows(ctx, filters=[{"column": "region", "operator": "==", "value": "France"}])
        assert filtered["matched"] == 3 and filtered["returned"] == 3

        agg = await main.group_by_aggregate(
            ctx,
            group_by=["region"],
            aggregations=[{"function": "sum", "column": "amount"}, {"function": "count"}],
        )
        assert agg["groups"] == 3
        rows = {r["region"]: r for r in agg["rows"]}
        assert abs(rows["France"]["sum(amount)"] - (12.50 + 199.99 + 15.75)) < 1e-9
        assert rows["France"]["count"] == 3
        assert rows["Spain"]["count"] == 2 and rows["Germany"]["count"] == 3

        overall = await main.group_by_aggregate(
            ctx, group_by=[], aggregations=[{"function": "sum", "column": "amount"}, {"function": "count"}]
        )
        assert overall["groups"] == 1
        assert abs(overall["rows"][0]["sum(amount)"] - frame["amount"].sum()) < 1e-9
        assert overall["rows"][0]["count"] == 8

        distinct = await main.group_by_aggregate(
            ctx, group_by=["category"], aggregations=[{"function": "count_distinct", "column": "region"}]
        )
        rows = {r["category"]: r for r in distinct["rows"]}
        assert rows["books"]["count_distinct(region)"] == 2

        try:
            await main.filter_rows(ctx, filters=[{"column": "nope", "operator": "==", "value": "x"}])
            raise AssertionError("expected ToolError")
        except agents.ToolError as error:
            assert "Unknown column" in str(error)

        empty_ctx = agents.ToolCallbackContext.local()
        try:
            await main.describe_columns(agents.ToolCallbackContext(agents.CallbackSession(id="other", turn_id="t"), tool_call_id="c"))
            raise AssertionError("expected ToolError")
        except agents.ToolError as error:
            assert "No CSV is loaded" in str(error)
        _ = empty_ctx
        print("tool results OK")

    asyncio.run(run())


if __name__ == "__main__":
    test_helpers()
    test_tool_results()
    asyncio.run(test_agent_scripted())
    print("ALL TESTS PASSED")
