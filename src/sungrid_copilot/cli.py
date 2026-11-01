"""Command-line interface: `sungrid <command>` (or `python -m sungrid_copilot <command>`)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SunGrid Cooperative member-support copilot.")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_path=False, show_time=False)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "fastembed", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build(force_reindex: bool = False):
    from .factory import build

    with console.status("Loading copilot (first run downloads the embedding model and builds the index)..."):
        return build(force_reindex=force_reindex)


def _print_trace(resp) -> None:
    t = Table(title="trace", show_lines=False, header_style="bold")
    t.add_column("step")
    t.add_column("ms", justify="right")
    t.add_column("detail", overflow="fold")
    for s in resp.trace:
        detail = {k: v for k, v in s.items() if k not in ("step", "ms", "hits", "scores")}
        if "hits" in s:
            detail["hits"] = [f"{h['citation']} (ev={h['evidence']})" for h in s["hits"]]
        t.add_row(s["step"], f"{s['ms']:.1f}", json.dumps(detail, default=str, ensure_ascii=False)[:600])
    console.print(t)


def _print_response(resp, show_trace: bool) -> None:
    cat = resp.category.get("label", "-")
    sub = f"route={resp.route} · category={cat} ({resp.category.get('confidence', 0):.2f}) · {resp.latency_ms:.0f} ms"
    console.print(
        Panel(
            resp.answer,
            title="SunGrid Copilot",
            subtitle=sub,
            border_style="green" if resp.route != "abstain" else "yellow",
        )
    )
    if show_trace:
        _print_trace(resp)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show info logs.")) -> None:
    _setup_logging(verbose)


@app.command()
def info() -> None:
    """Show the resolved configuration (no secrets)."""
    s = get_settings()
    rows = {
        "LLM provider": s.resolved_llm_provider(),
        "LLM model": s.resolved_llm_model() or "-",
        "Embedding provider": s.embedding_provider,
        "Knowledge base": str(s.kb_path),
        "Index dir": str(s.index_path),
        "top_k": str(s.top_k),
    }
    t = Table(show_header=False)
    for k, v in rows.items():
        t.add_row(k, v)
    console.print(t)


@app.command()
def ingest(force: bool = typer.Option(False, "--force", help="Rebuild the index from scratch.")) -> None:
    """Chunk, embed and index the knowledge base (incremental unless --force)."""
    comps = _build(force_reindex=force)
    rep = comps.ingest_report
    if rep is None:
        console.print(f"[green]Index is up to date[/] ({comps.store.count()} chunks, embedder {comps.embedder.name}).")
        return
    console.print_json(json.dumps(rep.to_dict()))


@app.command()
def ask(
    question: str = typer.Argument(..., help="The member question."),
    trace: bool = typer.Option(False, "--trace", "-t", help="Print the agent trace."),
    as_json: bool = typer.Option(False, "--json", help="Print the full response as JSON."),
) -> None:
    """Ask a single question."""
    copilot = _build().copilot
    resp = copilot.ask(question)
    if as_json:
        console.print_json(json.dumps(resp.to_dict(), default=str))
    else:
        _print_response(resp, trace)


@app.command()
def chat(trace: bool = typer.Option(False, "--trace", "-t", help="Print the agent trace after each answer.")) -> None:
    """Interactive multi-turn chat. Commands: /trace, /reset, /quit."""
    copilot = _build().copilot
    sid = "cli"
    console.print(
        "[bold]SunGrid Copilot[/] — ask about membership, rebates, billing, installation or company news. "
        "Try: [italic]Am I eligible for the rooftop rebate?[/]  (/trace, /reset, /quit)"
    )
    while True:
        try:
            q = console.input("[bold cyan]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/trace":
            trace = not trace
            console.print(f"trace {'on' if trace else 'off'}")
            continue
        if q == "/reset":
            copilot.sessions.reset(sid)
            console.print("session cleared")
            continue
        _print_response(copilot.ask(q, sid), trace)


@app.command("eval")
def evaluate_cmd(
    out_dir: Path = typer.Option(Path("reports"), "--out", help="Where to write the report."),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if a quality gate fails (CI)."),
) -> None:
    """Run the golden-set evaluation and write reports/eval_report.md + eval_results.json."""
    from .evaluation import evaluate, render_markdown

    s = get_settings()
    copilot = _build().copilot
    with console.status("Evaluating..."):
        report = evaluate(
            copilot, s.data_path / "eval" / "golden_set.jsonl", s.data_path / "classifier" / "seed_queries.jsonl"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(report)
    (out_dir / "eval_report.md").write_text(md, encoding="utf-8")
    (out_dir / "eval_results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    console.print(Markdown(md.split("## Retrieval filter ablation")[0]))
    console.print(f"Full report: {out_dir / 'eval_report.md'}")
    if strict and not all(report["quality_gates"].values()):
        raise typer.Exit(code=1)


@app.command()
def audit() -> None:
    """Show how a seed-only classifier sees each document (surfaces ambiguous documents)."""
    from .evaluation import category_audit

    s = get_settings()
    copilot = _build().copilot
    rows = category_audit(copilot, s.data_path / "classifier" / "seed_queries.jsonl")
    t = Table(header_style="bold")
    for col in ("document", "manifest", "classifier top-2", "entropy"):
        t.add_column(col)
    for r in rows:
        t.add_row(
            r["doc_id"] + (" ⚠" if r["ambiguous_flag"] else ""),
            ", ".join(r["manifest"]),
            ", ".join(f"{k} {v:.2f}" for k, v in r["top2"]),
            str(r["entropy"]),
        )
    console.print(t)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the HTTP API (POST /ask, GET /health)."""
    import uvicorn

    uvicorn.run("sungrid_copilot.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
