"""Build and execute the portable notebook and its local HTML preview."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def build_notebook(root: Path, config_path: Path | None = None):
    config_path = config_path or root / "configs/pilot-v1.json"
    config = json.loads(config_path.read_text())
    out = root / "artifacts" / config.get("artifact_subdir", "")
    config_relative = str(config_path.relative_to(root))
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "US earnings research (Python 3.12)",
                                    "language": "python", "name": "python3"}
    notebook.cells = [
        markdown("# US earnings strategy: reproducible pilot\n\n## tl;dr\n\n"
                 "This notebook reruns the registered strategy from cached original disclosures and price data. "
                 "It makes no network requests and places no orders."),
        code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import Markdown, display
from us_earnings.research import run_research
get_ipython().run_line_magic("matplotlib", "inline")

ROOT = Path.cwd()
if not (ROOT / "configs/pilot-v1.json").exists():
    ROOT = ROOT.parent
CONFIG_FILE = ROOT / "CONFIG_PATH_PLACEHOLDER"
result = run_research(ROOT, CONFIG_FILE)
later = result["summary"].loc[result["summary"].period.str.contains("later evaluation")].set_index("arm")
guidance = later.loc["guidance"]
ml = later.loc["ml"]
market = later.loc["spy"]
display(Markdown(
    f"**2024–2025 annualized returns after assumed costs:** raised guidance + price "
    f"{guidance.cagr:.2%}; ML guidance + price {ml.cagr:.2%}; SPY {market.cagr:.2%}. "
    f"The guidance strategy averaged {guidance.average_exposure:.1%} invested exposure "
    f"and closed {int(guidance.trades)} trades. These portfolios have different exposure and risk.\\n\\n"
    "**Capital decision: defer allocation.** This is a convenience-sample historical pilot, "
    "with weak text labels and no prospective performance record. The local ML model is a "
    "TF-IDF logistic classifier, not a generative LLM."
))'''),
        markdown("## Context & Methods\n\n"
                 "The question is whether an earnings event and a raised-guidance filter improve a simple price rule. "
                 "The portfolio starts at one normalized unit; no account-size assumption is needed.\n\n"
                 "### Key Assumptions\n\n"
                 "- Fixed sample of 24 surviving US companies; this is not a historical index universe.\n"
                 "- Research window: 2019–2025; development: 2019–2022; validation: 2023; later evaluation: 2024–2025.\n"
                 "- An event becomes a signal at the close of the first benchmark session strictly after its SEC filing date. "
                 "Entry occurs at the next session open. Actual earnings publication may have been earlier.\n"
                 "- Price confirmation: close above a trailing 50-session average, with that session's return above SPY's return.\n"
                 "- Position target: 10% of current opening NAV, up to 10 positions, no leverage. "
                 "Candidates rank by prior-session relative return, with ticker as a tie-breaker.\n"
                 "- Exit: close of the twentieth holding session, or a fixed stop two trailing 20-session ATRs below the entry open. "
                 "An opening gap through a stop fills at the worse open, before costs. Entry-day stops are allowed.\n"
                 "- Costs: 10 basis points each side; diagnostics use 5, 20 and 50 basis points. Cash earns zero; taxes are excluded.\n"
                 "- Alpaca SIP prices are adjusted for splits, cash dividends and spin-offs. Returns use fractional adjusted units. "
                 "Stops are total-return-unit approximations, not reconstructed historical broker orders.\n"
                 "- The lexical baseline detects explicit raised annual guidance language. Numeric-only revisions can be missed.\n"
                 "- The ML classifier fits only pre-2023 disclosure text and lexical weak labels. It never trains on returns. "
                 "The fixed 0.5 score threshold is not a calibrated probability.\n"
                 "- The later period is not a fully blind holdout: the universe was chosen today and text parsing was audited "
                 "across years. Versioned parsing repairs followed initial results. Sensitivities are diagnostics, not strategy selection."),
        code('''config = result["config"]
display(pd.DataFrame({"Parameter": ["Holding sessions", "Stop ATR multiple", "Cost per side (bps)",
                                   "Maximum positions", "Target position fraction"],
                      "Value": [config["holding_sessions"], config["stop_atr_multiple"],
                                config["cost_bps_per_side"], config["max_positions"],
                                config["position_fraction"]]}))'''),
        markdown("## Data\n\n"
                 "Original SEC Item 2.02 filings provide event dates and earnings exhibits. "
                 "Alpaca consolidated SIP bars provide daily OHLC. Raw responses, URLs, fetch times and SHA-256 hashes "
                 "are retained locally. Missing exhibits stay in the coverage denominator and cannot enter event strategies.\n\n"
                 "Source documentation: [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), "
                 "[Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)."),
        code('''manifest = result["manifest"]
display(Markdown(f"**Price rows:** {manifest['prices']['rows']:,}; "
                 f"**catalog filings:** {manifest['events']['catalog_filings']:,}; "
                 f"**retrieved exhibits:** {manifest['events']['retrieved']:,}; "
                 f"**explicit guidance raises:** {manifest['events']['explicit_raises']:,}."))
display(result["coverage"].rename(columns={"filings": "Catalog filings", "retrieved": "Retrieved",
                                          "explicit_raises": "Explicit raises", "missing": "Missing"}))'''),
        markdown("### Source preview\n\nThe preview preserves exact event identity and original-document links. "
                 "It is a sample, not a validation of every extracted label."),
        code('''preview = result["events"].loc[result["events"].raised_guidance,
                                  ["ticker", "filingDate", "accessionNumber", "guidance_state", "source_url"]].head(5)
source_lines = ["| Ticker | Filed | Accession | Guidance state | Original document |",
                "|---|---|---|---|---|"]
for row in preview.itertuples():
    source_lines.append(f"| {row.ticker} | {row.filingDate} | {row.accessionNumber} | "
                        f"{row.guidance_state} | [SEC exhibit]({row.source_url}) |")
display(Markdown("\\n".join(source_lines)))
display(pd.DataFrame([manifest["text_model"]], columns=["training_rows", "positive_training_rows",
                                                       "training_last_date", "later_predictions", "later_positives"]))'''),
        markdown("## Results\n\n### Portfolio comparison\n\n"
                 "These are funded, cash-constrained portfolio results. SPY and the sample buy-and-hold portfolio are fully "
                 "invested, so their returns are not risk-matched to sparse earnings strategies. Trade counts below refer to "
                 "trades closing in the selected period, including positions carried from the previous year."),
        code('''columns = ["label", "cagr", "max_drawdown", "average_exposure", "trades", "win_rate"]
table = later.reset_index()[columns].rename(columns={"label": "Strategy", "cagr": "Annualized return",
    "max_drawdown": "Maximum drawdown", "average_exposure": "Average exposure", "trades": "Closed trades",
    "win_rate": "Winning trades"})
display(table.style.format({"Annualized return": "{:.2%}", "Maximum drawdown": "{:.2%}",
                           "Average exposure": "{:.1%}", "Closed trades": "{:.0f}",
                           "Winning trades": "{:.1%}"}, na_rep="—").hide(axis="index"))'''),
        code('''import matplotlib.pyplot as plt
from us_earnings.figures import equity_figure, event_figure, exposure_figure
OUT = ROOT / "artifacts" / result["config"].get("artifact_subdir", "")
display(equity_figure(result, OUT))
plt.close("all")'''),
        markdown("The common plot starts in 2023 because the ML model is trained through 2022. "
                 "Every curve is rebased to wealth at the start of 2023. Earlier baseline histories remain in the CSV ledgers. "
                 "The sample passive portfolio retains the weights that evolved from its 2019 purchases."),
        markdown("### Comparable event horizons\n\n"
                 "For each price-confirmed earnings event, compare the stock's next-open-to-twentieth-close return with SPY "
                 "over the identical dates, applying the same assumed costs to both. These observations have no stops or "
                 "portfolio capacity constraints. They are not annualized or treated as independent trades. "
                 "Intervals resample whole signal months 2,000 times; they do not eliminate all dependence."),
        code('''display(event_figure(result, OUT))
plt.close("all")
display(result["intervals"].loc[result["intervals"].period.str.contains("later evaluation")]
        [["filter", "events", "months", "mean_excess", "lower_95", "upper_95"]]
        .style.format({"mean_excess": "{:+.2%}", "lower_95": "{:+.2%}", "upper_95": "{:+.2%}"}, na_rep="Unavailable").hide(axis="index"))'''),
        markdown("### Exposure and concentration\n\n"
                 "A low drawdown can follow from holding substantial cash. Issuer concentration and the number of events "
                 "matter when deciding whether a promising average is repeatable."),
        code('''display(exposure_figure(result, OUT))
plt.close("all")'''),
        markdown("### Cost and holding-period diagnostics\n\n"
                 "The registered primary horizon remains 20 sessions. These diagnostic variants are not used to choose "
                 "a winning configuration. ML cost results cover 2023–2025; other cost results cover 2019–2025."),
        code('''display(result["stress"][["arm", "cost_bps_per_side", "cagr", "max_drawdown", "trades"]]
        .sort_values(["arm", "cost_bps_per_side"]).style.format(
            {"cagr": "{:.2%}", "max_drawdown": "{:.2%}"}).hide(axis="index"))
display(pd.read_csv(OUT / "horizon-sensitivity.csv")
        [["arm", "holding_sessions", "cagr", "max_drawdown", "trades"]]
        .style.format({"cagr": "{:.2%}", "max_drawdown": "{:.2%}"}).hide(axis="index"))'''),
        markdown("### Inspect a trade and reconcile accounting"),
        code('''ledger = result["ledgers"]["guidance"]
display(ledger.head(8)[["ticker", "signal_date", "entry_date", "exit_date", "net_return", "exit_reason", "accession"]])
checks = []
for arm, trades in result["ledgers"].items():
    curve = result["curves"][arm]
    residual = float(curve.nav.iloc[-1] - (1 + trades.pnl.sum()))
    assert abs(residual) < 1e-9, (arm, residual)
    assert curve.cash.min() >= -1e-9
    assert curve.positions.max() <= config["max_positions"]
    assert (trades.entry_date > trades.signal_date).all()
    checks.append({"Strategy": arm, "Ending NAV minus 1 + realized P&L": residual,
                   "Maximum positions": int(curve.positions.max()), "Closed trades": len(trades)})
display(pd.DataFrame(checks).style.format({"Ending NAV minus 1 + realized P&L": "{:.2e}"}).hide(axis="index"))'''),
        markdown("## Takeaways\n\n"
                 "The pipeline supports reproducible historical comparisons, not a claim of profitable AI trading. "
                 "The main unresolved issues are survivorship bias, incomplete numeric guidance extraction, weak ML labels, "
                 "adjusted-unit execution assumptions, and absence of a prospective record.\n\n"
                 "Before deciding capital: obtain a historical universe including delisted firms; build a blinded human "
                 "label sample with normalized guidance periods and accounting bases; replay raw corporate actions and fills; "
                 "then freeze the strategy for prospective paper evaluation. "
                 "Changing rules after reading these results requires a new protocol and a new evaluation sample.\n\n"
                 "All figures, trade ledgers, event audits, source failures and the environment lockfile accompany this notebook. "
                 "Cached source data and fitted models are local artifacts and are excluded from Git."),
    ]
    for cell in notebook.cells:
        cell.source = cell.source.replace("CONFIG_PATH_PLACEHOLDER", config_relative)
    if "parser_repair_note" in config:
        notebook.cells.insert(3, markdown("### Version history\n\n" + config["parser_repair_note"] +
                                          " The initial v1 results remain in the parent artifacts directory."))
    suffix = "-v2" if config.get("artifact_subdir") == "pilot-v2" else ""
    path = root / f"notebooks/us-earnings-pilot{suffix}.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    manager = KernelManager(kernel_name="python3", transport="ipc")
    manager.kernel_spec.argv = [str(root / ".venv/bin/python"), "-m", "ipykernel_launcher",
                                "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=300, km=manager,
                            resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    html, _ = HTMLExporter(template_name="lab", exclude_input=True).from_notebook_node(notebook)
    preview = out / "us-earnings-pilot.html"
    preview.write_text(html)
    validation = {"executed_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                  "errors": sum(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code"
                                for output in cell.outputs), "notebook": str(path), "html": str(preview)}
    (out / "notebook-validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    return path
