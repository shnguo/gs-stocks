from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def render_report(run: Path) -> Path:
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "run.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Cannot report an incomplete experiment")
    from .storage import file_hash
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_hash(run / name) != expected:
            raise ValueError("Experiment artifact hash mismatch")
    rows, paths = [], []
    colors = ["#64748b", "#007c83", "#6554c0", "#b36e14", "#ca5373"]
    series = []
    for name, metric in summary["metrics"].items():
        portfolio = summary["portfolios"].get(name)
        ic = metric["mean_rank_ic"]
        trading_ic = summary.get("trading_universe_metrics", summary["metrics"])[name]["mean_rank_ic"]
        ic_text = "样本不足" if ic is None else f"{ic:.4f}"
        trading_ic_text = "样本不足" if trading_ic is None else f"{trading_ic:.4f}"
        start = f"<tr><th>{html.escape(name)}</th><td>{ic_text}</td><td>{trading_ic_text}</td>"
        if portfolio is None:
            rows.append(start + "<td colspan=\"4\">未运行组合模拟</td></tr>")
            continue
        rows.append(start + f"<td>{portfolio['total_return']:.2%}</td>"
                    f"<td>{portfolio['max_drawdown']:.2%}</td><td>{portfolio['fills']}</td>"
                    f"<td>{portfolio['fees']:.2f}</td></tr>")
        nav = pd.read_parquet(run / f"{name}-nav.parquet")
        capital = manifest["identity"]["config"]["portfolio"]["initial_cash"]
        series.append((name, nav.nav.to_numpy() / capital))
    lower = min((min(values) for _, values in series), default=0)
    upper = max((max(values) for _, values in series), default=0)
    for color, (name, values) in zip(colors, series):
        points = " ".join(f"{45 + 810 * i / max(1, len(values)-1):.1f},"
                          f"{220 - 185 * (v-lower) / max(1e-6, upper-lower):.1f}"
                          for i, v in enumerate(values))
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2">'
                     f'<title>{html.escape(name)}</title></polyline>')
    labels = " ".join(f'<span style="color:{c}">{html.escape(n)}</span>'
                      for c, (n, _) in zip(colors, series))
    chart = (f'<h2>模拟净值</h2><p>{labels}</p><svg viewBox="0 0 900 260" role="img" '
             f'aria-label="各模型在工程测试数据上的净值曲线"><text x="0" y="35">{upper:.3f}</text>'
             f'<text x="0" y="220">{lower:.3f}</text>{"".join(paths)}'
             f'<text x="45" y="252">{nav.date.iloc[0]}</text>'
             f'<text x="855" y="252" text-anchor="end">{nav.date.iloc[-1]}</text></svg>'
             if series else '<p>本次仅运行模型训练和排序评估。组合模拟需另行验收交易数据。</p>')
    blocker_labels = {
        "actions_verified_missing": "分红、送转等公司行动尚未核验",
        "historical_information_vintage_unverified": "历史数据在当时是否可用尚未核验",
        "historical_universe_verified_missing": "包含退市股票的历史证券池尚未核验",
        "industry_history_verified_missing": "历史行业归属尚未核验",
        "insufficient_multiyear_coverage": "尚未具备足够长的历史覆盖",
        "rules_verified_missing": "各时期交易规则和费用尚未核验",
        "st_history_verified_missing": "历史 ST 状态尚未完整核验",
        "risk_status_unknown_in_expected_universe": "非 ST 模拟候选池存在风险状态未知的证券日",
        "training_snapshot_has_no_execution_contract": "此快照仅用于训练，尚未提供成交模拟数据契约",
        "synthetic_data_is_engineering_validation_only": "本次数据为人工合成，仅用于工程验证",
        "historical_industry_portfolio_engine_pending": "回测尚需接入随时间变化的行业归属",
        "final_holdout_and_multiyear_walkforward_pending": "多年滚动验证和最终保留集尚未执行",
        "missing_label_endpoints_need_censoring_review": "部分样本缺少未来执行价格，需处理缺失偏差",
    }
    blockers = "".join(f"<li>{html.escape(blocker_labels.get(x, x))}</li>"
                       for x in summary["formal_blockers"])
    inclusive = summary.get("training_risk_policy", "exclude_st") == "include"
    pool_text = ("训练、验证及排序评估包含 ST、*ST 和风险状态未知的股票。"
                 "状态只用于审计，不输入模型。历史 ST 完整性不作为该训练池的数据门槛。"
                 if inclusive else "训练、验证及排序评估均按当时已知状态排除 ST、*ST 和未知状态。")
    pool_text += ("训练池 Rank IC 与已确认非 ST 信号子集 Rank IC 分列。"
                  if summary.get("training_only") else
                  "模拟买入仅限当时已确认正常的股票，并在执行前复核；"
                  "训练池 Rank IC 与非 ST 信号子集 Rank IC 分列，净值对应受限模拟组合。")
    risk_counts = summary.get("partition_risk_status_counts", {})
    counts_text = "；".join(f"{label}：" + "、".join(
        f"{html.escape(status)} {count}" for status, count in risk_counts.get(key, {}).items())
        for key, label in [("train", "训练"), ("validation", "验证"), ("test", "测试")]
        if key in risk_counts)
    source_counts = summary.get("partition_source_st_counts", {})
    source_counts_text = "；".join(
        f"{label}的来源 ST 标记 {source_counts[key]['flagged_st']} 条"
        for key, label in [("train", "训练"), ("validation", "验证"), ("test", "测试")]
        if key in source_counts)
    source_counts_text += ("。来源日标记仅供核对，不等同于已核验的历史公告可用时间。"
                           if source_counts else "")
    coverage = summary.get("dataset_coverage", {})
    coverage_text = (f"本次数据覆盖 {coverage['instruments']} 只股票、{coverage['bar_rows']} 条日线，"
                     f"从 {coverage['first_date']} 至 {coverage['last_date']}。"
                     "全 A 股覆盖尚未验收。" if coverage else "")
    label_coverage = summary.get("test_label_coverage", {})
    label_text = (f"测试评分保留 {label_coverage['scored_samples']} 条样本，"
                  f"其中 {label_coverage['available_labels']} 条有可核验标签，"
                  f"{label_coverage['unavailable_labels']} 条标签不可用。"
                  "排名指标只使用结果已知的样本，缺失结果可能造成偏差，不能忽略。"
                  if label_coverage else "")
    portfolio_text = ("已保存模型权重、预测、样本审计、训练日志及排序指标。"
                      if summary.get("training_only") else
                      "已保存逐日预测、过滤原因、成交、被阻止的订单、持仓、现金、费用、训练日志及两倍成本和延迟一天的压力测试。")
    limits_text = ("本次使用训练快照；交易费用、行业约束和完整风险状态尚不作为训练输入。"
                  if summary.get("training_only") else
                  f"本次模拟最多持有 {manifest['identity']['config']['portfolio']['max_positions']} 只股票，"
                  f"单只初始权重上限 {manifest['identity']['config']['portfolio']['max_weight']:.0%}。")
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>A 股模型工程验证</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#18232e;background:#f4f6f8;margin:0}}
main{{max-width:1040px;margin:40px auto;padding:32px;background:white;border-radius:12px}}
h1{{font-size:28px}}h2{{font-size:20px}}.notice{{padding:16px;background:#fff4d6;border-left:4px solid #b88020}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}td,th{{text-align:right;padding:12px;border-bottom:1px solid #e3e7eb}}th:first-child{{text-align:left}}
.meta{{color:#596776;overflow-wrap:anywhere}}svg{{width:100%;height:auto}}span{{margin-right:20px}}li{{overflow-wrap:anywhere}}</style>
<main><p class="meta">A 股 · 全市场排序研究框架 · {summary['horizon']} 个交易日</p>
<h1>模型工程验证已跑通</h1><p class="notice">本页使用{'人工合成数据' if summary['synthetic'] else '未通过正式验收的研究数据'}。
指标用于检查研究流程，不能作为投资表现或 Transformer 有效性的证据。</p>
<p>训练 {summary['partition_rows']['train']} 条，验证 {summary['partition_rows']['validation']} 条，
测试 {summary['partition_rows']['test']} 条。模型仅按验证集选择；未使用最终保留集。</p>
<p>{pool_text}</p><p class="meta">{counts_text}</p>
<p class="meta">{coverage_text}</p><p class="meta">{source_counts_text}</p><p>{label_text}</p>
<p>{limits_text}</p>
<h2>同一数据范围的模型对照</h2><table><thead><tr><th>模型</th><th>训练池 Rank IC</th><th>非 ST 子集 Rank IC</th><th>区间收益</th><th>最大回撤</th><th>成交笔数</th><th>费用</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><p class="meta">工程数据上的结果，不用于模型排名或实盘决策。</p>
{chart}
<h2>正式实验尚未满足的条件</h2><ul>{blockers}</ul>
<p>{portfolio_text}</p>
<p class="meta">数据版本：{html.escape(manifest['identity']['dataset_id'])}<br>实验版本：{html.escape(manifest['run_id'])}</p></main></html>'''
    path = run / "report.html"
    path.write_text(text)
    return path
