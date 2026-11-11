"""Build and execute the bounded evidence companion, using notebook dependencies."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replace-owned-output", action="store_true",
                        help="Explicitly rebuild this script's notebook after a presentation repair")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "artifacts/edge-opportunity-20261012-v2"
    notebook_path = root / "notebooks/edge-opportunity-audit-20261012.ipynb"
    if notebook_path.exists() and not args.replace_owned_output:
        raise FileExistsError("Preserve the executed notebook; use a new version for another build")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook()
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook.cells = [
        markdown("# 20万元A股研究：证据审计\n\n## tl;dr\n\n"
                 "本地保存的有效前瞻成熟日期为0；现有重排没有确立收益优势。"
                 "优先验证新增盈利改善公告后的20日延续，但该假设尚无本系统实测结果。"
                 "本笔记本复算既有证据，不运行新的策略回测。"),
        markdown("## Context & Methods\n\n"
                 "决策是下一阶段研究资源投向哪里。资金为用户提供的20万元人民币。\n\n"
                 "### Key Assumptions\n\n"
                 "最近价格对照是迟发回顾，股票×预测日不是独立交易。"
                 "上涨初期CSV的execution_return_pct含预测选定日最高价，"
                 "其名字不能作为可执行收益证明。全股票池结论仅审阅原报告及审计回执。"
                 "新协议的10%回撤和20日窗口为提案，未获用户风险偏好确认。"),
        markdown("### 1. 读取有哈希的本地证据\n\n从quant-research目录运行。"),
        code("from pathlib import Path\nimport sys, json\n"
             "root = Path.cwd()\nassert (root / 'src/quant_research').is_dir()\n"
             "sys.path.insert(0, str(root / 'scripts'))\n"
             "from audit_edge_opportunity import audit\n"
             "evidence = audit(root)\n"
             "output_dir = root / 'artifacts/edge-opportunity-20261012-v2'\n"
             "print('行情截止:', evidence['price_observed_through'])\n"
             "print('保存的前瞻成熟日期:', evidence['saved_prospective_mature_dates'])\n"
             "print('复核通过项:', len(evidence['checks_passed']))"),
        markdown("## Data\n\n数据仅来自本地既有日报、逐日重排结果、"
                 "全股票池审计及源代码。没有刷新市场数据，也没有读取封存测试标签。"),
        code("from IPython.display import display, HTML, Image\nfrom html import escape\n"
             "def show_table(rows, columns):\n"
             "    head = ''.join('<th>' + escape(label) + '</th>' for _, label in columns)\n"
             "    body = ''\n"
             "    for row in rows:\n"
             "        values = []\n"
             "        for key, _ in columns:\n"
             "            value = row.get(key)\n"
             "            text = format(value, '.4f') if isinstance(value, float) else str(value)\n"
             "            values.append('<td>' + escape(text) + '</td>')\n"
             "        body += '<tr>' + ''.join(values) + '</tr>'\n"
             "    display(HTML('<table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>'))\n"
             "print('逐日重排窗口:', evidence['early_stage_results_recomputed'][0]['start'],"
             " evidence['early_stage_results_recomputed'][0]['end'])\n"
             "print('证据文件数:', len(evidence['sources']))"),
        markdown("## Results\n\n### 2. 最近收盘预测对照"),
        code("price_rows = evidence['price_results_recomputed']\n"
             "show_table(price_rows, [('arm', '榜单'), ('known_stock_forecast_day_rows', '可对照记录'),"
             " ('model_mape_pct', '模型MAPE %'), ('baseline_mape_pct', '原收盘基线MAPE %')])"),
        markdown("主榜误差较原收盘基线更高；上涨初期基本持平。"
                 "两个榜单的股票、日期群体不同，表间差值不能证明筛选的增量收益。"),
        markdown("### 3. 同日期重排的情景差值"),
        code("scenario_rows = [r for r in evidence['early_stage_results_recomputed'] if r['arm'] != 'control']\n"
             "show_table(scenario_rows, [('arm', '重排'), ('date_count', '历史日期'),"
             " ('paired_mean_change_pp', '相对对照变化 pp'), ('improved_date_fraction', '改善日期比例')])\n"
             "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
             "labels = {'early_stage':'Early-stage', 'overheat_penalty':'Overheat penalty', 'combined':'Combined'}\n"
             "ordered = sorted(scenario_rows, key=lambda row: row['paired_mean_change_pp'])\n"
             "fig, ax = plt.subplots(figsize=(9, 3.7), layout='constrained')\n"
             "values = [r['paired_mean_change_pp'] for r in ordered]\n"
             "bars = ax.barh([labels[r['arm']] for r in ordered], values, color='#406a80', height=0.56)\n"
             "ax.axvline(0, color='#333333', linewidth=1)\n"
             "ax.set_xlim(-1.35, 0.08)\n"
             "ax.set_xlabel('Paired scenario change vs control (percentage points)')\n"
             "ax.set_title('Reranking comparison | 196 historical dates, Jul 2024–Jul 2025')\n"
             "ax.spines[['top','right']].set_visible(False)\n"
             "ax.grid(axis='x', alpha=0.18)\nax.set_axisbelow(True)\n"
             "for bar, value in zip(bars, values):\n"
             "    ax.text(value - 0.025, bar.get_y() + bar.get_height()/2, format(value, '+.3f'),"
             " ha='right', va='center', fontsize=11)\n"
             "fig.savefig(output_dir / 'reranking-scenario-deltas.png', dpi=160)\n"
             "display(Image(filename=str(output_dir / 'reranking-scenario-deltas.png')))\n"
             "plt.close(fig)"),
        markdown("这些是下一日开盘到模型选定日最高价的情景差异，"
                 "不是账户净收益。原比较的1e-10个百分点平局容差已保持一致。"
                 "三种重排的平均情景差均为负。"),
        markdown("### 4. 正式收益验证的数据资格"),
        code("show_table([{'field': k, 'value': v} for k, v in evidence['input_status']['declaration'].items()],"
             " [('field', '声明项'), ('value', '当前保存值')])\n"
             "print('formal_ready:', evidence['input_status']['formal_ready'])\n"
             "print('纳入的及时发布记录:', sum(r['prospective'] for r in evidence['publications']))"),
        markdown("声明字段不能代替源证据。当前保存输入为训练用途，"
                 "不能据此通过真实成本后收益验收。"),
        markdown("## Takeaways\n\n"
                 "1. 保留模型预测误差改善的证据，停止把高低价情景解释为盈利。\n"
                 "2. 优先补新增业绩预告数值与真实可见时间，验证事件本身的增量。\n"
                 "3. 新实验先四组固定对照，20万元完整账户，20日为主要持有期。\n"
                 "4. 120交易日只审查运营，252交易日做唯一正式统计判定。\n\n"
                 "完整方案见docs/edge-opportunity-and-validation-20261012.md；"
                 "结构化协议仍为设计规范，尚未接入现有runner。"),
        markdown("### 来源与复算轨迹"),
        code("show_table(evidence['sources'], [('file', '证据文件'), ('sha256', 'SHA-256')])"),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (output / "notebook.html").write_text(html)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error"
                   for cell in code_cells for output in cell.outputs)
    print(f"Executed {len(code_cells)} cells: {notebook_path}")


if __name__ == "__main__":
    main()
