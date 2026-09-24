from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


OUTPUT = Path("/Users/guo/Documents/stocks/tencent_xiaomi_comparison_2026-08-19.ipynb")

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """## tl;dr

- 长期基本面优先选择腾讯：二季度收入和核心利润仍同比增长，而小米同期收入与调整后净利润均下滑。
- 2026年8月19日11:34 HKT盘中，小米的量价和相对五日线位置明显强于腾讯，但属于财报后跳空，不宜追涨。
- 腾讯当前仅略高于正式五日线，且五日线仍向下；按收盘确认规则，尚不是高质量买点。"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

比较对象为腾讯控股 0700.HK 与小米集团 1810.HK。基本面采用两家公司截至2026年6月30日止季度的正式披露；行情采用腾讯公开港股行情及前复权日线接口，时间戳为2026年8月19日11:34 HKT。

### Key Assumptions

- 核心利润口径分别为腾讯非国际财务报告准则归母净利润、小米调整后净利润。
- 正式MA5使用截至2026年8月18日的五个已完成交易日收盘价；动态MA5使用前四个已完成收盘价加盘中价。
- 盘中信号必须等待收盘确认，不能直接视作正式买点。"""
    ),
    nbf.v4.new_markdown_cell("## Data\n\n### 1. Load reviewed fundamentals and market observations"),
    nbf.v4.new_code_cell(
        """import pandas as pd

as_of_hkt = "2026-08-19 11:34 HKT"

fundamentals = pd.DataFrame([
    {
        "company": "腾讯",
        "ticker": "0700.HK",
        "quarter": "2026Q2",
        "revenue_rmb_b": 204.785,
        "revenue_yoy_pct": 11.0,
        "adjusted_profit_rmb_b": 68.415,
        "adjusted_profit_yoy_pct": 9.0,
        "gross_margin_pct": 57.8,
        "capex_rmb_b": 52.8,
        "reported_fcf_rmb_b": -13.8,
        "fcf_ex_compute_prepay_rmb_b": 37.6,
    },
    {
        "company": "小米",
        "ticker": "1810.HK",
        "quarter": "2026Q2",
        "revenue_rmb_b": 108.9216,
        "revenue_yoy_pct": -6.1,
        "adjusted_profit_rmb_b": 6.2191,
        "adjusted_profit_yoy_pct": -42.6,
        "gross_margin_pct": 19.8,
        "capex_rmb_b": 3.6,
        "reported_fcf_rmb_b": None,
        "fcf_ex_compute_prepay_rmb_b": None,
    },
])

tencent_completed_closes = [461.6, 441.0, 440.0, 446.4, 442.4]
xiaomi_completed_closes = [26.36, 25.88, 25.62, 25.88, 26.18]
tencent_prior_ma5_closes = [470.8, 461.6, 441.0, 440.0, 446.4]
xiaomi_prior_ma5_closes = [26.66, 26.36, 25.88, 25.62, 25.88]

quotes = pd.DataFrame([
    {
        "company": "腾讯",
        "ticker": "0700.HK",
        "current_hkd": 446.4,
        "previous_close_hkd": 442.4,
        "intraday_change_pct": 0.90,
        "current_volume_m": 8.622627,
        "previous_full_day_volume_m": 23.218078,
        "completed_closes": tencent_completed_closes,
        "prior_ma5_closes": tencent_prior_ma5_closes,
        "prior_four_closes": [441.0, 440.0, 446.4, 442.4],
    },
    {
        "company": "小米",
        "ticker": "1810.HK",
        "current_hkd": 28.04,
        "previous_close_hkd": 26.18,
        "intraday_change_pct": 7.10,
        "current_volume_m": 213.701849,
        "previous_full_day_volume_m": 179.217011,
        "completed_closes": xiaomi_completed_closes,
        "prior_ma5_closes": xiaomi_prior_ma5_closes,
        "prior_four_closes": [25.88, 25.62, 25.88, 26.18],
    },
])

fundamentals"""
    ),
    nbf.v4.new_markdown_cell("## Results\n\n### 2. Compute MA5 status and volume context"),
    nbf.v4.new_code_cell(
        """quotes["formal_ma5_hkd"] = quotes["completed_closes"].apply(lambda x: sum(x) / 5)
quotes["prior_formal_ma5_hkd"] = quotes["prior_ma5_closes"].apply(lambda x: sum(x) / 5)
quotes["dynamic_ma5_hkd"] = quotes.apply(
    lambda r: (sum(r["prior_four_closes"]) + r["current_hkd"]) / 5,
    axis=1,
)
quotes["distance_to_formal_ma5_pct"] = (
    quotes["current_hkd"] / quotes["formal_ma5_hkd"] - 1
) * 100
quotes["formal_ma5_slope_pct"] = (
    quotes["formal_ma5_hkd"] / quotes["prior_formal_ma5_hkd"] - 1
) * 100
quotes["intraday_vs_previous_full_day_volume_pct"] = (
    quotes["current_volume_m"] / quotes["previous_full_day_volume_m"]
) * 100

technical = quotes[[
    "company",
    "ticker",
    "current_hkd",
    "intraday_change_pct",
    "formal_ma5_hkd",
    "dynamic_ma5_hkd",
    "distance_to_formal_ma5_pct",
    "formal_ma5_slope_pct",
    "current_volume_m",
    "intraday_vs_previous_full_day_volume_pct",
]].copy()
technical.round(3)"""
    ),
    nbf.v4.new_markdown_cell("### 3. Compare growth and margin quality"),
    nbf.v4.new_code_cell(
        """comparison = fundamentals[[
    "company",
    "revenue_yoy_pct",
    "adjusted_profit_yoy_pct",
    "gross_margin_pct",
]].copy()

comparison["growth_spread_pct_points"] = (
    comparison["adjusted_profit_yoy_pct"] - comparison["revenue_yoy_pct"]
)
comparison.round(1)"""
    ),
    nbf.v4.new_markdown_cell("### 4. Reasonableness checks"),
    nbf.v4.new_code_cell(
        """checks = {
    "腾讯毛利率重算": round(118.433 / 204.785 * 100, 1),
    "小米毛利率重算": round(21.6089 / 108.9216 * 100, 1),
    "腾讯正式MA5": round(sum(tencent_completed_closes) / 5, 3),
    "小米正式MA5": round(sum(xiaomi_completed_closes) / 5, 3),
}
assert checks["腾讯毛利率重算"] == 57.8
assert checks["小米毛利率重算"] == 19.8
assert checks["腾讯正式MA5"] == 446.28
assert checks["小米正式MA5"] == 25.984
checks"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

- 腾讯的二季度经营质量明显更强：收入同比增长11%，核心利润同比增长9%，毛利率约57.8%。主要风险是AI投入使资本开支升至528亿元，报告自由现金流转为负138亿元；剔除算力预付款后仍为正376亿元。
- 小米二季度汽车收入同比增长17.1%，但手机与AIoT仍承压，集团收入同比下降6.1%，调整后净利润同比下降42.6%，手机毛利率降至8.5%，汽车、AI及其他新业务经营亏损26亿元。
- 盘中技术面由小米领先：小米高于正式MA5约7.9%，且盘中成交量已超过前一完整交易日；腾讯仅略高于正式MA5约0.03%，五日线仍向下。
- 因此，1至3年期只选一只时优先腾讯；短线若参与小米，应等待收盘确认并避免在财报后跳空位置追价。"""
    ),
]

nbf.write(nb, OUTPUT)
client = NotebookClient(nb, timeout=120, kernel_name="python3")
executed = client.execute(cwd=str(OUTPUT.parent))
nbf.write(executed, OUTPUT)
print(OUTPUT)
