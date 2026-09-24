from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


OUT = Path("/Users/guo/Documents/stocks/wuxi_cxo_peer_analysis_2026-08-10.ipynb")

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
        "# 药明康德与 CXO 同行比较\n\n"
        "快照时间：2026-08-10 11:00 北京时间。行情与估值来自东方财富实时行情接口；"
        "经营数据来自公司季度报告、业绩预告及港交所披露。"
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "\n"
        "peers = [\n"
        "    {'company':'药明康德','ticker':'603259.SH','price':161.20,'currency':'CNY','market_cap_bn':480.98,'ttm_pe':21.70,'q1_revenue_growth':28.81,'q1_core_profit_growth':83.56,'profit_basis':'扣非归母','policy_risk':'高','view':'核心候选'},\n"
        "    {'company':'凯莱英','ticker':'002821.SZ','price':176.48,'currency':'CNY','market_cap_bn':63.67,'ttm_pe':52.31,'q1_revenue_growth':16.91,'q1_core_profit_growth':-10.88,'profit_basis':'扣非归母；经调整归母增长27.91%','policy_risk':'中','view':'首选替代，待中报'},\n"
        "    {'company':'康龙化成','ticker':'300759.SZ','price':47.74,'currency':'CNY','market_cap_bn':87.71,'ttm_pe':65.39,'q1_revenue_growth':15.48,'q1_core_profit_growth':17.45,'profit_basis':'扣非归母','policy_risk':'中','view':'增长确认但估值偏贵'},\n"
        "    {'company':'泰格医药','ticker':'300347.SZ','price':55.19,'currency':'CNY','market_cap_bn':47.52,'ttm_pe':242.24,'q1_revenue_growth':15.17,'q1_core_profit_growth':17.65,'profit_basis':'扣非归母；TTM利润处低谷','policy_risk':'低至中','view':'反转观察'},\n"
        "    {'company':'昭衍新药','ticker':'603127.SH','price':52.62,'currency':'CNY','market_cap_bn':39.43,'ttm_pe':41.36,'q1_revenue_growth':10.02,'q1_core_profit_growth':747.11,'profit_basis':'扣非归母；受低基数和生物资产影响','policy_risk':'低至中','view':'高弹性但低质量'},\n"
        "    {'company':'ST诺泰','ticker':'688076.SH','price':28.52,'currency':'CNY','market_cap_bn':9.04,'ttm_pe':17.45,'q1_revenue_growth':-2.23,'q1_core_profit_growth':-17.78,'profit_basis':'扣非归母','policy_risk':'低至中','view':'回避，治理风险'},\n"
        "    {'company':'药明生物','ticker':'2269.HK','price':46.50,'currency':'HKD','market_cap_bn':192.74,'ttm_pe':35.47,'q1_revenue_growth':None,'q1_core_profit_growth':None,'profit_basis':'港股不披露季度业绩；2025调整净利增长22%','policy_risk':'高','view':'成长候选但非政策对冲'},\n"
        "]\n"
        "df = pd.DataFrame(peers)\n"
        "df"
    ),
    nbf.v4.new_code_cell(
        "assert df['company'].is_unique\n"
        "assert (df['ttm_pe'] > 0).all()\n"
        "assert (df['market_cap_bn'] > 0).all()\n"
        "q1 = df.dropna(subset=['q1_revenue_growth']).copy()\n"
        "assert len(q1) == 6\n"
        "q1 = q1.sort_values('q1_revenue_growth', ascending=False)\n"
        "q1[['company','q1_revenue_growth','q1_core_profit_growth','profit_basis']]"
    ),
    nbf.v4.new_code_cell(
        "valuation = df.sort_values('ttm_pe')[['company','ttm_pe','market_cap_bn','view']]\n"
        "valuation"
    ),
    nbf.v4.new_markdown_cell(
        "## 解释边界\n\n"
        "TTM 市盈率来自第三方行情口径，仅用于同一时点的粗筛；凯莱英受汇兑影响、泰格处于利润低谷、"
        "昭衍受生物资产公允价值影响，因此不能机械比较。药明生物没有季度披露，未纳入一季度营收增速图。"
    ),
]

NotebookClient(nb, timeout=120, kernel_name="python3").execute()
nbf.write(nb, OUT)
print(OUT)
