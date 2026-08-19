"use client";

import { ChangeEvent, PointerEvent, useMemo, useState } from "react";

type Candle = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type IndicatorPoint = {
  score: number;
  raw: number;
  closePosition: number;
  volumeRatio: number;
  confirm: boolean;
};

type StockPreset = {
  code: string;
  name: string;
  market: string;
  base: number;
  seed: number;
  drift: number;
};

const STOCKS: StockPreset[] = [
  { code: "688008", name: "澜起科技", market: "科创板", base: 71.4, seed: 83, drift: 0.0011 },
  { code: "688981", name: "中芯国际", market: "科创板", base: 92.8, seed: 157, drift: 0.0007 },
  { code: "00700", name: "腾讯控股", market: "港股", base: 515.2, seed: 229, drift: 0.0005 },
];

const PERIODS = [
  { label: "60日", value: 60 },
  { label: "120日", value: 120 },
  { label: "全部", value: 180 },
];

const CHART_WIDTH = 1120;
const PRICE_HEIGHT = 410;
const INDICATOR_HEIGHT = 184;
const LEFT = 18;
const RIGHT = 74;
const TOP = 20;
const BOTTOM = 30;

function seededRandom(seed: number) {
  let value = seed % 2147483647;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

function tradingDates(count: number) {
  const dates: string[] = [];
  const cursor = new Date("2026-08-18T12:00:00");
  while (dates.length < count) {
    if (cursor.getDay() !== 0 && cursor.getDay() !== 6) {
      dates.unshift(cursor.toISOString().slice(0, 10));
    }
    cursor.setDate(cursor.getDate() - 1);
  }
  return dates;
}

function generateSeries(stock: StockPreset, count = 180): Candle[] {
  const random = seededRandom(stock.seed);
  const dates = tradingDates(count);
  const series: Candle[] = [];
  let previous = stock.base * 0.74;

  for (let index = 0; index < count; index += 1) {
    const cycle = Math.sin(index / 12) * 0.0024;
    let dailyMove = (random() - 0.47) * 0.028 + stock.drift + cycle;
    if (index === 64 || index === 121) dailyMove -= 0.047;
    if (index === 65 || index === 122) dailyMove += 0.028;

    let open = previous * (1 + (random() - 0.5) * 0.012);
    let close = previous * (1 + dailyMove);
    let high = Math.max(open, close) * (1 + random() * 0.015);
    let low = Math.min(open, close) * (1 - random() * 0.016);
    let volume = (7_600_000 + random() * 5_100_000) * (1 + Math.abs(dailyMove) * 10);

    if (index === count - 13) {
      open = previous * 0.962;
      low = previous * 0.892;
      close = previous * 1.018;
      high = close * 1.014;
      volume = 25_800_000;
    }

    series.push({
      date: dates[index],
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Math.round(volume),
    });
    previous = close;
  }
  return series;
}

function movingAverage(values: number[], length: number) {
  return values.map((_, index) => {
    const from = Math.max(0, index - length + 1);
    const window = values.slice(from, index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
}

function calculateIndicator(data: Candle[]): IndicatorPoint[] {
  const trueRanges = data.map((item, index) => {
    if (index === 0) return item.high - item.low;
    const previousClose = data[index - 1].close;
    return Math.max(
      item.high - item.low,
      Math.abs(item.high - previousClose),
      Math.abs(item.low - previousClose),
    );
  });
  const atr = movingAverage(trueRanges, 14);
  const volumeAverage = movingAverage(data.map((item) => item.volume), 20);
  const closeAverage = movingAverage(data.map((item) => item.close), 5);
  let smooth = 0;

  return data.map((item, index) => {
    if (index === 0) {
      return { score: 0, raw: 0, closePosition: 0.5, volumeRatio: 1, confirm: false };
    }
    const previousLow = data[index - 1].low;
    const probe = Math.max(previousLow - item.low, 0) / Math.max(atr[index], 0.01) * 100;
    const priorLows = data.slice(Math.max(0, index - 38), index).map((point) => point.low);
    const newLow = item.low <= Math.min(...priorLows);
    const closePosition = (item.close - item.low) / Math.max(item.high - item.low, 0.01);
    const volumeRatio = Math.min(item.volume / Math.max(volumeAverage[index], 1), 2);
    const raw = newLow ? probe * closePosition * volumeRatio : 0;
    smooth = index === 1 ? raw : raw * 0.5 + smooth * 0.5;
    const score = Math.min(smooth, 100);
    const trend = item.close > closeAverage[index] && closeAverage[index] > closeAverage[index - 1];
    const recover = item.close > item.open && item.close > data[index - 1].close && closePosition > 0.65;
    const volumeOk = item.volume > volumeAverage[index] * 1.2;
    return { score, raw, closePosition, volumeRatio, confirm: score > 30 && trend && recover && volumeOk };
  });
}

function formatNumber(value: number, digits = 2) {
  return value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatVolume(value: number) {
  if (value >= 100_000_000) return `${formatNumber(value / 100_000_000)}亿`;
  return `${formatNumber(value / 10_000, 0)}万`;
}

function formatDate(date: string) {
  return date.slice(5).replace("-", "/");
}

function axisTicks(min: number, max: number, count: number) {
  return Array.from({ length: count }, (_, index) => min + (max - min) * index / (count - 1));
}

export default function Home() {
  const [stockCode, setStockCode] = useState(STOCKS[0].code);
  const [period, setPeriod] = useState(120);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [importedData, setImportedData] = useState<Candle[] | null>(null);
  const [importName, setImportName] = useState("");

  const stock = STOCKS.find((item) => item.code === stockCode) ?? STOCKS[0];
  const allData = useMemo(
    () => importedData ?? generateSeries(stock),
    [importedData, stock],
  );
  const allIndicator = useMemo(() => calculateIndicator(allData), [allData]);
  const from = Math.max(0, allData.length - period);
  const data = allData.slice(from);
  const indicator = allIndicator.slice(from);
  const ma5 = movingAverage(allData.map((item) => item.close), 5).slice(from);
  const activeIndex = hoverIndex ?? data.length - 1;
  const active = data[activeIndex];
  const activeIndicator = indicator[activeIndex];
  const latest = data[data.length - 1];
  const previous = data[data.length - 2] ?? latest;
  const change = (latest.close / previous.close - 1) * 100;
  const latestSignal = [...indicator].reverse().findIndex((point) => point.confirm);
  const signalIndex = latestSignal < 0 ? -1 : indicator.length - 1 - latestSignal;

  const plotWidth = CHART_WIDTH - LEFT - RIGHT;
  const pricePlotHeight = PRICE_HEIGHT - TOP - BOTTOM;
  const lows = data.map((item) => item.low);
  const highs = data.map((item) => item.high);
  const rawMin = Math.min(...lows);
  const rawMax = Math.max(...highs);
  const margin = (rawMax - rawMin) * 0.08;
  const priceMin = rawMin - margin;
  const priceMax = rawMax + margin;
  const xAt = (index: number) => LEFT + (index + 0.5) * plotWidth / data.length;
  const yPrice = (value: number) => TOP + (priceMax - value) / (priceMax - priceMin) * pricePlotHeight;
  const candleWidth = Math.max(2, Math.min(8, plotWidth / data.length * 0.62));
  const dateTickIndexes = Array.from(new Set([0, 1, 2, 3, 4, 5].map((step) => Math.round(step * (data.length - 1) / 5))));
  const priceTicks = axisTicks(priceMin, priceMax, 5);
  const yScore = (value: number) => TOP + (100 - value) / 100 * (INDICATOR_HEIGHT - TOP - BOTTOM);

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const pointerX = (event.clientX - box.left) / box.width * CHART_WIDTH;
    const nextIndex = Math.round((pointerX - LEFT) / plotWidth * data.length - 0.5);
    setHoverIndex(Math.max(0, Math.min(data.length - 1, nextIndex)));
  }

  function handleStockChange(event: ChangeEvent<HTMLSelectElement>) {
    setStockCode(event.target.value);
    setImportedData(null);
    setImportName("");
    setHoverIndex(null);
  }

  async function handleCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const rows = text.trim().split(/\r?\n/).slice(1);
    const parsed = rows.map((row) => {
      const [date, open, high, low, close, volume] = row.split(",").map((value) => value.trim());
      return { date, open: Number(open), high: Number(high), low: Number(low), close: Number(close), volume: Number(volume) };
    }).filter((item) => item.date && [item.open, item.high, item.low, item.close, item.volume].every(Number.isFinite));
    if (parsed.length >= 20) {
      setImportedData(parsed.sort((a, b) => a.date.localeCompare(b.date)));
      setImportName(file.name);
      setPeriod(Math.min(120, parsed.length));
      setHoverIndex(null);
    }
    event.target.value = "";
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="低位承接研究台首页">
          <span className="brand-mark">承</span>
          <span><strong>低位承接研究台</strong><small>Price absorption research</small></span>
        </a>
        <div className="topbar-note"><span className="status-dot" aria-hidden="true" />本地计算 · 无未来函数</div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">日线级别 · 技术研究</p>
          <h1>把“主力吸货”拆成<br /><em>看得见的承接证据</em></h1>
          <p className="hero-copy">用阶段新低、下探幅度、收盘位置与成交量共同衡量低位承接，避免只凭一根神秘柱线下结论。</p>
        </div>
        <div className="hero-summary">
          <span>当前观察</span>
          <strong>{importedData ? "CSV 数据" : `${stock.name} ${stock.code}`}</strong>
          <small>{importName || `${stock.market} · 演示日线`}</small>
        </div>
      </section>

      <section className="workspace" aria-label="股票走势图和指标图">
        <div className="toolbar">
          <div className="security-title">
            <span className="market-tag">{importedData ? "CSV" : stock.market}</span>
            <div><h2>{importedData ? importName.replace(/\.csv$/i, "") : stock.name}</h2><span>{importedData ? "本地导入数据" : stock.code}</span></div>
          </div>
          <div className="price-summary">
            <strong className={change >= 0 ? "price-up" : "price-down"}>{formatNumber(latest.close)}</strong>
            <span className={change >= 0 ? "price-up" : "price-down"}>{change >= 0 ? "+" : ""}{formatNumber(change)}%</span>
          </div>
          <div className="toolbar-actions">
            <label className="select-wrap"><span className="sr-only">选择股票</span><select value={stockCode} onChange={handleStockChange} disabled={Boolean(importedData)}>{STOCKS.map((item) => <option value={item.code} key={item.code}>{item.name} {item.code}</option>)}</select></label>
            <div className="period-switch" aria-label="选择图表周期">
              {PERIODS.map((item) => <button className={period === item.value ? "active" : ""} key={item.value} onClick={() => { setPeriod(Math.min(item.value, allData.length)); setHoverIndex(null); }}>{item.label}</button>)}
            </div>
            <label className="upload-button">导入 CSV<input type="file" accept=".csv,text/csv" onChange={handleCsv} /></label>
          </div>
        </div>

        <div className="quote-strip" aria-live="polite">
          <span>{active.date}</span><span>开 <b>{formatNumber(active.open)}</b></span><span>高 <b>{formatNumber(active.high)}</b></span><span>低 <b>{formatNumber(active.low)}</b></span><span>收 <b>{formatNumber(active.close)}</b></span><span>量 <b>{formatVolume(active.volume)}</b></span><span className="ma-label">MA5 <b>{formatNumber(ma5[activeIndex])}</b></span>
        </div>

        <div className="chart-stack">
          <div className="chart-heading">
            <div><span className="section-number">01</span><h3>每日价格走势</h3></div>
            <div className="legend"><span className="legend-up" />上涨 <span className="legend-down" />下跌 <span className="legend-ma" />MA5</div>
          </div>

          <svg className="chart price-chart" viewBox={`0 0 ${CHART_WIDTH} ${PRICE_HEIGHT}`} role="img" aria-label={`${stock.name}日线蜡烛图`} onPointerMove={onPointerMove} onPointerLeave={() => setHoverIndex(null)}>
            {priceTicks.map((tick) => { const y = yPrice(tick); return <g key={tick}><line className="grid-line" x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={y} y2={y} /><text className="axis-label" x={CHART_WIDTH - RIGHT + 12} y={y + 4}>{formatNumber(tick)}</text></g>; })}
            {dateTickIndexes.map((index) => <g key={index}><line className="grid-line vertical" x1={xAt(index)} x2={xAt(index)} y1={TOP} y2={PRICE_HEIGHT - BOTTOM} /><text className="date-label" x={xAt(index)} y={PRICE_HEIGHT - 7} textAnchor="middle">{formatDate(data[index].date)}</text></g>)}
            <polyline className="ma-line" points={ma5.map((value, index) => `${xAt(index)},${yPrice(value)}`).join(" ")} />
            {data.map((item, index) => {
              const up = item.close >= item.open;
              const x = xAt(index);
              const bodyTop = yPrice(Math.max(item.open, item.close));
              const bodyBottom = yPrice(Math.min(item.open, item.close));
              return <g className={up ? "candle up" : "candle down"} key={item.date}><line x1={x} x2={x} y1={yPrice(item.high)} y2={yPrice(item.low)} /><rect x={x - candleWidth / 2} y={bodyTop} width={candleWidth} height={Math.max(1.5, bodyBottom - bodyTop)} /></g>;
            })}
            {hoverIndex !== null && <line className="crosshair" x1={xAt(hoverIndex)} x2={xAt(hoverIndex)} y1={TOP} y2={PRICE_HEIGHT - BOTTOM} />}
            <rect className="pointer-layer" x={LEFT} y={TOP} width={plotWidth} height={pricePlotHeight} />
          </svg>

          <div className="chart-divider" />
          <div className="chart-heading indicator-heading">
            <div><span className="section-number">02</span><h3>低位承接强度</h3></div>
            <div className="indicator-readout"><span>当前</span><strong>{formatNumber(activeIndicator.score, 1)}</strong><small>{activeIndicator.score >= 60 ? "强承接" : activeIndicator.score >= 30 ? "观察" : "中性"}</small></div>
          </div>

          <svg className="chart indicator-chart" viewBox={`0 0 ${CHART_WIDTH} ${INDICATOR_HEIGHT}`} role="img" aria-label="低位承接强度指标图" onPointerMove={onPointerMove} onPointerLeave={() => setHoverIndex(null)}>
            {[0, 30, 60, 100].map((tick) => <g key={tick}><line className={`score-grid score-${tick}`} x1={LEFT} x2={CHART_WIDTH - RIGHT} y1={yScore(tick)} y2={yScore(tick)} /><text className="axis-label" x={CHART_WIDTH - RIGHT + 12} y={yScore(tick) + 4}>{tick}</text></g>)}
            {indicator.map((point, index) => { const y = yScore(point.score); return <rect className={point.score >= 60 ? "score-bar strong" : "score-bar"} key={data[index].date} x={xAt(index) - candleWidth / 2} y={y} width={candleWidth} height={yScore(0) - y} rx={Math.min(2, candleWidth / 3)} />; })}
            <polyline className="score-line" points={indicator.map((point, index) => `${xAt(index)},${yScore(point.score)}`).join(" ")} />
            {indicator.map((point, index) => point.confirm ? <g className="confirm-marker" key={`confirm-${data[index].date}`}><circle cx={xAt(index)} cy={yScore(Math.min(point.score + 9, 96))} r="5" /><path d={`M ${xAt(index) - 2.5} ${yScore(Math.min(point.score + 9, 96))} l 2 2.5 l 4 -5`} /></g> : null)}
            {hoverIndex !== null && <line className="crosshair" x1={xAt(hoverIndex)} x2={xAt(hoverIndex)} y1={TOP} y2={INDICATOR_HEIGHT - BOTTOM} />}
            <rect className="pointer-layer" x={LEFT} y={TOP} width={plotWidth} height={INDICATOR_HEIGHT - TOP - BOTTOM} />
          </svg>
        </div>
      </section>

      <section className="evidence-grid" aria-label="指标解释">
        <article className="method-card">
          <p className="eyebrow">指标方法</p><h2>一次可信的承接，至少要留下四项证据。</h2>
          <p>指标只在创出近 38 日新低时启动，再用 ATR 归一化下探幅度，并结合收盘收复程度和相对成交量。结果经 3 日指数平滑后限制在 0 至 100。</p>
          <div className="formula-line" aria-label="指标计算关系"><span>阶段新低</span><i>×</i><span>下探幅度</span><i>×</i><span>收盘位置</span><i>×</i><span>相对量能</span></div>
        </article>
        <article className="factor-card">
          <div className="factor-card-head"><div><p className="eyebrow">光标日拆解</p><h3>{active.date}</h3></div>{activeIndicator.confirm && <span className="confirm-pill">确认信号</span>}</div>
          <div className="factor-row"><span>收盘位置</span><div><i style={{ width: `${Math.min(activeIndicator.closePosition * 100, 100)}%` }} /></div><b>{formatNumber(activeIndicator.closePosition * 100, 0)}%</b></div>
          <div className="factor-row"><span>相对量能</span><div><i style={{ width: `${Math.min(activeIndicator.volumeRatio / 2 * 100, 100)}%` }} /></div><b>{formatNumber(activeIndicator.volumeRatio, 2)}×</b></div>
          <div className="factor-row"><span>承接强度</span><div><i style={{ width: `${activeIndicator.score}%` }} /></div><b>{formatNumber(activeIndicator.score, 1)}</b></div>
          <p className="signal-note">{signalIndex >= 0 ? `窗口内最近一次确认出现在 ${data[signalIndex].date}。` : "当前窗口没有满足全部确认条件的日期。"}</p>
        </article>
      </section>

      <aside className="disclaimer">
        <strong>研究边界</strong><p>本页识别的是价格和成交量留下的低位承接迹象，不等同于真实大单资金流，也不构成投资建议。演示行情为模拟数据；实盘判断请导入复权后的日线数据。</p><span>CSV 列顺序：date, open, high, low, close, volume</span>
      </aside>
      <footer><span>低位承接研究台</span><span>2026 · Daily research view</span></footer>
    </main>
  );
}
