export const MIN_VISIBLE_CHART_SLOTS = 60;

export type ChartHorizontalLayout = {
  candleWidth: number;
  leadingSlots: number;
  slotCount: number;
  slotWidth: number;
};

export function chartHorizontalLayout(
  dataLength: number,
  plotWidth: number,
): ChartHorizontalLayout {
  const safeDataLength = Math.max(1, Math.floor(dataLength));
  const slotCount = Math.max(MIN_VISIBLE_CHART_SLOTS, safeDataLength);
  const slotWidth = plotWidth / slotCount;
  return {
    candleWidth: Math.max(2, Math.min(8, slotWidth * 0.62)),
    leadingSlots: 0,
    slotCount,
    slotWidth,
  };
}

export function chartXAt(
  index: number,
  left: number,
  layout: ChartHorizontalLayout,
): number {
  return left + (layout.leadingSlots + index + 0.5) * layout.slotWidth;
}

export function chartDataIndexAt(
  pointerX: number,
  left: number,
  dataLength: number,
  layout: ChartHorizontalLayout,
): number | null {
  const slotIndex = Math.round((pointerX - left) / layout.slotWidth - 0.5);
  const dataIndex = slotIndex - layout.leadingSlots;
  return dataIndex >= 0 && dataIndex < dataLength ? dataIndex : null;
}
