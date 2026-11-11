import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "研究复盘",
  description: "每日候选榜、价格预测对照和分析证据。",
};

export default function SectionLayout({ children }: { children: React.ReactNode }) {
  return children;
}
