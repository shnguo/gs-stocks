import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "行情图表",
  description: "日线蜡烛图与低位承接强度指标、自选股票和实时行情。",
};

export default function SectionLayout({ children }: { children: React.ReactNode }) {
  return children;
}
