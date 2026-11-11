import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "美股筛选",
  description: "美股漂移动量排名、走势形态和回测证据。",
};

export default function SectionLayout({ children }: { children: React.ReactNode }) {
  return children;
}
