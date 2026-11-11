import type { Metadata } from "next";
import "./globals.css";
import "./site.css";
import { SiteHeader } from "./components/site-header";

export const metadata: Metadata = {
  title: { default: "低位承接研究台", template: "%s｜低位承接研究台" },
  description: "行情图表、研究复盘与美股动量筛选：将日线蜡烛图与低位承接强度指标放在同一时间轴上观察。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <head>
        <link rel="preload" as="font" type="font/woff2" crossOrigin="anonymous" href="/fonts/geist/Geist-Variable.woff2" />
        <link
          rel="preload"
          as="fetch"
          crossOrigin="anonymous"
          fetchPriority="high"
          href="/api/chart-daily-bars?instrument_id=cn.xshg.688008&adjustment_basis=none"
        />
        <link
          rel="preload"
          as="fetch"
          crossOrigin="anonymous"
          fetchPriority="high"
          href="/api/realtime-quote?instrument_id=cn.xshg.688008"
        />
      </head>
      <body>
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
