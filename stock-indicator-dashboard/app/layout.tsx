import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "低位承接研究台｜蜡烛图与承接强度",
  description: "将日线蜡烛图与低位承接强度指标放在同一时间轴上观察。",
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
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
