"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { marketSessions, type MarketSession } from "../lib/market-sessions";

export const SECTIONS = [
  { href: "/", label: "总览" },
  { href: "/market", label: "行情" },
  { href: "/research", label: "研究复盘" },
  { href: "/us-screen", label: "美股筛选" },
] as const;

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

export function SiteHeader() {
  const pathname = usePathname() ?? "/";
  // Session state depends on the viewer's clock, so it renders after mount only.
  const [sessions, setSessions] = useState<MarketSession[] | null>(null);
  useEffect(() => {
    const update = () => setSessions(marketSessions());
    const first = window.setTimeout(update, 0);
    const timer = window.setInterval(update, 30_000);
    return () => { window.clearTimeout(first); window.clearInterval(timer); };
  }, []);

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link className="site-brand" href="/" aria-label="低位承接研究台首页">
          <span className="site-brand-mark" aria-hidden="true">承</span>
          <span className="site-brand-copy"><strong>低位承接研究台</strong><small>Market research workspace</small></span>
        </Link>
        <nav className="site-nav" aria-label="主导航">
          {SECTIONS.map((item) => (
            <Link key={item.href} href={item.href} className="site-nav-link" aria-current={isActive(pathname, item.href) ? "page" : undefined}>
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="site-status" aria-label="交易时段（按交易所时间估算，未计节假日）" title="按交易所时间估算，未计节假日">
          {sessions?.map((session) => (
            <span key={session.market} className={`site-session session-${session.state}`}>
              <i aria-hidden="true" />{session.label}<b>{session.stateLabel}</b>
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}
