"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Brand } from "./brand";
import { cn } from "@/lib/utils";
import { TABS } from "@/lib/tabs";

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto max-w-6xl px-8 h-14 flex items-center justify-between">
        <Brand />
        <nav className="hidden md:flex items-center gap-1">
          {TABS.map((t) => {
            const active = pathname === t.href;
            return (
              <Link
                key={t.href}
                href={t.href}
                className={cn(
                  "group relative flex items-baseline gap-1.5 px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/60 group-hover:text-primary/80 transition-colors">
                  {t.n}
                </span>
                <span>{t.label}</span>
                {active && (
                  <span className="absolute left-3 right-3 -bottom-[1px] h-px bg-primary" />
                )}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
