import { cn } from "@/lib/utils";

interface PageShellProps {
  kicker?: string;
  title: string;
  intro?: string;
  children?: React.ReactNode;
  className?: string;
}

/**
 * Shared page header: editorial kicker + big serif title + optional intro.
 * Every feature page lands on this — keeps the voice consistent across tabs.
 */
export function PageShell({ kicker, title, intro, children, className }: PageShellProps) {
  return (
    <section className={cn("mx-auto max-w-6xl px-8 pt-14 pb-10", className)}>
      {kicker && (
        <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-primary mb-6">
          {kicker}
        </p>
      )}
      <h1 className="font-display text-5xl md:text-6xl leading-[1.05] text-foreground">
        {title}
      </h1>
      {intro && (
        <p className="mt-6 max-w-2xl text-muted-foreground text-base leading-relaxed">
          {intro}
        </p>
      )}
      <div className="rule-hairline mt-10" />
      {children && <div className="mt-12">{children}</div>}
    </section>
  );
}
