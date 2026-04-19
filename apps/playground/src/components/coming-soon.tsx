import type { LucideIcon } from "lucide-react";

interface ComingSoonProps {
  icon: LucideIcon;
  title: string;
  description: string;
  bullets?: string[];
}

/**
 * Shared "not yet" panel for tabs whose features aren't in the current
 * build. Intentional-looking card with icon + pitch + bullets, so a
 * tester lands on something that reads as "we know this isn't here"
 * rather than "the page is broken."
 */
export function ComingSoon({ icon: Icon, title, description, bullets }: ComingSoonProps) {
  return (
    <div className="rounded-md border border-border bg-card/60 p-10 md:p-12">
      <div className="flex items-center gap-4">
        <span
          aria-hidden
          className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-md border border-primary/40 bg-primary/5 text-primary"
        >
          <Icon className="size-6" />
        </span>
        <div>
          <span className="inline-flex items-center rounded-sm border border-border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            Coming in a future update
          </span>
          <h2 className="mt-2 font-display text-3xl italic text-foreground">{title}</h2>
        </div>
      </div>
      <p className="mt-6 max-w-2xl text-base leading-relaxed text-muted-foreground">
        {description}
      </p>
      {bullets && bullets.length > 0 && (
        <ul className="mt-6 max-w-2xl space-y-2 text-sm text-muted-foreground">
          {bullets.map((b) => (
            <li key={b} className="flex items-start gap-2">
              <span className="mt-2 inline-block h-px w-3 shrink-0 bg-muted-foreground/60" />
              <span>{b}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
