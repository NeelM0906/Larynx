import Link from "next/link";

export function Brand() {
  return (
    <Link
      href="/"
      className="group flex items-baseline gap-2 text-foreground"
    >
      <span className="font-display text-2xl italic leading-none">Larynx</span>
      <span className="text-[10px] font-mono uppercase tracking-[0.25em] text-muted-foreground/80 group-hover:text-primary transition-colors">
        playground
      </span>
    </Link>
  );
}
