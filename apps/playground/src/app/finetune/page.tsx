import { PageShell } from "@/components/page-shell";

export default function FinetunePage() {
  return (
    <PageShell
      kicker="07 · Fine-tune"
      title="Not yet."
      intro="Training worker lands in M7. This tab is reserved so the URL doesn't change when it does."
    >
      <div className="rounded-md border border-dashed border-primary/40 p-10 text-center">
        <p className="font-display text-3xl italic text-primary">Coming soon.</p>
        <p className="mt-3 text-sm font-mono uppercase tracking-widest text-muted-foreground">
          M7 · on-demand GPU training
        </p>
      </div>
    </PageShell>
  );
}
