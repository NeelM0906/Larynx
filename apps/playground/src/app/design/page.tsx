import { PageShell } from "@/components/page-shell";

export default function DesignPage() {
  return (
    <PageShell
      kicker="03 · Voice design"
      title="Describe a voice. Hear it."
      intro="Prose in. Preview out. Save the keeper to the library."
    >
      <div className="rounded-md border border-dashed border-border p-10 text-center text-muted-foreground text-sm font-mono">
        Design prompt + preview + save — lands in M6.3
      </div>
    </PageShell>
  );
}
