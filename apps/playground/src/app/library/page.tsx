import { PageShell } from "@/components/page-shell";

export default function LibraryPage() {
  return (
    <PageShell
      kicker="04 · Voice library"
      title="The collection."
      intro="Every voice in the gateway's Postgres. Reference clip, use count, delete."
    >
      <div className="rounded-md border border-dashed border-border p-10 text-center text-muted-foreground text-sm font-mono">
        Voice card grid — lands in M6.4
      </div>
    </PageShell>
  );
}
