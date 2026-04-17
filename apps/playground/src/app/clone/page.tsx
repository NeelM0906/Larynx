import { PageShell } from "@/components/page-shell";

export default function ClonePage() {
  return (
    <PageShell
      kicker="02 · Voice cloning"
      title="Borrow a voice."
      intro="Upload a clean reference clip, give it a name, and test-synthesize a line with it."
    >
      <div className="rounded-md border border-dashed border-border p-10 text-center text-muted-foreground text-sm font-mono">
        Clone uploader + preview — lands in M6.2
      </div>
    </PageShell>
  );
}
