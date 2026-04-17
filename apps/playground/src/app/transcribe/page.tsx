import { PageShell } from "@/components/page-shell";

export default function TranscribePage() {
  return (
    <PageShell
      kicker="06 · Transcribe"
      title="Words from audio."
      intro="Upload a file or record the mic. Pick language, hotwords, punctuation. Read the transcript."
    >
      <div className="rounded-md border border-dashed border-border p-10 text-center text-muted-foreground text-sm font-mono">
        STT panel + recorder — lands in M6.6
      </div>
    </PageShell>
  );
}
