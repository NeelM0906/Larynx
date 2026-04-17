import { PageShell } from "@/components/page-shell";

export default function TtsPage() {
  return (
    <PageShell
      kicker="01 · Text to speech"
      title="Say something."
      intro="Type. Pick a voice. Nudge cfg_value and inference timesteps. Hit generate. Download the WAV."
    >
      <div className="rounded-md border border-dashed border-border p-10 text-center text-muted-foreground text-sm font-mono">
        TTS form + player — lands in M6.1
      </div>
    </PageShell>
  );
}
