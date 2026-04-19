import { CaptionsIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ComingSoon } from "@/components/coming-soon";

export default function TranscribePage() {
  return (
    <PageShell
      kicker="06 · Transcribe"
      title="Words from audio."
      intro="Upload a file or record the mic. Pick language, hotwords, punctuation. Read the transcript."
    >
      <ComingSoon
        icon={CaptionsIcon}
        title="Transcribe — not in this build"
        description="Transcription runs against POST /v1/audio/transcriptions (OpenAI-shaped) or
          POST /v1/stt for the native path. Both are live on the gateway today; the
          UI just hasn't landed yet."
        bullets={[
          "File-drop or record-from-mic input.",
          "Language picker + hotwords (comma-separated) for domain terms.",
          "Streaming partials via WS /v1/stt/stream for longer clips.",
        ]}
      />
    </PageShell>
  );
}
