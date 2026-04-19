import { MessagesSquareIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ComingSoon } from "@/components/coming-soon";

export default function ConversationPage() {
  return (
    <PageShell
      kicker="05 · Conversation"
      title="A proper duplex."
      intro="Mic in, VAD → STT → LLM → TTS, live state & transcript, with barge-in."
    >
      <ComingSoon
        icon={MessagesSquareIcon}
        title="Conversation — not in this build"
        description="Deliberately deferred this cycle. The backend duplex path (WS /v1/conversation)
          exists but hasn't been end-to-end verified on real hardware yet — building the
          client against an unvalidated backend would bake in the wrong shape. Once the
          backend gets its first clean real-model run, the UI lands on top."
        bullets={[
          "Full-duplex WebSocket with PCM mic capture + streamed TTS playback.",
          "Config drawer for the LLM + voice selection.",
          "Live state pill (listening / thinking / speaking) and rolling transcript.",
        ]}
      />
    </PageShell>
  );
}
