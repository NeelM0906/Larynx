import { PageShell } from "@/components/page-shell";

export default function ConversationPage() {
  return (
    <PageShell
      kicker="05 · Conversation"
      title="A proper duplex."
      intro="Port of M5's conversation.html — config drawer, live state, transcript panel, barge-in."
    >
      <div className="rounded-md border border-dashed border-border p-10 text-center text-muted-foreground text-sm font-mono">
        WS client + state machine + transcript — lands in M6.5
      </div>
    </PageShell>
  );
}
