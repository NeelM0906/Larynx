import Link from "next/link";
import { TABS, type TabLabel } from "@/lib/tabs";

const BLURBS: Record<TabLabel, string> = {
  TTS: "Prompt the gateway, pick a voice, tune cfg_value & temperature, hear it back.",
  Clone: "Upload a reference clip, name it, then test-synthesize with the fresh voice.",
  Design: "Describe the voice in prose, preview it, save it to the library.",
  Library: "Every voice in Postgres. Test one, delete one, or clone a fresh one.",
  Conversation: "Duplex WebSocket: mic in, VAD → STT → LLM → TTS, live state & transcript.",
  Transcribe: "Upload or record, pick language + hotwords, get the transcript.",
  "Fine-tune": "Drop a dataset, pick a name, watch a LoRA train, land in the library.",
};

const STATUS: Record<TabLabel, string> = {
  TTS: "Ready",
  Clone: "Ready",
  Library: "Ready",
  "Fine-tune": "Ready",
  Design: "Coming soon",
  Transcribe: "Coming soon",
  Conversation: "Coming soon",
};

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-8 pt-24 pb-16">
      <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-primary mb-8">
        §&nbsp;&nbsp;Larynx&nbsp;/&nbsp;M6&nbsp;/&nbsp;Internal&nbsp;Bench
      </p>
      <h1 className="font-display text-6xl md:text-7xl leading-[0.98] text-foreground max-w-4xl">
        A quiet place to poke at{" "}
        <span className="italic text-primary">voices,</span>{" "}
        <span className="italic">conversations,</span> and{" "}
        <span className="italic">transcripts</span> — without opening curl.
      </h1>
      <p className="mt-8 max-w-2xl text-muted-foreground text-lg leading-relaxed">
        Seven tabs, one token. Every page talks to the same gateway you&apos;d
        hit from your own code. Nothing here is for customers; everything here
        is for us.
      </p>
      <div className="rule-hairline mt-16" />
      <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-px bg-border/60 border border-border/60">
        {TABS.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className="group relative bg-background p-8 transition-colors hover:bg-card"
          >
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground/80 group-hover:text-primary transition-colors">
                {t.n}
              </span>
              <span
                className={`font-mono text-[10px] uppercase tracking-[0.2em] transition-colors ${
                  STATUS[t.label] === "Ready"
                    ? "text-muted-foreground/60 group-hover:text-muted-foreground"
                    : "text-muted-foreground/30 group-hover:text-muted-foreground/60"
                }`}
              >
                {STATUS[t.label]}
              </span>
            </div>
            <h2 className="mt-6 font-display text-3xl italic text-foreground">
              {t.label}
            </h2>
            <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
              {BLURBS[t.label]}
            </p>
            <span className="mt-6 inline-flex items-center gap-1 text-xs font-mono uppercase tracking-widest text-muted-foreground group-hover:text-primary transition-colors">
              Open
              <span aria-hidden className="transition-transform group-hover:translate-x-0.5">
                →
              </span>
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
