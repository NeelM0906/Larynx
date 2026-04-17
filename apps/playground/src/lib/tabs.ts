export const TABS = [
  { n: "01", label: "TTS", href: "/tts" },
  { n: "02", label: "Clone", href: "/clone" },
  { n: "03", label: "Design", href: "/design" },
  { n: "04", label: "Library", href: "/library" },
  { n: "05", label: "Conversation", href: "/conversation" },
  { n: "06", label: "Transcribe", href: "/transcribe" },
  { n: "07", label: "Fine-tune", href: "/finetune" },
] as const;

export type TabLabel = (typeof TABS)[number]["label"];
