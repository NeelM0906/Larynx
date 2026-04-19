import { SparklesIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ComingSoon } from "@/components/coming-soon";

export default function DesignPage() {
  return (
    <PageShell
      kicker="03 · Voice design"
      title="Describe a voice. Hear it."
      intro="Prose in. Preview out. Save the keeper to the library."
    >
      <ComingSoon
        icon={SparklesIcon}
        title="Voice design — not in this build"
        description="Design lets you conjure a voice from a short natural-language prompt —
          “warm, middle-aged female, slight southern lilt” — render a preview, then
          promote it to the library once it sounds right."
        bullets={[
          "Prompt-driven synthesis via POST /v1/voices/design.",
          "A listen-before-save preview step so you don't clutter the library.",
          "Same editorial card format as Clone, once it lands.",
        ]}
      />
    </PageShell>
  );
}
