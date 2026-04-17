import type { Metadata } from "next";
import { Instrument_Serif, Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthGate } from "@/components/auth-gate";
import { Nav } from "@/components/nav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: "Larynx Playground",
  description: "Internal bench for the Larynx voice gateway.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${instrumentSerif.variable} antialiased min-h-screen flex flex-col`}
      >
        <AuthGate>
          <Nav />
          <main className="flex-1 animate-rise">{children}</main>
          <footer className="border-t border-border/60 mt-24 py-6 px-8">
            <div className="mx-auto max-w-6xl flex items-center justify-between text-[11px] font-mono uppercase tracking-widest text-muted-foreground/70">
              <span>Larynx · internal bench</span>
              <span>M6 · playground</span>
            </div>
          </footer>
        </AuthGate>
      </body>
    </html>
  );
}
