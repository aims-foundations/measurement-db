import type { ReactNode } from "react";
import type { Metadata } from "next";
import type { Viewport } from "next";
import {
  Source_Serif_4,
  Source_Sans_3,
  Roboto_Mono,
  Google_Sans_Flex,
} from "next/font/google";
import { siteConfig } from "@/content/site";
import { MotionProvider } from "@/components/motion-provider";
import { RdHeader } from "@/components/redesign/rd-header";
import { RdFooter } from "@/components/redesign/rd-footer";
import { getNavItems } from "@/lib/nav";
import "./globals.css";
import "./redesign.css";

const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-source-serif",
});

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-source-sans",
});

const robotoMono = Roboto_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-roboto-mono",
});

const googleSans = Google_Sans_Flex({
  subsets: ["latin"],
  variable: "--font-rd-sans",
});

export const metadata: Metadata = {
  // Pages set relative canonicals like "/measurement-db/<slug>"; metadataBase
  // resolves them against the public origin (this site is proxied there under
  // /measurement-db, so those absolute paths are the real public URLs).
  metadataBase: new URL(siteConfig.url),
  title: "Measurement Data Bank",
  description:
    "The measurement data bank behind AIMS: benchmarks, model coverage, and response matrices.",
};

export const viewport: Viewport = {
  themeColor: "#8C1515",
};

export default async function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  const navItems = await getNavItems();
  return (
    <html lang="en">
      <body
        className={`${sourceSerif.variable} ${sourceSans.variable} ${robotoMono.variable} ${googleSans.variable} font-sans antialiased`}
      >
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-md focus:bg-[var(--digital-red)] focus:px-4 focus:py-2 focus:text-white"
        >
          Skip to main content
        </a>
        <div className={`rd-root ${googleSans.variable}`}>
          <MotionProvider>
            <RdHeader navItems={navItems} />
            {children}
            <RdFooter />
          </MotionProvider>
        </div>
      </body>
    </html>
  );
}
