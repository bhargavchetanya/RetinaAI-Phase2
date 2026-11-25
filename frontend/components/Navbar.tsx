"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ScanEye } from "lucide-react";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/screening/", label: "Screening" },
  { href: "/results/", label: "Training results" },
  { href: "/technical/", label: "Technical" },
];

export default function Navbar() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/85 backdrop-blur">
      <nav className="mx-auto flex max-w-7xl items-center gap-2 px-4 py-3 sm:gap-6 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold text-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/icon.svg" alt="" className="h-7 w-7" />
          <span className="hidden sm:inline">
            Retina<span className="text-cyan-400">AI</span>
          </span>
        </Link>
        <div className="flex flex-1 items-center gap-1 overflow-x-auto text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href.replace(/\/$/, ""));
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`whitespace-nowrap rounded-lg px-3 py-1.5 ${active ? "bg-slate-800 text-white" : "text-slate-400 hover:text-white"}`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
        <Link
          href="/screening/"
          className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3 py-1.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400"
        >
          <ScanEye className="h-4 w-4" /> <span className="hidden sm:inline">Try it</span>
        </Link>
      </nav>
    </header>
  );
}
