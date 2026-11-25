import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "RetinaAI – Diabetic Retinopathy Detection (5 models compared)",
  description: "A 5-layer CNN and 4 ImageNet pre-trained models detect and grade diabetic retinopathy in retina photos.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Navbar />
        {children}
        <footer className="mx-auto max-w-7xl px-4 pb-10 pt-6 text-xs text-slate-500 sm:px-6">
          RetinaAI · UML501 Machine Learning project, Thapar Institute of Engineering &amp; Technology · Student project, not a
          medical device.
        </footer>
      </body>
    </html>
  );
}
