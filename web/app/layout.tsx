import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = { title: 'Emotion Lab', description: 'Per-frame emotion classification across models' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">Emotion Lab</Link>
          <nav>
            <Link href="/">Clips</Link>
            <Link href="/leaderboard">Leaderboard</Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
