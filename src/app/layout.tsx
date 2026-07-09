import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Zano Extract Kenshi — Local OCR Phrase Extractor',
  description: 'Client-side OCR for extracting 23/24/26-word Zano recovery phrases from screenshots. No uploads, no server storage.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
