'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { createWorker, type Worker } from 'tesseract.js';
import { UploadCloud, FolderOpen, Image as ImageIcon, Copy, Download, Trash2, Zap, ShieldCheck, ScanText, CheckCircle2, AlertTriangle, ClipboardPaste } from 'lucide-react';

type Mode = 'fast' | 'accurate';
type Status = 'queued' | 'processing' | 'done' | 'error';

type OcrItem = {
  id: string;
  file: File;
  name: string;
  size: number;
  preview: string;
  status: Status;
  progress: number;
  rawText: string;
  normalizedWords: string[];
  phrases: string[];
  bestPhrase: string;
  error?: string;
};

const WORD_FIXES: Record<string, string> = {
  '0': 'o',
  '1': 'l',
  '3': 'e',
  '5': 's',
  '7': 't',
  rn: 'm',
};

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const SPECIAL_WORD_FIXES: Record<string, string> = {
  governmen: 'government',
  goverment: 'government',
  // Zano / common OCR misreads
  tangal: 'tangle',
  tangel: 'tangle',
  tangl: 'tangle',
  tahgle: 'tangle',
  angal: 'tangle',
  tange: 'tangle',
};

/** Strip leading number prefix from a cell token BEFORE normalization.
 *  OCR often reads "1.tangle", "1|tangle", "1 tangle", "1)tangle" etc.
 *  If we normalize first, "1" → "l" and corrupts the word. */
function stripNumberPrefix(raw: string) {
  let t = raw.trim();
  // Match patterns: "1.", "1)", "1|", "1:", "1-", "1 " followed by word
  // Also handle "1.tangle" (no space) and "1 tangle" (with space)
  t = t.replace(/^\d{1,2}\s*[.)|:\-]\s*/i, '');
  t = t.replace(/^\d{1,2}\s+/i, '');
  return t;
}

function normalizeToken(token: string) {
  let t = token.toLowerCase().trim();
  t = t.replace(/[’']/g, '');
  t = t.replace(/[^a-z0-9]/g, '');
  if (!t) return '';
  t = t.replace(/[01357]/g, (m) => WORD_FIXES[m] ?? m);
  t = SPECIAL_WORD_FIXES[t] ?? t;
  if (t.length < 2 || t.length > 18) return '';
  return t;
}

function extractWords(text: string) {
  return text
    .replace(/\n/g, ' ')
    .split(/\s+/)
    .map(stripNumberPrefix) // ← strip "1." "2)" from full-page OCR too
    .map(normalizeToken)
    .filter(Boolean);
}

function scoreWindow(words: string[]) {
  let score = 0;
  const unique = new Set(words).size;
  score += unique * 2;
  score += words.filter((w) => w.length >= 4 && w.length <= 10).length;
  score -= words.filter((w) => /(.)\1{2,}/.test(w)).length * 3;
  score -= words.filter((w) => w.length <= 2).length * 2;
  return score;
}

const PHRASE_LENGTHS = [26, 25, 24, 23];

function extractPhrases(words: string[]) {
  const windows: string[] = [];
  for (const len of PHRASE_LENGTHS) {
    if (words.length < len) continue;
    for (let i = 0; i <= words.length - len; i++) {
      const slice = words.slice(i, i + len);
      const unique = new Set(slice);
      if (unique.size < Math.min(18, len - 2)) continue;
      windows.push(slice.join(' '));
    }
  }
  return Array.from(new Set(windows))
    .sort((a, b) => scoreWindow(b.split(' ')) - scoreWindow(a.split(' ')))
    .slice(0, 8);
}

async function fileToDataUrl(file: File) {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function preprocessImage(file: File, variant: 'clean' | 'hard' = 'clean') {
  const dataUrl = await fileToDataUrl(file);
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });

  const maxSide = variant === 'hard' ? 2600 : 1900;
  const scale = Math.min(maxSide / Math.max(img.width, img.height), 3);
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return dataUrl;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = Math.round(d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114);
    const contrast = variant === 'hard' ? 1.85 : 1.35;
    const shifted = Math.max(0, Math.min(255, (gray - 128) * contrast + 128));
    const v = variant === 'hard' ? (shifted > 150 ? 255 : 0) : shifted;
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

async function zanoCellImages(file: File, y0Offset = 0) {
  const dataUrl = await fileToDataUrl(file);
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });

  const cols: Array<[number, number]> = [[0.035, 0.325], [0.365, 0.655], [0.695, 0.965]];
  const y0 = 0.412 + y0Offset;
  const step = 0.0495;
  const rowH = 0.043;
  const cells: string[] = [];

  for (let index = 0; index < 26; index += 1) {
    const row = Math.floor(index / 3);
    const col = index % 3;
    const sx = Math.round(img.width * cols[col][0]);
    const sy = Math.round(img.height * (y0 + row * step));
    const sw = Math.round(img.width * (cols[col][1] - cols[col][0]));
    const sh = Math.round(img.height * rowH);
    const scale = 4;
    const canvas = document.createElement('canvas');
    canvas.width = sw * scale;
    canvas.height = sh * scale;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) continue;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = imageData.data;
    for (let i = 0; i < d.length; i += 4) {
      const gray = Math.round(d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114);
      const shifted = Math.max(0, Math.min(255, (gray - 128) * 1.7 + 128));
      d[i] = d[i + 1] = d[i + 2] = shifted;
    }
    ctx.putImageData(imageData, 0, 0);
    cells.push(canvas.toDataURL('image/png'));
  }
  return cells;
}

function cellWord(text: string) {
  let tokens = text
    .split(/\s+/)
    .map(stripNumberPrefix) // ← strip "1." "2)" etc FIRST, before normalize
    .map(normalizeToken)
    .filter((token) => token && !/^\d+$/.test(token) && token.length <= 18);

  // If OCR read "1.tangle" as single token, stripNumberPrefix + normalize already handled it.
  // Handle multi-token: "1" "tangle" → stripNumberPrefix("1")="" → filtered out
  if (tokens.length > 1 && tokens[0].length <= 2) tokens = tokens.slice(1);
  if (tokens.length >= 2 && tokens[0].length + tokens[1].length <= 18) return `${tokens[0]}${tokens[1]}`;
  return tokens.sort((a, b) => b.length - a.length)[0] ?? '';
}

async function recognizeZanoGrid(worker: Worker, file: File, onProgress: (p: number) => void) {
  await worker.setParameters({ tessedit_pageseg_mode: '6' as never });
  const cells = await zanoCellImages(file);
  // Keep ALL 26 positions — empty string for failed cells (don't filter!)
  const words: string[] = new Array(26).fill('');

  for (let i = 0; i < cells.length; i += 1) {
    const result = await worker.recognize(cells[i]);
    words[i] = cellWord(result.data.text || '');
    onProgress(Math.round(5 + ((i + 1) / cells.length) * 75));
  }

  // Retry failed cells with slight y-offset shifts — first 3 cells (row 0) often
  // fail because the grid y0 doesn't match every screenshot variant.
  const failed = words.map((w, i) => (w ? -1 : i)).filter((i) => i >= 0);
  if (failed.length > 0 && failed.length <= 8) {
    // Try shifting y0 up slightly (first 3 cells = top row most affected)
    for (const offset of [-0.015, 0.015, -0.03, 0.03]) {
      const retryCells = await zanoCellImages(file, offset);
      let stillFailed = 0;
      for (const i of failed) {
        if (words[i]) continue;
        const result = await worker.recognize(retryCells[i]);
        const w = cellWord(result.data.text || '');
        if (w) words[i] = w;
        else stillFailed++;
      }
      if (stillFailed === 0) break;
      failed.length = 0;
      words.forEach((w, i) => { if (!w) failed.push(i); });
    }
  }

  return words; // array of 26, some may be '' (empty) — caller handles
}

async function recognize(worker: Worker, file: File, mode: Mode, onProgress: (p: number) => void) {
  // ── STRATEGY: Full-page OCR FIRST (proven accurate), cell-by-cell as FALLBACK ──
  // The old approach (grid first) produced garbage like "lalli wully sollouwcth"
  // because hardcoded cell coordinates don't match all screenshot variants.

  // Phase 1: Full-page OCR with preprocessing
  await worker.setParameters({
    tessedit_pageseg_mode: '6' as never,
    preserve_interword_spaces: '1',
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \\n-_,.;:|[](){}',
  });

  const clean = await preprocessImage(file, 'clean');
  const result1 = await worker.recognize(clean);
  onProgress(40);
  let combined = result1.data.text || '';

  if (mode === 'accurate') {
    const hard = await preprocessImage(file, 'hard');
    const result2 = await worker.recognize(hard);
    onProgress(70);
    combined += `\n${result2.data.text || ''}`;
  }

  // Check if full-page OCR got us 26 clean words
  const fullPageWords = extractWords(combined);
  if (fullPageWords.length >= 26) {
    onProgress(100);
    return fullPageWords.slice(0, 26).join(' ');
  }

  // Phase 2: Cell-by-cell grid as fallback (only if full-page missed words)
  const gridWords = await recognizeZanoGrid(worker, file, onProgress);
  const filled = gridWords.filter(Boolean);

  // Merge: prefer full-page words, use grid to fill if full-page was short
  if (fullPageWords.length < 26 && filled.length > fullPageWords.length) {
    combined = `${fullPageWords.join(' ')}\n${gridWords.join(' ')}`;
  }

  onProgress(100);
  return combined;
}

function itemFromFile(file: File): OcrItem {
  return {
    id: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(16).slice(2)}`,
    file,
    name: file.webkitRelativePath || file.name,
    size: file.size,
    preview: URL.createObjectURL(file),
    status: 'queued',
    progress: 0,
    rawText: '',
    normalizedWords: [],
    phrases: [],
    bestPhrase: '',
  };
}

export default function Home() {
  const [items, setItems] = useState<OcrItem[]>([]);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<Mode>('fast');
  const [workerProgress, setWorkerProgress] = useState('ready');
  const [pasteStatus, setPasteStatus] = useState('Ctrl+V ready');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const foundCount = items.filter((i) => i.bestPhrase).length;
  const totalPhraseText = useMemo(() => {
    return items
      .filter((i) => i.bestPhrase)
      .map((i) => `# ${i.name}\n${i.bestPhrase}`)
      .join('\n\n');
  }, [items]);

  function addFiles(files: FileList | File[]) {
    const imageFiles = Array.from(files).filter((f) => f.type.startsWith('image/'));
    if (!imageFiles.length) return;
    setItems((prev) => [...prev, ...imageFiles.map(itemFromFile)]);
    setPasteStatus(`${imageFiles.length} image${imageFiles.length > 1 ? 's' : ''} added`);
  }

  function addClipboardFiles(files: File[]) {
    if (!files.length) {
      setPasteStatus('No image found in clipboard');
      return;
    }
    addFiles(files);
  }

  function handlePasteEvent(event: ClipboardEvent) {
    const files: File[] = [];
    for (const item of Array.from(event.clipboardData?.items ?? [])) {
      if (!item.type.startsWith('image/')) continue;
      const blob = item.getAsFile();
      if (!blob) continue;
      files.push(new File([blob], `pasted-${Date.now()}-${files.length + 1}.${blob.type.split('/')[1] || 'png'}`, { type: blob.type || 'image/png' }));
    }
    if (files.length) {
      event.preventDefault();
      addClipboardFiles(files);
    }
  }

  async function pasteFromClipboard() {
    try {
      if (!navigator.clipboard?.read) {
        setPasteStatus('Browser does not expose clipboard images. Use Ctrl+V instead.');
        return;
      }
      const clipboardItems = await navigator.clipboard.read();
      const files: File[] = [];
      for (const clipboardItem of clipboardItems) {
        const imageType = clipboardItem.types.find((type) => type.startsWith('image/'));
        if (!imageType) continue;
        const blob = await clipboardItem.getType(imageType);
        files.push(new File([blob], `pasted-${Date.now()}-${files.length + 1}.${imageType.split('/')[1] || 'png'}`, { type: imageType }));
      }
      addClipboardFiles(files);
    } catch (err) {
      setPasteStatus(err instanceof Error ? err.message : 'Clipboard read failed. Try Ctrl+V.');
    }
  }

  useEffect(() => {
    window.addEventListener('paste', handlePasteEvent);
    return () => window.removeEventListener('paste', handlePasteEvent);
  }, []);

  function updateItem(id: string, patch: Partial<OcrItem>) {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  async function startOcr() {
    if (running) return;
    setRunning(true);
    setWorkerProgress('loading OCR engine');

    let worker: Worker | null = null;
    try {
      worker = await createWorker('eng', 1, {
        logger: (m) => {
          if (m.status) setWorkerProgress(`${m.status}${m.progress ? ` ${(m.progress * 100).toFixed(0)}%` : ''}`);
        },
      });
      await worker.setParameters({
        tessedit_pageseg_mode: '6' as never,
        preserve_interword_spaces: '1',
        tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n-_,.;:|[](){}',
      });

      const queue = items.filter((i) => i.status === 'queued' || i.status === 'error');
      for (const item of queue) {
        try {
          updateItem(item.id, { status: 'processing', progress: 8, error: undefined });
          const rawText = await recognize(worker, item.file, mode, (p) => updateItem(item.id, { progress: p }));
          const words = extractWords(rawText);
          const phrases = extractPhrases(words);
          updateItem(item.id, {
            status: 'done',
            progress: 100,
            rawText,
            normalizedWords: words,
            phrases,
            bestPhrase: phrases[0] ?? ([23, 24, 25, 26].includes(words.length) ? words.join(' ') : ''),
          });
        } catch (err) {
          updateItem(item.id, { status: 'error', error: err instanceof Error ? err.message : String(err), progress: 0 });
        }
      }
    } finally {
      await worker?.terminate();
      setWorkerProgress('ready');
      setRunning(false);
    }
  }

  async function copyText(text: string) {
    await navigator.clipboard.writeText(text);
  }

  function downloadResults() {
    const blob = new Blob([totalPhraseText || 'No phrases found yet.'], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'zano-phrases-extracted.txt';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#050816] text-white">
      <div className="absolute inset-0 bg-grid opacity-40" />
      <div className="absolute left-1/2 top-0 h-[520px] w-[520px] -translate-x-1/2 rounded-full bg-cyan-500/15 blur-3xl" />
      <div className="absolute right-[-140px] top-[220px] h-[420px] w-[420px] rounded-full bg-purple-600/15 blur-3xl" />

      <section className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-8 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-white/10 pb-6 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs font-semibold text-cyan-100">
              <ShieldCheck className="h-3.5 w-3.5" /> 100% browser-side OCR • no upload
            </div>
            <h1 className="mt-4 text-4xl font-black tracking-tight sm:text-6xl">
              Zano Extract <span className="bg-gradient-to-r from-cyan-300 via-blue-400 to-purple-400 bg-clip-text text-transparent">Kenshi</span>
            </h1>
            <p className="mt-3 max-w-2xl text-base leading-7 text-gray-300">
              Extract 23/24/25/26-word Zano recovery phrases from many screenshots at once. Drop images, paste screenshots, or upload a folder; OCR runs locally in your browser.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-sm text-gray-300 shadow-glow">
            <div className="flex items-center gap-2 text-white"><ScanText className="h-4 w-4 text-cyan-300" /> OCR engine</div>
            <div className="mt-2 font-mono text-xs text-cyan-100">{workerProgress}</div>
          </div>
        </header>

        <div className="grid flex-1 gap-6 py-8 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-5">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }}
              className="rounded-[2rem] border border-dashed border-cyan-300/30 bg-white/[0.045] p-6 shadow-[0_24px_80px_rgba(0,0,0,0.22)] backdrop-blur"
            >
              <div className="flex min-h-60 flex-col items-center justify-center rounded-[1.5rem] border border-white/10 bg-black/20 p-4 text-center">
                <UploadCloud className="h-12 w-12 text-cyan-300" />
                <h2 className="mt-4 text-2xl font-bold">Upload screenshots</h2>
                <p className="mt-2 max-w-md text-sm text-gray-400">Multiple images sekaligus bisa. Folder upload bisa di desktop. Screenshot juga bisa langsung Ctrl+V / Paste.</p>
                <div className="mt-5 flex flex-wrap justify-center gap-3">
                  <button onClick={() => fileInputRef.current?.click()} className="inline-flex items-center gap-2 rounded-xl bg-cyan-300 px-4 py-2 font-bold text-black hover:bg-cyan-200">
                    <ImageIcon className="h-4 w-4" /> Pick images
                  </button>
                  <button onClick={() => folderInputRef.current?.click()} className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/10 px-4 py-2 font-bold text-white hover:bg-white/15">
                    <FolderOpen className="h-4 w-4" /> Pick folder
                  </button>
                  <button onClick={pasteFromClipboard} className="inline-flex items-center gap-2 rounded-xl border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 font-bold text-cyan-100 hover:bg-cyan-300/15">
                    <ClipboardPaste className="h-4 w-4" /> Paste image
                  </button>
                </div>
                <div className="mt-3 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-gray-400">
                  {pasteStatus}
                </div>
              </div>

              <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => e.target.files && addFiles(e.target.files)} />
              <input ref={folderInputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => e.target.files && addFiles(e.target.files)} {...({ webkitdirectory: 'true', directory: 'true' } as Record<string, string>)} />

              <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
                <div className="flex rounded-xl border border-white/10 bg-black/25 p-1 text-sm">
                  <button onClick={() => setMode('fast')} className={`rounded-lg px-3 py-2 font-semibold ${mode === 'fast' ? 'bg-cyan-300 text-black' : 'text-gray-300'}`}>Fast</button>
                  <button onClick={() => setMode('accurate')} className={`rounded-lg px-3 py-2 font-semibold ${mode === 'accurate' ? 'bg-cyan-300 text-black' : 'text-gray-300'}`}>Accurate 2-pass</button>
                </div>
                <div className="flex gap-2">
                  <button disabled={!items.length || running} onClick={startOcr} className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-300 to-purple-400 px-4 py-2 font-black text-black disabled:cursor-not-allowed disabled:opacity-40">
                    <Zap className="h-4 w-4" /> {running ? 'Scanning...' : 'Start OCR'}
                  </button>
                  <button disabled={running || !items.length} onClick={() => setItems([])} className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-4 py-2 font-semibold text-white disabled:opacity-40">
                    <Trash2 className="h-4 w-4" /> Clear
                  </button>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <Stat label="Images" value={items.length} />
              <Stat label="Found" value={foundCount} />
              <Stat label="Mode" value={mode === 'fast' ? '1x' : '2x'} />
            </div>

            <div className="rounded-2xl border border-yellow-300/20 bg-yellow-300/10 p-4 text-sm leading-6 text-yellow-50">
              <div className="mb-1 flex items-center gap-2 font-bold"><AlertTriangle className="h-4 w-4" /> Security note</div>
              Jangan paste hasil phrase ke chat / server. Tool ini static client-side; tetap treat hasilnya sebagai secret.
            </div>
          </div>

          <div className="space-y-5">
            <div className="rounded-[2rem] border border-white/10 bg-white/[0.045] p-5 backdrop-blur">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-2xl font-bold">Extraction results</h2>
                  <p className="text-sm text-gray-400">Best candidate appears first per screenshot.</p>
                </div>
                <div className="flex gap-2">
                  <button disabled={!totalPhraseText} onClick={() => copyText(totalPhraseText)} className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/10 px-3 py-2 text-sm font-bold disabled:opacity-40"><Copy className="h-4 w-4" /> Copy all</button>
                  <button disabled={!items.length} onClick={downloadResults} className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/10 px-3 py-2 text-sm font-bold disabled:opacity-40"><Download className="h-4 w-4" /> TXT</button>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {items.length === 0 ? (
                <div className="rounded-[2rem] border border-white/10 bg-black/20 p-10 text-center text-gray-400">No screenshots yet.</div>
              ) : items.map((item) => (
                <ResultCard key={item.id} item={item} onCopy={copyText} />
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return <div className="rounded-2xl border border-white/10 bg-white/[0.045] p-4"><div className="text-xs uppercase tracking-[0.2em] text-gray-500">{label}</div><div className="mt-1 text-2xl font-black text-white">{value}</div></div>;
}

function ResultCard({ item, onCopy }: { item: OcrItem; onCopy: (text: string) => void }) {
  return (
    <article className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-white/[0.045] backdrop-blur">
      <div className="grid gap-4 p-4 sm:grid-cols-[120px_1fr]">
        <img src={item.preview} alt="uploaded screenshot" className="h-28 w-full rounded-xl object-cover sm:w-28" />
        <div className="min-w-0">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="min-w-0">
              <h3 className="truncate font-bold text-white">{item.name}</h3>
              <p className="text-xs text-gray-500">{formatBytes(item.size)} • {item.normalizedWords.length} OCR words</p>
            </div>
            <StatusBadge status={item.status} />
          </div>

          {item.status === 'processing' && <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-cyan-300 transition-all" style={{ width: `${item.progress}%` }} /></div>}
          {item.error && <p className="mt-3 text-sm text-red-300">{item.error}</p>}

          {item.bestPhrase ? (
            <div className="mt-4 rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-sm font-bold text-cyan-100"><CheckCircle2 className="h-4 w-4" /> Best phrase ({item.bestPhrase.split(' ').length} words)</div>
                <button onClick={() => onCopy(item.bestPhrase)} className="rounded-lg bg-cyan-300 px-2 py-1 text-xs font-black text-black">Copy</button>
              </div>
              <textarea
                readOnly
                value={item.bestPhrase}
                className="mb-3 h-20 w-full resize-none rounded-xl border border-white/10 bg-black/25 p-3 font-mono text-xs leading-5 text-cyan-50 outline-none"
                aria-label="Plain phrase without numbering"
              />
              <div className="flex flex-wrap gap-1.5">
                {item.bestPhrase.split(' ').map((word, i) => (
                  <span key={`${word}-${i}`} className="phrase-word rounded-lg border border-white/10 bg-black/25 px-2 py-1 font-mono text-xs text-white">
                    <span className="mr-1 text-gray-500 select-none" aria-hidden="true">{i + 1}.</span>
                    <span>{word}</span>
                  </span>
                ))}
              </div>
            </div>
          ) : item.status === 'done' ? (
            <div className="mt-4 rounded-2xl border border-red-300/20 bg-red-300/10 p-3 text-sm text-red-100">No 23–26 word phrase candidate found. Try Accurate 2-pass or crop the screenshot around the phrase.</div>
          ) : null}

          {item.phrases.length > 1 && (
            <details className="mt-3 rounded-xl border border-white/10 bg-black/20 p-3 text-sm text-gray-300">
              <summary className="cursor-pointer font-bold text-white">Alternative candidates ({item.phrases.length - 1})</summary>
              <div className="mt-3 space-y-2">
                {item.phrases.slice(1).map((p, idx) => <button key={idx} onClick={() => onCopy(p)} className="block w-full rounded-lg bg-white/5 p-2 text-left font-mono text-xs hover:bg-white/10">{p}</button>)}
              </div>
            </details>
          )}
        </div>
      </div>
    </article>
  );
}

function StatusBadge({ status }: { status: Status }) {
  const cls = status === 'done' ? 'bg-green-300/15 text-green-200 border-green-300/30' : status === 'error' ? 'bg-red-300/15 text-red-200 border-red-300/30' : status === 'processing' ? 'bg-cyan-300/15 text-cyan-200 border-cyan-300/30' : 'bg-white/10 text-gray-300 border-white/10';
  return <span className={`rounded-full border px-3 py-1 text-xs font-bold ${cls}`}>{status}</span>;
}
