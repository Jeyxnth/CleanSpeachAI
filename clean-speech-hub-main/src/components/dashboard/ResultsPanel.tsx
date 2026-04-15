import { Copy, Download, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

interface ResultsPanelProps {
  rawTranscript: string;
  cleanedTranscript: string;
  fillerWords: string[];
  censoredWords: string[];
  originalAudioUrl?: string;
  stats: {
    noiseReduction: number;
    fillersRemoved: number;
    wordsCensored: number;
    clarityScore: number;
    originalWordCount: number;
    cleanedWordCount: number;
  };
  cleanedAudioUrl?: string;
  noiseReductionApplied?: boolean;
  noiseReductionFailed?: boolean;
  rmsDiff?: number;
  snrDb?: number;
}

function AnimatedStat({ label, value, suffix = "" }: { label: string; value: number; suffix?: string }) {
  return (
    <div className="glass-card rounded-lg p-4 animate-count-up">
      <p className="text-xs text-text-muted mb-1">{label}</p>
      <p className="text-2xl font-display font-bold text-foreground">
        {value}{suffix}
      </p>
    </div>
  );
}

function highlightWords(text: string, fillerWords: string[], censoredWords: string[]) {
  if (!text) return null;
  const words = text.split(/(\s+)/);
  return words.map((word, i) => {
    const clean = word.toLowerCase().replace(/[^a-z]/g, "");
    if (censoredWords.map(w => w.toLowerCase()).includes(clean)) {
      return <span key={i} className="bg-danger/20 text-danger px-1 rounded">[BLEEP]</span>;
    }
    if (fillerWords.map(w => w.toLowerCase()).includes(clean)) {
      return <span key={i} className="bg-warning/20 text-warning px-1 rounded">{word}</span>;
    }
    return <span key={i}>{word}</span>;
  });
}

export default function ResultsPanel({
  rawTranscript,
  cleanedTranscript,
  fillerWords,
  censoredWords,
  originalAudioUrl,
  stats,
  cleanedAudioUrl,
  noiseReductionApplied,
  noiseReductionFailed,
  rmsDiff,
  snrDb,
}: ResultsPanelProps) {
  const { toast } = useToast();
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(cleanedTranscript);
    setCopied(true);
    toast({ title: "Copied to clipboard!" });
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadTranscript = () => {
    const blob = new Blob([cleanedTranscript], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cleaned_transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="glass-card rounded-xl p-6 animate-slide-up">
      <h2 className="text-xl font-display font-semibold mb-6 flex items-center gap-2">
        <CheckCircle className="h-5 w-5 text-success" />
        Results
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Transcripts */}
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-text-secondary mb-2">Original Transcript</h3>
            <div className="bg-muted/30 rounded-lg p-4 font-mono-code text-sm leading-relaxed max-h-48 overflow-y-auto">
              {highlightWords(rawTranscript, fillerWords, censoredWords)}
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-text-secondary">Cleaned Transcript</h3>
              <Button variant="ghost" size="sm" onClick={handleCopy} className="text-xs">
                {copied ? <CheckCircle className="h-3 w-3 mr-1 text-success" /> : <Copy className="h-3 w-3 mr-1" />}
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>
            <div className="bg-muted/30 rounded-lg p-4 font-mono-code text-sm leading-relaxed max-h-48 overflow-y-auto">
              {cleanedTranscript}
            </div>
          </div>
        </div>

        {/* Right: Stats + Audio */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <AnimatedStat label="Noise Reduction" value={stats.noiseReduction} suffix="%" />
            <AnimatedStat label="Fillers Removed" value={stats.fillersRemoved} />
            <AnimatedStat label="Words Censored" value={stats.wordsCensored} />
            <AnimatedStat label="Clarity Score" value={stats.clarityScore} suffix="%" />
            <AnimatedStat label="Original Words" value={stats.originalWordCount} />
            <AnimatedStat label="Cleaned Words" value={stats.cleanedWordCount} />
          </div>

          {(originalAudioUrl || cleanedAudioUrl) && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                {noiseReductionApplied && !noiseReductionFailed && (
                  <span className="text-xs px-2 py-1 rounded-full bg-success/20 text-success font-medium">
                    Noise Reduction Applied ✓
                  </span>
                )}
                {noiseReductionFailed && (
                  <span className="text-xs px-2 py-1 rounded-full bg-danger/20 text-danger font-medium">
                    Noise Reduction Failed ✗
                  </span>
                )}
                {!noiseReductionApplied && !noiseReductionFailed && (
                  <span className="text-xs px-2 py-1 rounded-full bg-muted/30 text-text-muted font-medium">
                    Noise Reduction Off
                  </span>
                )}
              </div>

              {typeof snrDb === "number" && (
                <p className="text-sm text-text-secondary">
                  Audio Quality Improved: <span className="font-semibold text-success">+{snrDb.toFixed(2)} dB SNR</span>
                </p>
              )}

              {typeof rmsDiff === "number" && (
                <p className="text-xs text-text-muted">RMS Difference: {rmsDiff.toFixed(6)}</p>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {originalAudioUrl && (
                  <div>
                    <h3 className="text-sm font-semibold text-text-secondary mb-2">Original Audio</h3>
                    <audio controls className="w-full" src={originalAudioUrl}>
                      Your browser does not support audio.
                    </audio>
                  </div>
                )}
                {cleanedAudioUrl && (
                  <div>
                    <h3 className="text-sm font-semibold text-text-secondary mb-2">Cleaned Audio</h3>
                    <audio controls className="w-full" src={cleanedAudioUrl}>
                      Your browser does not support audio.
                    </audio>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="flex gap-3">
            {cleanedAudioUrl && (
              <Button variant="outline" size="sm" asChild className="flex-1">
                <a href={cleanedAudioUrl} download>
                  <Download className="h-3 w-3 mr-1" /> Download Audio
                </a>
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={handleDownloadTranscript} className="flex-1">
              <Download className="h-3 w-3 mr-1" /> Download Transcript
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
