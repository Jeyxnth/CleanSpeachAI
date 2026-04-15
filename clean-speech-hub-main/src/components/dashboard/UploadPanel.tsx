import { useCallback, useRef, useState } from "react";
import { AudioWaveform, Upload, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface UploadPanelProps {
  onFileSelected: (file: File) => void;
  file: File | null;
  waveformCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  transcriptionMode: "normal" | "clear";
  onTranscriptionModeChange: (mode: "normal" | "clear") => void;
  applyNoiseReduction: boolean;
  onApplyNoiseReductionChange: (enabled: boolean) => void;
  onRunPipeline: () => void;
  pipelineRunning: boolean;
}

const ACCEPTED = ".wav,.mp3,.m4a,.ogg";

export default function UploadPanel({
  onFileSelected,
  file,
  waveformCanvasRef,
  transcriptionMode,
  onTranscriptionModeChange,
  applyNoiseReduction,
  onApplyNoiseReductionChange,
  onRunPipeline,
  pipelineRunning,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFileSelected(f);
  }, [onFileSelected]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onFileSelected(f);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="glass-card rounded-xl p-6">
      <h2 className="text-xl font-display font-semibold mb-4 flex items-center gap-2">
        <Upload className="h-5 w-5 text-primary" />
        Upload Audio
      </h2>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
          dragOver ? "border-primary bg-primary/5 animate-pulse-glow" : "border-border hover:border-primary/50"
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <AudioWaveform className="h-12 w-12 text-primary/60 mx-auto mb-3" />
        <p className="text-foreground font-medium">Drop your audio file here or click to browse</p>
        <p className="text-text-muted text-sm mt-1">Supports WAV, MP3, M4A, OGG</p>
        <input ref={inputRef} type="file" accept={ACCEPTED} className="hidden" onChange={handleChange} />
      </div>

      {/* File info + waveform */}
      {file && (
        <div className="mt-4 animate-slide-up">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-sm font-medium">{file.name}</p>
              <p className="text-xs text-text-muted">{formatSize(file.size)}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
            <label className="bg-muted/20 rounded-lg p-3 border border-border/60">
              <span className="block text-xs text-text-muted mb-1">Transcription Mode</span>
              <select
                className="w-full bg-background/60 border border-border rounded-md px-2 py-1.5 text-sm"
                value={transcriptionMode}
                onChange={(e) => onTranscriptionModeChange(e.target.value as "normal" | "clear")}
                disabled={pipelineRunning}
              >
                <option value="normal">Normal (faster)</option>
                <option value="clear">Clear (higher accuracy)</option>
              </select>
            </label>

            <label className="bg-muted/20 rounded-lg p-3 border border-border/60 flex items-center justify-between gap-4">
              <div>
                <span className="block text-xs text-text-muted">Noise Reduction</span>
                <span className="text-sm">Preprocess audio before STT</span>
              </div>
              <input
                type="checkbox"
                checked={applyNoiseReduction}
                onChange={(e) => onApplyNoiseReductionChange(e.target.checked)}
                disabled={pipelineRunning}
                className="h-4 w-4"
              />
            </label>

          </div>

          <canvas
            ref={waveformCanvasRef}
            className="w-full h-24 rounded-lg bg-muted/30"
          />
          <Button
            className="w-full mt-4"
            size="lg"
            onClick={onRunPipeline}
            disabled={pipelineRunning}
          >
            {pipelineRunning ? (
              <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...</>
            ) : (
              <><AudioWaveform className="mr-2 h-4 w-4" /> Run CleanSpeech Pipeline</>
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
