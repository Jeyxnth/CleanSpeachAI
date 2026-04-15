import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import UploadPanel from "@/components/dashboard/UploadPanel";
import PipelineVisualizer, { type StageData } from "@/components/dashboard/PipelineVisualizer";
import ResultsPanel from "@/components/dashboard/ResultsPanel";
import { useToast } from "@/hooks/use-toast";

// ========== PIPELINE SIMULATION CONFIG ==========
const DEFAULT_FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "literally", "right", "so", "actually"];
const BACKEND_BASE_URL = (import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

interface ProcessApiResponse {
  transcript?: string;
  original_transcript?: string;
  raw_transcript: string;
  cleaned_transcript: string;
  removed_words: string[];
  filler_count: number;
  flagged_count: number;
  audio_download_url: string;
  original_audio_url?: string;
  cleaned_audio_url?: string;
  mode: "normal" | "clear";
  noise_applied: boolean;
  noise_reduction_percent?: number;
  fillers_removed?: number;
  original_word_count?: number;
  cleaned_word_count?: number;
  clarity_score?: number;
  noise_reduction_success?: boolean;
  noise_reduction_error?: string | null;
  noise_reduction_applied?: boolean;
  noise_reduction_failed?: boolean;
  rms_diff?: number;
  snr_db?: number;
  model_used: string;
  chunked_vad_applied: boolean;
  chunk_count: number;
  large_model_requested: boolean;
  model_fallback_applied: boolean;
}

function getWordCount(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

function toAbsoluteAudioUrl(audioDownloadUrl: string): string {
  if (/^https?:\/\//i.test(audioDownloadUrl)) return audioDownloadUrl;
  const normalizedPath = audioDownloadUrl.startsWith("/") ? audioDownloadUrl : `/${audioDownloadUrl}`;
  return `${BACKEND_BASE_URL}${normalizedPath}`;
}
// ================================================

const initialStages = (): StageData[] => Array.from({ length: 5 }, () => ({ status: "idle" as const }));

export default function Dashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stages, setStages] = useState<StageData[]>(initialStages());
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [resultData, setResultData] = useState<any>(null);
  const [transcriptionMode, setTranscriptionMode] = useState<"normal" | "clear">("clear");
  const [applyNoiseReduction, setApplyNoiseReduction] = useState(true);

  // Draw waveform from audio file
  const drawWaveform = useCallback((audioFile: File) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const audioCtx = new AudioContext();
        const buffer = await audioCtx.decodeAudioData(e.target?.result as ArrayBuffer);
        const data = buffer.getChannelData(0);
        const width = canvas.offsetWidth * window.devicePixelRatio;
        const height = canvas.offsetHeight * window.devicePixelRatio;
        canvas.width = width;
        canvas.height = height;

        ctx.clearRect(0, 0, width, height);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;

        ctx.beginPath();
        ctx.strokeStyle = "hsl(263, 84%, 58%)";
        ctx.lineWidth = 1.5;

        for (let i = 0; i < width; i++) {
          let min = 1.0, max = -1.0;
          for (let j = 0; j < step; j++) {
            const datum = data[i * step + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
          }
          ctx.moveTo(i, (1 + min) * amp);
          ctx.lineTo(i, (1 + max) * amp);
        }
        ctx.stroke();
        audioCtx.close();
      } catch {
        // fallback: draw placeholder bars
        const width = canvas.offsetWidth * window.devicePixelRatio;
        const height = canvas.offsetHeight * window.devicePixelRatio;
        canvas.width = width;
        canvas.height = height;
        ctx.fillStyle = "hsl(263, 84%, 58%)";
        for (let i = 0; i < width; i += 4) {
          const h = Math.random() * height * 0.8;
          ctx.fillRect(i, (height - h) / 2, 2, h);
        }
      }
    };
    reader.readAsArrayBuffer(audioFile);
  }, []);

  const handleFileSelected = useCallback((f: File) => {
    setFile(f);
    setShowResults(false);
    setStages(initialStages());
    setTimeout(() => drawWaveform(f), 100);
  }, [drawWaveform]);

  const handleTranscriptionModeChange = useCallback((mode: "normal" | "clear") => {
    setTranscriptionMode(mode);
  }, []);

  const runPipeline = useCallback(async () => {
    if (!file || !user) return;
    setPipelineRunning(true);
    setShowResults(false);
    setStages(initialStages());

    const liveStages = initialStages();
    liveStages[0] = { status: "processing" };
    setStages([...liveStages]);

    // Upload to storage
    const filePath = `${user.id}/${Date.now()}_${file.name}`;
    const { error: uploadError } = await supabase.storage
      .from("cleanspeech-audio")
      .upload(filePath, file);

    if (uploadError) {
      toast({ title: "Upload failed", description: uploadError.message, variant: "destructive" });
      setPipelineRunning(false);
      return;
    }

    liveStages[0] = { status: "complete", stats: { "Storage Upload": "Done" } };
    liveStages[1] = { status: "processing" };
    setStages([...liveStages]);

    // Create job record
    const { data: job, error: jobError } = await supabase
      .from("audio_jobs")
      .insert({
        user_id: user.id,
        original_filename: file.name,
        original_audio_url: filePath,
        status: "processing",
      })
      .select()
      .single();

    if (jobError || !job) {
      toast({ title: "Failed to create job", variant: "destructive" });
      setPipelineRunning(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("mode", transcriptionMode);
      formData.append("apply_noise_reduction", String(applyNoiseReduction));
      formData.append("language", "en");

      const response = await fetch(`${BACKEND_BASE_URL}/process`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload?.detail === "string" ? payload.detail : "Audio processing failed";
        throw new Error(detail);
      }

      const processResult = payload as ProcessApiResponse;
      const rawTranscript = processResult.original_transcript || processResult.raw_transcript || processResult.transcript || "";
      const cleanedTranscript = processResult.transcript || processResult.cleaned_transcript || rawTranscript;
      const rawCount = processResult.original_word_count ?? getWordCount(rawTranscript);
      const cleanedCount = processResult.cleaned_word_count ?? getWordCount(cleanedTranscript);
      const fillersRemoved = processResult.fillers_removed ?? processResult.filler_count ?? 0;
      const clarityScore = processResult.clarity_score ?? Math.max(0, Math.min(100, 60 + fillersRemoved * 2));

      const removedUnique = Array.from(new Set((processResult.removed_words || []).map((word) => word.toLowerCase())));
      const fillerWords = removedUnique.filter((word) => DEFAULT_FILLER_WORDS.includes(word));
      const censoredWords = removedUnique.filter((word) => !DEFAULT_FILLER_WORDS.includes(word));
      const cleanedAudioUrl = toAbsoluteAudioUrl(processResult.cleaned_audio_url || processResult.audio_download_url);
      const originalAudioUrl = processResult.original_audio_url ? toAbsoluteAudioUrl(processResult.original_audio_url) : undefined;
      const noiseReductionPercent = processResult.noise_reduction_percent ?? 0;
      const noiseWasApplied = Boolean(processResult.noise_reduction_success ?? processResult.noise_reduction_applied);
      const noiseReductionFailed = Boolean(processResult.noise_reduction_error || processResult.noise_reduction_failed);
      const rmsDiff = processResult.rms_diff ?? 0;
      const snrDb = processResult.snr_db ?? 0;
      const activeMode = processResult.mode || transcriptionMode;

      const completedStages: StageData[] = [
        {
          status: "complete",
          stats: {
            "Storage Upload": "Done",
            "Noise Reduction": noiseWasApplied ? `${noiseReductionPercent}%` : "Bypassed",
          },
        },
        {
          status: "complete",
          stats: {
            "Words Detected": rawCount,
            "Language": "English",
            "Model": processResult.model_used,
          },
        },
        { status: "complete", stats: { "Filler Words": fillersRemoved, "Flagged Words": processResult.flagged_count ?? 0 } },
        { status: "complete", stats: { "Fillers Removed": fillersRemoved, "Words Censored": processResult.flagged_count ?? 0 } },
        {
          status: "complete",
          stats: {
            "Mode": activeMode,
            "SNR": `${snrDb} dB`,
            "Original Words": rawCount,
            "Cleaned Words": cleanedCount,
            "Audio URL": "Ready",
          },
        },
      ];
      setStages(completedStages);

      const results = {
        rawTranscript,
        cleanedTranscript,
        fillerWords,
        censoredWords,
        originalAudioUrl,
        cleanedAudioUrl,
        noiseReductionApplied: noiseWasApplied,
        noiseReductionFailed,
        rmsDiff,
        snrDb,
        stats: {
          noiseReduction: noiseReductionPercent,
          fillersRemoved,
          wordsCensored: processResult.flagged_count ?? 0,
          clarityScore,
          originalWordCount: rawCount,
          cleanedWordCount: cleanedCount,
        },
      };

      await supabase.from("audio_jobs").update({
        raw_transcript: rawTranscript,
        cleaned_transcript: cleanedTranscript,
        cleaned_audio_url: cleanedAudioUrl,
        filler_count: fillersRemoved,
        bleep_count: processResult.flagged_count ?? 0,
        noise_reduction_pct: noiseReductionPercent,
        clarity_score: clarityScore,
        word_count_original: rawCount,
        word_count_cleaned: cleanedCount,
        status: "complete",
      }).eq("id", job.id);

      setResultData(results);
      setShowResults(true);
    } catch (error: any) {
      liveStages[1] = { status: "error" };
      setStages([...liveStages]);

      await supabase.from("audio_jobs").update({ status: "error" }).eq("id", job.id);
      toast({
        title: "Audio processing failed",
        description: error?.message || "Could not generate cleaned audio",
        variant: "destructive",
      });
    } finally {
      setPipelineRunning(false);
    }
  }, [file, user, toast, transcriptionMode, applyNoiseReduction]);

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-display font-bold">Dashboard</h1>
        <p className="text-text-secondary mt-1">Upload audio and run the CleanSpeech pipeline</p>
      </div>

      <UploadPanel
        onFileSelected={handleFileSelected}
        file={file}
        waveformCanvasRef={canvasRef}
        transcriptionMode={transcriptionMode}
        onTranscriptionModeChange={handleTranscriptionModeChange}
        applyNoiseReduction={applyNoiseReduction}
        onApplyNoiseReductionChange={setApplyNoiseReduction}
        onRunPipeline={runPipeline}
        pipelineRunning={pipelineRunning}
      />

      <PipelineVisualizer stages={stages} />

      {showResults && resultData && (
        <ResultsPanel {...resultData} />
      )}
    </div>
  );
}
