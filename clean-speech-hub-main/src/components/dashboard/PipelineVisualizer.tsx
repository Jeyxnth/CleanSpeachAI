import PipelineStage, { type StageStatus } from "./PipelineStage";
import { AudioLines, Mic, Brain, Filter, Sparkles } from "lucide-react";

export interface StageData {
  status: StageStatus;
  stats?: Record<string, string | number>;
}

interface PipelineVisualizerProps {
  stages: StageData[];
}

const stageConfig = [
  { icon: AudioLines, label: "DSP Filter", subtitle: "Noise suppression" },
  { icon: Mic, label: "STT Engine", subtitle: "Speech to text" },
  { icon: Brain, label: "NLP / ML", subtitle: "Language analysis" },
  { icon: Filter, label: "Word Filter", subtitle: "Content sanitization" },
  { icon: Sparkles, label: "Output", subtitle: "Clean audio + transcript" },
];

export default function PipelineVisualizer({ stages }: PipelineVisualizerProps) {
  return (
    <div className="glass-card rounded-xl p-6 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[200px] bg-primary/5 blur-[80px] rounded-full pointer-events-none" />

      <h2 className="text-xl font-display font-semibold mb-6 relative z-10">Processing Pipeline</h2>

      <div className="flex flex-col md:flex-row items-stretch gap-2 md:gap-0 relative z-10">
        {stageConfig.map((cfg, i) => (
          <div key={cfg.label} className="flex items-center flex-1 min-w-0">
            <PipelineStage
              icon={cfg.icon}
              label={cfg.label}
              subtitle={cfg.subtitle}
              status={stages[i]?.status ?? "idle"}
              stats={stages[i]?.stats}
            />
            {i < 4 && (
              <svg className="hidden md:block w-8 h-2 shrink-0 mx-1" viewBox="0 0 32 8">
                <line
                  x1="0" y1="4" x2="32" y2="4"
                  stroke={stages[i]?.status === "complete" ? "hsl(160,60%,52%)" : "hsl(263,50%,20%)"}
                  strokeWidth="2"
                  strokeDasharray="6 4"
                  className={stages[i]?.status === "complete" ? "animate-dash-flow" : ""}
                />
              </svg>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
