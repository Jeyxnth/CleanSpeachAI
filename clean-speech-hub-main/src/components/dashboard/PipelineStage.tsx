import { Check, Loader2, AlertCircle } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type StageStatus = "idle" | "processing" | "complete" | "error";

interface PipelineStageProps {
  icon: LucideIcon;
  label: string;
  subtitle: string;
  status: StageStatus;
  stats?: Record<string, string | number>;
}

export default function PipelineStage({ icon: Icon, label, subtitle, status, stats }: PipelineStageProps) {
  const borderClass = {
    idle: "border-border",
    processing: "animated-border animate-pulse-glow",
    complete: "border-success/50 shadow-[0_0_20px_rgba(52,211,153,0.15)]",
    error: "border-danger/50",
  }[status];

  const badgeClass = {
    idle: "hidden",
    processing: "bg-warning/20 text-warning",
    complete: "bg-success/20 text-success",
    error: "bg-danger/20 text-danger",
  }[status];

  return (
    <div className={`glass-card rounded-xl p-4 transition-all duration-300 hover:scale-[1.02] border ${borderClass} min-w-[140px] flex-1`}>
      <div className="flex items-center gap-2 mb-2">
        <div className="relative">
          <Icon className={`h-6 w-6 ${status === "idle" ? "text-text-muted" : status === "processing" ? "text-warning" : status === "complete" ? "text-success" : "text-danger"}`} />
          {status === "processing" && (
            <Loader2 className="h-4 w-4 animate-spin text-warning absolute -top-1 -right-1" />
          )}
          {status === "complete" && (
            <Check className="h-3 w-3 text-success absolute -top-1 -right-1 bg-background rounded-full" />
          )}
          {status === "error" && (
            <AlertCircle className="h-3 w-3 text-danger absolute -top-1 -right-1" />
          )}
        </div>
        <div>
          <p className={`text-sm font-semibold ${status === "idle" ? "text-text-muted" : "text-foreground"}`}>{label}</p>
          <p className="text-xs text-text-muted">{subtitle}</p>
        </div>
      </div>

      {status !== "idle" && (
        <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${badgeClass}`}>
          {status === "processing" ? "Processing..." : status === "complete" ? "Done" : "Error"}
        </span>
      )}

      {status === "complete" && stats && (
        <div className="mt-2 space-y-1 animate-count-up">
          {Object.entries(stats).map(([key, value]) => (
            <div key={key} className="flex justify-between text-xs">
              <span className="text-text-muted">{key}</span>
              <span className="font-mono text-foreground">{value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
