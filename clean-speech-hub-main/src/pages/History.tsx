import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";
import { History as HistoryIcon, Download, FileAudio } from "lucide-react";
import { Button } from "@/components/ui/button";

const BACKEND_BASE_URL = (import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

interface AudioJob {
  id: string;
  created_at: string;
  original_filename: string;
  filler_count: number | null;
  bleep_count: number | null;
  clarity_score: number | null;
  status: string;
  cleaned_transcript: string | null;
  cleaned_audio_url: string | null;
}

function resolveCleanedAudioUrl(rawUrl: string | null): string | null {
  if (!rawUrl) return null;
  if (/^https?:\/\//i.test(rawUrl)) return rawUrl;
  const normalizedPath = rawUrl.startsWith("/") ? rawUrl : `/${rawUrl}`;
  return `${BACKEND_BASE_URL}${normalizedPath}`;
}

export default function HistoryPage() {
  const { user } = useAuth();
  const [jobs, setJobs] = useState<AudioJob[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;
    supabase
      .from("audio_jobs")
      .select("id, created_at, original_filename, filler_count, bleep_count, clarity_score, status, cleaned_transcript, cleaned_audio_url")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .then(({ data }) => {
        setJobs(data ?? []);
        setLoading(false);
      });
  }, [user]);

  const statusBadge = (status: string) => {
    const map: Record<string, string> = {
      complete: "bg-success/20 text-success",
      processing: "bg-warning/20 text-warning",
      pending: "bg-muted text-text-muted",
      error: "bg-danger/20 text-danger",
    };
    return <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${map[status] ?? map.pending}`}>{status}</span>;
  };

  const handleDownload = (job: AudioJob) => {
    if (!job.cleaned_transcript) return;
    const blob = new Blob([job.cleaned_transcript], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${job.original_filename}_cleaned.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadAudio = (job: AudioJob) => {
    const cleanedAudioUrl = resolveCleanedAudioUrl(job.cleaned_audio_url);
    if (!cleanedAudioUrl) return;
    const anchor = document.createElement("a");
    anchor.href = cleanedAudioUrl;
    anchor.download = `${job.original_filename}_cleaned.wav`;
    anchor.click();
  };

  return (
    <div className="max-w-6xl mx-auto animate-fade-in">
      <div className="mb-6">
        <h1 className="text-3xl font-display font-bold flex items-center gap-2">
          <HistoryIcon className="h-7 w-7 text-primary" />
          History
        </h1>
        <p className="text-text-secondary mt-1">View your past audio processing jobs</p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-text-muted">Loading...</div>
      ) : jobs.length === 0 ? (
        <div className="glass-card rounded-xl p-12 text-center">
          <FileAudio className="h-16 w-16 text-text-muted mx-auto mb-4" />
          <h3 className="text-lg font-display font-semibold text-text-secondary">No jobs yet</h3>
          <p className="text-text-muted mt-1">Upload an audio file from the Dashboard to get started</p>
        </div>
      ) : (
        <div className="glass-card rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left text-xs text-text-muted font-medium p-4">Date</th>
                  <th className="text-left text-xs text-text-muted font-medium p-4">Filename</th>
                  <th className="text-center text-xs text-text-muted font-medium p-4">Fillers</th>
                  <th className="text-center text-xs text-text-muted font-medium p-4">Censored</th>
                  <th className="text-center text-xs text-text-muted font-medium p-4">Clarity</th>
                  <th className="text-center text-xs text-text-muted font-medium p-4">Status</th>
                  <th className="text-left text-xs text-text-muted font-medium p-4">Audio</th>
                  <th className="text-right text-xs text-text-muted font-medium p-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id} className="border-b border-border/50 hover:bg-muted/20 transition-colors align-top">
                    <td className="p-4 text-sm">{new Date(job.created_at).toLocaleDateString()}</td>
                    <td className="p-4 text-sm font-mono-code">{job.original_filename}</td>
                    <td className="p-4 text-sm text-center">{job.filler_count ?? "—"}</td>
                    <td className="p-4 text-sm text-center">{job.bleep_count ?? "—"}</td>
                    <td className="p-4 text-sm text-center">{job.clarity_score ? `${job.clarity_score}%` : "—"}</td>
                    <td className="p-4 text-center">{statusBadge(job.status)}</td>
                    <td className="p-4">
                      {job.status === "complete" && resolveCleanedAudioUrl(job.cleaned_audio_url) ? (
                        <audio
                          controls
                          className="w-52"
                          src={resolveCleanedAudioUrl(job.cleaned_audio_url) || undefined}
                          preload="none"
                        >
                          Your browser does not support audio.
                        </audio>
                      ) : (
                        <span className="text-xs text-text-muted">—</span>
                      )}
                    </td>
                    <td className="p-4 text-right">
                      {job.status === "complete" && (
                        <div className="flex justify-end gap-2">
                          <Button variant="ghost" size="sm" onClick={() => handleDownload(job)}>
                            <Download className="h-3 w-3 mr-1" /> Transcript
                          </Button>
                          {resolveCleanedAudioUrl(job.cleaned_audio_url) && (
                            <Button variant="ghost" size="sm" onClick={() => handleDownloadAudio(job)}>
                              <Download className="h-3 w-3 mr-1" /> Audio
                            </Button>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
