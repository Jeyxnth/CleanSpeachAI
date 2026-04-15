import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AudioWaveform, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function Auth() {
  const { user, loading: authLoading } = useAuth();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);
  const { signUp, signIn } = useAuth();
  const { toast } = useToast();

  if (authLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (user) return <Navigate to="/" replace />;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    if (isSignUp) {
      if (password !== confirmPassword) {
        toast({ title: "Passwords don't match", variant: "destructive" });
        setLoading(false);
        return;
      }
      const { error } = await signUp(email, password, fullName);
      if (error) {
        toast({ title: "Sign up failed", description: error.message, variant: "destructive" });
      } else {
        toast({ title: "Check your email", description: "We sent you a confirmation link." });
      }
    } else {
      const { error } = await signIn(email, password);
      if (error) {
        toast({ title: "Sign in failed", description: error.message, variant: "destructive" });
      }
    }
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen animate-fade-in">
      {/* Left side - branding */}
      <div className="hidden lg:flex lg:w-1/2 flex-col items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-background to-background" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full bg-primary/10 blur-[120px]" />
        <div className="relative z-10 text-center px-12">
          <div className="flex items-center justify-center gap-3 mb-6">
            <AudioWaveform className="h-12 w-12 text-primary" />
            <h1 className="text-5xl font-display font-bold text-gradient">CleanSpeech</h1>
          </div>
          <p className="text-xl text-text-secondary max-w-md">
            Turn noisy speech into crystal-clear communication
          </p>
          {/* Pipeline decoration */}
          <div className="mt-12 flex items-center justify-center gap-4">
            {["DSP", "STT", "NLP", "Filter", "Output"].map((stage, i) => (
              <div key={stage} className="flex items-center gap-4">
                <div className="glass-card rounded-lg px-4 py-2 text-sm text-text-secondary font-mono">
                  {stage}
                </div>
                {i < 4 && (
                  <svg width="24" height="2" className="text-primary/40">
                    <line x1="0" y1="1" x2="24" y2="1" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" />
                  </svg>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right side - auth form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="lg:hidden flex items-center gap-2 mb-8 justify-center">
            <AudioWaveform className="h-8 w-8 text-primary" />
            <span className="text-2xl font-display font-bold text-gradient">CleanSpeech</span>
          </div>

          <h2 className="text-3xl font-display font-bold mb-2">
            {isSignUp ? "Create Account" : "Welcome Back"}
          </h2>
          <p className="text-text-secondary mb-8">
            {isSignUp ? "Sign up to start processing audio" : "Sign in to your account"}
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignUp && (
              <div>
                <Label htmlFor="fullName" className="text-text-secondary">Full Name</Label>
                <Input
                  id="fullName"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                  className="mt-1 bg-input border-border focus:border-primary"
                  placeholder="John Doe"
                />
              </div>
            )}
            <div>
              <Label htmlFor="email" className="text-text-secondary">Email</Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="mt-1 bg-input border-border focus:border-primary"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <Label htmlFor="password" className="text-text-secondary">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="mt-1 bg-input border-border focus:border-primary"
                placeholder="••••••••"
              />
            </div>
            {isSignUp && (
              <div>
                <Label htmlFor="confirmPassword" className="text-text-secondary">Confirm Password</Label>
                <Input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  className="mt-1 bg-input border-border focus:border-primary"
                  placeholder="••••••••"
                />
              </div>
            )}
            <Button type="submit" className="w-full" size="lg" disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isSignUp ? "Create Account" : "Sign In"}
            </Button>
          </form>

          <p className="mt-6 text-center text-text-secondary text-sm">
            {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
            <button
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-primary hover:underline font-medium"
            >
              {isSignUp ? "Sign In" : "Sign Up"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
