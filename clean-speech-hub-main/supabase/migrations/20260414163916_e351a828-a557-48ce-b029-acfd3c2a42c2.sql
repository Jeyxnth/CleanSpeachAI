
-- Create audio_jobs table
CREATE TABLE public.audio_jobs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  original_filename TEXT NOT NULL,
  original_audio_url TEXT,
  cleaned_audio_url TEXT,
  raw_transcript TEXT,
  cleaned_transcript TEXT,
  filler_count INT DEFAULT 0,
  bleep_count INT DEFAULT 0,
  noise_reduction_pct FLOAT DEFAULT 0,
  clarity_score FLOAT DEFAULT 0,
  word_count_original INT DEFAULT 0,
  word_count_cleaned INT DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'complete', 'error'))
);

-- Enable RLS
ALTER TABLE public.audio_jobs ENABLE ROW LEVEL SECURITY;

-- Users can only see their own jobs
CREATE POLICY "Users can view own jobs" ON public.audio_jobs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can create own jobs" ON public.audio_jobs FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own jobs" ON public.audio_jobs FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own jobs" ON public.audio_jobs FOR DELETE USING (auth.uid() = user_id);

-- Create profiles table
CREATE TABLE public.profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON public.profiles FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can create own profile" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = user_id);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id, full_name)
  VALUES (NEW.id, NEW.raw_user_meta_data ->> 'full_name');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Storage bucket for audio files
INSERT INTO storage.buckets (id, name, public) VALUES ('cleanspeech-audio', 'cleanspeech-audio', false);

CREATE POLICY "Users can upload own audio" ON storage.objects FOR INSERT WITH CHECK (bucket_id = 'cleanspeech-audio' AND auth.uid()::text = (storage.foldername(name))[1]);
CREATE POLICY "Users can view own audio" ON storage.objects FOR SELECT USING (bucket_id = 'cleanspeech-audio' AND auth.uid()::text = (storage.foldername(name))[1]);
CREATE POLICY "Users can delete own audio" ON storage.objects FOR DELETE USING (bucket_id = 'cleanspeech-audio' AND auth.uid()::text = (storage.foldername(name))[1]);
