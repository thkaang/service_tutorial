import { useState } from 'react';
import { Button } from '@/components/ui/button';
import FileUploader from '@/components/FileUploader';
import FilePreview from '@/components/FilePreview';
import ProcessingState from '@/components/ProcessingState';
import TranscriptionResults from '@/components/TranscriptionResults';
import ThemeToggle from '@/components/ThemeToggle';

type AppState = 'upload' | 'preview' | 'processing' | 'results';

export default function Home() {
  const [state, setState] = useState<AppState>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [transcriptionText, setTranscriptionText] = useState('');

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setState('preview');
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setState('upload');
  };

  const handleStartTranscription = async () => {
    setState('processing');

    if (!selectedFile) {
      console.error("No file selected.");
      return;
    }

    try {
      // 파일 전송을 위해 FormData 사용
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Flask 서버로 POST 요청 보내기
      const response = await fetch("http://127.0.0.1:5000/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      // Flask에서 반환한 텍스트
      const transcription = await response.text();

      setTranscriptionText(transcription);
      setState('results');
    } catch (error) {
      console.error(error);
      setTranscriptionText("Error occurred while processing the audio.");
      setState('results');
    }
    
//     setTimeout(() => {
//       const mockTranscription = `This is a sample transcription for the file "${selectedFile?.name}". In a production environment, this text would come from your custom audio-to-text transcription service.
//
// The audio file has been successfully processed and converted to text. The transcription system has analyzed the audio content and provided an accurate text representation of the spoken words.
//
// You can now copy this text, download it, or start a new transcription by uploading another audio file.`;
//
//       setTranscriptionText(mockTranscription);
//       setState('results');
//     }, 3000);
  };

  const handleNewTranscription = () => {
    setSelectedFile(null);
    setTranscriptionText('');
    setState('upload');
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-semibold" data-testid="text-app-title">
            Audio Transcription
          </h1>
          <ThemeToggle />
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        {state === 'upload' && (
          <FileUploader onFileSelect={handleFileSelect} />
        )}

        {state === 'preview' && selectedFile && (
          <div className="space-y-6">
            <FilePreview
              fileName={selectedFile.name}
              fileSize={selectedFile.size}
              onRemove={handleRemoveFile}
            />
            <div className="flex justify-end">
              <Button
                size="lg"
                onClick={handleStartTranscription}
                data-testid="button-start-transcription"
              >
                Start Transcription
              </Button>
            </div>
          </div>
        )}

        {state === 'processing' && (
          <ProcessingState />
        )}

        {state === 'results' && (
          <div className="space-y-6">
            <TranscriptionResults
              text={transcriptionText}
              wordCount={transcriptionText.split(/\s+/).length}
              processingTime={3.2}
            />
            <div className="flex justify-center">
              <Button
                variant="outline"
                onClick={handleNewTranscription}
                data-testid="button-new-transcription"
              >
                Upload Another File
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
