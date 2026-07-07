import { useCallback, useState } from 'react';
import { CloudUpload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  acceptedFormats?: string[];
  maxSizeMB?: number;
}

export default function FileUploader({
  onFileSelect,
  acceptedFormats = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/m4a', 'audio/x-m4a'],
  maxSizeMB = 100
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  }, []);

  const handleFile = (file: File) => {
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > maxSizeMB) {
      alert(`File size exceeds ${maxSizeMB}MB limit`);
      return;
    }
    
    onFileSelect(file);
  };

  return (
    <Card 
      className={`transition-all duration-300 ${isDragging ? 'border-primary bg-accent' : 'border-dashed'}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center justify-center min-h-64 p-8 text-center">
        <div className="mb-6 p-4 rounded-full bg-accent">
          <CloudUpload className="w-12 h-12 text-primary" />
        </div>
        
        <h2 className="text-xl font-semibold mb-2" data-testid="text-upload-title">
          Upload Audio File
        </h2>
        
        <p className="text-muted-foreground mb-6 max-w-sm" data-testid="text-upload-description">
          Drag and drop your audio file here, or click to browse
        </p>

        <input
          type="file"
          id="file-input"
          className="hidden"
          accept={acceptedFormats.join(',')}
          onChange={handleFileInput}
          data-testid="input-file"
        />
        
        <Button 
          onClick={() => document.getElementById('file-input')?.click()}
          size="lg"
          data-testid="button-select-file"
        >
          Select Audio File
        </Button>

        <div className="mt-6 space-y-1">
          <p className="text-sm text-muted-foreground" data-testid="text-supported-formats">
            Supported formats: MP3, WAV, M4A
          </p>
          <p className="text-sm text-muted-foreground" data-testid="text-file-limit">
            Maximum file size: {maxSizeMB}MB
          </p>
        </div>
      </div>
    </Card>
  );
}
