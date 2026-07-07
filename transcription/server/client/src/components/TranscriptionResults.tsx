import { useState } from 'react';
import { Clipboard, Download, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

interface TranscriptionResultsProps {
  text: string;
  wordCount?: number;
  processingTime?: number;
}

export default function TranscriptionResults({ 
  text, 
  wordCount, 
  processingTime 
}: TranscriptionResultsProps) {
  const [copied, setCopied] = useState(false);
  const [useMonospace, setUseMonospace] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transcription.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold" data-testid="text-results-title">
            Transcription Results
          </h2>
          {wordCount !== undefined && (
            <span className="text-sm text-muted-foreground" data-testid="text-word-count">
              {wordCount} words
            </span>
          )}
          {processingTime !== undefined && (
            <span className="text-sm text-muted-foreground" data-testid="text-processing-time">
              Processed in {processingTime}s
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <Switch
              id="monospace"
              checked={useMonospace}
              onCheckedChange={setUseMonospace}
              data-testid="switch-monospace"
            />
            <Label htmlFor="monospace" className="text-sm">
              Monospace
            </Label>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleCopy}
            data-testid="button-copy"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4 mr-2" />
                Copied
              </>
            ) : (
              <>
                <Clipboard className="w-4 h-4 mr-2" />
                Copy
              </>
            )}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleDownload}
            data-testid="button-download"
          >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>
      </div>

      <Card className="p-6">
        <div 
          className={`whitespace-pre-wrap leading-relaxed max-w-prose ${useMonospace ? 'font-mono text-sm' : ''}`}
          data-testid="text-transcription-content"
        >
          {text}
        </div>
      </Card>
    </div>
  );
}
