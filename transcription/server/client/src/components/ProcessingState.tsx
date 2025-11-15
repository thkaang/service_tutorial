import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface ProcessingStateProps {
  progress?: number;
}

export default function ProcessingState({ progress = 50 }: ProcessingStateProps) {
  return (
    <Card className="p-8">
      <div className="flex flex-col items-center gap-6">
        <div className="w-full max-w-md space-y-2">
          <div className="flex items-center justify-between">
            <p className="font-medium" data-testid="text-processing-status">
              Transcribing audio...
            </p>
            <p className="text-sm text-muted-foreground" data-testid="text-processing-progress">
              {progress}%
            </p>
          </div>
          <Progress value={progress} data-testid="progress-transcription" />
        </div>
        
        <p className="text-sm text-muted-foreground text-center" data-testid="text-processing-message">
          This may take a few moments depending on the audio length
        </p>
      </div>
    </Card>
  );
}
