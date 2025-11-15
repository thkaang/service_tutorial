import { FileText, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface FilePreviewProps {
  fileName: string;
  fileSize: number;
  onRemove: () => void;
}

export default function FilePreview({ fileName, fileSize, onRemove }: FilePreviewProps) {
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Card className="p-4">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-lg bg-accent">
          <FileText className="w-6 h-6 text-primary" />
        </div>
        
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate" data-testid="text-file-name">
            {fileName}
          </p>
          <p className="text-sm text-muted-foreground" data-testid="text-file-size">
            {formatFileSize(fileSize)}
          </p>
        </div>

        <Button
          variant="ghost"
          size="icon"
          onClick={onRemove}
          data-testid="button-remove-file"
        >
          <X className="w-5 h-5" />
        </Button>
      </div>
    </Card>
  );
}
