import FilePreview from '../FilePreview';

export default function FilePreviewExample() {
  const handleRemove = () => {
    console.log('Remove file triggered');
  };

  return (
    <div className="max-w-4xl mx-auto p-8">
      <FilePreview 
        fileName="interview_recording.mp3" 
        fileSize={5242880}
        onRemove={handleRemove}
      />
    </div>
  );
}
