import FileUploader from '../FileUploader';

export default function FileUploaderExample() {
  const handleFileSelect = (file: File) => {
    console.log('File selected:', file.name);
  };

  return (
    <div className="max-w-4xl mx-auto p-8">
      <FileUploader onFileSelect={handleFileSelect} />
    </div>
  );
}
