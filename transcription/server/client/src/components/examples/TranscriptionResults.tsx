import TranscriptionResults from '../TranscriptionResults';

export default function TranscriptionResultsExample() {
  const sampleText = `Welcome to our audio transcription service. This is a sample transcription that demonstrates how the results will be displayed after processing your audio file.

The transcription system accurately converts spoken words into written text, preserving the natural flow and structure of the conversation. You can easily copy this text to your clipboard or download it as a text file for further use.

This technology is useful for interviews, meetings, podcasts, and any other audio content that needs to be converted to text format.`;

  return (
    <div className="max-w-4xl mx-auto p-8">
      <TranscriptionResults 
        text={sampleText}
        wordCount={92}
        processingTime={3.5}
      />
    </div>
  );
}
