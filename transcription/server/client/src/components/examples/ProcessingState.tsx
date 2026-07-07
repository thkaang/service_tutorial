import { useState, useEffect } from 'react';
import ProcessingState from '../ProcessingState';

export default function ProcessingStateExample() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => (prev >= 100 ? 0 : prev + 10));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-8">
      <ProcessingState progress={progress} />
    </div>
  );
}
