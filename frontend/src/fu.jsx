import { useState } from 'react';
import { Upload, AlertCircle, CheckCircle } from 'lucide-react';

export default function VideoUpload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">DeepFake Detection</h1>

      <form onSubmit={handleUpload} className="space-y-4">
        <div className="border-2 border-dashed rounded-lg p-6 text-center">
          <input
            type="file"
            accept=".mp4,.avi,.mov"
            onChange={(e) => setFile(e.target.files[0])}
            className="hidden"
            id="video-upload"
          />
          <label
            htmlFor="video-upload"
            className="flex flex-col items-center cursor-pointer"
          >
            <Upload className="w-12 h-12 mb-2" />
            <span className="text-sm">
              {file ? file.name : 'Click to upload video'}
            </span>
          </label>
        </div>

        <button
          type="submit"
          disabled={!file || loading}
          className="w-full bg-blue-500 text-white py-2 rounded-lg disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Analyze Video'}
        </button>
      </form>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-4" role="alert">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 mr-2" />
            <span className="block sm:inline">{error}</span>
          </div>
        </div>
      )}

      {result && (
        <div 
          className={`
            ${result.is_fake ? 'bg-red-100 border-red-400 text-red-700' : 'bg-green-100 border-green-400 text-green-700'} 
            px-4 py-3 rounded relative mt-4
          `}
        >
          <div className="flex items-center">
            <CheckCircle className="h-5 w-5 mr-2" />
            <div>
              <p className="font-bold">
                {result.is_fake ? 'Potential DeepFake Detected' : 'Likely Authentic Video'}
              </p>
              <p className="text-sm">
                Confidence score: {(result.prediction * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}