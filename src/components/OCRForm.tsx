import React, { useState } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Upload, Loader2 } from 'lucide-react';

type OCRModel = 'trocr' | 'ar-ocr' | 'tesseract';

export const OCRForm: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<OCRModel>('trocr');
  const [error, setError] = useState<string | null>(null);

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    },
    maxFiles: 1
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select an image first');
      return;
    }
    
    setLoading(true);
    setResult('');
    setError(null);
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', selectedModel);
    
    try {
      const response = await axios.post('http://localhost:5000/ocr', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setResult(response.data.text);
    } catch (err) {
      console.error('Error processing OCR:', err);
      setError('Error processing the image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select OCR Model
          </label>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => setSelectedModel('trocr')}
              className={`px-4 py-2 rounded-md text-sm font-medium ${
                selectedModel === 'trocr'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              TrOCR
            </button>
            <button
              type="button"
              onClick={() => setSelectedModel('ar-ocr')}
              className={`px-4 py-2 rounded-md text-sm font-medium ${
                selectedModel === 'ar-ocr'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              AR-OCR
            </button>
            <button
              type="button"
              onClick={() => setSelectedModel('tesseract')}
              className={`px-4 py-2 rounded-md text-sm font-medium ${
                selectedModel === 'tesseract'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              Tesseract
            </button>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload Arabic Text Image
          </label>
          <div
            {...getRootProps()}
            className={`mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-dashed rounded-md ${
              isDragActive ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'
            }`}
          >
            <div className="space-y-1 text-center">
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <div className="flex text-sm text-gray-600">
                <input {...getInputProps()} />
                <p className="pl-1">
                  Drag and drop an image here, or click to select a file
                </p>
              </div>
              <p className="text-xs text-gray-500">
                PNG, JPG, JPEG up to 10MB
              </p>
            </div>
          </div>
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
        </div>

        {preview && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Preview
            </label>
            <div className="mt-1 flex justify-center">
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 rounded-md"
              />
            </div>
          </div>
        )}

        <div>
          <button
            type="submit"
            disabled={loading}
            className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              loading
                ? 'bg-indigo-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                Processing...
              </>
            ) : (
              'Extract Text'
            )}
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Extracted Text
          </label>
          <div className="mt-1 p-4 bg-gray-50 rounded-md border border-gray-300 min-h-[100px] text-right" dir="rtl">
            {result}
          </div>
        </div>
      )}
    </div>
  );
};