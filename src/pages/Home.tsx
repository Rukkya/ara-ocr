import React from 'react';
import { OCRForm } from '../components/OCRForm';

export const Home: React.FC = () => {
  return (
    <div className="py-10">
      <header className="mb-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900 text-center">
            Arabic OCR System
          </h1>
          <p className="mt-2 text-lg text-gray-600 text-center">
            Extract text from Arabic images using multiple OCR models
          </p>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
          <OCRForm />
        </div>
      </main>
    </div>
  );
};