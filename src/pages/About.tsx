import React from 'react';
import { BookOpenText, FileText, Cpu, BarChart4 } from 'lucide-react';

export const About: React.FC = () => {
  return (
    <div className="py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">About the Project</h1>
          <p className="mt-4 text-lg text-gray-500">
            Learn more about our Arabic OCR system and the technologies behind it
          </p>
        </div>

        <div className="mt-12">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                    <BookOpenText className="h-6 w-6 text-white" />
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg font-medium text-gray-900">KHATT Dataset</h3>
                  </div>
                </div>
                <div className="mt-4 text-gray-500">
                  <p>
                    The KHATT (KFUPM Handwritten Arabic TexT) dataset is a comprehensive collection of handwritten Arabic text images. 
                    It contains thousands of text samples from different writers with various writing styles, making it ideal for training 
                    and evaluating Arabic OCR systems.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                    <Cpu className="h-6 w-6 text-white" />
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg font-medium text-gray-900">OCR Models</h3>
                  </div>
                </div>
                <div className="mt-4 text-gray-500">
                  <p>
                    Our system integrates multiple state-of-the-art OCR models:
                  </p>
                  <ul className="list-disc list-inside mt-2">
                    <li><strong>TrOCR:</strong> A transformer-based OCR model that combines image understanding with text generation.</li>
                    <li><strong>AR-OCR:</strong> A specialized model for Arabic text recognition with support for diacritics and complex ligatures.</li>
                    <li><strong>Tesseract:</strong> An open-source OCR engine with Arabic language support.</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                    <FileText className="h-6 w-6 text-white" />
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg font-medium text-gray-900">Technology Stack</h3>
                  </div>
                </div>
                <div className="mt-4 text-gray-500">
                  <p>
                    Our application is built using modern technologies:
                  </p>
                  <ul className="list-disc list-inside mt-2">
                    <li><strong>Frontend:</strong> React with TypeScript, Tailwind CSS for styling</li>
                    <li><strong>Backend:</strong> Flask API for model integration and processing</li>
                    <li><strong>Model Deployment:</strong> Python with specialized OCR libraries</li>
                    <li><strong>Data Processing:</strong> Custom preprocessing pipelines for Arabic text</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                    <BarChart4 className="h-6 w-6 text-white" />
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg font-medium text-gray-900">Performance</h3>
                  </div>
                </div>
                <div className="mt-4 text-gray-500">
                  <p>
                    Our system achieves state-of-the-art performance on Arabic text recognition:
                  </p>
                  <ul className="list-disc list-inside mt-2">
                    <li>High accuracy on printed and handwritten text</li>
                    <li>Support for various Arabic fonts and calligraphy styles</li>
                    <li>Robust handling of diacritics and special characters</li>
                    <li>Fast processing time with optimized model inference</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};