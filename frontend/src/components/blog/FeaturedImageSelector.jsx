/**
 * Featured Image Selector with Drag & Drop
 * Features:
 * - Drag & drop image upload
 * - Click to select
 * - Image preview with cropping indicator
 * - Remove/replace functionality
 * - Aspect ratio guidelines
 */
import React, { useState, useCallback, useRef } from 'react';

const FeaturedImageSelector = ({
  featuredImage,
  onImageSelect,
  onImageRemove,
  uploadedImages = [],
  onUploadImage,
  aspectRatio = '16:9'
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      await uploadImage(imageFile);
    }
  }, []);

  const handleFileSelect = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      await uploadImage(file);
    }
  }, []);

  const uploadImage = async (file) => {
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 100);
      
      // Upload the image
      if (onUploadImage) {
        const url = await onUploadImage(file);
        if (url) {
          onImageSelect(url);
        }
      } else {
        // Fallback: create local URL
        const url = URL.createObjectURL(file);
        onImageSelect(url);
      }
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
      }, 500);
    } catch (err) {
      console.error('Image upload failed:', err);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const getAspectRatioClass = () => {
    switch (aspectRatio) {
      case '16:9':
        return 'aspect-video';
      case '4:3':
        return 'aspect-[4/3]';
      case '1:1':
        return 'aspect-square';
      case '3:2':
        return 'aspect-[3/2]';
      default:
        return 'aspect-video';
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl border-2 border-gray-700 p-5">
      <label className="block text-sm font-medium text-gray-200 mb-3">
        Featured Image
      </label>

      {/* Main Image Area */}
      {featuredImage ? (
        <div className="relative group">
          <div className={`${getAspectRatioClass()} rounded-lg overflow-hidden bg-gray-900`}>
            <img
              src={featuredImage}
              alt="Featured"
              className="w-full h-full object-cover"
            />
          </div>
          
          {/* Overlay Actions */}
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Replace
            </button>
            <button
              type="button"
              onClick={onImageRemove}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Remove
            </button>
          </div>

          {/* Aspect Ratio Badge */}
          <div className="absolute top-2 right-2 px-2 py-1 bg-black/50 rounded text-xs text-white">
            {aspectRatio}
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`${getAspectRatioClass()} rounded-lg border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center ${
            isDragging 
              ? 'border-blue-500 bg-blue-500/10' 
              : 'border-gray-600 hover:border-gray-500 bg-gray-900/50 hover:bg-gray-900'
          }`}
        >
          {isUploading ? (
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-3 relative">
                <svg className="w-full h-full animate-spin text-blue-500" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              </div>
              <p className="text-sm text-gray-300">Uploading... {uploadProgress}%</p>
            </div>
          ) : (
            <>
              <svg className="w-12 h-12 text-gray-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-gray-300 text-sm font-medium mb-1">
                {isDragging ? 'Drop image here' : 'Click or drag to upload'}
              </p>
              <p className="text-gray-500 text-xs">
                Recommended: {aspectRatio} aspect ratio, max 10MB
              </p>
            </>
          )}
        </div>
      )}

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Gallery from uploaded images */}
      {uploadedImages.length > 0 && !featuredImage && (
        <div className="mt-4">
          <p className="text-xs text-gray-400 mb-2">Or select from uploaded images:</p>
          <div className="grid grid-cols-4 gap-2">
            {uploadedImages.slice(0, 8).map((img, index) => (
              <button
                key={index}
                type="button"
                onClick={() => onImageSelect(img.url)}
                className="aspect-square rounded-lg overflow-hidden border-2 border-transparent hover:border-blue-500 transition-colors"
              >
                <img
                  src={img.url}
                  alt={`Option ${index + 1}`}
                  className="w-full h-full object-cover"
                />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="mt-4 p-3 bg-gray-900 rounded-lg">
        <p className="text-xs text-gray-400">
          <span className="font-medium text-gray-300">Tips:</span> Use high-quality images (1200x675px recommended). 
          Featured images appear in blog listings, social shares, and search results.
        </p>
      </div>
    </div>
  );
};

export default FeaturedImageSelector;
