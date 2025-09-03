import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { createBlogPost, uploadBlogImage } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';

const NewBlogPost = () => {
  const { darkMode } = useTheme();
  const navigate = useNavigate();
  const fileInputRef = useRef();
  
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    district: '',
    authorName: '',
    authorPhoto: ''
  });
  
  const [images, setImages] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState({});

  // Istanbul districts that are supported by the AI chatbot
  const chatbotDistricts = [
    'Beyoğlu', 'Sultanahmet', 'Fatih', 'Kadıköy', 'Beşiktaş', 'Şişli', 
    'Üsküdar', 'Bakırköy', 'Galata', 'Taksim', 'Ortaköy', 'Karaköy', 'Eminönü'
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleImageSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    // Validate file types and sizes
    const validFiles = files.filter(file => {
      const isValidType = file.type.startsWith('image/');
      const isValidSize = file.size <= 10 * 1024 * 1024; // 10MB
      
      if (!isValidType) {
        setError('Please select only image files');
        return false;
      }
      
      if (!isValidSize) {
        setError('Image size should be less than 10MB');
        return false;
      }
      
      return true;
    });

    if (validFiles.length > 0) {
      setError(null);
      uploadImages(validFiles);
    }
  };

  const uploadImages = async (files) => {
    setUploading(true);
    
    try {
      for (const file of files) {
        const fileName = `${Date.now()}-${file.name}`;
        setUploadProgress(prev => ({ ...prev, [fileName]: 0 }));
        
        try {
          const uploadedImage = await uploadBlogImage(file);
          
          setImages(prev => [...prev, {
            id: Date.now() + Math.random(),
            file,
            url: uploadedImage.url,
            alt_text: ''
          }]);
          
          setUploadProgress(prev => ({ ...prev, [fileName]: 100 }));
        } catch (err) {
          console.error('Failed to upload image:', err);
          setError(`Failed to upload ${file.name}: ${err.message}`);
        }
      }
    } finally {
      setUploading(false);
      // Clear progress after a delay
      setTimeout(() => setUploadProgress({}), 2000);
    }
  };

  const removeImage = (imageId) => {
    setImages(prev => prev.filter(img => img.id !== imageId));
  };

  const updateImageAltText = (imageId, altText) => {
    setImages(prev => prev.map(img =>
      img.id === imageId ? { ...img, alt_text: altText } : img
    ));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.title.trim() || !formData.content.trim()) {
      setError('Title and content are required');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const postData = {
        title: formData.title.trim(),
        content: formData.content.trim(),
        district: formData.district || null,
        author_name: formData.authorName.trim() || null,
        author_photo: formData.authorPhoto.trim() || null,
        images: images.map(img => ({
          url: img.url,
          alt_text: img.alt_text || formData.title
        }))
      };

      const newPost = await createBlogPost(postData);
      navigate(`/blog/${newPost.id}`);
    } catch (err) {
      setError(err.message || 'Failed to create blog post');
      console.error('Failed to create post:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const insertText = (before, after = '') => {
    const textarea = document.querySelector('textarea[name="content"]');
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = formData.content.substring(start, end);
    const newText = formData.content.substring(0, start) + 
                   before + selectedText + after + 
                   formData.content.substring(end);
    
    setFormData(prev => ({ ...prev, content: newText }));
    
    // Restore cursor position
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(
        start + before.length,
        end + before.length
      );
    }, 0);
  };

  return (
    <div className={`min-h-screen pt-24 px-4 transition-colors duration-200 ${
      darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
    }`}>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl md:text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Share Your Istanbul Story
          </h1>
          <p className="text-gray-300 text-lg">
            Tell fellow travelers about your experiences, discoveries, and recommendations in Istanbul.
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className={`mb-6 p-4 rounded-lg border transition-colors duration-200 ${
            darkMode 
              ? 'bg-red-900/20 border-red-500/20'
              : 'bg-red-50 border-red-200'
          }`}>
            <p className={darkMode ? 'text-red-400' : 'text-red-700'}>{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Title */}
          <div className={`rounded-lg p-6 transition-colors duration-200 ${
            darkMode ? 'bg-gray-800' : 'bg-white shadow-lg border border-gray-200'
          }`}>
            <label htmlFor="title" className={`block text-sm font-semibold mb-2 transition-colors duration-200 ${
              darkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>
              Title *
            </label>
            <input
              type="text"
              id="title"
              name="title"
              value={formData.title}
              onChange={handleInputChange}
              placeholder="Give your story a compelling title..."
              className={`w-full px-4 py-3 border rounded-lg focus:outline-none text-lg transition-colors duration-200 ${
                darkMode 
                  ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500'
                  : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
              }`}
              required
            />
          </div>

          {/* Content */}
          <div className="bg-gray-800 rounded-lg p-6">
            <label htmlFor="content" className="block text-sm font-semibold text-gray-200 mb-2">
              Your Story *
            </label>
            
            {/* Simple text formatting toolbar */}
            <div className="flex flex-wrap gap-2 mb-4 p-3 bg-gray-700 rounded-t-lg border-b border-gray-600">
              <button
                type="button"
                onClick={() => insertText('**', '**')}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm transition-colors duration-200"
                title="Bold"
              >
                <strong>B</strong>
              </button>
              <button
                type="button"
                onClick={() => insertText('*', '*')}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm transition-colors duration-200"
                title="Italic"
              >
                <em>I</em>
              </button>
              <button
                type="button"
                onClick={() => insertText('\n## ', '')}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm transition-colors duration-200"
                title="Heading"
              >
                H2
              </button>
              <button
                type="button"
                onClick={() => insertText('\n- ', '')}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm transition-colors duration-200"
                title="Bullet Point"
              >
                • List
              </button>
            </div>

            <textarea
              id="content"
              name="content"
              value={formData.content}
              onChange={handleInputChange}
              placeholder="Share your Istanbul experience... Tell us about the places you visited, the food you tried, the people you met, and any tips you have for other travelers."
              rows={15}
              className="w-full px-4 py-3 bg-gray-700 text-white border border-gray-600 rounded-b-lg focus:outline-none focus:border-indigo-500 resize-y"
              required
            />
            <div className="mt-2 text-sm text-gray-400">
              You can use **bold**, *italic*, and ## headings in your text.
            </div>
          </div>

          {/* Images */}
          <div className="bg-gray-800 rounded-lg p-6">
            <label className="block text-sm font-semibold text-gray-200 mb-4">
              Images
            </label>
            
            <div className="mb-4">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageSelect}
                accept="image/*"
                multiple
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg transition-colors duration-200"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                {uploading ? 'Uploading...' : 'Add Images'}
              </button>
              <p className="mt-2 text-sm text-gray-400">
                Upload photos to illustrate your story. Max 10MB per image.
              </p>
            </div>

            {/* Upload Progress */}
            {Object.keys(uploadProgress).length > 0 && (
              <div className="mb-4 space-y-2">
                {Object.entries(uploadProgress).map(([fileName, progress]) => (
                  <div key={fileName} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="flex justify-between text-sm text-gray-300 mb-1">
                        <span>{fileName.split('-').slice(1).join('-')}</span>
                        <span>{progress}%</span>
                      </div>
                      <div className="w-full bg-gray-600 rounded-full h-2">
                        <div 
                          className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Image Preview */}
            {images.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {images.map((image, index) => (
                  <div key={image.id} className="relative bg-gray-700 rounded-lg overflow-hidden">
                    <div className="aspect-video">
                      <img
                        src={URL.createObjectURL(image.file)}
                        alt={`Preview ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    
                    <div className="p-3">
                      <input
                        type="text"
                        value={image.alt_text}
                        onChange={(e) => updateImageAltText(image.id, e.target.value)}
                        placeholder="Describe this image..."
                        className="w-full px-3 py-2 bg-gray-600 text-white text-sm border border-gray-500 rounded focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    
                    <button
                      type="button"
                      onClick={() => removeImage(image.id)}
                      className="absolute top-2 right-2 p-1 bg-red-600 hover:bg-red-700 text-white rounded-full transition-colors duration-200"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* District */}
          <div className="bg-gray-800 rounded-lg p-6">
            <label htmlFor="district" className="block text-sm font-semibold text-gray-200 mb-2">
              District or Area in Istanbul
            </label>
            <select
              id="district"
              name="district"
              value={formData.district}
              onChange={handleInputChange}
              className="w-full px-4 py-3 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:border-indigo-500"
            >
              <option value="">Select a district or area...</option>
              {chatbotDistricts.map(district => (
                <option key={district} value={district}>{district}</option>
              ))}
            </select>
          </div>

          {/* Author Information */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-200 mb-4">Author Information</h3>
            
            <div className="space-y-4">
              <div>
                <label htmlFor="authorName" className="block text-sm font-semibold text-gray-200 mb-2">
                  Your Name
                </label>
                <input
                  type="text"
                  id="authorName"
                  name="authorName"
                  value={formData.authorName}
                  onChange={handleInputChange}
                  placeholder="Enter your name..."
                  className="w-full px-4 py-3 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:border-indigo-500"
                />
              </div>
              
              <div>
                <label htmlFor="authorPhoto" className="block text-sm font-semibold text-gray-200 mb-2">
                  Profile Photo URL (Optional)
                </label>
                <input
                  type="url"
                  id="authorPhoto"
                  name="authorPhoto"
                  value={formData.authorPhoto}
                  onChange={handleInputChange}
                  placeholder="https://example.com/your-photo.jpg"
                  className="w-full px-4 py-3 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:border-indigo-500"
                />
                <p className="mt-2 text-sm text-gray-400">
                  Add a link to your profile photo (optional). Use services like Gravatar, LinkedIn, or any public image URL.
                </p>
              </div>
            </div>
          </div>

          {/* Submit Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-end">
            <button
              type="button"
              onClick={() => navigate('/blog')}
              className="px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors duration-200"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting || uploading}
              className="px-8 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold rounded-lg transition-colors duration-200"
            >
              {submitting ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Publishing...
                </div>
              ) : (
                'Publish Story'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NewBlogPost;
