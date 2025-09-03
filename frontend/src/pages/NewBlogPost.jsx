import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createBlogPost, uploadBlogImage } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import '../App.css';

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
  const [showScrollTop, setShowScrollTop] = useState(false);

  // Istanbul districts that are supported by the AI chatbot
  const chatbotDistricts = [
    'Beyoğlu', 'Sultanahmet', 'Fatih', 'Kadıköy', 'Beşiktaş', 'Şişli', 
    'Üsküdar', 'Bakırköy', 'Galata', 'Taksim', 'Ortaköy', 'Karaköy', 'Eminönü'
  ];

  // Handle scroll events for scroll-to-top button
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 400);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

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

    if (!formData.authorName.trim()) {
      setError('Author name is required');
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
    <div 
      className={`min-h-screen pt-56 sm:pt-64 md:pt-72 px-2 sm:px-4 pb-8 transition-colors duration-200 overflow-y-auto ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}
      style={{ scrollBehavior: 'smooth' }}
    >
      {/* AI Istanbul Logo - Top Left */}
      <Link to="/" style={{textDecoration: 'none'}} className="fixed z-50">
        <div className="logo-istanbul logo-move-top-left">
          <span className="logo-text">
            A/<span style={{fontWeight: 400}}>STANBUL</span>
          </span>
        </div>
      </Link>

      <div className="max-w-4xl mx-auto">
        {/* Simple Header */}
        <div className="mb-16 text-center">
          <h1 className="text-3xl sm:text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Share Your Istanbul Story
          </h1>
        </div>

        {/* Error Alert */}
        {error && (
          <div className={`mb-6 p-4 rounded-lg border-l-4 transition-colors duration-200 ${
            darkMode 
              ? 'bg-red-900/20 border-red-500 border-l-red-500'
              : 'bg-red-50 border-red-200 border-l-red-500'
          }`}>
            <p className={`font-medium ${darkMode ? 'text-red-400' : 'text-red-800'}`}>
              {error}
            </p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          
          {/* Username Input Row */}
          <div className={`p-6 rounded-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-600' : 'bg-white shadow-lg border border-gray-200'
          }`}>
            <div className="flex flex-col md:flex-row md:items-end gap-6">
              <div className="flex-1">
                <label htmlFor="authorName" className={`block text-lg font-semibold mb-3 transition-colors duration-200 ${
                  darkMode ? 'text-gray-200' : 'text-gray-700'
                }`}>
                  Your Name
                </label>
                <input
                  type="text"
                  id="authorName"
                  name="authorName"
                  value={formData.authorName}
                  onChange={handleInputChange}
                  placeholder="Enter your name or username..."
                  className={`w-full px-4 py-4 border rounded-xl focus:outline-none text-lg transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-700 text-white border-gray-600 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 placeholder-gray-400'
                      : 'bg-gray-50 text-gray-900 border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 placeholder-gray-500'
                  }`}
                  required
                  autoComplete="name"
                />
              </div>
              
              {/* Profile Photo Upload */}
              <div className="flex flex-col items-center">
                <label className={`block text-sm font-medium mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-gray-200' : 'text-gray-700'
                }`}>
                  Profile Photo (Optional)
                </label>
                <div className="relative">
                  <input
                    type="file"
                    id="authorPhotoUpload"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onloadend = () => {
                          setFormData(prev => ({ ...prev, authorPhoto: reader.result }));
                        };
                        reader.readAsDataURL(file);
                      }
                    }}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    type="button"
                    onClick={() => document.getElementById('authorPhotoUpload').click()}
                    className={`w-20 h-20 rounded-full border-2 border-dashed flex items-center justify-center transition-all duration-200 ${
                      darkMode
                        ? 'border-gray-500 hover:border-gray-400 bg-gray-700'
                        : 'border-gray-300 hover:border-gray-400 bg-gray-100'
                    }`}
                  >
                    {formData.authorPhoto ? (
                      <img
                        src={formData.authorPhoto}
                        alt="Profile"
                        className="w-full h-full rounded-full object-cover"
                      />
                    ) : (
                      <svg className={`w-8 h-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Post Title Input Row */}
          <div className={`p-6 rounded-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-600' : 'bg-white shadow-lg border border-gray-200'
          }`}>
            <label htmlFor="title" className={`block text-lg font-semibold mb-3 transition-colors duration-200 ${
              darkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>
              Post Title
            </label>
            <input
              type="text"
              id="title"
              name="title"
              value={formData.title}
              onChange={handleInputChange}
              placeholder="Give your Istanbul story a compelling title..."
              className={`w-full px-4 py-4 border rounded-xl focus:outline-none text-lg font-medium transition-all duration-200 ${
                darkMode 
                  ? 'bg-gray-700 text-white border-gray-600 focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20 placeholder-gray-400'
                  : 'bg-gray-50 text-gray-900 border-gray-300 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 placeholder-gray-500'
              }`}
              required
              autoComplete="off"
            />
          </div>

          {/* District Selection */}
          <div className={`p-6 rounded-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-600' : 'bg-white shadow-lg border border-gray-200'
          }`}>
            <label htmlFor="district" className={`block text-lg font-semibold mb-3 transition-colors duration-200 ${
              darkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>
              Istanbul District
            </label>
            <select
              id="district"
              name="district"
              value={formData.district}
              onChange={handleInputChange}
              className={`w-full px-4 py-4 border rounded-xl focus:outline-none text-lg transition-all duration-200 ${
                darkMode
                  ? 'bg-gray-700 text-white border-gray-600 focus:border-orange-400 focus:ring-2 focus:ring-orange-400/20'
                  : 'bg-gray-50 text-gray-900 border-gray-300 focus:border-orange-500 focus:ring-2 focus:ring-orange-500/20'
              }`}
            >
              <option value="">Select a district (optional)</option>
              {chatbotDistricts.map(district => (
                <option key={district} value={district}>
                  {district}
                </option>
              ))}
            </select>
            <p className={`mt-2 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Choose the main Istanbul district your story is about (helps others find relevant content)
            </p>
          </div>

          {/* Essay Section - Large */}
          <div className={`p-6 rounded-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-600' : 'bg-white shadow-lg border border-gray-200'
          }`}>
            <div className="flex items-center justify-between mb-4">
              <label htmlFor="content" className={`text-lg font-semibold transition-colors duration-200 ${
                darkMode ? 'text-gray-200' : 'text-gray-700'
              }`}>
                Your Story
              </label>
              
              {/* Essay Settings */}
              <div className="flex items-center space-x-4">
                <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Words: {formData.content.split(' ').filter(word => word.length > 0).length}
                </div>
                <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Characters: {formData.content.length}
                </div>
                <select
                  onChange={(e) => {
                    const fontSize = e.target.value;
                    const textarea = document.querySelector('textarea[name="content"]');
                    if (textarea) {
                      textarea.style.fontSize = fontSize;
                    }
                  }}
                  className={`text-sm px-3 py-1 rounded border ${
                    darkMode
                      ? 'bg-gray-700 text-white border-gray-600'
                      : 'bg-white text-gray-900 border-gray-300'
                  }`}
                >
                  <option value="16px">Normal</option>
                  <option value="18px">Large</option>
                  <option value="20px">Extra Large</option>
                </select>
              </div>
            </div>

            {/* Simple Formatting Tools */}
            <div className={`mb-4 p-3 rounded-lg border transition-colors duration-200 ${
              darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => insertText('**', '**')}
                  className={`px-3 py-2 rounded-lg text-sm font-bold transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Bold text"
                >
                  Bold
                </button>
                <button
                  type="button"
                  onClick={() => insertText('*', '*')}
                  className={`px-3 py-2 rounded-lg text-sm italic transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Italic text"
                >
                  Italic
                </button>
                <button
                  type="button"
                  onClick={() => insertText('\n## ', '')}
                  className={`px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Add heading"
                >
                  Heading
                </button>
                <button
                  type="button"
                  onClick={() => insertText('\n- ', '')}
                  className={`px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Add list item"
                >
                  List
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const textarea = document.querySelector('textarea[name="content"]');
                    textarea?.focus();
                  }}
                  className={`px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                    darkMode
                      ? 'bg-blue-600 hover:bg-blue-500 text-white'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                  title="Focus on writing"
                >
                  Focus
                </button>
              </div>
            </div>

            <textarea
              id="content"
              name="content"
              value={formData.content}
              onChange={handleInputChange}
              placeholder="Start writing your Istanbul story here...

Share your experiences, discoveries, and recommendations. Tell us about the places you visited, the food you tried, the people you met, and any tips you have for other travelers.

Use **bold** and *italic* text to highlight important points, ## for headings, and - for lists."
              rows={25}
              className={`w-full px-4 py-4 border rounded-xl focus:outline-none resize-y text-base leading-relaxed transition-all duration-200 ${
                darkMode
                  ? 'bg-gray-700 text-white border-gray-600 focus:border-green-400 focus:ring-2 focus:ring-green-400/20 placeholder-gray-400'
                  : 'bg-gray-50 text-gray-900 border-gray-300 focus:border-green-500 focus:ring-2 focus:ring-green-500/20 placeholder-gray-500'
              }`}
              required
              autoComplete="off"
              style={{ minHeight: '500px', fontSize: '16px' }}
            />
          </div>

          {/* Add Image Button */}
          <div className={`p-6 rounded-xl text-center transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-600' : 'bg-white shadow-lg border border-gray-200'
          }`}>
            <input
              type="file"
              id="imageUpload"
              name="imageUpload"
              ref={fileInputRef}
              onChange={handleImageSelect}
              accept="image/*"
              multiple
              className="hidden"
              aria-label="Upload images for your blog post"
            />
            
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 disabled:opacity-50 text-white font-semibold rounded-xl transition-all duration-200 text-lg shadow-lg hover:shadow-xl"
            >
              <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              {uploading ? 'Uploading Images...' : 'Add Images'}
            </button>
            <p className={`mt-2 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Upload photos to illustrate your story (JPG, PNG, WebP • Max 10MB each)
            </p>
          </div>

          {/* Upload Progress */}
          {Object.keys(uploadProgress).length > 0 && (
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
              <h4 className={`text-sm font-medium mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Uploading Images...
              </h4>
              <div className="space-y-3">
                {Object.entries(uploadProgress).map(([fileName, progress]) => (
                  <div key={fileName}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                        {fileName.split('-').slice(1).join('-')}
                      </span>
                      <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
                        {progress}%
                      </span>
                    </div>
                    <div className={`w-full rounded-full h-2 ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`}>
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Image Gallery */}
          {images.length > 0 && (
            <div className={`p-6 rounded-xl transition-colors duration-200 ${
              darkMode ? 'bg-gray-800 border border-gray-600' : 'bg-white shadow-lg border border-gray-200'
            }`}>
              <h4 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                Your Images ({images.length})
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {images.map((image, index) => (
                  <div key={image.id} className={`relative rounded-lg overflow-hidden ${
                    darkMode ? 'bg-gray-700 border border-gray-600' : 'bg-gray-50 border border-gray-200'
                  }`}>
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
                        placeholder="Image description..."
                        className={`w-full px-2 py-1 text-sm border rounded ${
                          darkMode
                            ? 'bg-gray-600 text-white border-gray-500 placeholder-gray-400'
                            : 'bg-white text-gray-900 border-gray-300 placeholder-gray-500'
                        }`}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => removeImage(image.id)}
                      className="absolute top-2 right-2 p-1 bg-red-600 hover:bg-red-700 text-white rounded-full"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Hidden Author Photo Field - now handled by file upload above */}
          <input type="hidden" name="authorPhoto" value={formData.authorPhoto} />

          {/* Submit Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8">
            <button
              type="button"
              onClick={() => navigate('/blog')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-200 ${
                darkMode
                  ? 'bg-gray-600 hover:bg-gray-700 text-white shadow-lg'
                  : 'bg-gray-300 hover:bg-gray-400 text-gray-800 shadow-lg hover:shadow-xl'
              }`}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting || uploading}
              className="px-12 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl text-lg"
            >
              {submitting ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                  Publishing...
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <span className="mr-2">🚀</span>
                  Publish Story
                </div>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Scroll to Top Button */}
      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className={`fixed bottom-6 right-6 p-3 rounded-full shadow-lg transition-all duration-300 z-40 ${
            darkMode
              ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white'
          }`}
          title="Scroll to top"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
        </button>
      )}

      {/* Simple Progress Indicator */}
      <div className="fixed top-32 right-4 z-30 hidden lg:block">
        <div className={`rounded-lg p-3 shadow-lg transition-all duration-200 ${
          darkMode ? 'bg-gray-800 text-white border border-gray-600' : 'bg-white text-gray-900 shadow-xl border border-gray-200'
        }`}>
          <div className="text-sm font-medium mb-2">Progress</div>
          <div className="space-y-2">
            <div className={`flex items-center text-xs ${
              formData.authorName ? 'text-green-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                formData.authorName ? 'bg-green-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Name {formData.authorName && '✓'}
            </div>
            <div className={`flex items-center text-xs ${
              formData.title ? 'text-green-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                formData.title ? 'bg-green-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Title {formData.title && '✓'}
            </div>
            <div className={`flex items-center text-xs ${
              formData.district ? 'text-orange-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                formData.district ? 'bg-orange-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              District {formData.district && '✓'}
            </div>
            <div className={`flex items-center text-xs ${
              formData.content.length > 50 ? 'text-green-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                formData.content.length > 50 ? 'bg-green-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Story {formData.content.length > 50 && '✓'}
            </div>
            <div className={`flex items-center text-xs ${
              images.length > 0 ? 'text-blue-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                images.length > 0 ? 'bg-blue-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Images ({images.length})
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewBlogPost;