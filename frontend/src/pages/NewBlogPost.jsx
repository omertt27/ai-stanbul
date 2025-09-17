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
    heading: '',
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
    'General',
    'Beyoğlu', 'Sultanahmet', 'Fatih', 'Kadıköy', 'Beşiktaş', 'Şişli', 
    'Üsküdar', 'Bakırköy', 'Galata', 'Taksim', 'Ortaköy', 'Karaköy', 'Eminönü'
  ];

  const getWordCount = (text) => {
    if (!text || text.trim().length === 0) return 0;
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
  };

  const getCharacterCount = (text) => {
    return text ? text.length : 0;
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
        heading: formData.heading.trim() || null,
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
    if (!textarea) return;
    
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = formData.content.substring(start, end);
    
    // If no text is selected, provide default text for demonstration
    const textToWrap = selectedText || (
      before === '**' ? 'bold text' :
      before === '*' ? 'italic text' :
      before === '\n## ' ? 'Your Heading' :
      before === '\n- ' ? 'list item' :
      'text'
    );
    
    const newText = formData.content.substring(0, start) + 
                   before + textToWrap + after + 
                   formData.content.substring(end);
    
    setFormData(prev => ({ ...prev, content: newText }));
    
    // Restore cursor position and select the inserted text
    setTimeout(() => {
      textarea.focus();
      if (selectedText) {
        // If text was selected, place cursor after the formatting
        textarea.setSelectionRange(
          start + before.length + textToWrap.length + after.length,
          start + before.length + textToWrap.length + after.length
        );
      } else {
        // If no text was selected, select the default text so user can replace it
        textarea.setSelectionRange(
          start + before.length,
          start + before.length + textToWrap.length
        );
      }
    }, 0);
  };

  return (
    <div 
      className={`min-h-screen w-full pt-96 px-6 pb-12 transition-colors duration-200 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}
    >
      <div className="max-w-6xl mx-auto">

      <div className="pt-12 px-6 pb-10">
        <div className="max-w-4xl mx-auto">
          {/* Share Your Story Section */}
          <div className={`mb-6 p-6 rounded-xl border-2 transition-all duration-200 ${
            darkMode 
              ? 'bg-gray-800 border-gray-700' 
              : 'bg-white border-blue-200 shadow-md'
          }`}>
          <h2 className={`text-xl font-bold mb-2 transition-colors duration-200 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>
            Share Your Istanbul Story
          </h2>
          <p className={`text-sm transition-colors duration-200 ${
            darkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            Create your travel blog post about Istanbul. Share your experiences, discoveries, and recommendations with fellow travelers.
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className={`mb-2 p-2 rounded border-l-4 transition-colors duration-200 ${
            darkMode 
              ? 'bg-red-900/20 border-red-500 border-l-red-500'
              : 'bg-red-50 border-red-200 border-l-red-500'
          }`}>
            <p className={`text-sm font-medium ${darkMode ? 'text-red-400' : 'text-red-800'}`}>
              {error}
            </p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          
          {/* Compact Username Row */}
          <div className={`p-5 rounded-xl border-2 transition-all duration-200 ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
          }`}>
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <label htmlFor="authorName" className={`block text-sm font-medium mb-1 transition-colors duration-200 ${
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
                  placeholder="Enter your name..."
                  className="w-full px-2 py-2 border rounded text-sm focus:outline-none focus:ring-2 transition-all duration-200 bg-gray-700/50 text-white border-gray-600 focus:border-blue-400 focus:ring-blue-400/20 placeholder-gray-400"
                  required
                />
              </div>
              
              {/* Compact Profile Photo */}
              <div className="flex flex-col items-center">
                <label htmlFor="authorPhotoUpload" className={`text-xs mb-1 transition-colors duration-200 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  Photo
                </label>
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
                  className="sr-only"
                />
                <button
                  type="button"
                  onClick={() => document.getElementById('authorPhotoUpload').click()}
                  className={`w-10 h-10 rounded-full border border-dashed flex items-center justify-center transition-all duration-200 ${
                    darkMode
                      ? 'border-gray-600 hover:border-gray-500 bg-gray-700/50'
                      : 'border-gray-300 hover:border-gray-400 bg-gray-100'
                  }`}
                  aria-label="Upload profile photo"
                >
                  {formData.authorPhoto ? (
                    <img
                      src={formData.authorPhoto}
                      alt="Profile"
                      className="w-full h-full rounded-full object-cover"
                    />
                  ) : (
                    <svg className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Compact Title Row */}
          <div className={`p-5 rounded-xl border-2 transition-all duration-200 ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
          }`}>
            <label htmlFor="title" className={`block text-sm font-medium mb-1 transition-colors duration-200 ${
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
              placeholder="Your Istanbul story title..."
              className="w-full px-2 py-2 border rounded text-sm focus:outline-none focus:ring-2 transition-all duration-200 bg-gray-700/50 text-white border-gray-600 focus:border-blue-400 focus:ring-blue-400/20 placeholder-gray-400"
              required
            />
          </div>

          {/* Compact Heading Row */}
          <div className={`p-5 rounded-xl border-2 transition-all duration-200 ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
          }`}>
            <label htmlFor="heading" className={`block text-sm font-medium mb-1 transition-colors duration-200 ${
              darkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>
              Story Heading (optional)
            </label>
            <input
              type="text"
              id="heading"
              name="heading"
              value={formData.heading}
              onChange={handleInputChange}
              placeholder="Add a heading..."
              className="w-full px-2 py-2 border rounded text-sm focus:outline-none focus:ring-2 transition-all duration-200 bg-gray-700/50 text-white border-gray-600 focus:border-blue-400 focus:ring-blue-400/20 placeholder-gray-400"
            />
          </div>

          {/* District Selection - Compact */}
          <div className="flex items-center gap-4">
            <label htmlFor="district" className={`text-sm font-medium transition-colors duration-200 ${
              darkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>
              District:
            </label>
            <select
              id="district"
              name="district"
              value={formData.district}
              onChange={handleInputChange}
              className="px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 text-sm transition-all duration-200 min-w-[180px] bg-gray-700/50 text-white border-gray-600 focus:border-blue-400 focus:ring-blue-400/20"
            >
              <option value="">Choose district...</option>
              {chatbotDistricts.map(district => (
                <option key={district} value={district}>
                  {district}
                </option>
              ))}
            </select>
            <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              (optional)
            </span>
          </div>

          {/* Compact Essay Section */}
          <div className={`p-6 rounded-xl border-2 transition-all duration-200 ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <label htmlFor="content" className={`text-sm font-medium transition-colors duration-200 ${
                darkMode ? 'text-gray-200' : 'text-gray-700'
              }`}>
                Your Story
              </label>
              
              {/* Compact Settings */}
              <div className="flex items-center space-x-4 text-xs">
                <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                  {getWordCount(formData.content)} words
                </span>
                <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                  {getCharacterCount(formData.content)} characters
                </span>
              </div>
            </div>

            {/* Improved Formatting Tools */}
            <div className={`mb-3 p-2 rounded border transition-colors duration-200 ${
              darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
            }`}>
              <div className="flex flex-wrap gap-1">
                <button
                  type="button"
                  onClick={() => insertText('**', '**')}
                  className={`px-2 py-1 rounded text-xs font-bold transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Make text bold (select text first, or click to add sample)"
                >
                  <strong>Bold</strong>
                </button>
                <button
                  type="button"
                  onClick={() => insertText('*', '*')}
                  className={`px-2 py-1 rounded text-xs italic transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Make text italic (select text first, or click to add sample)"
                >
                  <em>Italic</em>
                </button>
                <button
                  type="button"
                  onClick={() => insertText('\n## ', '')}
                  className={`px-2 py-1 rounded text-xs transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Add a heading"
                >
                  Heading
                </button>
                <button
                  type="button"
                  onClick={() => insertText('\n- ', '')}
                  className={`px-2 py-1 rounded text-xs transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Add a bullet point"
                >
                  • List
                </button>
                <button
                  type="button"
                  onClick={() => insertText('\n\n', '')}
                  className={`px-2 py-1 rounded text-xs transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-600 hover:bg-gray-500 text-white'
                      : 'bg-white hover:bg-gray-100 text-gray-800 border border-gray-300'
                  }`}
                  title="Add a new paragraph"
                >
                  ¶ Paragraph
                </button>
              </div>
              <p className={`text-xs mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Tip: Select text first, then click formatting buttons. Or click buttons to insert examples.
              </p>
            </div>

            <textarea
              id="content"
              name="content"
              value={formData.content}
              onChange={handleInputChange}
              placeholder="Write your Istanbul story here...

Use the formatting buttons above or type:
**bold text** for bold
*italic text* for italic  
## Your Heading for headings
- list item for bullet points

Share your experiences, discoveries, and recommendations!"
              rows={10}
              className="w-full px-3 py-2 border rounded focus:outline-none resize-y text-sm leading-relaxed focus:ring-2 transition-all duration-200 bg-gray-700 text-white border-gray-600 focus:border-blue-400 focus:ring-blue-400/20 placeholder-gray-400"
              required
              style={{ minHeight: '250px' }}
            />
          </div>

          {/* Compact Action Buttons */}
          <div className={`p-5 rounded-xl border-2 text-center transition-all duration-200 ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
          }`}>
            <input
              type="file"
              id="imageUpload"
              name="imageUpload"
              ref={fileInputRef}
              onChange={handleImageSelect}
              accept="image/*"
              multiple
              className="sr-only"
              aria-label="Upload images for your blog post"
            />
            
            <div className="flex flex-wrap justify-center gap-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 disabled:opacity-50 text-white font-medium rounded text-sm transition-all duration-200"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 002 2v12a2 2 0 002 2z" />
                </svg>
              {uploading ? 'Uploading...' : 'Add Images'}
            </button>

            <button
              type="submit"
              disabled={submitting || uploading}
              className="inline-flex items-center px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded text-sm transition-all duration-200"
            >
              {submitting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Publishing...
                </>
              ) : (
                <>
                  <span className="mr-1">◆</span>
                  Publish Story
                </>
              )}
            </button>

            <button
              type="button"
              onClick={() => navigate('/blog')}
              className={`px-4 py-2 rounded font-medium text-sm transition-all duration-200 ${
                darkMode
                  ? 'bg-gray-600 hover:bg-gray-700 text-white'
                  : 'bg-gray-300 hover:bg-gray-400 text-gray-800'
              }`}
            >
              Cancel
            </button>
            </div>

            <p className={`mt-3 text-xs text-center ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Upload photos (JPG, PNG • Max 10MB each)
            </p>
          </div>

          {/* Upload Progress */}
          {Object.keys(uploadProgress).length > 0 && (
            <div className={`p-6 rounded-xl border-2 ${
              darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
            }`}>
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
            <div className={`p-6 rounded-xl border-2 transition-all duration-200 ${
              darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-blue-200 shadow-md'
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

          {/* Hidden Author Photo Field */}
          <input type="hidden" name="authorPhoto" value={formData.authorPhoto} />
        </form>
      </div>

      {/* Compact Progress Indicator - Mobile Hidden */}
      <div className="fixed top-20 right-2 z-30 hidden lg:block">
        <div className={`rounded-lg p-2 shadow-lg transition-all duration-200 ${
          darkMode ? 'bg-gray-800/90 text-white border border-gray-700' : 'bg-white text-gray-900 shadow-xl border border-gray-200'
        }`}>
          <div className="text-xs font-medium mb-1">Progress</div>
          <div className="space-y-1">
            <div className={`flex items-center text-xs ${
              formData.authorName ? 'text-green-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                formData.authorName ? 'bg-green-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Name
            </div>
            <div className={`flex items-center text-xs ${
              formData.title ? 'text-green-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                formData.title ? 'bg-green-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Title
            </div>
            <div className={`flex items-center text-xs ${
              formData.content.length > 50 ? 'text-green-500' : darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <div className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                formData.content.length > 50 ? 'bg-green-500' : darkMode ? 'bg-gray-600' : 'bg-gray-300'
              }`}></div>
              Story
            </div>
          </div>
        </div>
        </div>
      </div>
      </div>
    </div>
  );
};

export default NewBlogPost;