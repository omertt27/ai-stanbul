/**
 * Industry-Level Blog Post Creator
 * 
 * Features:
 * ✅ Rich Text Editor (TipTap WYSIWYG)
 * ✅ Auto-save with draft recovery
 * ✅ SEO & metadata management
 * ✅ Featured image with drag & drop
 * ✅ Category & tag management
 * ✅ Publishing options (draft, schedule, publish)
 * ✅ Live preview
 * ✅ Image gallery management
 * ✅ AI content assistant
 * ✅ Accessibility compliant
 * 
 * Author: AI Istanbul Team
 * Date: January 2026
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { createBlogPost, updateBlogPost, fetchBlogPost, uploadBlogImage } from '../api/blogApi';
import { useTheme } from '../contexts/ThemeContext';
import SEOHead from '../components/SEOHead';

// Industry-level components
import RichTextEditor from '../components/blog/RichTextEditor';
import SEOMetadataEditor from '../components/blog/SEOMetadataEditor';
import FeaturedImageSelector from '../components/blog/FeaturedImageSelector';
import CategoryTagsManager from '../components/blog/CategoryTagsManager';
import PublishingPanel from '../components/blog/PublishingPanel';
import { useAutoSave, AutoSaveIndicator, DraftRecoveryModal } from '../hooks/useAutoSave.jsx';
import AIContentAssistant from '../components/AIContentAssistant';
import { generateAutoTags } from '../utils/contentAI';
import '../App.css';

const NewBlogPostPro = () => {
  const { darkMode } = useTheme();
  const navigate = useNavigate();
  const { id: postId } = useParams(); // Get post ID from URL for edit mode
  const fileInputRef = useRef();
  const isEditMode = Boolean(postId);
  const [loading, setLoading] = useState(isEditMode); // Loading state for edit mode
  
  // Form state
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    authorName: '',
    authorPhoto: '',
    metaDescription: '',
    slug: '',
    focusKeyword: '',
    featuredImage: '',
    category: null,
    subcategories: [],
    status: 'draft',
    scheduledDate: null,
    visibility: 'public'
  });
  
  const [tags, setTags] = useState([]);
  const [images, setImages] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showDraftModal, setShowDraftModal] = useState(false);
  const [pendingDraft, setPendingDraft] = useState(null);

  // Auto-save hook - use different key for edit mode
  const autoSaveKey = isEditMode ? `edit-post-${postId}` : 'new-post';
  const { 
    saveStatus, 
    lastSaved, 
    hasDraft, 
    saveDraft, 
    loadDraft, 
    clearDraft 
  } = useAutoSave({ ...formData, tags }, autoSaveKey);

  // Load existing post data in edit mode
  useEffect(() => {
    if (isEditMode && postId) {
      const loadPost = async () => {
        try {
          setLoading(true);
          const post = await fetchBlogPost(postId);
          setFormData({
            title: post.title || '',
            content: post.content || '',
            authorName: post.author_name || '',
            authorPhoto: post.author_photo || '',
            metaDescription: post.meta_description || '',
            slug: post.slug || '',
            focusKeyword: post.focus_keyword || '',
            featuredImage: post.featured_image || '',
            category: post.category || null,
            subcategories: post.subcategories || [],
            status: post.status || 'draft',
            scheduledDate: post.scheduled_at || null,
            visibility: post.visibility || 'public'
          });
          setTags(post.tags || []);
          if (post.images) {
            setImages(post.images.map((img, idx) => ({
              id: idx,
              url: typeof img === 'string' ? img : img.url,
              alt_text: typeof img === 'string' ? '' : img.alt_text || ''
            })));
          }
        } catch (err) {
          console.error('Failed to load post:', err);
          setError('Failed to load post for editing');
        } finally {
          setLoading(false);
        }
      };
      loadPost();
    }
  }, [isEditMode, postId]);

  // Check for existing draft on mount (only for new posts)
  useEffect(() => {
    if (!isEditMode && hasDraft) {
      const draft = loadDraft();
      if (draft && (draft.title || draft.content)) {
        setPendingDraft(draft);
        setShowDraftModal(true);
      }
    }
  }, [hasDraft, loadDraft, isEditMode]);

  // Handle draft recovery
  const handleRecoverDraft = useCallback(() => {
    if (pendingDraft) {
      setFormData({
        title: pendingDraft.title || '',
        content: pendingDraft.content || '',
        authorName: pendingDraft.authorName || '',
        authorPhoto: pendingDraft.authorPhoto || '',
        metaDescription: pendingDraft.metaDescription || '',
        slug: pendingDraft.slug || '',
        focusKeyword: pendingDraft.focusKeyword || '',
        featuredImage: pendingDraft.featuredImage || '',
        category: pendingDraft.category || null,
        subcategories: pendingDraft.subcategories || [],
        status: pendingDraft.status || 'draft',
        scheduledDate: pendingDraft.scheduledDate || null,
        visibility: pendingDraft.visibility || 'public'
      });
      setTags(pendingDraft.tags || []);
    }
    setShowDraftModal(false);
    setPendingDraft(null);
  }, [pendingDraft]);

  const handleDiscardDraft = useCallback(() => {
    clearDraft();
    setShowDraftModal(false);
    setPendingDraft(null);
  }, [clearDraft]);

  // Form field handlers
  const updateFormData = useCallback((field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  // Image upload handler
  const handleImageUpload = useCallback(async (file) => {
    try {
      const result = await uploadBlogImage(file);
      const imageData = {
        id: Date.now() + Math.random(),
        url: result.url,
        alt_text: ''
      };
      setImages(prev => [...prev, imageData]);
      return result.url;
    } catch (err) {
      console.error('Image upload failed:', err);
      setError(`Failed to upload image: ${err.message}`);
      return null;
    }
  }, []);

  // Auto-generate tags
  const handleAutoGenerateTags = useCallback(() => {
    const plainContent = formData.content.replace(/<[^>]*>/g, '');
    const autoTags = generateAutoTags(formData.title, plainContent, formData.category);
    const newTags = autoTags.filter(tag => !tags.includes(tag));
    setTags(prev => [...prev, ...newTags]);
  }, [formData.title, formData.content, formData.category, tags]);

  // Validation
  const getValidationErrors = useCallback(() => {
    const errors = [];
    if (!formData.title.trim()) errors.push('Title is required');
    if (!formData.content.trim() || formData.content === '<p></p>') errors.push('Content is required');
    if (!formData.authorName.trim()) errors.push('Author name is required');
    
    const plainContent = formData.content.replace(/<[^>]*>/g, '');
    const wordCount = plainContent.split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount < 100) errors.push('Content should be at least 100 words');
    
    return errors;
  }, [formData]);

  const canPublish = getValidationErrors().length === 0;

  // Preview handler
  const handlePreview = useCallback(() => {
    setShowPreview(true);
  }, []);

  // Save draft handler
  const handleSaveDraft = useCallback(() => {
    updateFormData('status', 'draft');
    saveDraft();
  }, [updateFormData, saveDraft]);

  // Publish handler
  const handlePublish = useCallback(async () => {
    const errors = getValidationErrors();
    if (errors.length > 0) {
      setError(errors.join('. '));
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const plainContent = formData.content.replace(/<[^>]*>/g, '');
      
      const postData = {
        title: formData.title.trim(),
        content: formData.content.trim(),
        author_name: formData.authorName.trim(),
        author_photo: formData.authorPhoto || null,
        district: formData.category || null,
        featured_image: formData.featuredImage || null,
        category: formData.category || null,
        tags: tags,
        status: formData.scheduledDate ? 'scheduled' : 'published',
        scheduled_at: formData.scheduledDate || null,
        meta_description: formData.metaDescription || plainContent.substring(0, 160),
        slug: formData.slug || null,
        images: images.map(img => ({
          url: img.url,
          alt_text: img.alt_text || formData.title
        }))
      };

      let savedPost;
      if (isEditMode) {
        // Update existing post
        savedPost = await updateBlogPost(postId, postData);
      } else {
        // Create new post
        savedPost = await createBlogPost(postData);
      }
      
      // Clear draft after successful publish
      clearDraft();
      
      navigate(`/blog/${savedPost.id || postId}`);
    } catch (err) {
      setError(err.message || `Failed to ${isEditMode ? 'update' : 'publish'} blog post`);
      console.error('Save failed:', err);
    } finally {
      setSubmitting(false);
    }
  }, [formData, tags, images, navigate, clearDraft, getValidationErrors, isEditMode, postId]);

  // Check if form has been modified
  const isDirty = formData.title || formData.content !== '' && formData.content !== '<p></p>';

  // Loading state for edit mode
  if (loading) {
    return (
      <div className="min-h-screen w-full bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">Loading post...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen w-full bg-gray-900 text-white">
      <SEOHead
        title={isEditMode ? "Edit Blog Post" : "Create New Blog Post"}
        description="Share your Istanbul travel experience and insights with fellow travelers."
        keywords={['create blog post', 'Istanbul travel', 'travel writing']}
        url={isEditMode ? `/blog/edit/${postId}` : "/blog/new"}
        type="website"
      />

      {/* Draft Recovery Modal */}
      {showDraftModal && (
        <DraftRecoveryModal
          draft={pendingDraft}
          onRecover={handleRecoverDraft}
          onDiscard={handleDiscardDraft}
        />
      )}

      {/* Preview Modal */}
      {showPreview && (
        <div className="fixed inset-0 bg-black/80 z-50 overflow-y-auto">
          <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="bg-gray-800 rounded-xl overflow-hidden">
              <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
                <h2 className="text-lg font-semibold">Preview</h2>
                <button
                  onClick={() => setShowPreview(false)}
                  className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <article className="p-6">
                {formData.featuredImage && (
                  <img 
                    src={formData.featuredImage} 
                    alt={formData.title}
                    className="w-full aspect-video object-cover rounded-lg mb-6"
                  />
                )}
                <h1 className="text-3xl font-bold mb-4">{formData.title || 'Untitled Post'}</h1>
                <div className="flex items-center gap-3 mb-6 text-gray-400 text-sm">
                  {formData.authorPhoto && (
                    <img src={formData.authorPhoto} alt="" className="w-8 h-8 rounded-full" />
                  )}
                  <span>{formData.authorName || 'Anonymous'}</span>
                  <span>•</span>
                  <span>{new Date().toLocaleDateString()}</span>
                </div>
                <div 
                  className="prose prose-invert max-w-none"
                  dangerouslySetInnerHTML={{ __html: formData.content || '<p>No content yet...</p>' }}
                />
                {tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-8 pt-6 border-t border-gray-700">
                    {tags.map(tag => (
                      <span key={tag} className="px-3 py-1 bg-blue-600/20 text-blue-400 text-sm rounded-full">
                        #{tag}
                      </span>
                    ))}
                  </div>
                )}
              </article>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">{isEditMode ? 'Edit Post' : 'Create New Post'}</h1>
            <p className="text-gray-400">
              {isEditMode ? 'Update your blog post' : 'Share your Istanbul story with the world'}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <AutoSaveIndicator saveStatus={saveStatus} lastSaved={lastSaved} />
            <button
              onClick={() => navigate('/blog')}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-700 rounded-xl">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-red-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <p className="text-red-400">{error}</p>
              <button 
                onClick={() => setError(null)}
                className="ml-auto text-red-400 hover:text-red-300"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Main Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Author Info */}
            <div className="bg-gray-800 rounded-xl border-2 border-gray-700 p-5">
              <div className="flex items-center gap-4">
                <div className="flex-shrink-0">
                  <input
                    type="file"
                    id="authorPhotoUpload"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onloadend = () => updateFormData('authorPhoto', reader.result);
                        reader.readAsDataURL(file);
                      }
                    }}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    type="button"
                    onClick={() => document.getElementById('authorPhotoUpload').click()}
                    className="w-14 h-14 rounded-full border-2 border-dashed border-gray-600 hover:border-gray-500 flex items-center justify-center overflow-hidden transition-colors"
                  >
                    {formData.authorPhoto ? (
                      <img src={formData.authorPhoto} alt="Author" className="w-full h-full object-cover" />
                    ) : (
                      <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    )}
                  </button>
                </div>
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Author Name
                  </label>
                  <input
                    type="text"
                    value={formData.authorName}
                    onChange={(e) => updateFormData('authorName', e.target.value)}
                    placeholder="Your name..."
                    className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>

            {/* Title */}
            <div className="bg-gray-800 rounded-xl border-2 border-gray-700 p-5">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Post Title
              </label>
              <input
                type="text"
                value={formData.title}
                onChange={(e) => updateFormData('title', e.target.value)}
                placeholder="Enter your post title..."
                className="w-full px-4 py-3 bg-gray-700 text-white text-xl font-semibold border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-500"
              />
              <p className="text-xs text-gray-400 mt-2">
                {formData.title.length}/60 characters (recommended for SEO)
              </p>
            </div>

            {/* Rich Text Editor */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Content
              </label>
              <RichTextEditor
                content={formData.content}
                onChange={(content) => updateFormData('content', content)}
                onImageUpload={handleImageUpload}
                placeholder="Start writing your Istanbul story..."
                minHeight="450px"
              />
            </div>

            {/* Image Gallery */}
            {images.length > 0 && (
              <div className="bg-gray-800 rounded-xl border-2 border-gray-700 p-5">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Uploaded Images ({images.length})
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {images.map((img) => (
                    <div key={img.id} className="relative group aspect-square rounded-lg overflow-hidden">
                      <img src={img.url} alt="" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <button
                          type="button"
                          onClick={() => setImages(prev => prev.filter(i => i.id !== img.id))}
                          className="p-2 bg-red-600 hover:bg-red-700 rounded-full"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* AI Content Assistant */}
            <AIContentAssistant
              title={formData.title}
              content={formData.content.replace(/<[^>]*>/g, '')}
              district={formData.category}
              currentTags={tags}
              onTagSuggestion={(tag) => {
                if (!tags.includes(tag)) {
                  setTags(prev => [...prev, tag]);
                }
              }}
            />

            {/* SEO & Metadata */}
            <SEOMetadataEditor
              title={formData.title}
              content={formData.content}
              metaDescription={formData.metaDescription}
              slug={formData.slug}
              focusKeyword={formData.focusKeyword}
              featuredImage={formData.featuredImage}
              onMetaDescriptionChange={(value) => updateFormData('metaDescription', value)}
              onSlugChange={(value) => updateFormData('slug', value)}
              onFocusKeywordChange={(value) => updateFormData('focusKeyword', value)}
            />
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Publishing Panel */}
            <PublishingPanel
              status={formData.status}
              scheduledDate={formData.scheduledDate}
              visibility={formData.visibility}
              onStatusChange={(status) => updateFormData('status', status)}
              onScheduledDateChange={(date) => updateFormData('scheduledDate', date)}
              onVisibilityChange={(vis) => updateFormData('visibility', vis)}
              onSaveDraft={handleSaveDraft}
              onPreview={handlePreview}
              onPublish={handlePublish}
              isSubmitting={submitting}
              isDirty={isDirty}
              canPublish={canPublish}
              validationErrors={getValidationErrors()}
              isEditMode={isEditMode}
            />

            {/* Featured Image */}
            <FeaturedImageSelector
              featuredImage={formData.featuredImage}
              onImageSelect={(url) => updateFormData('featuredImage', url)}
              onImageRemove={() => updateFormData('featuredImage', '')}
              uploadedImages={images}
              onUploadImage={handleImageUpload}
            />

            {/* Categories & Tags */}
            <CategoryTagsManager
              selectedCategory={formData.category}
              selectedSubcategories={formData.subcategories}
              tags={tags}
              onCategoryChange={(cat) => updateFormData('category', cat)}
              onSubcategoriesChange={(subs) => updateFormData('subcategories', subs)}
              onTagsChange={setTags}
              onAutoGenerateTags={handleAutoGenerateTags}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewBlogPostPro;
