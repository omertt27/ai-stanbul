import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createBlogPost, updateBlogPost, deleteBlogPost, getBlogPosts } from '../api/blogApi';
import { Logger } from '../utils/logger';
const logger = new Logger('BlogAdmin');

const BlogAdmin = () => {
  const navigate = useNavigate();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(null);
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    excerpt: '',
    author_name: 'AI Istanbul Team',
    category: '',
    district: '',
    tags: '',
    featured_image: '',
    status: 'published'
  });

  useEffect(() => {
    loadPosts();
  }, []);

  const loadPosts = async () => {
    try {
      setLoading(true);
      const data = await getBlogPosts({ limit: 100 });
      setPosts(data.posts || []);
      logger.info('Loaded posts for admin', { count: data.posts?.length });
    } catch (error) {
      logger.error('Failed to load posts', error);
      alert('Failed to load posts: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const postData = {
        ...formData,
        tags: formData.tags.split(',').map(t => t.trim()).filter(Boolean)
      };

      if (editing) {
        await updateBlogPost(editing.id, postData);
        logger.info('Updated post', { id: editing.id });
        alert('Post updated successfully!');
      } else {
        await createBlogPost(postData);
        logger.info('Created new post');
        alert('Post created successfully!');
      }

      // Reset form and reload
      setFormData({
        title: '',
        content: '',
        excerpt: '',
        author_name: 'AI Istanbul Team',
        category: '',
        district: '',
        tags: '',
        featured_image: '',
        status: 'published'
      });
      setEditing(null);
      loadPosts();
    } catch (error) {
      logger.error('Failed to save post', error);
      alert('Failed to save post: ' + error.message);
    }
  };

  const handleEdit = (post) => {
    setEditing(post);
    setFormData({
      title: post.title,
      content: post.content,
      excerpt: post.excerpt || '',
      author_name: post.author_name,
      category: post.category || '',
      district: post.district || '',
      tags: Array.isArray(post.tags) ? post.tags.join(', ') : '',
      featured_image: post.featured_image || '',
      status: post.status || 'published'
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleDelete = async (post) => {
    if (!confirm(`Delete post "${post.title}"?`)) return;

    try {
      await deleteBlogPost(post.id);
      logger.info('Deleted post', { id: post.id });
      alert('Post deleted successfully!');
      loadPosts();
    } catch (error) {
      logger.error('Failed to delete post', error);
      alert('Failed to delete post: ' + error.message);
    }
  };

  const districts = ['Fatih', 'Beyoğlu', 'Kadıköy', 'Beşiktaş', 'Üsküdar', 'Şişli', 'Bakırköy', 'Sarıyer'];
  const categories = ['Travel Guide', 'Food & Drink', 'Culture', 'District Guide', 'Events', 'History', 'Tips'];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
          📝 Blog Admin Panel
        </h1>
        <p style={{ color: '#666' }}>
          Create, edit, and manage blog posts
        </p>
      </div>

      {/* Form Section */}
      <div style={{ 
        background: 'white', 
        padding: '2rem', 
        borderRadius: '8px', 
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '2rem'
      }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>
          {editing ? '✏️ Edit Post' : '➕ New Post'}
        </h2>
        
        <form onSubmit={handleSubmit}>
          <div style={{ display: 'grid', gap: '1rem' }}>
            {/* Title */}
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                Title *
              </label>
              <input
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                required
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '1rem'
                }}
                placeholder="Enter post title..."
              />
            </div>

            {/* Excerpt */}
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                Excerpt
              </label>
              <input
                type="text"
                value={formData.excerpt}
                onChange={(e) => setFormData({ ...formData, excerpt: e.target.value })}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '1rem'
                }}
                placeholder="Short description..."
              />
            </div>

            {/* Content */}
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                Content *
              </label>
              <textarea
                value={formData.content}
                onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                required
                rows={10}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '1rem',
                  fontFamily: 'inherit'
                }}
                placeholder="Write your blog post content here..."
              />
            </div>

            {/* Row: Author, Category, District */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Author
                </label>
                <input
                  type="text"
                  value={formData.author_name}
                  onChange={(e) => setFormData({ ...formData, author_name: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '1rem'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Category
                </label>
                <select
                  value={formData.category}
                  onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '1rem'
                  }}
                >
                  <option value="">Select...</option>
                  {categories.map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  District
                </label>
                <select
                  value={formData.district}
                  onChange={(e) => setFormData({ ...formData, district: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '1rem'
                  }}
                >
                  <option value="">Select...</option>
                  {districts.map(dist => (
                    <option key={dist} value={dist}>{dist}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Row: Tags, Featured Image */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Tags (comma-separated)
                </label>
                <input
                  type="text"
                  value={formData.tags}
                  onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '1rem'
                  }}
                  placeholder="tag1, tag2, tag3"
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                  Featured Image URL
                </label>
                <input
                  type="text"
                  value={formData.featured_image}
                  onChange={(e) => setFormData({ ...formData, featured_image: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '1rem'
                  }}
                  placeholder="/static/blog/image.jpg"
                />
              </div>
            </div>

            {/* Status */}
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>
                Status
              </label>
              <select
                value={formData.status}
                onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                style={{
                  width: '200px',
                  padding: '0.75rem',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '1rem'
                }}
              >
                <option value="published">Published</option>
                <option value="draft">Draft</option>
              </select>
            </div>

            {/* Buttons */}
            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <button
                type="submit"
                style={{
                  padding: '0.75rem 2rem',
                  background: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '1rem',
                  cursor: 'pointer',
                  fontWeight: '500'
                }}
              >
                {editing ? '💾 Update Post' : '✨ Create Post'}
              </button>
              
              {editing && (
                <button
                  type="button"
                  onClick={() => {
                    setEditing(null);
                    setFormData({
                      title: '',
                      content: '',
                      excerpt: '',
                      author_name: 'AI Istanbul Team',
                      category: '',
                      district: '',
                      tags: '',
                      featured_image: '',
                      status: 'published'
                    });
                  }}
                  style={{
                    padding: '0.75rem 2rem',
                    background: '#666',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '1rem',
                    cursor: 'pointer'
                  }}
                >
                  ❌ Cancel
                </button>
              )}
            </div>
          </div>
        </form>
      </div>

      {/* Posts List */}
      <div style={{ 
        background: 'white', 
        padding: '2rem', 
        borderRadius: '8px', 
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>
          📚 Existing Posts ({posts.length})
        </h2>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
            Loading posts...
          </div>
        ) : posts.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
            No posts yet. Create your first one above!
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '1rem' }}>
            {posts.map(post => (
              <div 
                key={post.id}
                style={{
                  padding: '1.5rem',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  background: '#f9f9f9'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                  <div style={{ flex: 1 }}>
                    <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
                      {post.title}
                    </h3>
                    <p style={{ color: '#666', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                      {post.excerpt || post.content.substring(0, 150)}...
                    </p>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.85rem', color: '#888' }}>
                      <span>👤 {post.author_name}</span>
                      {post.category && <span>📁 {post.category}</span>}
                      {post.district && <span>📍 {post.district}</span>}
                      <span>❤️ {post.likes_count || 0} likes</span>
                      <span>
                        {post.status === 'published' ? '🟢 Published' : '🟡 Draft'}
                      </span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', marginLeft: '1rem' }}>
                    <button
                      onClick={() => navigate(`/blog/${post.id}`)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#2196F3',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.9rem'
                      }}
                    >
                      👁️ View
                    </button>
                    <button
                      onClick={() => handleEdit(post)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#FF9800',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.9rem'
                      }}
                    >
                      ✏️ Edit
                    </button>
                    <button
                      onClick={() => handleDelete(post)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#f44336',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.9rem'
                      }}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BlogAdmin;
