import React, { useState, useEffect } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { fetchComments, createComment } from '../api/blogApi';

const Comments = ({ postId }) => {
  const { darkMode } = useTheme();
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');
  const [authorName, setAuthorName] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadComments();
  }, [postId]);

  const loadComments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchComments(postId);
      setComments(response.comments || []);
      console.log(`✅ Loaded ${response.comments?.length || 0} comments for post ${postId}`);
    } catch (err) {
      setError('Failed to load comments');
      console.error('Error loading comments:', err);
      // Set empty array on error
      setComments([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!newComment.trim() || !authorName.trim()) {
      setError('Please fill in both your name and comment');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const commentData = {
        author_name: authorName,
        content: newComment,
        author_email: null // Optional
      };
      
      const response = await createComment(postId, commentData);
      
      // Add new comment to the list
      setComments(prev => [response, ...prev]);
      setNewComment('');
      setAuthorName('');
      
      console.log('✅ Comment posted successfully');
      
    } catch (err) {
      setError('Failed to post comment. Please try again.');
      console.error('Error posting comment:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMs = now - date;
    const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
    const diffInDays = Math.floor(diffInHours / 24);

    if (diffInHours < 1) {
      return 'Just now';
    } else if (diffInHours < 24) {
      return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
    } else if (diffInDays < 7) {
      return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    }
  };

  return (
    <div className={`mt-8 border-t pt-6 transition-colors duration-200 ${
      darkMode ? 'border-gray-700' : 'border-gray-200'
    }`}>
      <h3 className={`text-xl font-semibold mb-4 transition-colors duration-200 ${
        darkMode ? 'text-white' : 'text-gray-900'
      }`}>
        💬 Comments ({comments.length})
      </h3>

      {/* Comment Form */}
      <div className={`rounded-lg p-4 mb-6 transition-colors duration-200 ${
        darkMode ? 'bg-gray-800/50 border border-gray-700/50' : 'bg-gray-50/80 border border-gray-200/50'
      }`}>
        <h4 className={`text-base font-medium mb-3 transition-colors duration-200 ${
          darkMode ? 'text-white' : 'text-gray-900'
        }`}>
          Leave a comment
        </h4>
        
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <input
              type="text"
              value={authorName}
              onChange={(e) => setAuthorName(e.target.value)}
              placeholder="Your name"
              className={`w-full px-3 py-2 text-sm rounded-md border transition-colors duration-200 ${
                darkMode
                  ? 'bg-gray-700/70 border-gray-600/70 text-white placeholder-gray-400 focus:border-indigo-500'
                  : 'bg-white/90 border-gray-300/70 text-gray-900 placeholder-gray-500 focus:border-indigo-500'
              } focus:outline-none focus:ring-1 focus:ring-indigo-500/30`}
              required
            />
          </div>
          
          <div>
            <textarea
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              placeholder="Share your thoughts..."
              rows={3}
              className={`w-full px-3 py-2 text-sm rounded-md border transition-colors duration-200 resize-none ${
                darkMode
                  ? 'bg-gray-700/70 border-gray-600/70 text-white placeholder-gray-400 focus:border-indigo-500'
                  : 'bg-white/90 border-gray-300/70 text-gray-900 placeholder-gray-500 focus:border-indigo-500'
              } focus:outline-none focus:ring-1 focus:ring-indigo-500/30`}
              required
            />
          </div>

          {error && (
            <div className="text-red-500 text-xs">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !newComment.trim() || !authorName.trim()}
            className={`px-4 py-2 text-sm rounded-md font-medium transition-all duration-200 ${
              submitting || !newComment.trim() || !authorName.trim()
                ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white hover:shadow-md'
            }`}
          >
            {submitting ? (
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-2"></div>
                Posting...
              </div>
            ) : (
              'Post Comment'
            )}
          </button>
        </form>
      </div>

      {/* Comments List */}
      {loading ? (
        <div className="flex justify-center py-6">
          <div className={`animate-spin rounded-full h-6 w-6 border-b-2 ${
            darkMode ? 'border-indigo-500' : 'border-indigo-600'
          }`}></div>
        </div>
      ) : comments.length > 0 ? (
        <div className="space-y-4">
          {comments.map((comment) => (
            <div
              key={comment.id}
              className={`rounded-lg p-4 transition-colors duration-200 ${
                darkMode ? 'bg-gray-800/50 border border-gray-700/50' : 'bg-white/90 border border-gray-200/50 shadow-sm'
              }`}
            >
              <div className="flex items-start space-x-3">
                {/* Avatar */}
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-sm font-semibold">
                    {comment.author_name.charAt(0).toUpperCase()}
                  </div>
                </div>

                <div className="flex-1">
                  {/* Author and date */}
                  <div className="flex items-center justify-between mb-1">
                    <h5 className={`font-medium text-sm transition-colors duration-200 ${
                      darkMode ? 'text-white' : 'text-gray-900'
                    }`}>
                      {comment.author_name}
                    </h5>
                    <span className={`text-xs transition-colors duration-200 ${
                      darkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {formatDate(comment.created_at)}
                    </span>
                  </div>

                  {/* Comment content */}
                  <p className={`text-sm leading-relaxed transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    {comment.content}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className={`text-center py-6 transition-colors duration-200 ${
          darkMode ? 'text-gray-400' : 'text-gray-500'
        }`}>
          <div className="text-3xl mb-2">💭</div>
          <p className="text-sm">No comments yet. Be the first to share your thoughts!</p>
        </div>
      )}
    </div>
  );
};

export default Comments;
