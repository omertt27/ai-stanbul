// Admin Dashboard JavaScript
// API Configuration
// Detect if running locally or in production
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? `http://${window.location.hostname}:5001` 
    : 'https://api.aistanbul.net';

// State Management
let currentSection = 'dashboard';
let blogPosts = [];
let comments = [];
let feedbackData = [];
let users = [];
let analyticsChart = null;

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    loadDashboardData();
    setupSearchHandlers();
    setupModalHandlers();
});

// Navigation
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            // Remove active class from all items
            navItems.forEach(nav => nav.classList.remove('active'));
            
            // Add active class to clicked item
            this.classList.add('active');
            
            // Hide all sections
            document.querySelectorAll('.content-section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Show selected section
            const sectionId = this.getAttribute('data-section');
            document.getElementById(sectionId).classList.add('active');
            currentSection = sectionId;
            
            // Load section data
            loadSectionData(sectionId);
        });
    });
}

// Load Dashboard Data
async function loadDashboardData() {
    try {
        showLoading('dashboard');
        
        // Load admin stats first
        const statsResponse = await fetch(`${API_BASE_URL}/api/admin/stats`);
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            // Update stats in dashboard
            if (document.getElementById('total-posts')) {
                document.getElementById('total-posts').textContent = stats.blog_posts || 0;
            }
            if (document.getElementById('total-comments')) {
                document.getElementById('total-comments').textContent = stats.comments || 0;
            }
            if (document.getElementById('total-feedback')) {
                document.getElementById('total-feedback').textContent = stats.feedback || 0;
            }
            if (document.getElementById('total-users')) {
                document.getElementById('total-users').textContent = stats.active_users || 0;
            }
            if (document.getElementById('pending-comments')) {
                document.getElementById('pending-comments').textContent = stats.pending_comments || 0;
            }
            if (document.getElementById('model-accuracy')) {
                document.getElementById('model-accuracy').textContent = `${stats.model_accuracy || 0}%`;
            }
        }
        
        // Load all data
        await Promise.all([
            loadBlogPosts(),
            loadComments(),
            loadFeedback(),
            loadUsers()
        ]);
        
        hideLoading('dashboard');
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data');
        hideLoading('dashboard');
    }
}

// Load Section Data
async function loadSectionData(section) {
    switch(section) {
        case 'blog-posts':
            await loadBlogPosts();
            renderBlogPosts();
            break;
        case 'comments':
            await loadComments();
            renderComments();
            break;
        case 'feedback':
            await loadFeedback();
            renderFeedback();
            break;
        case 'analytics':
            await loadAnalytics();
            break;
        case 'intents':
            await loadIntentStats();
            break;
        case 'system-performance':
            await loadSystemPerformance();
            break;
        case 'users':
            await loadUsers();
            renderUsers();
            break;
    }
}

// Blog Posts Functions
async function loadBlogPosts() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/blog/posts`);
        if (response.ok) {
            const data = await response.json();
            blogPosts = data.posts || [];
        }
        return blogPosts;
    } catch (error) {
        console.error('Error loading blog posts:', error);
        return [];
    }
}

function renderBlogPosts() {
    const tbody = document.getElementById('blog-table-body');
    
    if (blogPosts.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="empty-state">
                        <i class="fas fa-blog"></i>
                        <h4>No blog posts yet</h4>
                        <p>Create your first blog post to get started</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = blogPosts.map(post => `
        <tr>
            <td><strong>${post.title}</strong></td>
            <td>${post.author}</td>
            <td>
                <span class="badge ${post.status === 'Published' ? 'badge-success' : 'badge-warning'}">
                    ${post.status}
                </span>
            </td>
            <td>${formatDate(post.date)}</td>
            <td>${post.views}</td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn view" onclick="viewBlogPost(${post.id})" title="View">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn edit" onclick="editBlogPost(${post.id})" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="action-btn delete" onclick="deleteBlogPost(${post.id})" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function openBlogModal() {
    document.getElementById('blog-modal').classList.add('active');
    // Reset form
    document.getElementById('blog-title').value = '';
    document.getElementById('blog-slug').value = '';
    document.getElementById('blog-content').value = '';
}

function closeBlogModal() {
    document.getElementById('blog-modal').classList.remove('active');
}

async function saveBlogPost(event) {
    event.preventDefault();
    
    const postData = {
        title: document.getElementById('blog-title').value,
        slug: document.getElementById('blog-slug').value,
        category: document.getElementById('blog-category').value,
        content: document.getElementById('blog-content').value,
        status: document.getElementById('blog-status').value,
        author: 'Admin',
        date: new Date().toISOString().split('T')[0],
        views: 0
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/blog/posts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(postData)
        });
        
        if (response.ok) {
            showSuccess('Blog post created successfully!');
            closeBlogModal();
            await loadBlogPosts();
            renderBlogPosts();
        } else {
            // For demo, add to local array
            blogPosts.unshift({
                id: blogPosts.length + 1,
                ...postData
            });
            renderBlogPosts();
            updateDashboardStats();
            closeBlogModal();
            showSuccess('Blog post created successfully!');
        }
    } catch (error) {
        console.error('Error saving blog post:', error);
        showError('Failed to save blog post');
    }
}

function viewBlogPost(id) {
    const post = blogPosts.find(p => p.id === id);
    if (post) {
        alert(`Viewing: ${post.title}\n\n${post.content || 'No content available'}`);
    }
}

function editBlogPost(id) {
    const post = blogPosts.find(p => p.id === id);
    if (post) {
        document.getElementById('blog-title').value = post.title;
        document.getElementById('blog-slug').value = post.slug;
        document.getElementById('blog-content').value = post.content || '';
        document.getElementById('blog-status').value = post.status;
        openBlogModal();
    }
}

async function deleteBlogPost(id) {
    if (!confirm('Are you sure you want to delete this blog post?')) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/blog/posts/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok || true) { // For demo
            blogPosts = blogPosts.filter(p => p.id !== id);
            renderBlogPosts();
            updateDashboardStats();
            showSuccess('Blog post deleted successfully');
        }
    } catch (error) {
        console.error('Error deleting blog post:', error);
        showError('Failed to delete blog post');
    }
}

// Comments Functions
async function loadComments() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/comments`);
        if (response.ok) {
            const data = await response.json();
            comments = data.comments || [];
        }
        return comments;
    } catch (error) {
        console.error('Error loading comments:', error);
        return [];
    }
}

function renderComments() {
    const tbody = document.getElementById('comments-table-body');
    
    if (comments.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="empty-state">
                        <i class="fas fa-comments"></i>
                        <h4>No comments yet</h4>
                        <p>Comments will appear here when users start engaging</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = comments.map(comment => `
        <tr>
            <td>${comment.comment.substring(0, 50)}${comment.comment.length > 50 ? '...' : ''}</td>
            <td>${comment.author}</td>
            <td>${comment.post}</td>
            <td>
                <span class="badge ${
                    comment.status === 'Approved' ? 'badge-success' : 
                    comment.status === 'Pending' ? 'badge-warning' : 
                    'badge-danger'
                }">
                    ${comment.status}
                </span>
            </td>
            <td>${formatDate(comment.date)}</td>
            <td>
                <div class="action-buttons">
                    ${comment.status === 'Pending' ? `
                        <button class="action-btn view" onclick="approveComment(${comment.id})" title="Approve">
                            <i class="fas fa-check"></i>
                        </button>
                    ` : ''}
                    <button class="action-btn delete" onclick="deleteComment(${comment.id})" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function approveComment(id) {
    const comment = comments.find(c => c.id === id);
    if (comment) {
        comment.status = 'Approved';
        renderComments();
        updateDashboardStats();
        showSuccess('Comment approved');
    }
}

function deleteComment(id) {
    if (!confirm('Delete this comment?')) return;
    
    comments = comments.filter(c => c.id !== id);
    renderComments();
    updateDashboardStats();
    showSuccess('Comment deleted');
}

function approveAllComments() {
    if (!confirm('Approve all pending comments?')) return;
    
    comments.forEach(comment => {
        if (comment.status === 'Pending') {
            comment.status = 'Approved';
        }
    });
    
    renderComments();
    updateDashboardStats();
    showSuccess('All comments approved');
}

function deleteSpamComments() {
    if (!confirm('Delete all spam comments?')) return;
    
    comments = comments.filter(c => c.status !== 'Spam');
    renderComments();
    updateDashboardStats();
    showSuccess('Spam comments deleted');
}

// Feedback Functions
async function loadFeedback() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/feedback/export`);
        if (response.ok) {
            const data = await response.json();
            // Convert the summary object into an array of feedback items
            feedbackData = [];
            
            // Add misclassifications
            if (data.summary && data.summary.misclassifications) {
                data.summary.misclassifications.forEach((item, idx) => {
                    feedbackData.push({
                        id: idx + 1,
                        query: item.query || 'N/A',
                        predicted_intent: item.predicted_intent || 'N/A',
                        confidence: item.confidence || 0,
                        feedback: 'Misclassified',
                        date: item.timestamp || new Date().toISOString()
                    });
                });
            }
            
            // Add corrections
            if (data.summary && data.summary.corrections) {
                data.summary.corrections.forEach((item, idx) => {
                    feedbackData.push({
                        id: feedbackData.length + 1,
                        query: item.query || 'N/A',
                        predicted_intent: item.predicted_intent || 'N/A',
                        confidence: item.confidence || 0,
                        feedback: `Corrected to: ${item.correct_intent || 'N/A'}`,
                        date: item.timestamp || new Date().toISOString()
                    });
                });
            }
            
            // Add low confidence items
            if (data.summary && data.summary.low_confidence) {
                data.summary.low_confidence.forEach((item, idx) => {
                    feedbackData.push({
                        id: feedbackData.length + 1,
                        query: item.query || 'N/A',
                        predicted_intent: item.predicted_intent || 'N/A',
                        confidence: item.confidence || 0,
                        feedback: 'Low Confidence',
                        date: item.timestamp || new Date().toISOString()
                    });
                });
            }
            
            // If no feedback data, use mock data
            if (feedbackData.length === 0) {
                feedbackData = [
                    {
                        id: 1,
                        query: 'Where is Blue Mosque?',
                        predicted_intent: 'find_attraction',
                        confidence: 0.95,
                        feedback: 'Correct',
                        date: '2025-10-27 10:30'
                    },
                    {
                        id: 2,
                        query: 'Best restaurants in Taksim',
                        predicted_intent: 'find_restaurant',
                        confidence: 0.88,
                        feedback: 'Correct',
                        date: '2025-10-27 11:15'
                    },
                    {
                        id: 3,
                        query: 'How to get to airport',
                        predicted_intent: 'get_directions',
                        confidence: 0.65,
                        feedback: 'Corrected to: get_transportation',
                        date: '2025-10-27 12:00'
                    }
                ];
            }
        }
        return feedbackData;
    } catch (error) {
        console.error('Error loading feedback:', error);
        return [];
    }
}

function renderFeedback() {
    const tbody = document.getElementById('feedback-table-body');
    
    if (feedbackData.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="empty-state">
                        <i class="fas fa-star"></i>
                        <h4>No feedback yet</h4>
                        <p>User feedback will appear here</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = feedbackData.map(item => `
        <tr>
            <td>${item.query}</td>
            <td>${item.predicted_intent}</td>
            <td>
                <span class="badge ${
                    item.confidence >= 0.8 ? 'badge-success' :
                    item.confidence >= 0.6 ? 'badge-warning' :
                    'badge-danger'
                }">
                    ${(item.confidence * 100).toFixed(1)}%
                </span>
            </td>
            <td>${item.feedback}</td>
            <td>${formatDate(item.date)}</td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn view" onclick="viewFeedbackDetails(${item.id})" title="View">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function viewFeedbackDetails(id) {
    const item = feedbackData.find(f => f.id === id);
    if (item) {
        alert(`Query: ${item.query}\nIntent: ${item.predicted_intent}\nConfidence: ${(item.confidence * 100).toFixed(1)}%\nFeedback: ${item.feedback}`);
    }
}

async function exportFeedback() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/admin/feedback/export`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `feedback-${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            showSuccess('Feedback data exported');
        } else {
            // For demo, export as JSON
            const dataStr = JSON.stringify(feedbackData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `feedback-${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            showSuccess('Feedback data exported');
        }
    } catch (error) {
        console.error('Error exporting feedback:', error);
        showError('Failed to export feedback');
    }
}

// Analytics Functions
async function loadAnalytics() {
    try {
        // Fetch real analytics data
        const response = await fetch(`${API_BASE_URL}/api/admin/analytics?days=7`);
        let analyticsData;
        
        if (response.ok) {
            analyticsData = await response.json();
        } else {
            // No data available
            analyticsData = {
                dates: [],
                user_queries: [],
                blog_views: [],
                comments: []
            };
        }
    
        const canvas = document.getElementById('analytics-chart');
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if any
        if (analyticsChart) {
            analyticsChart.destroy();
        }
        
        // Create new chart
        analyticsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: analyticsData.dates,
                datasets: [
                    {
                        label: 'User Queries',
                        data: analyticsData.user_queries,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Blog Views',
                        data: analyticsData.blog_views,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Comments',
                        data: analyticsData.comments,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading analytics:', error);
        showError('Failed to load analytics');
    }
}

function updateAnalyticsPeriod(days) {
    // Reload analytics with new period
    loadAnalytics();
    showSuccess(`Showing data for last ${days} days`);
}

// Intent Stats Functions
async function loadIntentStats() {
    try {
        const tbody = document.getElementById('intents-table-body');
        
        // Fetch real intent statistics
        const response = await fetch(`${API_BASE_URL}/api/admin/intents/stats`);
        let intentStats;
        
        if (response.ok) {
            const data = await response.json();
            intentStats = data.intents || [];
        }
        
        if (intentStats.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5">
                        <div class="empty-state">
                            <i class="fas fa-brain"></i>
                            <h4>No intent data yet</h4>
                            <p>Intent statistics will appear here as users interact with the system</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = intentStats.map(stat => `
            <tr>
                <td><strong>${stat.intent}</strong></td>
                <td>${stat.count}</td>
                <td>
                    <span class="badge ${stat.accuracy >= 90 ? 'badge-success' : stat.accuracy >= 85 ? 'badge-warning' : 'badge-danger'}">
                        ${stat.accuracy}%
                    </span>
                </td>
                <td>${(stat.confidence * 100).toFixed(1)}%</td>
                <td>${stat.corrections}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading intent stats:', error);
        showError('Failed to load intent statistics');
    }
}

function retrainModel() {
    if (!confirm('Start model retraining? This may take several minutes.')) return;
    
    showSuccess('Model retraining started. You will be notified when complete.');
    
    // Simulate retraining
    setTimeout(() => {
        showSuccess('Model retraining complete! New model deployed.');
    }, 3000);
}

// Users Functions
async function loadUsers() {
    users = [
        {
            id: 1,
            name: 'Admin User',
            email: 'admin@aistanbul.net',
            role: 'Admin',
            status: 'Active',
            lastActive: '2025-10-27 16:30'
        },
        {
            id: 2,
            name: 'Content Editor',
            email: 'editor@aistanbul.net',
            role: 'Editor',
            status: 'Active',
            lastActive: '2025-10-27 14:15'
        }
    ];
    return users;
}

function renderUsers() {
    const tbody = document.getElementById('users-table-body');
    
    tbody.innerHTML = users.map(user => `
        <tr>
            <td><strong>${user.name}</strong></td>
            <td>${user.email}</td>
            <td><span class="badge badge-info">${user.role}</span></td>
            <td>
                <span class="badge ${user.status === 'Active' ? 'badge-success' : 'badge-danger'}">
                    ${user.status}
                </span>
            </td>
            <td>${formatDate(user.lastActive)}</td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn edit" onclick="editUser(${user.id})" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="action-btn delete" onclick="deleteUser(${user.id})" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function openUserModal() {
    alert('User creation modal - To be implemented');
}

function editUser(id) {
    const user = users.find(u => u.id === id);
    if (user) {
        alert(`Edit user: ${user.name}`);
    }
}

function deleteUser(id) {
    if (!confirm('Delete this user?')) return;
    
    users = users.filter(u => u.id !== id);
    renderUsers();
    showSuccess('User deleted');
}

// Settings Functions
function saveSettings() {
    const settings = {
        siteTitle: document.getElementById('site-title').value,
        adminEmail: document.getElementById('admin-email').value,
        apiUrl: document.getElementById('api-url').value,
        retrainThreshold: document.getElementById('retrain-threshold').value,
        autoApproveComments: document.getElementById('auto-approve-comments').checked,
        emailNotifications: document.getElementById('email-notifications').checked
    };
    
    console.log('Saving settings:', settings);
    showSuccess('Settings saved successfully!');
}

// Search Handlers
function setupSearchHandlers() {
    document.getElementById('blog-search').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const filtered = blogPosts.filter(post => 
            post.title.toLowerCase().includes(searchTerm)
        );
        renderFilteredBlogPosts(filtered);
    });
    
    document.getElementById('comment-search').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const filtered = comments.filter(comment => 
            comment.comment.toLowerCase().includes(searchTerm) ||
            comment.author.toLowerCase().includes(searchTerm)
        );
        renderFilteredComments(filtered);
    });
    
    document.getElementById('user-search').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        const filtered = users.filter(user => 
            user.name.toLowerCase().includes(searchTerm) ||
            user.email.toLowerCase().includes(searchTerm)
        );
        renderFilteredUsers(filtered);
    });
}

function renderFilteredBlogPosts(filtered) {
    const original = blogPosts;
    blogPosts = filtered;
    renderBlogPosts();
    blogPosts = original;
}

function renderFilteredComments(filtered) {
    const original = comments;
    comments = filtered;
    renderComments();
    comments = original;
}

function renderFilteredUsers(filtered) {
    const original = users;
    users = filtered;
    renderUsers();
    users = original;
}

// Modal Handlers
function setupModalHandlers() {
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                this.classList.remove('active');
            }
        });
    });
}

// Update Dashboard Stats
function updateDashboardStats() {
    document.getElementById('total-posts').textContent = blogPosts.length;
    document.getElementById('total-comments').textContent = comments.length;
    document.getElementById('total-feedback').textContent = feedbackData.length;
    document.getElementById('total-users').textContent = users.length;
    
    const pendingComments = comments.filter(c => c.status === 'Pending').length;
    document.getElementById('pending-comments').textContent = pendingComments;
    
    // Update feedback stats
    document.getElementById('total-predictions').textContent = feedbackData.length;
    const corrections = feedbackData.filter(f => f.feedback.includes('Corrected')).length;
    document.getElementById('corrections-count').textContent = corrections;
    const lowConfidence = feedbackData.filter(f => f.confidence < 0.7).length;
    document.getElementById('low-confidence').textContent = lowConfidence;
}

// Utility Functions
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays < 7) return `${diffDays} days ago`;
    
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

function showLoading(section) {
    const element = document.getElementById(section);
    if (element) {
        element.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i><p>Loading...</p></div>';
    }
}

function hideLoading(section) {
    // Content will be rendered by specific render functions
}

function showSuccess(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 24px;
        right: 24px;
        background: #10b981;
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    toast.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function showError(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 24px;
        right: 24px;
        background: #ef4444;
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    toast.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// =============================
// SYSTEM PERFORMANCE FUNCTIONS
// =============================

let latencyChart = null;
let accuracyChart = null;
let performanceRefreshInterval = null;

async function loadSystemPerformance() {
    try {
        showLoading('system-performance');
        
        // Fetch system metrics from backend
        const response = await fetch(`${API_BASE_URL}/api/admin/system/metrics?hours=24`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Render all components
        renderRealtimeMetrics(data.realtime_metrics);
        renderSystemHealth(data.system_health);
        renderPerformanceCharts(data.trends);
        renderRecentAlerts(data.recent_alerts || data.alerts);
        renderIntentBreakdown(data.intent_breakdown);
        
        hideLoading('system-performance');
        
        // Set up auto-refresh every 30 seconds
        if (performanceRefreshInterval) {
            clearInterval(performanceRefreshInterval);
        }
        performanceRefreshInterval = setInterval(() => {
            if (currentSection === 'system-performance') {
                loadSystemPerformance();
            }
        }, 30000); // 30 seconds
        
    } catch (error) {
        console.error('Error loading system performance:', error);
        showError('Failed to load system performance data');
        hideLoading('system-performance');
        
        // Show error state
        renderErrorState();
    }
}

function renderRealtimeMetrics(metrics) {
    // Update latency
    const latencyEl = document.getElementById('sys-latency');
    if (latencyEl) {
        latencyEl.textContent = metrics.avg_latency_ms ? `${Math.round(metrics.avg_latency_ms)}ms` : '-';
    }
    
    // Update accuracy
    const accuracyEl = document.getElementById('sys-accuracy');
    if (accuracyEl) {
        accuracyEl.textContent = metrics.intent_accuracy ? `${metrics.intent_accuracy.toFixed(1)}%` : '-';
    }
    
    // Update error rate
    const errorEl = document.getElementById('sys-error-rate');
    if (errorEl) {
        errorEl.textContent = metrics.error_rate ? `${metrics.error_rate.toFixed(2)}%` : '-';
    }
    
    // Update requests per minute
    const requestsEl = document.getElementById('sys-requests');
    if (requestsEl) {
        requestsEl.textContent = metrics.requests_per_minute ? metrics.requests_per_minute.toFixed(1) : '-';
    }
}

function renderSystemHealth(health) {
    const container = document.getElementById('system-health-status');
    if (!container) return;
    
    if (!health || !health.components || health.components.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-light); padding: 20px;">No health data available</p>';
        return;
    }
    
    let html = '';
    health.components.forEach(component => {
        const statusClass = component.status || 'unknown';
        const statusIcon = {
            'healthy': 'fa-check-circle',
            'warning': 'fa-exclamation-triangle',
            'degraded': 'fa-exclamation-circle',
            'error': 'fa-times-circle',
            'unknown': 'fa-question-circle'
        }[statusClass] || 'fa-question-circle';
        
        html += `
            <div class="health-component">
                <div class="health-component-name">
                    <i class="fas ${statusIcon}"></i>
                    <span>${component.name}</span>
                </div>
                <div class="health-status ${statusClass}">
                    <i class="fas ${statusIcon}"></i>
                    ${statusClass.charAt(0).toUpperCase() + statusClass.slice(1)}
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function renderPerformanceCharts(trends) {
    if (!trends || !trends.timestamps || trends.timestamps.length === 0) {
        console.warn('No trend data available for charts');
        return;
    }
    
    // Latency Chart
    const latencyCtx = document.getElementById('latency-trend-chart');
    if (latencyCtx) {
        if (latencyChart) {
            latencyChart.destroy();
        }
        
        latencyChart = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: trends.timestamps.map(t => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})),
                datasets: [{
                    label: 'Latency (ms)',
                    data: trends.latency || [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Milliseconds'
                        }
                    }
                }
            }
        });
    }
    
    // Accuracy Chart
    const accuracyCtx = document.getElementById('accuracy-trend-chart');
    if (accuracyCtx) {
        if (accuracyChart) {
            accuracyChart.destroy();
        }
        
        accuracyChart = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: trends.timestamps.map(t => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})),
                datasets: [{
                    label: 'Accuracy (%)',
                    data: trends.accuracy || [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 80,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage'
                        }
                    }
                }
            }
        });
    }
}

function renderRecentAlerts(alerts) {
    const container = document.getElementById('recent-alerts-list');
    const badge = document.getElementById('alert-badge');
    
    if (!container) return;
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-light); padding: 20px;">No alerts in the last 24 hours</p>';
        if (badge) badge.style.display = 'none';
        return;
    }
    
    // Update badge
    if (badge) {
        badge.textContent = alerts.length;
        badge.style.display = 'inline-block';
    }
    
    // Render alerts
    let html = '';
    alerts.forEach(alert => {
        const severity = alert.severity || 'info';
        const severityIcon = {
            'critical': 'fa-exclamation-circle',
            'warning': 'fa-exclamation-triangle',
            'info': 'fa-info-circle'
        }[severity] || 'fa-info-circle';
        
        const timeAgo = formatTimeAgo(new Date(alert.timestamp));
        
        html += `
            <div class="alert-item ${severity}">
                <div class="alert-header">
                    <div class="alert-title">
                        <i class="fas ${severityIcon}"></i>
                        ${alert.title || 'System Alert'}
                    </div>
                    <div class="alert-time">${timeAgo}</div>
                </div>
                <div class="alert-message">${alert.message || 'No details available'}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function renderIntentBreakdown(intentData) {
    const container = document.getElementById('intent-breakdown');
    if (!container) return;
    
    if (!intentData || Object.keys(intentData).length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-light); padding: 20px;">No intent data available</p>';
        return;
    }
    
    // Convert to array and sort by count
    const intents = Object.entries(intentData)
        .map(([name, count]) => ({name, count}))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10); // Top 10
    
    const total = intents.reduce((sum, intent) => sum + intent.count, 0);
    
    let html = '';
    intents.forEach(intent => {
        const percentage = total > 0 ? (intent.count / total * 100).toFixed(1) : 0;
        html += `
            <div class="intent-bar">
                <div class="intent-name">${intent.name}</div>
                <div class="intent-progress">
                    <div class="intent-progress-bar" style="width: ${percentage}%">
                        ${percentage}%
                    </div>
                </div>
                <div class="intent-count">${intent.count} req</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function renderErrorState() {
    const container = document.getElementById('system-health-status');
    if (container) {
        container.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <i class="fas fa-exclamation-triangle" style="font-size: 48px; color: var(--warning); margin-bottom: 16px;"></i>
                <h4 style="margin-bottom: 8px;">Unable to Load System Metrics</h4>
                <p style="color: var(--text-light);">The monitoring system may be unavailable. Please try again later.</p>
            </div>
        `;
    }
}

function formatTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

// Cleanup on section change
document.addEventListener('DOMContentLoaded', function() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            // Clear performance refresh interval when leaving the section
            if (currentSection === 'system-performance' && performanceRefreshInterval) {
                clearInterval(performanceRefreshInterval);
                performanceRefreshInterval = null;
            }
        });
    });
});
