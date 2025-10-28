# 🎛️ AI Istanbul Admin Dashboard

A beautiful, modern admin dashboard for managing your AI Istanbul guide platform.

## 🌟 Features

### 📊 Dashboard Overview
- Real-time statistics cards
- Blog posts, comments, feedback counts
- Model accuracy monitoring
- Pending moderation items

### 📝 Blog Management
- Create, edit, delete blog posts
- Rich text content editing
- Category management
- Draft/Published status control
- View counts tracking
- Search and filter functionality

### 💬 Comment Moderation
- Approve/reject comments
- Spam detection
- Bulk actions (approve all, delete spam)
- Filter by status (pending, approved, spam)
- Quick actions for each comment

### ⭐ User Feedback & ML Performance
- View all user feedback
- Intent classification accuracy
- Confidence scores
- User corrections tracking
- Export feedback data for retraining
- Low confidence predictions monitoring

### 📈 Analytics & Insights
- Interactive charts (Chart.js)
- User query trends
- Blog view statistics
- Comment activity
- Customizable time periods (7, 30, 90 days)

### 🧠 Intent Classification Stats
- Per-intent accuracy metrics
- Average confidence scores
- Correction counts
- Trigger model retraining
- Performance monitoring

### 👥 User Management
- User list with roles
- Status tracking (active/inactive)
- Last activity timestamps
- Add/edit/delete users

### ⚙️ Settings
- Site configuration
- Admin email
- API URLs
- Model retraining thresholds
- Auto-approve settings
- Email notifications

## 🚀 Quick Start

### 1. Access the Dashboard

Open in your browser:
```
http://localhost:5000/admin/dashboard.html
```

Or in production:
```
https://aistanbul.net/admin/dashboard.html
https://www.aistanbul.net/admin/dashboard.html
```

### 2. Default Credentials

**Username:** `admin@aistanbul.net`  
**Password:** `admin123` (change this in production!)

### 3. First Time Setup

1. Navigate to **Settings**
2. Update your admin email
3. Configure API base URL
4. Set retraining thresholds
5. Enable/disable auto-approvals
6. Save changes

## 📁 File Structure

```
admin/
├── dashboard.html          # Main dashboard UI
├── dashboard.js           # Dashboard functionality
└── README.md             # This file
```

## 🔌 API Endpoints

The dashboard connects to these backend endpoints:

### Dashboard Stats
```
GET /api/admin/stats
```

### Blog Management
```
GET    /api/admin/blog/posts
POST   /api/admin/blog/posts
PUT    /api/admin/blog/posts/{id}
DELETE /api/admin/blog/posts/{id}
```

### Comment Management
```
GET    /api/admin/comments
PUT    /api/admin/comments/{id}/approve
DELETE /api/admin/comments/{id}
```

### Feedback
```
GET /api/admin/feedback/export
GET /api/feedback/stats
```

### Analytics
```
GET /api/admin/analytics?days=30
```

### Intent Stats
```
GET /api/admin/intents/stats
```

### Model Retraining
```
POST /api/admin/model/retrain
```

## 🎨 Customization

### Changing Colors

Edit the CSS variables in `dashboard.html`:

```css
:root {
    --primary: #3b82f6;       /* Blue */
    --success: #10b981;       /* Green */
    --warning: #f59e0b;       /* Orange */
    --danger: #ef4444;        /* Red */
    --secondary: #8b5cf6;     /* Purple */
}
```

### Adding New Sections

1. Add nav item in sidebar:
```html
<a class="nav-item" data-section="my-section">
    <i class="fas fa-icon"></i>
    <span>My Section</span>
</a>
```

2. Add content section:
```html
<div class="content-section" id="my-section">
    <div class="section-header">
        <h3>My Section</h3>
    </div>
    <!-- Your content here -->
</div>
```

3. Add handler in `dashboard.js`:
```javascript
case 'my-section':
    await loadMySection();
    break;
```

## 🔐 Security

### Important Security Measures:

1. **Authentication Required**
   - Add JWT or session-based auth
   - Protect all `/api/admin/*` endpoints
   - Use HTTPS in production

2. **CORS Configuration**
   ```python
   # backend/main.py
   CORS_ORIGINS = [
       "https://aistanbul.net",
       "https://www.aistanbul.net"
   ]
   ```

3. **Rate Limiting**
   - Add rate limits to admin endpoints
   - Prevent brute force attacks

4. **Input Validation**
   - Sanitize all user inputs
   - Validate data types
   - Check permissions

### Example Auth Implementation:

```python
# backend/main.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Verify JWT token
    if not is_valid_admin_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

@app.get("/api/admin/stats", dependencies=[Depends(verify_admin_token)])
async def get_admin_stats():
    # ... existing code
```

## 📊 Data Storage

Dashboard data is stored in JSON files:

```
data/
├── blog_posts.json       # Blog posts
├── comments.json         # User comments
├── user_feedback.jsonl   # ML feedback (already exists)
└── users.json           # Admin users
```

### Example Blog Post Structure:

```json
{
  "posts": [
    {
      "id": 1,
      "title": "Best Places to Visit in Istanbul",
      "slug": "best-places-istanbul",
      "author": "Admin",
      "category": "Travel Tips",
      "content": "...",
      "status": "published",
      "date": "2025-10-28T12:00:00",
      "views": 1250,
      "featured_image": "https://...",
      "meta_description": "...",
      "tags": ["istanbul", "travel", "attractions"]
    }
  ]
}
```

### Example Comment Structure:

```json
{
  "comments": [
    {
      "id": 1,
      "post_id": 1,
      "comment": "Great article!",
      "author": "John Doe",
      "email": "john@example.com",
      "status": "approved",
      "date": "2025-10-27T14:30:00",
      "ip_address": "xxx.xxx.xxx.xxx"
    }
  ]
}
```

## 🔄 Backup & Restore

### Backup Data

```bash
# Create backup
tar -czf admin_backup_$(date +%Y%m%d).tar.gz data/
```

### Restore Data

```bash
# Extract backup
tar -xzf admin_backup_20251028.tar.gz
```

### Automated Backups

Add to crontab:
```bash
# Daily backup at 2 AM
0 2 * * * cd /path/to/ai-stanbul && tar -czf backups/admin_backup_$(date +\%Y\%m\%d).tar.gz data/
```

## 📱 Mobile Responsive

The dashboard is fully responsive and works on:
- 📱 Mobile phones (iOS, Android)
- 📱 Tablets (iPad, Android tablets)
- 💻 Laptops
- 🖥️ Desktop computers

Mobile features:
- Collapsible sidebar
- Touch-friendly buttons
- Responsive tables
- Optimized layouts

## 🎯 Best Practices

### 1. Regular Monitoring
- Check dashboard daily
- Review pending comments
- Monitor feedback statistics
- Track model accuracy

### 2. Content Management
- Keep blog posts updated
- Respond to comments promptly
- Moderate spam regularly
- Update categories as needed

### 3. Performance Optimization
- Review low-confidence predictions
- Retrain model monthly
- Export feedback data
- Analyze intent statistics

### 4. Security
- Change default passwords
- Use strong authentication
- Enable HTTPS
- Regular backups
- Monitor login attempts

## 🐛 Troubleshooting

### Dashboard not loading?
1. Check if backend is running
2. Verify API_BASE_URL in `dashboard.js`
3. Check browser console for errors
4. Ensure CORS is configured

### Data not showing?
1. Check API endpoint responses
2. Verify data files exist in `/data` folder
3. Check file permissions
4. Review backend logs

### Charts not rendering?
1. Ensure Chart.js is loaded
2. Check canvas element exists
3. Verify data format
4. Check browser console

### Export not working?
1. Verify feedback system is enabled
2. Check data file exists
3. Ensure proper permissions
4. Check browser download settings

## 🆕 Future Enhancements

Planned features:
- [ ] Rich text editor (TinyMCE/Quill)
- [ ] Image upload and management
- [ ] Advanced analytics dashboard
- [ ] Email notification system
- [ ] User roles and permissions
- [ ] Scheduled post publishing
- [ ] SEO optimization tools
- [ ] Multi-language support
- [ ] Activity logs
- [ ] API usage statistics

## 📚 Resources

- **Chart.js Documentation:** https://www.chartjs.org/docs/
- **Font Awesome Icons:** https://fontawesome.com/icons
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Inter Font:** https://rsms.me/inter/

## 💡 Tips

1. **Use keyboard shortcuts:**
   - `Ctrl+F` / `Cmd+F` to search within sections

2. **Bulk operations:**
   - Select multiple items for batch actions

3. **Quick filters:**
   - Use status filters to focus on specific items

4. **Export regularly:**
   - Export feedback data monthly for analysis

5. **Monitor trends:**
   - Check analytics weekly for insights

## 📞 Support

Need help?
- Check the [Production Retraining Guide](../PRODUCTION_RETRAINING_GUIDE.md)
- Review [Backend Documentation](../backend/README.md)
- Contact: admin@aistanbul.net

## 📄 License

This admin dashboard is part of the AI Istanbul project.

---

**Version:** 1.0.0  
**Last Updated:** October 28, 2025  
**Maintained by:** AI Istanbul Team
