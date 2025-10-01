import React from 'react';
import { Helmet } from 'react-helmet-async';

const SEOHead = ({ 
  title, 
  description, 
  keywords = [], 
  image, 
  url, 
  type = 'article', 
  author,
  publishedDate,
  modifiedDate,
  category,
  tags = [],
  structuredData
}) => {
  const defaultTitle = 'AI Istanbul - Your Intelligent Travel Guide';
  const defaultDescription = 'Discover Istanbul with our AI-powered travel guide. Get personalized recommendations, real-time information, and insider tips for the best Istanbul experience.';
  const defaultImage = '/images/ai-istanbul-og.jpg';
  const baseUrl = window.location.origin;
  
  const fullTitle = title ? `${title} - AI Istanbul` : defaultTitle;
  const fullDescription = description || defaultDescription;
  const fullImage = image ? (image.startsWith('http') ? image : `${baseUrl}${image}`) : `${baseUrl}${defaultImage}`;
  const fullUrl = url ? `${baseUrl}${url}` : window.location.href;
  
  // Create JSON-LD structured data
  const createStructuredData = () => {
    const baseSchema = {
      "@context": "https://schema.org",
      "@type": type === 'article' ? 'Article' : 'WebPage',
      "headline": title,
      "description": fullDescription,
      "url": fullUrl,
      "image": fullImage,
      "datePublished": publishedDate,
      "dateModified": modifiedDate || publishedDate,
      "author": author ? {
        "@type": "Person",
        "name": author
      } : {
        "@type": "Organization",
        "name": "AI Istanbul"
      },
      "publisher": {
        "@type": "Organization",
        "name": "AI Istanbul",
        "logo": {
          "@type": "ImageObject",
          "url": `${baseUrl}/images/ai-istanbul-logo.png`
        }
      }
    };

    // Add article-specific fields
    if (type === 'article') {
      baseSchema.articleSection = category;
      baseSchema.keywords = [...keywords, ...tags].join(', ');
      
      if (tags.length > 0) {
        baseSchema.about = tags.map(tag => ({
          "@type": "Thing",
          "name": tag
        }));
      }
    }

    // Merge with custom structured data
    return structuredData ? { ...baseSchema, ...structuredData } : baseSchema;
  };

  return (
    <Helmet>
      {/* Basic Meta Tags */}
      <title>{fullTitle}</title>
      <meta name="description" content={fullDescription} />
      {keywords.length > 0 && <meta name="keywords" content={keywords.join(', ')} />}
      <meta name="author" content={author || 'AI Istanbul'} />
      <link rel="canonical" href={fullUrl} />

      {/* Open Graph Tags */}
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={fullDescription} />
      <meta property="og:image" content={fullImage} />
      <meta property="og:url" content={fullUrl} />
      <meta property="og:type" content={type} />
      <meta property="og:site_name" content="AI Istanbul" />
      <meta property="og:locale" content="en_US" />
      <meta property="og:locale:alternate" content="tr_TR" />

      {/* Article-specific Open Graph */}
      {type === 'article' && (
        <>
          {publishedDate && <meta property="article:published_time" content={publishedDate} />}
          {modifiedDate && <meta property="article:modified_time" content={modifiedDate} />}
          {author && <meta property="article:author" content={author} />}
          {category && <meta property="article:section" content={category} />}
          {tags.map((tag, index) => (
            <meta key={index} property="article:tag" content={tag} />
          ))}
        </>
      )}

      {/* Twitter Card Tags */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={fullDescription} />
      <meta name="twitter:image" content={fullImage} />
      <meta name="twitter:site" content="@AIIstanbul" />
      <meta name="twitter:creator" content={author ? `@${author.replace(/\s+/g, '')}` : '@AIIstanbul'} />

      {/* Additional SEO Meta Tags */}
      <meta name="robots" content="index, follow, max-snippet:-1, max-image-preview:large, max-video-preview:-1" />
      <meta name="googlebot" content="index, follow" />
      <meta name="language" content="English" />
      <meta name="geo.region" content="TR-34" />
      <meta name="geo.placename" content="Istanbul" />
      <meta name="geo.position" content="41.0082;28.9784" />
      <meta name="ICBM" content="41.0082, 28.9784" />

      {/* Structured Data */}
      <script type="application/ld+json">
        {JSON.stringify(createStructuredData())}
      </script>

      {/* Preconnect for performance */}
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
    </Helmet>
  );
};

export default SEOHead;
