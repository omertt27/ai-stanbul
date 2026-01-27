/**
 * Keep-Warm Service for Google Cloud Run
 * Pings the backend every 5 minutes to prevent cold starts
 */

const BACKEND_URL = 'https://ai-stanbul-509659445005.europe-west1.run.app';

const keepWarm = async () => {
    try {
        const response = await fetch(`${BACKEND_URL}/api/health`);
        const data = await response.json();
        console.log(`✅ Keep-warm ping successful: ${new Date().toISOString()}`, data);
    } catch (error) {
        console.log(`❌ Keep-warm ping failed: ${error.message}`);
    }
};

// Ping every 5 minutes (300,000ms)
setInterval(keepWarm, 5 * 60 * 1000);

// Initial ping
keepWarm();

console.log('🔥 Keep-warm service started - pinging every 5 minutes');
