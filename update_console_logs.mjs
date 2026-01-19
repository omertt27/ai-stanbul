/**
 * Script to update all console.log statements to production-safe logger
 * Run this to clean up all development console logs
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, relative } from 'path';

const filesToUpdate = [
  'frontend/src/services/gpsLocationService.js',
  'frontend/src/services/offlineMapTileCache.js',
  'frontend/src/services/offlineEnhancementManager.js',
  'frontend/src/services/locationApi.js',
  'frontend/src/App.jsx',
];

const serviceNameMap = {
  'gpsLocationService.js': 'GPS',
  'offlineMapTileCache.js': 'MapTileCache',
  'offlineEnhancementManager.js': 'OfflineManager',
  'locationApi.js': 'LocationAPI',
  'App.jsx': 'App',
};

filesToUpdate.forEach(file => {
  try {
    let content = readFileSync(file, 'utf8');
    const filename = file.split('/').pop();
    const serviceName = serviceNameMap[filename] || 'Service';
    
    // Check if logger import already exists
    const hasLoggerImport = content.includes("import logger from");
    
    if (!hasLoggerImport) {
      // Add logger import after the first import or at the beginning
      const firstImportMatch = content.match(/^import .+from.+;$/m);
      if (firstImportMatch) {
        const importLine = firstImportMatch[0];
        const relativePath = file.includes('App.jsx') ? './utils/logger.js' : '../utils/logger.js';
        content = content.replace(
          importLine,
          `${importLine}\nimport logger from '${relativePath}';\nconst log = logger.namespace('${serviceName}');`
        );
      }
    }
    
    // Replace console statements
    content = content.replace(/console\.log\(/g, 'log.debug(');
    content = content.replace(/console\.error\(/g, 'log.error(');
    content = content.replace(/console\.warn\(/g, 'log.warn(');
    content = content.replace(/console\.info\(/g, 'log.info(');
    
    writeFileSync(file, content);
    console.log(`✅ Updated ${file}`);
  } catch (error) {
    console.error(`❌ Error updating ${file}:`, error.message);
  }
});

console.log('\n✅ All files updated with production-safe logger!');
console.log('\n📝 Summary:');
console.log('  - console.log → log.debug (only in development)');
console.log('  - console.error → log.error (always shown, sanitized in production)');
console.log('  - console.warn → log.warn (only in development)');
console.log('  - console.info → log.info (only in development)');
