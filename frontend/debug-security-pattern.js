// Test which security pattern blocks the sample card text
import re from 'regexparam';

const testInput = "show me the best attractions and landmarks in Istanbul";

console.log('Testing input:', testInput);
console.log('Length:', testInput.length);
console.log('Characters:', testInput.split('').map(c => `'${c}'`).join(' '));

// Test the same patterns from backend
const patterns = {
    sql: [
        /[';]/,
        /--/,
        /\/\*|\*\//,
        /\b(UNION|SELECT|DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|DECLARE)\b/i,
        /\b(OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?/i,
        /['\"]?\s*(OR|AND)\s+['\"]?/i
    ],
    xss: [
        /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/i,
        /<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/i,
        /<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>/i,
        /<embed\b[^>]*>/i,
        /<link\b[^>]*>/i,
        /<meta\b[^>]*>/i,
        /<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/i,
        /javascript:/i,
        /vbscript:/i,
        /data:/i,
        /on\w+\s*=/i,
        /expression\s*\(/i
    ],
    command: [
        /[;&|`]/,
        /\$\([^)]*\)/,
        /`[^`]*`/,
        /\${[^}]*}/,
        /\|\s*\w+/
    ],
    template: [
        /\{\{[^}]*\}\}/,
        /<%[^%]*%>/,
        /\{%[^%]*%\}/
    ],
    path: [
        /\.\.\//,
        /\.\.\\{1,2}/,
        /~\//
    ]
};

for (const [category, patternList] of Object.entries(patterns)) {
    console.log(`\nTesting ${category.toUpperCase()} patterns:`);
    for (let i = 0; i < patternList.length; i++) {
        const pattern = patternList[i];
        if (pattern.test(testInput)) {
            console.log(`❌ MATCHED: ${pattern} in category ${category}`);
            console.log(`   This pattern is blocking the input!`);
        } else {
            console.log(`✅ OK: ${pattern}`);
        }
    }
}

console.log('\n🔍 Detailed character analysis:');
console.log('Checking for special characters that might trigger patterns...');

const specialChars = testInput.match(/[^a-zA-Z0-9\s]/g);
if (specialChars) {
    console.log('Special characters found:', specialChars);
} else {
    console.log('No special characters found - this should be safe!');
}
