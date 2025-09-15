# VS Code Performance Fixes

## Immediate Actions:

### 1. Restart VS Code Completely
- Close all VS Code windows
- Force quit if needed: `Cmd + Opt + Esc`
- Reopen only necessary projects

### 2. Reduce Extensions Load
Current heavy extensions detected:
- GitHub Copilot (high memory usage)
- GitLens (can be heavy)
- Python extensions

### 3. Update Settings for Better Performance
Add these to your settings.json:

```json
{
  // Reduce file watching
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/.git/**": true,
    "**/dist/**": true,
    "**/build/**": true,
    "**/__pycache__/**": true
  },
  
  // Reduce search scope
  "search.exclude": {
    "**/node_modules": true,
    "**/bower_components": true,
    "**/*.code-search": true,
    "**/dist": true,
    "**/build": true
  },
  
  // Limit TypeScript/JavaScript services
  "typescript.preferences.includePackageJsonAutoImports": "off",
  "typescript.suggest.autoImports": false,
  
  // Reduce Copilot aggressiveness
  "github.copilot.advanced": {
    "inlineSuggestCount": 1
  },
  
  // Performance optimizations
  "editor.semanticHighlighting.enabled": false,
  "editor.bracketPairColorization.enabled": false,
  "extensions.autoUpdate": false,
  "terminal.integrated.gpuAcceleration": "off",
  
  // Reduce background processes
  "git.autofetch": false,
  "git.autorefresh": false
}
```

### 4. Close Unused Tabs and Windows
- Use `Cmd + W` to close tabs
- Keep only 1-2 projects open at once
- Use "File > Close Folder" to fully close projects

### 5. System-level Fixes
- Free up disk space (you have only 921M unused RAM)
- Close Chrome tabs (Chrome is using 1GB+)
- Restart your Mac if the issue persists

### 6. Emergency Reset
If VS Code keeps freezing:
1. Quit VS Code completely
2. Remove workspace state: `rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage`
3. Restart VS Code

## Prevention:
- Don't open too many large projects simultaneously
- Regularly restart VS Code
- Monitor Activity Monitor for memory usage
- Consider using lighter alternatives for some tasks
