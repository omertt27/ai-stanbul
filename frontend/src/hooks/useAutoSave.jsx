/**
 * Auto-Save Hook for Blog Posts
 * Features:
 * - Auto-save to localStorage every 30 seconds
 * - Draft recovery on page load
 * - Save status indicator
 * - Manual save trigger
 */
import { useState, useEffect, useCallback, useRef } from 'react';

const DRAFT_KEY_PREFIX = 'blog_draft_';
const AUTO_SAVE_INTERVAL = 30000; // 30 seconds

export const useAutoSave = (formData, draftId = 'new') => {
  const [saveStatus, setSaveStatus] = useState('idle'); // idle, saving, saved, error
  const [lastSaved, setLastSaved] = useState(null);
  const [hasDraft, setHasDraft] = useState(false);
  const timeoutRef = useRef(null);
  const draftKey = `${DRAFT_KEY_PREFIX}${draftId}`;

  // Check for existing draft on mount
  useEffect(() => {
    const existingDraft = localStorage.getItem(draftKey);
    if (existingDraft) {
      setHasDraft(true);
    }
  }, [draftKey]);

  // Save draft to localStorage
  const saveDraft = useCallback(() => {
    try {
      setSaveStatus('saving');
      
      const draftData = {
        ...formData,
        savedAt: new Date().toISOString(),
        version: 1
      };
      
      localStorage.setItem(draftKey, JSON.stringify(draftData));
      
      setLastSaved(new Date());
      setSaveStatus('saved');
      setHasDraft(true);
      
      // Reset status after 3 seconds
      setTimeout(() => setSaveStatus('idle'), 3000);
      
      return true;
    } catch (err) {
      console.error('Auto-save failed:', err);
      setSaveStatus('error');
      return false;
    }
  }, [formData, draftKey]);

  // Load draft from localStorage
  const loadDraft = useCallback(() => {
    try {
      const draftData = localStorage.getItem(draftKey);
      if (draftData) {
        const parsed = JSON.parse(draftData);
        return parsed;
      }
      return null;
    } catch (err) {
      console.error('Failed to load draft:', err);
      return null;
    }
  }, [draftKey]);

  // Clear draft from localStorage
  const clearDraft = useCallback(() => {
    try {
      localStorage.removeItem(draftKey);
      setHasDraft(false);
      setLastSaved(null);
      setSaveStatus('idle');
    } catch (err) {
      console.error('Failed to clear draft:', err);
    }
  }, [draftKey]);

  // Auto-save effect
  useEffect(() => {
    // Only auto-save if there's content
    const hasContent = formData.title?.trim() || formData.content?.trim();
    
    if (hasContent) {
      // Clear existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      // Set new timeout
      timeoutRef.current = setTimeout(() => {
        saveDraft();
      }, AUTO_SAVE_INTERVAL);
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [formData, saveDraft]);

  // Save on beforeunload
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      const hasContent = formData.title?.trim() || formData.content?.trim();
      if (hasContent) {
        saveDraft();
        e.preventDefault();
        e.returnValue = '';
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [formData, saveDraft]);

  return {
    saveStatus,
    lastSaved,
    hasDraft,
    saveDraft,
    loadDraft,
    clearDraft
  };
};

/**
 * Auto-Save Status Indicator Component
 */
export const AutoSaveIndicator = ({ saveStatus, lastSaved }) => {
  const getStatusDisplay = () => {
    switch (saveStatus) {
      case 'saving':
        return {
          icon: (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ),
          text: 'Saving...',
          color: 'text-yellow-400'
        };
      case 'saved':
        return {
          icon: (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          ),
          text: 'Saved',
          color: 'text-green-400'
        };
      case 'error':
        return {
          icon: (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ),
          text: 'Save failed',
          color: 'text-red-400'
        };
      default:
        return null;
    }
  };

  const status = getStatusDisplay();
  
  if (!status && !lastSaved) return null;

  return (
    <div className="flex items-center gap-2 text-xs">
      {status ? (
        <span className={`flex items-center gap-1 ${status.color}`}>
          {status.icon}
          {status.text}
        </span>
      ) : lastSaved ? (
        <span className="text-gray-400">
          Last saved {new Date(lastSaved).toLocaleTimeString()}
        </span>
      ) : null}
    </div>
  );
};

/**
 * Draft Recovery Modal Component
 */
export const DraftRecoveryModal = ({ draft, onRecover, onDiscard }) => {
  if (!draft) return null;

  const savedDate = draft.savedAt ? new Date(draft.savedAt) : null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-xl border border-gray-700 max-w-md w-full p-6 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-blue-600/20 flex items-center justify-center">
            <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Recover Draft?</h3>
            <p className="text-sm text-gray-400">
              We found an unsaved draft from your last session
            </p>
          </div>
        </div>

        <div className="bg-gray-900 rounded-lg p-4 mb-4">
          <p className="text-sm text-gray-300 font-medium truncate">
            {draft.title || 'Untitled Post'}
          </p>
          {savedDate && (
            <p className="text-xs text-gray-500 mt-1">
              Saved on {savedDate.toLocaleDateString()} at {savedDate.toLocaleTimeString()}
            </p>
          )}
          {draft.content && (
            <p className="text-xs text-gray-400 mt-2 line-clamp-2">
              {draft.content.replace(/<[^>]*>/g, '').substring(0, 100)}...
            </p>
          )}
        </div>

        <div className="flex gap-3">
          <button
            onClick={onDiscard}
            className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            Start Fresh
          </button>
          <button
            onClick={onRecover}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Recover Draft
          </button>
        </div>
      </div>
    </div>
  );
};

export default useAutoSave;
