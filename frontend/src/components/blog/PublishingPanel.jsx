/**
 * Publishing Options Panel
 * Features:
 * - Publish now / Schedule for later
 * - Save as draft
 * - Preview before publish
 * - Visibility settings
 * - Social sharing options
 */
import React, { useState } from 'react';

const PublishingPanel = ({
  status = 'draft', // draft, scheduled, published
  scheduledDate,
  visibility = 'public', // public, private, password
  onStatusChange,
  onScheduledDateChange,
  onVisibilityChange,
  onSaveDraft,
  onPreview,
  onPublish,
  isSubmitting = false,
  isDirty = false,
  canPublish = true,
  validationErrors = [],
  isEditMode = false // NEW: edit mode flag
}) => {
  const [showScheduler, setShowScheduler] = useState(false);
  const [showVisibilityOptions, setShowVisibilityOptions] = useState(false);

  const formatDate = (date) => {
    if (!date) return '';
    const d = new Date(date);
    return d.toLocaleString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    });
  };

  const getMinDateTime = () => {
    const now = new Date();
    now.setMinutes(now.getMinutes() + 5);
    return now.toISOString().slice(0, 16);
  };

  return (
    <div className="bg-gray-800 rounded-xl border-2 border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Publish
        </h3>
      </div>

      <div className="p-5 space-y-4">
        {/* Validation Errors */}
        {validationErrors.length > 0 && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-3">
            <p className="text-sm text-red-400 font-medium mb-1">Please fix the following:</p>
            <ul className="text-xs text-red-300 list-disc list-inside space-y-1">
              {validationErrors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Status Indicator */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-300">Status:</span>
          <span className={`px-2 py-1 rounded text-xs font-medium ${
            status === 'published' ? 'bg-green-600/20 text-green-400' :
            status === 'scheduled' ? 'bg-blue-600/20 text-blue-400' :
            'bg-gray-600/20 text-gray-400'
          }`}>
            {status === 'published' ? 'Published' :
             status === 'scheduled' ? 'Scheduled' :
             'Draft'}
          </span>
        </div>

        {/* Visibility */}
        <div>
          <button
            type="button"
            onClick={() => setShowVisibilityOptions(!showVisibilityOptions)}
            className="w-full flex items-center justify-between py-2 text-sm"
          >
            <span className="text-gray-300">Visibility:</span>
            <span className="flex items-center gap-1 text-gray-400">
              {visibility === 'public' && (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Public
                </>
              )}
              {visibility === 'private' && (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                  Private
                </>
              )}
              <svg className={`w-4 h-4 transition-transform ${showVisibilityOptions ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </span>
          </button>

          {showVisibilityOptions && (
            <div className="mt-2 space-y-2 p-3 bg-gray-900 rounded-lg">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="visibility"
                  checked={visibility === 'public'}
                  onChange={() => onVisibilityChange('public')}
                  className="w-4 h-4 text-blue-600"
                />
                <div>
                  <span className="text-sm text-white">Public</span>
                  <p className="text-xs text-gray-400">Visible to everyone</p>
                </div>
              </label>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="visibility"
                  checked={visibility === 'private'}
                  onChange={() => onVisibilityChange('private')}
                  className="w-4 h-4 text-blue-600"
                />
                <div>
                  <span className="text-sm text-white">Private</span>
                  <p className="text-xs text-gray-400">Only you can see this</p>
                </div>
              </label>
            </div>
          )}
        </div>

        {/* Schedule */}
        <div>
          <button
            type="button"
            onClick={() => setShowScheduler(!showScheduler)}
            className="w-full flex items-center justify-between py-2 text-sm"
          >
            <span className="text-gray-300">Publish:</span>
            <span className="flex items-center gap-1 text-gray-400">
              {scheduledDate ? (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  {formatDate(scheduledDate)}
                </>
              ) : (
                'Immediately'
              )}
              <svg className={`w-4 h-4 transition-transform ${showScheduler ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </span>
          </button>

          {showScheduler && (
            <div className="mt-2 p-3 bg-gray-900 rounded-lg space-y-3">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="schedule"
                  checked={!scheduledDate}
                  onChange={() => onScheduledDateChange(null)}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm text-white">Publish immediately</span>
              </label>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="radio"
                  name="schedule"
                  checked={!!scheduledDate}
                  onChange={() => onScheduledDateChange(getMinDateTime())}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm text-white">Schedule for later</span>
              </label>
              
              {scheduledDate && (
                <input
                  type="datetime-local"
                  value={scheduledDate}
                  min={getMinDateTime()}
                  onChange={(e) => onScheduledDateChange(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 text-white border border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>
          )}
        </div>

        {/* Unsaved Changes Warning */}
        {isDirty && (
          <div className="flex items-center gap-2 text-xs text-yellow-400">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            You have unsaved changes
          </div>
        )}

        {/* Action Buttons */}
        <div className="space-y-2 pt-2">
          {/* Preview Button */}
          <button
            type="button"
            onClick={onPreview}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            Preview
          </button>

          {/* Save Draft Button */}
          <button
            type="button"
            onClick={onSaveDraft}
            disabled={isSubmitting}
            className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
            </svg>
            Save Draft
          </button>

          {/* Publish Button */}
          <button
            type="button"
            onClick={onPublish}
            disabled={isSubmitting || !canPublish}
            className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
          >
            {isSubmitting ? (
              <>
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                {isEditMode ? 'Updating...' : 'Publishing...'}
              </>
            ) : scheduledDate ? (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                {isEditMode ? 'Update & Schedule' : 'Schedule Post'}
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                {isEditMode ? 'Update Post' : 'Publish Now'}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PublishingPanel;
