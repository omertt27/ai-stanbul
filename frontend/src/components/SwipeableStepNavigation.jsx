import React, { useState, useEffect, useRef } from 'react';

/**
 * SwipeableStepNavigation - Step-by-step navigation with swipe gestures
 * Swipe left/right to navigate between steps
 */
const SwipeableStepNavigation = ({ steps, initialStep = 0, onStepChange }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(initialStep);
  const [touchStart, setTouchStart] = useState(null);
  const [touchEnd, setTouchEnd] = useState(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const containerRef = useRef(null);

  const minSwipeDistance = 50; // Minimum swipe distance to trigger navigation

  const currentStep = steps[currentStepIndex];
  const totalSteps = steps.length;
  const progress = ((currentStepIndex + 1) / totalSteps) * 100;

  // Handle touch start
  const onTouchStart = (e) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  // Handle touch move
  const onTouchMove = (e) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  // Handle touch end
  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe && currentStepIndex < totalSteps - 1) {
      // Swipe left = Next step
      handleNext();
    } else if (isRightSwipe && currentStepIndex > 0) {
      // Swipe right = Previous step
      handlePrevious();
    }
  };

  const handleNext = () => {
    if (currentStepIndex < totalSteps - 1) {
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentStepIndex(prev => prev + 1);
        setIsTransitioning(false);
        if (onStepChange) onStepChange(currentStepIndex + 1);
      }, 150);
    }
  };

  const handlePrevious = () => {
    if (currentStepIndex > 0) {
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentStepIndex(prev => prev - 1);
        setIsTransitioning(false);
        if (onStepChange) onStepChange(currentStepIndex - 1);
      }, 150);
    }
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight') handleNext();
      if (e.key === 'ArrowLeft') handlePrevious();
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentStepIndex]);

  if (!currentStep) return null;

  // Helper to get step icon
  const getStepIcon = (mode) => {
    const icons = {
      walk: '🚶',
      metro: '🚇',
      bus: '🚌',
      tram: '🚋',
      ferry: '⛴️',
      transfer: '🔄',
      funicular: '🚡',
      default: '➡️'
    };
    return icons[mode?.toLowerCase()] || icons.default;
  };

  return (
    <div 
      ref={containerRef}
      className="w-full h-full flex flex-col"
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
    >
      {/* Swipe Indicator */}
      <div className="text-center py-2 text-xs text-gray-400">
        ← Swipe to navigate →
      </div>

      {/* Progress Bar */}
      <div className="px-4 pb-3">
        <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
          <span>Step {currentStepIndex + 1} of {totalSteps}</span>
          <span>{Math.round(progress)}% complete</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Current Step - Large Display */}
      <div 
        className={`flex-1 px-4 py-6 transition-all duration-150 ${
          isTransitioning ? 'opacity-50 scale-95' : 'opacity-100 scale-100'
        }`}
      >
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0 w-14 h-14 bg-indigo-600 text-white rounded-full flex items-center justify-center text-2xl font-bold shadow-lg">
            {currentStepIndex + 1}
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-4xl">{getStepIcon(currentStep.mode)}</span>
              {currentStep.mode && (
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide px-2 py-1 bg-gray-100 rounded">
                  {currentStep.mode}
                </span>
              )}
            </div>
            <p className="text-2xl font-bold text-gray-900 leading-tight mb-3">
              {currentStep.instruction || currentStep.description}
            </p>
            {currentStep.details && (
              <p className="text-lg text-gray-600 mb-3">
                {currentStep.details}
              </p>
            )}
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              {currentStep.duration && (
                <span className="flex items-center bg-gray-100 px-3 py-1 rounded-full">
                  <span className="mr-1">⏱️</span>
                  ~{Math.round(currentStep.duration / 60)} min
                </span>
              )}
              {currentStep.mode === 'transfer' && currentStep.accessibility && (
                <span className="flex items-center bg-blue-50 px-3 py-1 rounded-full text-blue-600">
                  <span className="mr-1">♿</span>
                  {currentStep.accessibility}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Buttons */}
      <div className="px-4 pb-4 border-t bg-white">
        <div className="flex items-center space-x-3 pt-4">
          <button
            onClick={handlePrevious}
            disabled={currentStepIndex === 0}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all flex items-center justify-center space-x-2 ${
              currentStepIndex === 0
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-white border-2 border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700 active:scale-95'
            }`}
          >
            <span>←</span>
            <span>Previous</span>
          </button>
          <button
            onClick={handleNext}
            disabled={currentStepIndex === totalSteps - 1}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all flex items-center justify-center space-x-2 ${
              currentStepIndex === totalSteps - 1
                ? 'bg-green-600 text-white'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg active:scale-95'
            }`}
          >
            {currentStepIndex === totalSteps - 1 ? (
              <>
                <span>✓</span>
                <span>Complete</span>
              </>
            ) : (
              <>
                <span>Next</span>
                <span>→</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Step Indicators */}
      <div className="px-4 pb-4 flex items-center justify-center space-x-2">
        {steps.map((_, idx) => (
          <button
            key={idx}
            onClick={() => {
              setIsTransitioning(true);
              setTimeout(() => {
                setCurrentStepIndex(idx);
                setIsTransitioning(false);
                if (onStepChange) onStepChange(idx);
              }, 150);
            }}
            className={`h-2 rounded-full transition-all ${
              idx === currentStepIndex
                ? 'bg-indigo-600 w-8'
                : idx < currentStepIndex
                ? 'bg-green-400 w-2'
                : 'bg-gray-300 w-2'
            }`}
            aria-label={`Go to step ${idx + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

export default SwipeableStepNavigation;
