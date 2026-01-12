import React, { useState, useEffect, useRef } from 'react';

/**
 * RouteBottomSheet - Google Maps-style draggable bottom sheet for mobile
 * Optimized for mobile devices with touch gestures
 */
const RouteBottomSheet = ({ 
  children, 
  isOpen = true,
  onClose,
  snapPoints = [0.3, 0.5, 0.9], // Collapsed, Half, Expanded (as % of screen height)
  initialSnapPoint = 0.5
}) => {
  const [currentSnapPoint, setCurrentSnapPoint] = useState(initialSnapPoint);
  const [isDragging, setIsDragging] = useState(false);
  const [startY, setStartY] = useState(0);
  const [currentY, setCurrentY] = useState(0);
  const sheetRef = useRef(null);

  // Convert snap point percentage to pixels
  const getSnapPointPx = (percentage) => {
    return window.innerHeight * percentage;
  };

  // Find closest snap point
  const findClosestSnapPoint = (currentHeight) => {
    const currentPercentage = currentHeight / window.innerHeight;
    return snapPoints.reduce((prev, curr) => 
      Math.abs(curr - currentPercentage) < Math.abs(prev - currentPercentage) ? curr : prev
    );
  };

  // Touch event handlers
  const handleTouchStart = (e) => {
    // Only handle drags from the handle area
    if (e.target.closest('.bottom-sheet-handle')) {
      setIsDragging(true);
      setStartY(e.touches[0].clientY);
    }
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;
    
    const touch = e.touches[0];
    setCurrentY(touch.clientY);
    
    // Calculate new height
    const deltaY = startY - touch.clientY;
    const newHeight = Math.max(
      getSnapPointPx(snapPoints[0]), // Min height
      Math.min(
        getSnapPointPx(snapPoints[snapPoints.length - 1]), // Max height
        getSnapPointPx(currentSnapPoint) + deltaY
      )
    );
    
    if (sheetRef.current) {
      sheetRef.current.style.height = `${newHeight}px`;
    }
  };

  const handleTouchEnd = () => {
    if (!isDragging) return;
    
    setIsDragging(false);
    
    // Snap to closest point
    const currentHeight = sheetRef.current?.offsetHeight || getSnapPointPx(currentSnapPoint);
    const closestSnap = findClosestSnapPoint(currentHeight);
    
    setCurrentSnapPoint(closestSnap);
    
    // Close if dragged below minimum
    if (closestSnap === snapPoints[0] && currentY > startY + 100) {
      if (onClose) onClose();
    }
  };

  // Mouse events for desktop testing
  const handleMouseDown = (e) => {
    if (e.target.closest('.bottom-sheet-handle')) {
      setIsDragging(true);
      setStartY(e.clientY);
    }
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    setCurrentY(e.clientY);
    const deltaY = startY - e.clientY;
    const newHeight = Math.max(
      getSnapPointPx(snapPoints[0]),
      Math.min(
        getSnapPointPx(snapPoints[snapPoints.length - 1]),
        getSnapPointPx(currentSnapPoint) + deltaY
      )
    );
    
    if (sheetRef.current) {
      sheetRef.current.style.height = `${newHeight}px`;
    }
  };

  const handleMouseUp = () => {
    if (!isDragging) return;
    
    setIsDragging(false);
    const currentHeight = sheetRef.current?.offsetHeight || getSnapPointPx(currentSnapPoint);
    const closestSnap = findClosestSnapPoint(currentHeight);
    setCurrentSnapPoint(closestSnap);
  };

  // Add mouse event listeners for desktop
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && onClose) {
        onClose();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-30 z-40 md:hidden"
        onClick={onClose}
        style={{ 
          opacity: isDragging ? 0.5 : 0.3,
          transition: isDragging ? 'none' : 'opacity 0.3s ease'
        }}
      />
      
      {/* Bottom Sheet */}
      <div
        ref={sheetRef}
        className="fixed bottom-0 left-0 right-0 bg-white rounded-t-3xl shadow-2xl z-50 md:hidden"
        style={{
          height: `${getSnapPointPx(currentSnapPoint)}px`,
          transition: isDragging ? 'none' : 'height 0.3s ease',
          maxHeight: '95vh',
          touchAction: 'none'
        }}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onMouseDown={handleMouseDown}
      >
        {/* Drag Handle */}
        <div className="bottom-sheet-handle flex items-center justify-center py-3 cursor-grab active:cursor-grabbing">
          <div className="w-12 h-1.5 bg-gray-300 rounded-full"></div>
        </div>
        
        {/* Snap Point Indicators (optional) */}
        <div className="flex items-center justify-center space-x-2 pb-2">
          {snapPoints.map((point, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentSnapPoint(point)}
              className={`w-2 h-2 rounded-full transition-all ${
                Math.abs(currentSnapPoint - point) < 0.05
                  ? 'bg-indigo-600 w-6'
                  : 'bg-gray-300'
              }`}
              aria-label={`Snap to ${point * 100}%`}
            />
          ))}
        </div>
        
        {/* Content */}
        <div className="overflow-y-auto h-full pb-20 px-4">
          {children}
        </div>
      </div>
    </>
  );
};

export default RouteBottomSheet;
