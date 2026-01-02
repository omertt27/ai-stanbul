"""
Route Service Custom Exceptions
================================

Custom exception hierarchy for better error handling and debugging
in the Istanbul AI route planning system.
"""


class RouteServiceError(Exception):
    """Base exception for all route service errors"""
    pass


class LocationExtractionError(RouteServiceError):
    """Failed to extract valid locations from query"""
    
    def __init__(self, message: str, query: str = None, error_response: dict = None):
        self.query = query
        self.error_response = error_response
        super().__init__(message)


class InsufficientLocationsError(LocationExtractionError):
    """Not enough locations extracted for route planning"""
    
    def __init__(self, message: str = None, found_count: int = None, required_count: int = 2, error_response: dict = None):
        self.found_count = found_count
        self.required_count = required_count
        self.error_response = error_response
        
        # Use custom message if provided, otherwise generate default
        if message is None and found_count is not None:
            message = f"Found {found_count} location(s), but need at least {required_count} for route planning"
        
        super().__init__(message or "Insufficient locations for route planning", error_response=error_response)


class ServiceUnavailableError(RouteServiceError):
    """External service (OSRM, GPS, etc.) is unavailable"""
    
    def __init__(self, service_name: str, message: str, original_error: Exception = None):
        self.service_name = service_name
        self.original_error = original_error
        super().__init__(f"{service_name} unavailable: {message}")


class GeocodingError(RouteServiceError):
    """Failed to geocode a location"""
    
    def __init__(self, location_query: str, message: str = None):
        self.location_query = location_query
        super().__init__(message or f"Failed to geocode location: '{location_query}'")


class InvalidCoordinatesError(RouteServiceError):
    """Invalid latitude/longitude coordinates"""
    
    def __init__(self, coords, message: str = None):
        self.coords = coords
        super().__init__(message or f"Invalid coordinates: {coords}")


class RouteCalculationError(RouteServiceError):
    """Failed to calculate route between locations"""
    
    def __init__(self, start, end, message: str = None):
        self.start = start
        self.end = end
        super().__init__(
            message or f"Failed to calculate route from {start} to {end}"
        )


class GPSPermissionRequiredError(RouteServiceError):
    """User needs to grant GPS permission"""
    
    def __init__(self, message: str = None, error_response: dict = None):
        self.error_response = error_response
        super().__init__(
            message or "GPS permission required to determine current location"
        )


class NavigationError(RouteServiceError):
    """Error during turn-by-turn navigation"""
    
    def __init__(self, message: str, error_response: dict = None):
        self.error_response = error_response
        super().__init__(message)
    
    def __init__(self, message: str, session_id: str = None):
        self.session_id = session_id
        super().__init__(message)


class FallbackRoutingUsedWarning(UserWarning):
    """Warning when fallback routing (Haversine) is used instead of OSRM"""
    
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Fallback routing used: {reason}")


class LowConfidenceWarning(UserWarning):
    """Warning when location match or route has low confidence"""
    
    def __init__(self, confidence: float, threshold: float = 0.7):
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"Low confidence result: {confidence:.2f} (threshold: {threshold})"
        )
