# Quick Start: Using the Route Visualizer

## Basic Example

```python
from services.transportation_directions_service import TransportationDirectionsService
from services.route_visualizer import RouteVisualizer

# Initialize
directions = TransportationDirectionsService()
visualizer = RouteVisualizer()

# Get a route
route = directions.get_directions(
    start=(41.0370, 28.9850),  # Taksim
    end=(40.9900, 29.0250),     # Kadıköy
    start_name="Taksim",
    end_name="Kadıköy"
)

# Show ASCII diagram
print(visualizer.visualize_route(route, format='ascii'))

# Save SVG file
svg = visualizer.visualize_route(route, format='svg')
with open('route.svg', 'w') as f:
    f.write(svg)

# Get JSON for frontend
json_data = visualizer.visualize_route(route, format='json')
print(json_data)
```

## Test It Now

```bash
# Run comprehensive tests
python test_route_visualizer.py

# View generated SVG files
open /tmp/route_*.svg
```

## See Documentation

- Full guide: `ROUTE_VISUALIZER_COMPLETE.md`
- Roadmap: `WHATS_NEXT_IMPLEMENTATION_ROADMAP.md`
