"""
Route Visualizer Service
========================

Generates visual representations of transportation routes:
- ASCII text diagrams for terminals/logs
- SVG diagrams for web display
- JSON structure for frontend rendering

Supports:
- Single routes with step-by-step visualization
- Alternative routes comparison
- Transfer point highlights
- Time and distance annotations
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VisualSegment:
    """A visual segment in the route diagram"""
    segment_type: str  # 'transit', 'transfer', 'walk'
    mode: str  # 'metro', 'tram', 'ferry', 'marmaray', 'funicular', 'walk'
    line_name: Optional[str]
    from_station: str
    to_station: str
    duration: int  # minutes
    stops_count: Optional[int] = None
    color: Optional[str] = None


class RouteVisualizer:
    """Generate visual representations of transportation routes"""
    
    # Mode emojis and symbols
    MODE_EMOJI = {
        'metro': '🚇',
        'tram': '🚊',
        'ferry': '⛴️',
        'marmaray': '🚉',
        'funicular': '🚡',
        'walk': '🚶'
    }
    
    # Line colors (Istanbul transit system colors)
    LINE_COLORS = {
        'M1A': '#ED1C24',  # Red
        'M1B': '#ED1C24',  # Red
        'M2': '#00A651',   # Green
        'M3': '#00A4E4',   # Light Blue
        'M4': '#F69320',   # Orange
        'M5': '#8B5A99',   # Purple
        'M6': '#C1A875',   # Brown
        'M7': '#FF69B4',   # Pink
        'M8': '#808080',   # Gray
        'M9': '#FFD700',   # Gold
        'T1': '#808080',   # Gray
        'T2': '#808080',
        'T3': '#808080',
        'T4': '#808080',
        'T5': '#808080',
        'T6': '#808080',
        'F1': '#808080',
        'F2': '#808080',
        'Marmaray': '#8B0000',  # Dark Red
        'Ferry': '#0066CC'      # Blue
    }
    
    def __init__(self):
        """Initialize route visualizer"""
        logger.info("✅ Route Visualizer initialized")
    
    def visualize_route(self, route, format: str = 'ascii') -> str:
        """
        Generate visual representation of a route
        
        Args:
            route: TransportRoute object
            format: 'ascii', 'svg', or 'json'
            
        Returns:
            String representation in requested format
        """
        try:
            # Extract visual segments from route
            segments = self._extract_segments(route)
            
            if format == 'ascii':
                return self._generate_ascii(segments, route)
            elif format == 'svg':
                return self._generate_svg(segments, route)
            elif format == 'json':
                return self._generate_json(segments, route)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Route visualization failed: {e}", exc_info=True)
            return self._generate_fallback(route)
    
    def visualize_alternatives(self, alternatives: List, format: str = 'ascii') -> str:
        """
        Generate comparison visualization of alternative routes
        
        Args:
            alternatives: List of RouteOption objects
            format: 'ascii', 'svg', or 'json'
            
        Returns:
            Comparison visualization
        """
        try:
            if format == 'ascii':
                return self._generate_ascii_comparison(alternatives)
            elif format == 'svg':
                return self._generate_svg_comparison(alternatives)
            elif format == 'json':
                return self._generate_json_comparison(alternatives)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Alternatives visualization failed: {e}", exc_info=True)
            return "Visualization unavailable"
    
    def _extract_segments(self, route) -> List[VisualSegment]:
        """Extract visual segments from route"""
        segments = []
        
        for step in route.steps:
            if step.mode == 'walk' and step.line_name is None:
                # Transfer/walking segment
                segment_type = 'transfer' if 'Transfer' in step.instruction else 'walk'
            else:
                # Transit segment
                segment_type = 'transit'
            
            # Extract station names from instruction or use locations
            from_station = self._extract_station_name(step.instruction, is_start=True)
            to_station = self._extract_station_name(step.instruction, is_start=False)
            
            # Get line color
            line_id = self._extract_line_id(step.line_name) if step.line_name else None
            color = self.LINE_COLORS.get(line_id, '#808080')
            
            segments.append(VisualSegment(
                segment_type=segment_type,
                mode=step.mode,
                line_name=step.line_name,
                from_station=from_station,
                to_station=to_station,
                duration=step.duration,
                stops_count=step.stops_count,
                color=color
            ))
        
        return segments
    
    def _extract_station_name(self, instruction: str, is_start: bool) -> str:
        """Extract station name from instruction"""
        try:
            if 'from' in instruction and 'to' in instruction:
                parts = instruction.split(' from ')
                if len(parts) > 1:
                    route_part = parts[1]
                    if ' to ' in route_part:
                        from_station, to_station = route_part.split(' to ', 1)
                        # Clean up stop count if present
                        to_station = to_station.split(' (')[0]
                        return from_station.strip() if is_start else to_station.strip()
            
            # Fallback: extract from "Transfer to X" or "Walk to X"
            if not is_start and ' to ' in instruction:
                return instruction.split(' to ', 1)[1].strip()
            
            return "Station"
        except:
            return "Station"
    
    def _extract_line_id(self, line_name: str) -> Optional[str]:
        """Extract line ID from line name"""
        if not line_name:
            return None
        
        # Check for known line patterns
        for line_id in self.LINE_COLORS.keys():
            if line_id in line_name:
                return line_id
        
        return None
    
    def _generate_ascii(self, segments: List[VisualSegment], route) -> str:
        """Generate ASCII art diagram"""
        lines = []
        lines.append("")
        lines.append("┌" + "─" * 68 + "┐")
        lines.append("│" + f" Route Diagram - {route.total_duration} min, ₺{route.estimated_cost:.2f}".ljust(68) + "│")
        lines.append("├" + "─" * 68 + "┤")
        
        for i, segment in enumerate(segments):
            emoji = self.MODE_EMOJI.get(segment.mode, '🚉')
            
            if segment.segment_type == 'transit':
                # Transit segment
                stops_text = f" ({segment.stops_count} stops)" if segment.stops_count else ""
                line_text = f"{emoji} {segment.line_name}" if segment.line_name else f"{emoji} {segment.mode.capitalize()}"
                
                lines.append("│" + " " * 68 + "│")
                lines.append("│  " + segment.from_station.ljust(64) + "  │")
                lines.append("│  " + "│".ljust(64) + "  │")
                lines.append("│  " + f"│ {line_text}{stops_text}".ljust(64) + "  │")
                lines.append("│  " + f"│ {segment.duration} min".ljust(64) + "  │")
                lines.append("│  " + "▼".ljust(64) + "  │")
                lines.append("│  " + segment.to_station.ljust(64) + "  │")
                
            elif segment.segment_type == 'transfer':
                # Transfer
                lines.append("│  " + "↓".ljust(64) + "  │")
                lines.append("│  " + f"Transfer ({segment.duration} min)".ljust(64) + "  │")
                lines.append("│  " + "↓".ljust(64) + "  │")
                
            else:
                # Walking
                lines.append("│  " + "↓".ljust(64) + "  │")
                lines.append("│  " + f"🚶 Walk ({segment.duration} min)".ljust(64) + "  │")
                lines.append("│  " + "↓".ljust(64) + "  │")
        
        lines.append("└" + "─" * 68 + "┘")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_svg(self, segments: List[VisualSegment], route) -> str:
        """Generate SVG diagram"""
        width = 400
        height = 60 + (len(segments) * 80)
        
        svg_parts = []
        svg_parts.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
        svg_parts.append('  <defs>')
        svg_parts.append('    <style>')
        svg_parts.append('      .station { font: 14px sans-serif; fill: #333; }')
        svg_parts.append('      .line-label { font: 12px sans-serif; fill: #666; }')
        svg_parts.append('      .duration { font: 11px sans-serif; fill: #999; }')
        svg_parts.append('      .transfer { font: 12px sans-serif; fill: #666; font-style: italic; }')
        svg_parts.append('    </style>')
        svg_parts.append('  </defs>')
        
        # Title
        svg_parts.append(f'  <text x="10" y="25" class="station" style="font-weight: bold;">')
        svg_parts.append(f'    Route: {route.total_duration} min, ₺{route.estimated_cost:.2f}')
        svg_parts.append('  </text>')
        
        y = 60
        for i, segment in enumerate(segments):
            if segment.segment_type == 'transit':
                # Station circle
                svg_parts.append(f'  <circle cx="30" cy="{y}" r="6" fill="{segment.color}" />')
                svg_parts.append(f'  <text x="45" y="{y + 5}" class="station">{segment.from_station}</text>')
                
                # Line
                line_y = y + 15
                svg_parts.append(f'  <line x1="30" y1="{y + 6}" x2="30" y2="{line_y + 30}" stroke="{segment.color}" stroke-width="4" />')
                
                # Line label
                line_text = segment.line_name or segment.mode.capitalize()
                stops_text = f" ({segment.stops_count} stops)" if segment.stops_count else ""
                svg_parts.append(f'  <text x="45" y="{line_y + 10}" class="line-label">{line_text}{stops_text}</text>')
                svg_parts.append(f'  <text x="45" y="{line_y + 25}" class="duration">{segment.duration} min</text>')
                
                # End station circle
                end_y = line_y + 45
                svg_parts.append(f'  <circle cx="30" cy="{end_y}" r="6" fill="{segment.color}" />')
                svg_parts.append(f'  <text x="45" y="{end_y + 5}" class="station">{segment.to_station}</text>')
                
                y = end_y + 15
                
            elif segment.segment_type == 'transfer':
                # Transfer arrow
                svg_parts.append(f'  <text x="45" y="{y}" class="transfer">↓ Transfer ({segment.duration} min)</text>')
                svg_parts.append(f'  <line x1="30" y1="{y - 10}" x2="30" y2="{y + 5}" stroke="#999" stroke-width="2" stroke-dasharray="3,3" />')
                y += 20
                
            else:
                # Walking
                svg_parts.append(f'  <text x="45" y="{y}" class="transfer">🚶 Walk ({segment.duration} min)</text>')
                svg_parts.append(f'  <line x1="30" y1="{y - 10}" x2="30" y2="{y + 5}" stroke="#999" stroke-width="2" stroke-dasharray="3,3" />')
                y += 20
        
        svg_parts.append('</svg>')
        
        return "\n".join(svg_parts)
    
    def _generate_json(self, segments: List[VisualSegment], route) -> Dict:
        """Generate JSON structure for frontend rendering"""
        return {
            'total_duration': route.total_duration,
            'total_cost': route.estimated_cost,
            'summary': route.summary,
            'segments': [
                {
                    'type': seg.segment_type,
                    'mode': seg.mode,
                    'line_name': seg.line_name,
                    'from_station': seg.from_station,
                    'to_station': seg.to_station,
                    'duration': seg.duration,
                    'stops_count': seg.stops_count,
                    'color': seg.color
                }
                for seg in segments
            ]
        }
    
    def _generate_ascii_comparison(self, alternatives: List) -> str:
        """Generate ASCII comparison of alternatives"""
        lines = []
        lines.append("")
        lines.append("═" * 80)
        lines.append(f" ROUTE ALTERNATIVES ({len(alternatives)} options)")
        lines.append("═" * 80)
        
        for i, alt in enumerate(alternatives, 1):
            route = alt.route
            lines.append("")
            lines.append(f"Option {i}: {alt.preference.value.upper()}")
            lines.append("─" * 80)
            lines.append(f"Duration: {route.total_duration} min  |  Cost: ₺{route.estimated_cost:.2f}  |  {route.summary}")
            
            # Show highlights
            for highlight in alt.highlights:
                lines.append(f"  {highlight}")
            
            # Show simplified path
            transit_steps = [s for s in route.steps if s.mode != 'walk']
            if transit_steps:
                path_parts = []
                for step in transit_steps:
                    emoji = self.MODE_EMOJI.get(step.mode, '🚉')
                    line = step.line_name.split()[0] if step.line_name else step.mode
                    path_parts.append(f"{emoji} {line}")
                
                path_visual = "  →  ".join(path_parts)
                lines.append(f"  {path_visual}")
        
        lines.append("")
        lines.append("═" * 80)
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_svg_comparison(self, alternatives: List) -> str:
        """Generate SVG comparison of alternatives"""
        # For now, generate individual SVGs side by side
        # TODO: Implement side-by-side comparison layout
        return "SVG comparison not yet implemented"
    
    def _generate_json_comparison(self, alternatives: List) -> List[Dict]:
        """Generate JSON comparison structure"""
        return [
            {
                'option': i + 1,
                'preference': alt.preference.value,
                'score': alt.score,
                'highlights': alt.highlights,
                'route': self._generate_json(
                    self._extract_segments(alt.route),
                    alt.route
                )
            }
            for i, alt in enumerate(alternatives)
        ]
    
    def _generate_fallback(self, route) -> str:
        """Generate simple fallback visualization"""
        return f"""
Route Summary:
--------------
Duration: {route.total_duration} minutes
Cost: ₺{route.estimated_cost:.2f}
Summary: {route.summary}

Steps:
{chr(10).join([f"  {i+1}. {step.instruction}" for i, step in enumerate(route.steps)])}
"""


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from services.transportation_directions_service import TransportationDirectionsService
    
    # Initialize services
    directions_service = TransportationDirectionsService()
    visualizer = RouteVisualizer()
    
    # Get a route
    route = directions_service.get_directions(
        start=(41.0370, 28.9850),  # Taksim
        end=(40.9900, 29.0250),     # Kadıköy
        start_name="Taksim",
        end_name="Kadıköy"
    )
    
    if route:
        print("="*80)
        print("ASCII Diagram:")
        print("="*80)
        ascii_diagram = visualizer.visualize_route(route, format='ascii')
        print(ascii_diagram)
        
        print("\n" + "="*80)
        print("SVG Diagram:")
        print("="*80)
        svg_diagram = visualizer.visualize_route(route, format='svg')
        print(svg_diagram[:500] + "...")  # Preview
        
        # Save SVG to file
        with open('/tmp/route_diagram.svg', 'w') as f:
            f.write(svg_diagram)
        print("\n✅ SVG saved to /tmp/route_diagram.svg")
