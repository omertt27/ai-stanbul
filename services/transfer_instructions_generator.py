#!/usr/bin/env python3
"""
Transfer Instructions Generator - Google Maps Style
===================================================

Generates detailed, step-by-step transfer instructions with visual guidance
similar to Google Maps for Istanbul transportation routes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TransferType(Enum):
    """Types of transfers between transportation modes"""
    METRO_TO_METRO = "metro_to_metro"
    METRO_TO_TRAM = "metro_to_tram"
    METRO_TO_BUS = "metro_to_bus"
    METRO_TO_FERRY = "metro_to_ferry"
    MARMARAY_TO_METRO = "marmaray_to_metro"
    TRAM_TO_METRO = "tram_to_metro"
    BUS_TO_METRO = "bus_to_metro"
    FERRY_TO_METRO = "ferry_to_metro"
    SAME_STATION = "same_station"
    CROSS_PLATFORM = "cross_platform"


@dataclass
class TransferInstruction:
    """Detailed transfer instruction with visual guidance"""
    step_number: int
    instruction_text: str
    detailed_steps: List[str]
    transfer_type: TransferType
    from_line: str
    to_line: str
    station_name: str
    walking_distance: int  # meters
    estimated_time: int  # minutes
    platform_info: Optional[str] = None
    exit_number: Optional[str] = None
    landmarks: Optional[List[str]] = None
    accessibility_info: Optional[str] = None
    visual_cues: Optional[List[str]] = None


@dataclass
class StationTransferData:
    """Comprehensive transfer data for a station"""
    station_name: str
    transfer_available: List[str]  # Available lines
    transfer_time: Dict[str, int]  # Transfer time between lines
    platform_layout: str
    signage_info: str
    accessibility: List[str]
    amenities: List[str]
    coordinates: Tuple[float, float]


class TransferInstructionsGenerator:
    """Generates detailed transfer instructions for Istanbul transportation"""
    
    def __init__(self):
        """Initialize the transfer instructions generator"""
        
        # Comprehensive station transfer database
        self.transfer_stations = self._load_transfer_stations()
        
        # Platform and exit information
        self.station_layouts = self._load_station_layouts()
        
        # Visual cues and landmarks
        self.visual_landmarks = self._load_visual_landmarks()
        
        logger.info("🗺️ Transfer Instructions Generator initialized")
    
    def _load_transfer_stations(self) -> Dict[str, StationTransferData]:
        """Load comprehensive transfer station data"""
        return {
            'Yenikapı': StationTransferData(
                station_name='Yenikapı',
                transfer_available=['M1A', 'M1B', 'M2', 'MARMARAY'],
                transfer_time={
                    'M1A_to_M2': 5,
                    'M1B_to_M2': 5,
                    'M1A_to_MARMARAY': 3,
                    'M2_to_MARMARAY': 3,
                },
                platform_layout='Multi-level interchange',
                signage_info='Clear multilingual signs, follow blue M2 signs for Taksim direction',
                accessibility=['Elevators', 'Escalators', 'Wheelchair accessible'],
                amenities=['Restrooms', 'Kiosks', 'ATMs', 'Info desk'],
                coordinates=(41.0085, 28.9512)
            ),
            'Ataköy-Şirinevler': StationTransferData(
                station_name='Ataköy-Şirinevler',
                transfer_available=['M1A', 'M1B'],
                transfer_time={'M1A_to_M1B': 2},
                platform_layout='Same platform transfer (cross-platform)',
                signage_info='Simply cross the platform for the other direction',
                accessibility=['Elevators', 'Escalators'],
                amenities=['Restrooms', 'Cafe'],
                coordinates=(40.9963, 28.8614)
            ),
            'Mecidiyeköy': StationTransferData(
                station_name='Mecidiyeköy',
                transfer_available=['M2', 'M7'],
                transfer_time={'M2_to_M7': 4},
                platform_layout='Connected underground passages',
                signage_info='Follow yellow M7 signs, take escalator down one level',
                accessibility=['Elevators', 'Escalators', 'Wheelchair accessible'],
                amenities=['Restrooms', 'Kiosks', 'Shopping area'],
                coordinates=(41.0639, 28.9986)
            ),
            'Gayrettepe': StationTransferData(
                station_name='Gayrettepe',
                transfer_available=['M2', 'M11'],
                transfer_time={'M2_to_M11': 3},
                platform_layout='Modern interchange with clear signage',
                signage_info='Follow purple M11 signs for Airport',
                accessibility=['Elevators', 'Escalators', 'Wheelchair accessible'],
                amenities=['Restrooms', 'Kiosks', 'ATMs', 'Food court'],
                coordinates=(41.0688, 29.0140)
            ),
            'Kabataş': StationTransferData(
                station_name='Kabataş',
                transfer_available=['T1', 'F1', 'FERRY'],
                transfer_time={
                    'T1_to_F1': 2,
                    'T1_to_FERRY': 5,
                    'F1_to_FERRY': 3,
                },
                platform_layout='Multi-modal hub (tram, funicular, ferry)',
                signage_info='Funicular entrance at tram stop, ferry terminal 100m walk',
                accessibility=['Elevators for F1', 'Tram is step-free'],
                amenities=['Restrooms', 'Kiosks', 'Ticket machines'],
                coordinates=(41.0298, 29.0064)
            ),
            'Sirkeci': StationTransferData(
                station_name='Sirkeci',
                transfer_available=['MARMARAY', 'T1'],
                transfer_time={'MARMARAY_to_T1': 4},
                platform_layout='Exit station, tram stop above ground',
                signage_info='Follow tram signs, exit to street level',
                accessibility=['Elevators', 'Escalators'],
                amenities=['Restrooms', 'Kiosks', 'Historic train station'],
                coordinates=(41.0170, 28.9767)
            ),
            'Üsküdar': StationTransferData(
                station_name='Üsküdar',
                transfer_available=['MARMARAY', 'M5', 'FERRY'],
                transfer_time={
                    'MARMARAY_to_M5': 3,
                    'MARMARAY_to_FERRY': 5,
                    'M5_to_FERRY': 4,
                },
                platform_layout='Major Asian-side interchange',
                signage_info='Clear signs for metro and ferry',
                accessibility=['Elevators', 'Escalators', 'Wheelchair accessible'],
                amenities=['Restrooms', 'Kiosks', 'Shopping', 'Food court'],
                coordinates=(41.0243, 29.0159)
            ),
            'Kadıköy': StationTransferData(
                station_name='Kadıköy',
                transfer_available=['M4', 'FERRY', 'BUS'],
                transfer_time={
                    'M4_to_FERRY': 8,
                    'M4_to_BUS': 5,
                },
                platform_layout='Metro underground, ferry at waterfront',
                signage_info='Exit metro, follow ferry signs, 300m walk to pier',
                accessibility=['Elevators in metro'],
                amenities=['Restrooms', 'Shopping area', 'Restaurants'],
                coordinates=(40.9904, 29.0261)
            ),
        }
    
    def _load_station_layouts(self) -> Dict[str, Dict[str, Any]]:
        """Load detailed station layout information"""
        return {
            'Yenikapı': {
                'levels': 3,
                'platforms': {
                    'M1A/M1B': 'Level -2',
                    'M2': 'Level -3',
                    'MARMARAY': 'Level -4 (deepest)',
                },
                'exits': {
                    'A': 'North exit - Sahil Yolu',
                    'B': 'South exit - Millet Caddesi',
                    'C': 'East exit - Kennedy Caddesi',
                },
                'navigation': [
                    'From M1 to M2: Take escalator down one level, follow blue signs',
                    'From M2 to Marmaray: Take elevator/escalator down one more level',
                    'From M1 to Marmaray: Follow yellow Marmaray signs, 2-3 minutes walk',
                ]
            },
            'Mecidiyeköy': {
                'levels': 2,
                'platforms': {
                    'M2': 'Level -2',
                    'M7': 'Level -3',
                },
                'exits': {
                    'A': 'Büyükdere Caddesi exit',
                    'B': 'Shopping mall connection',
                },
                'navigation': [
                    'From M2 to M7: Exit M2 platform, follow yellow M7 signs',
                    'Take escalator down, walk through passage (approx. 100m)',
                ]
            },
            'Kabataş': {
                'levels': 'Multi-level',
                'platforms': {
                    'T1': 'Ground level',
                    'F1': 'Underground funicular station',
                    'FERRY': 'Waterfront terminal',
                },
                'exits': {
                    'Tram': 'Main tram platform',
                    'Funicular': 'Inside building, marked entrance',
                    'Ferry': 'Walk towards Bosphorus, 100m',
                },
                'navigation': [
                    'From Tram to Funicular: Enter building at tram stop, immediate access',
                    'From Tram to Ferry: Walk towards waterfront, follow ferry signs',
                ]
            },
        }
    
    def _load_visual_landmarks(self) -> Dict[str, List[str]]:
        """Load visual landmarks for navigation"""
        return {
            'Yenikapı': [
                'Large modern station with high ceilings',
                'Look for the digital arrival boards',
                'Blue mosaic artwork on M2 level',
                'Yellow Marmaray signage with modern design',
            ],
            'Mecidiyeköy': [
                'Shopping mall entrance visible',
                'Modern LED information displays',
                'Yellow M7 line has newer trains',
            ],
            'Kabataş': [
                'See the Bosphorus from tram platform',
                'Red funicular cars visible from entrance',
                'Modern glass building for F1',
            ],
            'Gayrettepe': [
                'Large interchange with purple M11 signs',
                'Airport direction clearly marked',
                'Modern station with plenty of natural light',
            ],
        }
    
    def generate_transfer_instructions(
        self,
        from_line: str,
        to_line: str,
        station: str,
        direction_on_new_line: Optional[str] = None
    ) -> TransferInstruction:
        """
        Generate detailed transfer instructions for a specific station
        
        Args:
            from_line: Current line (e.g., 'M2')
            to_line: Destination line (e.g., 'MARMARAY')
            station: Station name (e.g., 'Yenikapı')
            direction_on_new_line: Direction to take on new line (e.g., 'towards Taksim')
            
        Returns:
            Detailed transfer instruction with step-by-step guidance
        """
        
        # Get station data
        station_data = self.transfer_stations.get(station)
        if not station_data:
            return self._generate_generic_transfer(from_line, to_line, station)
        
        # Determine transfer type
        transfer_type = self._determine_transfer_type(from_line, to_line)
        
        # Get transfer time
        transfer_key = f"{from_line}_to_{to_line}"
        transfer_time = station_data.transfer_time.get(transfer_key, 5)
        
        # Generate detailed steps
        detailed_steps = self._generate_detailed_steps(
            from_line, to_line, station, station_data, direction_on_new_line
        )
        
        # Main instruction text
        instruction_text = f"Transfer at {station} from {from_line} to {to_line}"
        if direction_on_new_line:
            instruction_text += f" (towards {direction_on_new_line})"
        
        # Get visual cues
        visual_cues = self.visual_landmarks.get(station, [])
        
        # Get platform info
        layout = self.station_layouts.get(station, {})
        platform_info = None
        if 'platforms' in layout:
            platform_info = f"{from_line}: {layout['platforms'].get(from_line, 'N/A')} → {to_line}: {layout['platforms'].get(to_line, 'N/A')}"
        
        return TransferInstruction(
            step_number=0,  # Will be set by caller
            instruction_text=instruction_text,
            detailed_steps=detailed_steps,
            transfer_type=transfer_type,
            from_line=from_line,
            to_line=to_line,
            station_name=station,
            walking_distance=self._estimate_walking_distance(station, from_line, to_line),
            estimated_time=transfer_time,
            platform_info=platform_info,
            landmarks=self._get_landmarks_for_transfer(station, from_line, to_line),
            accessibility_info=', '.join(station_data.accessibility),
            visual_cues=visual_cues
        )
    
    def _generate_detailed_steps(
        self,
        from_line: str,
        to_line: str,
        station: str,
        station_data: StationTransferData,
        direction: Optional[str]
    ) -> List[str]:
        """Generate detailed step-by-step transfer instructions"""
        
        steps = []
        
        # Step 1: Exit current train
        steps.append(f"1️⃣ Exit the {from_line} train at {station}")
        
        # Step 2: Platform navigation
        if station == 'Yenikapı':
            if from_line in ['M1A', 'M1B'] and to_line == 'M2':
                steps.append("2️⃣ Look for the blue M2 signs on the platform")
                steps.append("3️⃣ Take the escalator or stairs down one level")
                steps.append("4️⃣ Follow the blue corridor for approximately 100 meters")
                steps.append("5️⃣ You will see the M2 platform on your right")
            elif from_line in ['M1A', 'M1B'] and to_line == 'MARMARAY':
                steps.append("2️⃣ Look for yellow Marmaray signs")
                steps.append("3️⃣ Take the elevator or escalator down two levels (deepest level)")
                steps.append("4️⃣ Follow the yellow signs through the passage")
                steps.append("5️⃣ Marmaray platform will be straight ahead")
            elif from_line == 'M2' and to_line == 'MARMARAY':
                steps.append("2️⃣ Exit M2 platform and look for yellow Marmaray signs")
                steps.append("3️⃣ Take escalator down one level")
                steps.append("4️⃣ Follow the passage (approximately 50 meters)")
        
        elif station == 'Mecidiyeköy':
            if from_line == 'M2' and to_line == 'M7':
                steps.append("2️⃣ Exit M2 platform towards the center")
                steps.append("3️⃣ Look for bright yellow M7 signs")
                steps.append("4️⃣ Take escalator down one level")
                steps.append("5️⃣ Walk through the connecting passage (about 100m)")
                steps.append("6️⃣ M7 platform will be on your left")
        
        elif station == 'Gayrettepe':
            if from_line == 'M2' and to_line == 'M11':
                steps.append("2️⃣ Exit M2 platform")
                steps.append("3️⃣ Look for purple M11 airport signs - very visible")
                steps.append("4️⃣ Take the escalator/elevator following purple signs")
                steps.append("5️⃣ Short walking distance (approximately 50m)")
        
        elif station == 'Kabataş':
            if from_line == 'T1' and to_line == 'F1':
                steps.append("2️⃣ Get off tram at Kabataş (final stop)")
                steps.append("3️⃣ Look for the modern glass building immediately next to tram stop")
                steps.append("4️⃣ Enter the funicular station (marked with F1 signs)")
                steps.append("5️⃣ Very quick transfer - less than 1 minute walk")
        
        elif station == 'Üsküdar':
            if from_line == 'MARMARAY' and to_line == 'M5':
                steps.append("2️⃣ Exit Marmaray platform")
                steps.append("3️⃣ Follow green M5 signs upward")
                steps.append("4️⃣ Take escalator up to M5 level")
                steps.append("5️⃣ M5 station entrance is directly connected")
        
        # Direction step
        if direction:
            steps.append(f"✅ Board {to_line} train towards {direction}")
        else:
            steps.append(f"✅ Board {to_line} train")
        
        # Add estimated time
        transfer_time = station_data.transfer_time.get(f"{from_line}_to_{to_line}", 5)
        steps.append(f"⏱️ Total transfer time: approximately {transfer_time} minutes")
        
        return steps
    
    def _determine_transfer_type(self, from_line: str, to_line: str) -> TransferType:
        """Determine the type of transfer"""
        
        if from_line.startswith('M') and to_line.startswith('M'):
            return TransferType.METRO_TO_METRO
        elif from_line == 'MARMARAY' and to_line.startswith('M'):
            return TransferType.MARMARAY_TO_METRO
        elif from_line.startswith('M') and to_line == 'MARMARAY':
            return TransferType.METRO_TO_METRO  # Treated as metro transfer
        elif from_line.startswith('M') and to_line.startswith('T'):
            return TransferType.METRO_TO_TRAM
        elif from_line.startswith('T') and to_line.startswith('M'):
            return TransferType.TRAM_TO_METRO
        elif from_line.startswith('M') and to_line.startswith('F'):
            return TransferType.METRO_TO_FERRY
        else:
            return TransferType.SAME_STATION
    
    def _estimate_walking_distance(self, station: str, from_line: str, to_line: str) -> int:
        """Estimate walking distance for transfer in meters"""
        
        # Known distances for major stations
        distances = {
            'Yenikapı': {
                'M1A_M2': 150,
                'M1B_M2': 150,
                'M2_MARMARAY': 80,
                'M1A_MARMARAY': 120,
            },
            'Mecidiyeköy': {
                'M2_M7': 120,
            },
            'Gayrettepe': {
                'M2_M11': 70,
            },
            'Kabataş': {
                'T1_F1': 30,
                'T1_FERRY': 120,
            },
        }
        
        station_distances = distances.get(station, {})
        key = f"{from_line}_{to_line}"
        return station_distances.get(key, 100)  # Default 100m
    
    def _get_landmarks_for_transfer(self, station: str, from_line: str, to_line: str) -> List[str]:
        """Get visual landmarks to help with navigation"""
        
        landmarks = {
            'Yenikapı': ['Large digital information boards', 'Blue mosaic art', 'Modern ceiling design'],
            'Mecidiyeköy': ['Shopping mall entrance', 'Food court area', 'Yellow M7 signage'],
            'Gayrettepe': ['Purple airport signs', 'Modern architecture', 'Natural lighting'],
            'Kabataş': ['Bosphorus view', 'Glass funicular building', 'Tram turnaround'],
        }
        
        return landmarks.get(station, ['Follow station signage'])
    
    def _generate_generic_transfer(self, from_line: str, to_line: str, station: str) -> TransferInstruction:
        """Generate generic transfer instruction for stations without detailed data"""
        
        detailed_steps = [
            f"1️⃣ Exit the {from_line} train at {station}",
            f"2️⃣ Follow signs for {to_line}",
            "3️⃣ Use escalators or elevators as needed",
            f"4️⃣ Board {to_line} train",
            "⏱️ Estimated transfer time: 5-7 minutes",
        ]
        
        return TransferInstruction(
            step_number=0,
            instruction_text=f"Transfer at {station} from {from_line} to {to_line}",
            detailed_steps=detailed_steps,
            transfer_type=self._determine_transfer_type(from_line, to_line),
            from_line=from_line,
            to_line=to_line,
            station_name=station,
            walking_distance=100,
            estimated_time=5,
            platform_info="Follow station signage",
            landmarks=["Station signs and information boards"],
            accessibility_info="Elevators available",
            visual_cues=["Follow line color-coded signs"]
        )
    
    def format_transfer_instruction_for_display(self, instruction: TransferInstruction) -> str:
        """Format transfer instruction for user-friendly display (Google Maps style)"""
        
        output = []
        
        # Header with emoji and main instruction
        transfer_emoji = {
            TransferType.METRO_TO_METRO: "🚇➡️🚇",
            TransferType.MARMARAY_TO_METRO: "🚆➡️🚇",
            TransferType.METRO_TO_TRAM: "🚇➡️🚊",
            TransferType.TRAM_TO_METRO: "🚊➡️🚇",
            TransferType.METRO_TO_FERRY: "🚇➡️⛴️",
        }.get(instruction.transfer_type, "🔄")
        
        output.append(f"\n{transfer_emoji} **TRANSFER AT {instruction.station_name.upper()}**")
        output.append(f"From {instruction.from_line} to {instruction.to_line}")
        output.append(f"⏱️ {instruction.estimated_time} min • 🚶 {instruction.walking_distance}m")
        output.append("\n**Step-by-Step Instructions:**")
        
        # Detailed steps
        for step in instruction.detailed_steps:
            output.append(f"  {step}")
        
        # Platform information
        if instruction.platform_info:
            output.append(f"\n📍 **Platform Info:** {instruction.platform_info}")
        
        # Visual cues
        if instruction.visual_cues:
            output.append(f"\n👁️ **Look for:**")
            for cue in instruction.visual_cues:
                output.append(f"  • {cue}")
        
        # Accessibility
        if instruction.accessibility_info:
            output.append(f"\n♿ **Accessibility:** {instruction.accessibility_info}")
        
        return '\n'.join(output)
    
    def get_all_transfer_stations(self) -> List[str]:
        """Get list of all stations with transfer capabilities"""
        return list(self.transfer_stations.keys())
    
    def get_station_info(self, station_name: str) -> Optional[StationTransferData]:
        """Get comprehensive information about a transfer station"""
        return self.transfer_stations.get(station_name)


# Factory function
def get_transfer_instructions_generator() -> TransferInstructionsGenerator:
    """Get instance of transfer instructions generator"""
    return TransferInstructionsGenerator()


def test_transfer_instructions():
    """Test the transfer instructions generator"""
    print("🗺️ TRANSFER INSTRUCTIONS GENERATOR TEST")
    print("=" * 60)
    
    generator = TransferInstructionsGenerator()
    
    # Test various transfers
    test_transfers = [
        ('M1A', 'M2', 'Yenikapı', 'Taksim'),
        ('M2', 'MARMARAY', 'Yenikapı', 'Gebze'),
        ('M2', 'M7', 'Mecidiyeköy', 'Mahmutbey'),
        ('M2', 'M11', 'Gayrettepe', 'Istanbul Airport'),
        ('T1', 'F1', 'Kabataş', 'Taksim'),
    ]
    
    for from_line, to_line, station, direction in test_transfers:
        print(f"\n{'='*60}")
        print(f"Testing: {from_line} → {to_line} at {station}")
        print('='*60)
        
        instruction = generator.generate_transfer_instructions(
            from_line, to_line, station, direction
        )
        
        formatted = generator.format_transfer_instruction_for_display(instruction)
        print(formatted)
    
    # Test station info
    print(f"\n{'='*60}")
    print("TRANSFER STATION INFORMATION")
    print('='*60)
    
    station_info = generator.get_station_info('Yenikapı')
    if station_info:
        print(f"\n📍 **{station_info.station_name}**")
        print(f"Available lines: {', '.join(station_info.transfer_available)}")
        print(f"Layout: {station_info.platform_layout}")
        print(f"Accessibility: {', '.join(station_info.accessibility)}")
        print(f"Amenities: {', '.join(station_info.amenities)}")
    
    print(f"\n✅ Transfer instructions generator test completed!")


if __name__ == "__main__":
    test_transfer_instructions()
