#!/usr/bin/env python3
"""
Trace how the full route is being built to find the stop count issue.
"""
import sys
sys.path.insert(0, '/Users/omer/Desktop/ai-stanbul/backend')

from services.transportation_rag_system import get_transportation_rag
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize RAG
rag = get_transportation_rag()

# Patch the _build_route_from_path_weighted to add detailed logging
original_build = rag._build_route_from_path_weighted

def traced_build(path, lines_used, transfers, total_time, confidences):
    print("\n" + "="*80)
    print("TRACING _build_route_from_path_weighted")
    print("="*80)
    print(f"Path ({len(path)} stations):")
    for i, station_id in enumerate(path):
        station = rag.stations[station_id]
        print(f"  {i}: {station.name} ({station.line})")
    
    print(f"\nLines used: {lines_used}")
    print(f"Transfers: {transfers}")
    print(f"Total time: {total_time}")
    
    # Call original with detailed step-by-step logging
    if not path or len(path) < 2:
        return None
    
    steps = []
    current_line = rag.stations[path[0]].line
    segment_start = 0
    segment_time = 0.0
    overall_confidences = []
    
    print(f"\nStarting segment 0: line={current_line}, segment_start={segment_start}")
    
    # Build segments by detecting line changes
    for i in range(1, len(path)):
        station_id = path[i]
        station = rag.stations[station_id]
        prev_station_id = path[i-1]
        
        print(f"\n--- Loop i={i}: {station.name} ({station.line}) ---")
        
        # Get travel time for this hop
        if i < len(confidences) + 1:
            travel_time, confidence = rag.travel_time_db.get_travel_time(
                prev_station_id,
                station_id
            )
            overall_confidences.append(confidence)
        else:
            travel_time = 2.5  # default
            confidence = "low"
        
        print(f"  Travel time from {rag.stations[prev_station_id].name}: {travel_time} min")
        
        # Line change = transfer
        if station.line != current_line:
            print(f"  ⚡ TRANSFER DETECTED: {current_line} → {station.line}")
            
            # Create transit step for previous segment
            start_station = rag.stations[path[segment_start]]
            end_station = rag.stations[path[i-1]]
            
            stops_count = i - segment_start - 1
            print(f"  Creating transit step:")
            print(f"    From: {start_station.name} (index {segment_start})")
            print(f"    To: {end_station.name} (index {i-1})")
            print(f"    Stops calculation: {i} - {segment_start} - 1 = {stops_count}")
            print(f"    Duration: {segment_time}")
            
            steps.append({
                "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
                "line": current_line,
                "from": start_station.name,
                "to": end_station.name,
                "duration": round(segment_time, 1),
                "type": "transit",
                "stops": stops_count
            })
            
            # Add transfer step
            transfer_penalty = rag.travel_time_db.get_transfer_penalty(
                current_line,
                station.line
            )
            
            print(f"  Creating transfer step:")
            print(f"    At: {end_station.name}")
            print(f"    Duration: {transfer_penalty} min")
            
            steps.append({
                "instruction": f"Transfer to {station.line} at {end_station.name}",
                "line": station.line,
                "from": end_station.name,
                "to": end_station.name,
                "duration": round(transfer_penalty, 1),
                "type": "transfer"
            })
            
            # Start new segment
            print(f"  Starting new segment: segment_start = {i-1} (was {segment_start})")
            current_line = station.line
            segment_start = i - 1
            segment_time = 0.0
        else:
            # Continue on same line
            segment_time += travel_time
            print(f"  Continuing on {current_line}, segment_time now {segment_time}")
    
    # Final segment
    print(f"\n--- FINAL SEGMENT ---")
    start_station = rag.stations[path[segment_start]]
    end_station = rag.stations[path[-1]]
    stops_count = len(path) - segment_start - 1
    
    print(f"  From: {start_station.name} (index {segment_start})")
    print(f"  To: {end_station.name} (index {len(path)-1})")
    print(f"  Stops calculation: {len(path)} - {segment_start} - 1 = {stops_count}")
    print(f"  Duration: {segment_time}")
    
    steps.append({
        "instruction": f"Take {current_line} from {start_station.name} to {end_station.name}",
        "line": current_line,
        "from": start_station.name,
        "to": end_station.name,
        "duration": round(segment_time, 1),
        "type": "transit",
        "stops": stops_count
    })
    
    print("\n" + "="*80)
    print("FINAL STEPS:")
    print("="*80)
    for idx, step in enumerate(steps):
        print(f"{idx+1}. {step['instruction']}")
        if step['type'] == 'transit':
            print(f"   Stops: {step['stops']}, Duration: {step['duration']} min")
        else:
            print(f"   Duration: {step['duration']} min")
    
    # Return the actual route object
    return original_build(path, lines_used, transfers, total_time, confidences)

rag._build_route_from_path_weighted = traced_build

# Clear cache
print("Clearing Redis cache...")
if rag.redis:
    rag.redis.flushdb()

# Test the full route
print("\n" + "="*80)
print("TESTING FULL ROUTE: Sultanahmet → Kadıköy")
print("="*80)

route = rag.find_route("Sultanahmet", "Kadıköy")

if route:
    print("\n" + "="*80)
    print("FINAL ROUTE OUTPUT:")
    print("="*80)
    print(f"Total time: {route.total_time} min")
    print(f"Transfers: {route.transfers}")
    print(f"\nSteps:")
    for i, step in enumerate(route.steps, 1):
        print(f"{i}. {step['instruction']}")
        if 'stops' in step:
            print(f"   Stops: {step['stops']}")
else:
    print("\n❌ No route found!")
