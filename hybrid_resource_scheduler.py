#!/usr/bin/env python3
"""
Hybrid GPU/CPU Resource Manager for Istanbul AI
Intelligent scheduling between NVIDIA T4 (16h), C3 VM (8h), and CPU fallback
"""

import torch
import logging
from datetime import datetime, time
from typing import Dict, Any, Optional, Literal
from enum import Enum
import psutil
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """Available compute backends"""
    T4_GPU = "nvidia_t4"
    C3_VM = "gcp_c3"
    CPU_FALLBACK = "cpu_fallback"

@dataclass
class BackendStatus:
    """Backend availability and performance status"""
    backend: BackendType
    available: bool
    latency_ms: float
    utilization_percent: float
    memory_available_gb: float
    last_check: datetime

class HybridResourceScheduler:
    """
    Intelligent backend selection for Istanbul AI
    
    Schedule:
    - T4 GPU: 06:00-22:00 (16 hours) - Peak performance
    - C3 VM:  22:00-06:00 (8 hours)  - Extended coverage  
    - CPU:    Always available        - Emergency fallback
    
    Features:
    - Automatic backend switching
    - Health monitoring
    - Graceful degradation
    - Performance tracking
    """
    
    def __init__(self):
        """Initialize hybrid scheduler"""
        # Schedule configuration
        self.gpu_start_hour = 6  # 06:00
        self.gpu_end_hour = 22   # 22:00
        self.c3_start_hour = 22  # 22:00
        self.c3_end_hour = 6     # 06:00
        
        # Backend status tracking
        self.backend_status: Dict[BackendType, BackendStatus] = {}
        
        # Performance counters
        self.request_count = {backend: 0 for backend in BackendType}
        self.error_count = {backend: 0 for backend in BackendType}
        
        # Initialize backends
        self._initialize_backends()
        
        logger.info("🚀 Hybrid Resource Scheduler initialized")
        logger.info(f"   GPU Hours: {self.gpu_start_hour}:00-{self.gpu_end_hour}:00")
        logger.info(f"   C3 Hours: {self.c3_start_hour}:00-{self.c3_end_hour}:00")
        logger.info(f"   CPU: Always available")
    
    def _initialize_backends(self):
        """Check and initialize all backends"""
        # Check T4 GPU
        gpu_available = self._check_gpu_available()
        self.backend_status[BackendType.T4_GPU] = BackendStatus(
            backend=BackendType.T4_GPU,
            available=gpu_available,
            latency_ms=0.0,
            utilization_percent=0.0,
            memory_available_gb=0.0,
            last_check=datetime.now()
        )
        
        # Check C3 VM (assumes we're running on it if not GPU)
        c3_available = not gpu_available and self._check_c3_available()
        self.backend_status[BackendType.C3_VM] = BackendStatus(
            backend=BackendType.C3_VM,
            available=c3_available,
            latency_ms=0.0,
            utilization_percent=0.0,
            memory_available_gb=0.0,
            last_check=datetime.now()
        )
        
        # CPU is always available
        self.backend_status[BackendType.CPU_FALLBACK] = BackendStatus(
            backend=BackendType.CPU_FALLBACK,
            available=True,
            latency_ms=0.0,
            utilization_percent=0.0,
            memory_available_gb=0.0,
            last_check=datetime.now()
        )
    
    def _check_gpu_available(self) -> bool:
        """Check if NVIDIA T4 GPU is available"""
        try:
            if not torch.cuda.is_available():
                return False
            
            # Check if we have at least one GPU
            if torch.cuda.device_count() == 0:
                return False
            
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(0)
            
            # Check if it's a T4 (or any GPU for now)
            logger.info(f"✅ GPU detected: {gpu_name}")
            
            # Check memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   GPU Memory: {memory_gb:.1f} GB")
            
            return True
            
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def _check_c3_available(self) -> bool:
        """Check if running on C3 VM"""
        try:
            # Check CPU count (C3 typically has 8+ cores)
            cpu_count = psutil.cpu_count()
            
            # Check memory (C3 typically has 32+ GB)
            memory_gb = psutil.virtual_memory().total / 1e9
            
            # Simple heuristic: if we have decent resources, assume C3
            is_c3 = cpu_count >= 8 and memory_gb >= 30
            
            if is_c3:
                logger.info(f"✅ C3 VM detected: {cpu_count} CPUs, {memory_gb:.1f} GB RAM")
            
            return is_c3
            
        except Exception as e:
            logger.warning(f"C3 check failed: {e}")
            return False
    
    def get_optimal_backend(self, force: Optional[BackendType] = None) -> BackendType:
        """
        Select optimal backend based on time and availability
        
        Args:
            force: Force specific backend (for testing)
            
        Returns:
            BackendType to use for processing
        """
        # Force specific backend if requested
        if force and self.backend_status[force].available:
            return force
        
        current_hour = datetime.now().hour
        
        # GPU hours (06:00-22:00)
        if self.gpu_start_hour <= current_hour < self.gpu_end_hour:
            if self._is_backend_healthy(BackendType.T4_GPU):
                return BackendType.T4_GPU
            
            logger.warning("⚠️ GPU scheduled but unavailable, falling back")
            return self._get_fallback_backend()
        
        # C3 hours (22:00-06:00)
        elif self._is_c3_hours(current_hour):
            if self._is_backend_healthy(BackendType.C3_VM):
                return BackendType.C3_VM
            
            logger.warning("⚠️ C3 VM scheduled but unavailable, falling back")
            return self._get_fallback_backend()
        
        # Shouldn't reach here, but fallback to CPU
        return BackendType.CPU_FALLBACK
    
    def _is_c3_hours(self, hour: int) -> bool:
        """Check if current hour is in C3 schedule (22:00-06:00)"""
        if self.c3_start_hour > self.c3_end_hour:
            # Wraps around midnight
            return hour >= self.c3_start_hour or hour < self.c3_end_hour
        else:
            return self.c3_start_hour <= hour < self.c3_end_hour
    
    def _is_backend_healthy(self, backend: BackendType) -> bool:
        """Check if backend is healthy and available"""
        status = self.backend_status.get(backend)
        
        if not status or not status.available:
            return False
        
        # Check error rate
        total_requests = self.request_count[backend]
        if total_requests > 100:  # Only check after 100 requests
            error_rate = self.error_count[backend] / total_requests
            if error_rate > 0.1:  # 10% error threshold
                logger.warning(f"⚠️ {backend.value} has high error rate: {error_rate:.1%}")
                return False
        
        # Check utilization (don't use if overloaded)
        if status.utilization_percent > 95:
            logger.warning(f"⚠️ {backend.value} overloaded: {status.utilization_percent:.1f}%")
            return False
        
        return True
    
    def _get_fallback_backend(self) -> BackendType:
        """Get best available fallback backend"""
        # Try C3 VM first
        if self._is_backend_healthy(BackendType.C3_VM):
            return BackendType.C3_VM
        
        # Always fallback to CPU
        return BackendType.CPU_FALLBACK
    
    def record_request(self, backend: BackendType, success: bool, latency_ms: float):
        """Record request for performance tracking"""
        self.request_count[backend] += 1
        
        if not success:
            self.error_count[backend] += 1
        
        # Update latency (exponential moving average)
        status = self.backend_status[backend]
        if status.latency_ms == 0:
            status.latency_ms = latency_ms
        else:
            status.latency_ms = 0.9 * status.latency_ms + 0.1 * latency_ms
    
    def update_backend_status(self):
        """Update status of all backends"""
        # Update GPU status
        if self.backend_status[BackendType.T4_GPU].available:
            try:
                gpu_util = torch.cuda.utilization()
                gpu_memory_free = torch.cuda.mem_get_info()[0] / 1e9
                
                self.backend_status[BackendType.T4_GPU].utilization_percent = gpu_util
                self.backend_status[BackendType.T4_GPU].memory_available_gb = gpu_memory_free
                self.backend_status[BackendType.T4_GPU].last_check = datetime.now()
            except Exception as e:
                logger.error(f"Failed to update GPU status: {e}")
        
        # Update C3 status
        if self.backend_status[BackendType.C3_VM].available:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_available = psutil.virtual_memory().available / 1e9
                
                self.backend_status[BackendType.C3_VM].utilization_percent = cpu_percent
                self.backend_status[BackendType.C3_VM].memory_available_gb = memory_available
                self.backend_status[BackendType.C3_VM].last_check = datetime.now()
            except Exception as e:
                logger.error(f"Failed to update C3 status: {e}")
        
        # Update CPU status
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_available = psutil.virtual_memory().available / 1e9
            
            self.backend_status[BackendType.CPU_FALLBACK].utilization_percent = cpu_percent
            self.backend_status[BackendType.CPU_FALLBACK].memory_available_gb = memory_available
            self.backend_status[BackendType.CPU_FALLBACK].last_check = datetime.now()
        except Exception as e:
            logger.error(f"Failed to update CPU status: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        self.update_backend_status()
        
        stats = {
            'current_time': datetime.now().isoformat(),
            'active_backend': self.get_optimal_backend().value,
            'backends': {}
        }
        
        for backend_type, status in self.backend_status.items():
            total_requests = self.request_count[backend_type]
            error_rate = (self.error_count[backend_type] / total_requests * 100 
                         if total_requests > 0 else 0)
            
            stats['backends'][backend_type.value] = {
                'available': status.available,
                'latency_ms': round(status.latency_ms, 2),
                'utilization_percent': round(status.utilization_percent, 1),
                'memory_available_gb': round(status.memory_available_gb, 1),
                'total_requests': total_requests,
                'error_rate': round(error_rate, 2),
                'last_check': status.last_check.isoformat()
            }
        
        return stats
    
    def get_status_report(self) -> str:
        """Get human-readable status report"""
        stats = self.get_statistics()
        
        report = ["=" * 60]
        report.append("🖥️  HYBRID RESOURCE SCHEDULER STATUS")
        report.append("=" * 60)
        report.append(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 Active Backend: {stats['active_backend'].upper()}")
        report.append("")
        
        for backend_name, backend_stats in stats['backends'].items():
            status_icon = "✅" if backend_stats['available'] else "❌"
            report.append(f"{status_icon} {backend_name.upper()}")
            report.append(f"   Available: {backend_stats['available']}")
            report.append(f"   Latency: {backend_stats['latency_ms']}ms")
            report.append(f"   Utilization: {backend_stats['utilization_percent']}%")
            report.append(f"   Memory Available: {backend_stats['memory_available_gb']}GB")
            report.append(f"   Requests: {backend_stats['total_requests']} (errors: {backend_stats['error_rate']}%)")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# Singleton instance
_scheduler_instance: Optional[HybridResourceScheduler] = None

def get_scheduler() -> HybridResourceScheduler:
    """Get global scheduler instance"""
    global _scheduler_instance
    
    if _scheduler_instance is None:
        _scheduler_instance = HybridResourceScheduler()
    
    return _scheduler_instance


if __name__ == "__main__":
    # Test the scheduler
    logging.basicConfig(level=logging.INFO)
    
    scheduler = HybridResourceScheduler()
    
    print("\n" + scheduler.get_status_report())
    
    # Test backend selection
    print("\n🧪 Testing backend selection:")
    for hour in [7, 12, 18, 23, 2]:
        print(f"   Hour {hour:02d}:00 → {scheduler.get_optimal_backend().value}")
    
    # Simulate some requests
    print("\n🧪 Simulating requests...")
    backend = scheduler.get_optimal_backend()
    for i in range(10):
        import random
        success = random.random() > 0.1
        latency = random.uniform(20, 100)
        scheduler.record_request(backend, success, latency)
    
    print("\n" + scheduler.get_status_report())
