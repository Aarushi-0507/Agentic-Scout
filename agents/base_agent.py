"""
Base Agent Interface for AGENTIC SCOUT specialized autonomous agents.
Provides unified state management, lifecycle tracking, and structured logging.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
from core.state import AnalysisState


class BaseAgent(ABC):
    """Abstract base class defining the agent execution protocol."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = "IDLE"
        self.duration_ms = 0.0

    def run(self, state: AnalysisState, *args, **kwargs) -> Dict[str, Any]:
        """Executes the agent task with full lifecycle instrumentation."""
        self.status = "RUNNING"
        state.log_agent_action(self.name, "RUNNING", f"Starting {self.role}...")
        start_time = time.time()
        
        try:
            result = self.execute(state, *args, **kwargs)
            self.duration_ms = (time.time() - start_time) * 1000.0
            self.status = "SUCCESS"
            state.log_agent_action(
                self.name,
                "SUCCESS",
                f"Successfully completed {self.role}.",
                details={"summary": result.get("summary", "")},
                duration_ms=self.duration_ms
            )
            return result
        except Exception as e:
            self.duration_ms = (time.time() - start_time) * 1000.0
            self.status = "FAILED"
            state.log_agent_action(
                self.name,
                "FAILED",
                f"Failed during {self.role}: {str(e)}",
                details={"error": str(e)},
                duration_ms=self.duration_ms
            )
            raise e

    @abstractmethod
    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        """Core domain task logic implemented by specialized agent."""
        pass
