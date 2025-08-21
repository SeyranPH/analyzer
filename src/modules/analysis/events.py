
from typing import Callable, Optional, Dict, Any

def make_emit(cb: Optional[Callable[[str, Dict[str, Any]], None]] = None):
    def emit(event: str, data: Dict[str, Any]):
        if cb:
            try:
                cb(event, data)
            except Exception:
                pass
    return emit
