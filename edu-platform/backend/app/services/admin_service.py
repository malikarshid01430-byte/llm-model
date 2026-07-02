from __future__ import annotations

from typing import Any, Dict


class AdminService:
    def get_dashboard(self, user_id: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "overview": {
                "active_users": 482,
                "courses_live": 38,
                "storage_used_gb": 182,
                "monthly_revenue": 18420,
            },
            "alerts": [{"title": "System health nominal", "severity": "info"}],
        }
