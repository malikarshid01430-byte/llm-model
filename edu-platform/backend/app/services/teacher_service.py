from __future__ import annotations

from typing import Any, Dict, List


class TeacherService:
    def get_dashboard(self, user_id: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "overview": {
                "active_courses": 3,
                "submissions_pending": 18,
                "attendance_marked": 92,
                "avg_response_time": "2h",
            },
            "alerts": [
                {"title": "Assignment review due", "priority": "high"},
                {"title": "Parent meeting scheduled", "priority": "medium"},
            ],
        }

    def list_assignments(self, user_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": "assignment-1",
                "title": "Prompt Engineering Lab",
                "submissions": 24,
            },
            {"id": "assignment-2", "title": "Model Evaluation Quiz", "submissions": 17},
        ]
