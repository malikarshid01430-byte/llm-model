from __future__ import annotations

from typing import Any, Dict, List


class StudentService:
    def get_dashboard(self, user_id: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "overview": {
                "courses_enrolled": 4,
                "assignments_due": 2,
                "attendance_rate": 96,
                "average_grade": 91,
            },
            "recent_activity": [
                {"title": "Completed AI Ethics lesson", "status": "done"},
                {"title": "Reviewed feedback for homework", "status": "pending"},
            ],
        }

    def list_courses(self, user_id: str) -> List[Dict[str, Any]]:
        return [
            {"id": "course-1", "title": "AI for Education", "progress": 82},
            {"id": "course-2", "title": "Applied Learning Science", "progress": 67},
        ]
