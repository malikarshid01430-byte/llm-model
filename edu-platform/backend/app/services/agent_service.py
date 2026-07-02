from __future__ import annotations

from typing import Any, Dict


class AgentService:
    def run(self, agent: str, task: str) -> Dict[str, Any]:
        if agent == "tutor":
            steps = [
                "Assess the learner's current level",
                "Create a personalized study plan",
                "Generate practice questions and feedback",
            ]
        elif agent == "research":
            steps = [
                "Gather relevant sources",
                "Summarize evidence",
                "Draft a structured research brief",
            ]
        else:
            steps = [
                "Break the task into milestones",
                "Execute the plan",
                "Review and improve the result",
            ]

        return {"agent": agent, "task": task, "steps": steps}
