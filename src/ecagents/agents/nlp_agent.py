"""NLP Interpretation Agent: natural-language request -> validated TaskIntent -> TaskSpec(s)."""
from __future__ import annotations

import re

from ..contracts import Priority, TaskIntent, TaskSpec, TaskType
from .llm import LLMClient

# Giga-cycles of CPU work per MB of input, by task type (profiling assumptions).
CYCLES_PER_MB = {
    TaskType.VIDEO_ANALYTICS: 1.6,
    TaskType.ML_INFERENCE: 1.2,
    TaskType.BATCH_ANALYTICS: 3.0,
    TaskType.IOT_AGGREGATION: 0.3,
    TaskType.WEB_REQUEST: 0.2,
}

SYSTEM = """You convert requests sent to an edge-cloud computing platform into a JSON task intent.
Fields:
- task_type: one of video_analytics (camera/video/frames/object detection on video),
  ml_inference (running a trained model on images, text or audio), batch_analytics
  (reports, ETL, aggregation over logs or large datasets, training), iot_aggregation
  (sensor/telemetry readings), web_request (API calls, page or small query).
- input_size_mb: data that must be uploaded, in megabytes (1 GB = 1024 MB, 1 KB = 1/1024 MB).
- deadline_ms: the latest acceptable completion time in milliseconds. "real-time" with no number
  means 100 ms; "within a minute" means 60000; if no time constraint is given use 3600000.
- priority: high if urgent, safety-critical or explicitly high; low if it says whenever, no rush,
  background or low priority; otherwise normal.
- preferred_location: edge if the user requires data to stay local/on-premise/near the device,
  cloud if they ask for the cloud; otherwise any.
- count: how many independent tasks of this kind (default 1).
- max_cost_usd: only if the user states a budget, else null.
Return only the JSON object."""


class NLPInterpretationAgent:
    name = "nlp-llm"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self.fallbacks = 0

    def interpret(self, request: str) -> TaskIntent:
        intent = self.llm.generate_json(f"Request: {request}", TaskIntent, system=SYSTEM)
        if intent is None:
            self.fallbacks += 1
            return RuleBasedParser().interpret(request)
        return intent


class RuleBasedParser:
    """Keyword/regex baseline (the non-agentic alternative to the NLP agent)."""

    name = "nlp-rules"
    KEYWORDS = [
        (TaskType.VIDEO_ANALYTICS, r"video|camera|cctv|frame|footage|stream"),
        (TaskType.ML_INFERENCE, r"infer|classif|detect|model|recogn|predict|transcrib|translat"),
        (TaskType.BATCH_ANALYTICS, r"report|batch|etl|aggregate logs|training|train|dataset|analytics"),
        (TaskType.IOT_AGGREGATION, r"sensor|telemetry|iot|reading|temperature|meter"),
        (TaskType.WEB_REQUEST, r"api|web|page|query|request|lookup"),
    ]

    def interpret(self, request: str) -> TaskIntent:
        text = request.lower()
        ttype = TaskType.WEB_REQUEST
        for t, pattern in self.KEYWORDS:
            if re.search(pattern, text):
                ttype = t
                break
        size = 1.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|mb|kb)", text)
        if m:
            size = float(m.group(1)) * {"gb": 1024, "mb": 1, "kb": 1 / 1024}[m.group(2)]
        deadline = 3_600_000.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*(ms|millisecond|s\b|sec|second|min|minute|h\b|hour)", text)
        if m:
            unit = m.group(2)
            mult = 1 if unit.startswith("m") and unit in ("ms", "millisecond") else \
                1000 if unit.startswith("s") else 60_000 if unit.startswith("min") else 3_600_000
            deadline = float(m.group(1)) * mult
        elif "real-time" in text or "real time" in text:
            deadline = 100.0
        priority = Priority.NORMAL
        if re.search(r"urgent|critical|asap|high priority|emergency", text):
            priority = Priority.HIGH
        elif re.search(r"no rush|whenever|background|low priority", text):
            priority = Priority.LOW
        loc = "any"
        if re.search(r"on-prem|on prem|stay local|locally|must not leave|edge", text):
            loc = "edge"
        elif "cloud" in text:
            loc = "cloud"
        return TaskIntent(task_type=ttype, input_size_mb=max(size, 1e-3), deadline_ms=deadline,
                          priority=priority, preferred_location=loc)


def intent_to_tasks(intent: TaskIntent, first_id: int, arrival_s: float, source_edge: int) -> list[TaskSpec]:
    work = max(CYCLES_PER_MB[intent.task_type] * intent.input_size_mb, 0.05)
    return [TaskSpec(task_id=first_id + k, arrival_s=arrival_s, work_gcycles=work,
                     input_mb=intent.input_size_mb, output_mb=intent.input_size_mb * 0.1,
                     deadline_s=intent.deadline_ms / 1000, source_edge=source_edge,
                     task_type=intent.task_type, priority=intent.priority,
                     preferred_location=intent.preferred_location)
            for k in range(intent.count)]
