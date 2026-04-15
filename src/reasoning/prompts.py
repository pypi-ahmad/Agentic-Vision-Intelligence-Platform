"""Prompt templates for various reasoning tasks."""

SYSTEM_ANALYST = (
    "You are a professional visual scene analyst. Provide precise, structured, "
    "factual observations. Highlight anything notable or unusual."
)

SYSTEM_REPORT = (
    "You are a professional report writer for a visual operations monitoring system. "
    "Generate clear, structured reports with timestamps and source references."
)

SYSTEM_QA = (
    "You are a helpful assistant for a visual intelligence system. Answer questions "
    "accurately and concisely based on available data. Do not fabricate information."
)

SCENE_DESCRIPTION = """Analyze the current scene from CV detection data and the provided image.

**Detection Summary:** {detection_summary}
**Object Counts:** {object_counts}
**Tracked Objects:** {tracked_objects}

Provide:
1. A clear description of the scene
2. Notable objects or activities
3. Spatial layout observations
4. Anything unusual or noteworthy"""

EVENT_SUMMARY = """Summarize the following events from a monitoring window:

**Events:** {events_text}
**Scene State:** {scene_state}

Provide:
1. Summary of what happened
2. Patterns or trends
3. Notable incidents
4. Overall assessment"""

QUESTION_ANSWER = """Answer the user's question using the scene context.

**Scene:** {scene_description}
**Recent Events:** {recent_events}
**Detections:** {detection_data}

**Question:** {question}

Provide a clear, accurate answer."""

INCIDENT_REPORT = """Generate a monitoring report.

**Session:** {session_id} | **Duration:** {duration} | **Frames:** {total_frames}
**Scene:** {scene_summary}
**Events:** {events_text}
**Stats:** {object_stats}

Sections:
1. Executive Summary
2. Key Observations
3. Event Timeline Summary
4. Notable Incidents
5. Recommendations"""

ANOMALY_REASONING = """Analyze for anomalies:

**Detections:** {detection_data}
**Events:** {events_text}
**Context:** {scene_description}

Assess:
1. Anything unusual?
2. Concerning patterns?
3. Should alerts be raised?
4. Concern level (low/medium/high)?"""

ALERT_EXPLANATION = """Explain this alert:

**Type:** {alert_type} | **Severity:** {severity}
**Description:** {description}
**Context:** {scene_context}
**Recent Events:** {recent_events}

Explain:
1. What happened
2. Why it triggered an alert
3. Recommended actions"""
