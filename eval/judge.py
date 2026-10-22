"""
VLM Success Judge (Option A).

judge(video_path, instruction) -> Verdict

The judge reads a rollout video (or a sequence of frames) and:
  1. Determines success or failure.
  2. Identifies which stage degraded (failed_stage).
  3. Provides a natural-language explanation.

It is validated against ground-truth sim stage_flags: when a SimBackend
provides stage_flags, agreement() measures judge vs ground-truth.

Implementation:
  - Primary: Gemini / GPT-4o vision API (lazy import, requires GOOGLE_API_KEY or OPENAI_API_KEY).
  - Fallback: DummyJudge that reads from EpisodeResult.success directly (used in tests).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Verdict:
    """Output of the VLM judge for one rollout."""

    success: bool
    failed_stage: Optional[str]  # one of STAGES or None if success
    explanation: str
    model: str = "unknown"
    confidence: float = 1.0  # [0, 1]; higher = more certain

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "failed_stage": self.failed_stage,
            "explanation": self.explanation,
            "model": self.model,
            "confidence": self.confidence,
        }


class DummyJudge:
    """Test-safe judge: echoes ground-truth success from the EpisodeResult fields.

    Used when no VLM API key is available (CI, offline tests).
    """

    model_name = "dummy"

    def judge(
        self,
        video_path: str | Path | None,
        instruction: str,
        ground_truth_success: bool | None = None,
        ground_truth_failed_stage: str | None = None,
    ) -> Verdict:
        """Return a Verdict based on ground-truth (no VLM call)."""
        success = ground_truth_success if ground_truth_success is not None else True
        return Verdict(
            success=success,
            failed_stage=ground_truth_failed_stage if not success else None,
            explanation=(
                "Ground-truth success." if success
                else f"Ground-truth failure at stage '{ground_truth_failed_stage}'."
            ),
            model=self.model_name,
            confidence=1.0,
        )


class GeminiJudge:
    """VLM judge using Google Gemini vision API.

    Requires: GOOGLE_API_KEY in environment.
    Falls back to DummyJudge if the key is missing.
    """

    model_name = "gemini-1.5-flash"

    _PROMPT_TEMPLATE = """\
You are evaluating a robot manipulation task: "{instruction}"

The robot (Franka FR3 Duo bimanual) must pick the named object from a cluttered table
and place it on a handover table. The task has 5 ordered stages:
  1. reach  – end-effector approaches the object
  2. grasp  – gripper closes on the object
  3. lift   – object is lifted above the table
  4. transport – object moves over the handover table
  5. place  – object is placed and gripper opens

Watch the rollout video / images and answer:
1. Did the robot SUCCEED (object placed on handover table, gripper open)? YES or NO.
2. If NO, at which stage did it FAIL? (reach / grasp / lift / transport / place)
3. Brief explanation (1–2 sentences) of what went wrong.

Respond in this exact JSON format:
{{"success": true/false, "failed_stage": "stage_name_or_null", "explanation": "..."}}
"""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                self._client = genai.GenerativeModel(self.model_name)
            except (ImportError, KeyError):
                return None
        return self._client

    def _encode_video_frames(self, video_path: str | Path, n_frames: int = 8) -> list[dict]:
        """Extract n_frames evenly spaced frames from video as base64 JPEG."""
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = [int(i * total / n_frames) for i in range(n_frames)]
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame)
                    frames.append({
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buf.tobytes()).decode(),
                    })
            cap.release()
            return frames
        except Exception:
            return []

    def judge(
        self,
        video_path: str | Path | None,
        instruction: str,
        ground_truth_success: bool | None = None,
        ground_truth_failed_stage: str | None = None,
    ) -> Verdict:
        """Call Gemini to judge the rollout. Falls back to DummyJudge on error."""
        import json

        client = self._get_client()
        if client is None:
            return DummyJudge().judge(
                video_path, instruction, ground_truth_success, ground_truth_failed_stage
            )

        prompt = self._PROMPT_TEMPLATE.format(instruction=instruction)
        parts: list = [prompt]

        if video_path is not None:
            frames = self._encode_video_frames(video_path)
            for f in frames:
                parts.append({"inline_data": f})

        try:
            response = client.generate_content(parts)
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return Verdict(
                success=bool(data.get("success", False)),
                failed_stage=data.get("failed_stage") or None,
                explanation=data.get("explanation", ""),
                model=self.model_name,
                confidence=0.9,
            )
        except Exception as exc:
            return Verdict(
                success=False,
                failed_stage=None,
                explanation=f"Judge error: {exc}",
                model=self.model_name,
                confidence=0.0,
            )


def agreement(verdicts: list[Verdict], ground_truths: list[bool]) -> float:
    """Compute judge–ground-truth agreement rate.

    Args:
        verdicts: List of Verdict from the judge.
        ground_truths: Corresponding ground-truth success booleans from SimBackend.

    Returns:
        Fraction in [0, 1].
    """
    if not verdicts:
        return 0.0
    matches = sum(v.success == gt for v, gt in zip(verdicts, ground_truths))
    return matches / len(verdicts)


def get_judge(use_vlm: bool = False) -> GeminiJudge | DummyJudge:
    """Factory: returns GeminiJudge if GOOGLE_API_KEY is set, else DummyJudge."""
    if use_vlm and os.environ.get("GOOGLE_API_KEY"):
        return GeminiJudge()
    return DummyJudge()
