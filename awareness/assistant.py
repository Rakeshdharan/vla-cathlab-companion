from typing import Callable, Iterable, Iterator, Protocol
import numpy as np
from awareness.engine import AwarenessEngine


class HasInstructions(Protocol):
    def instructions(self, obj: str) -> list[str]: ...


class Handover(str):
    """The instruction text, which also remembers the tool. A rollout needs both: the object picks the scene
    target and the success check, the text is what the policy hears."""

    tool: str

    def __new__(cls, instruction: str, tool: str) -> "Handover":
        obj = super().__new__(cls, instruction)
        obj.tool = tool
        return obj

    @property
    def instruction(self) -> str:
        return str(self)

    def rollout_params(self) -> dict[str, str]:
        return {"object": self.tool, "instruction": str(self)}


class Assistant:
    def __init__(
        self,
        engine: AwarenessEngine,
        task: HasInstructions,
        confirm: Callable[[str], bool],
        speak: Callable[[str], None] = lambda _: None,
        cooldown_s: float = 20.0,
        phrase: str = "Doctor, do you need the {tool}?",
    ):
        self.engine = engine
        self.task = task
        self.confirm = confirm
        self.speak = speak
        self.cooldown_s = cooldown_s
        self.phrase = phrase
        self._last_proposal_t: float | None = None
        self._accepted: set[str] = set()
        self._declined: set[tuple[str | None, str]] = set()

    def run(self, stream: Iterable[tuple[float, np.ndarray]]) -> Iterator[Handover]:
        for t, frame in stream:
            state = self.engine.observe(frame, t)
            tool = state.next_tool
            if tool is None or tool in self._accepted or (state.step, tool) in self._declined:
                continue
            if self._last_proposal_t is not None and t - self._last_proposal_t < self.cooldown_s:
                continue
            self._last_proposal_t = t
            question = self.phrase.format(tool=tool)
            self.speak(question)
            if self.confirm(question):
                self._accepted.add(tool)
                yield Handover(self.task.instructions(tool)[0], tool)
            else:
                self._declined.add((state.step, tool))
