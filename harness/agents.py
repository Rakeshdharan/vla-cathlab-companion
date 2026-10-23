"""Pluggable coding-agent CLIs (configs/agents.yaml)."""
from pathlib import Path
import yaml
from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    argv: list[str]           # placeholders: {workdir} {prompt} {prompt_file}
    description: str = ""


class AgentsConfig(BaseModel):
    default: str
    agents: dict[str, AgentConfig]

    def get(self, name: str | None) -> AgentConfig:
        key = name or self.default
        if key not in self.agents:
            raise KeyError(f"unknown agent '{key}'; configured: {sorted(self.agents)}")
        return self.agents[key]


def load_agents(path: str | Path) -> AgentsConfig:
    raw = yaml.safe_load(Path(path).read_text())
    agents = {k: AgentConfig(name=k, **v) for k, v in raw["agents"].items()}
    return AgentsConfig(default=raw["default"], agents=agents)


def build_argv(agent: AgentConfig, workdir: Path, prompt: str, prompt_file: Path) -> list[str]:
    values = {"workdir": str(workdir), "prompt": prompt, "prompt_file": str(prompt_file)}
    return [a.format(**values) for a in agent.argv]
