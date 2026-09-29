import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
)

from agent.llm_client import ToolSpec
from models.agent_events import (
    GuidelineSources,
    LookupChunks,
    RagSources,
    ToolResult,
    ToolSubstep,
)
from pydantic import BaseModel, TypeAdapter, ValidationError
from services.dataset_configuration import ConfigError
from utils.logger import get_logger

LOGGER = get_logger(__name__)

AGENT_TOOLS_FILE = os.getenv("AGENT_TOOLS_FILE", "agent_tools.json")
AGENT_MAX_CALLS_PER_TOOL = int(os.getenv("AGENT_MAX_CALLS_PER_TOOL", "3"))


class ToolTexts(BaseModel):
    display_name: str
    description: str
    parameters: Dict[str, str] = {}


@dataclass
class NeedsUserInput:
    question: str
    options: List[str]


@dataclass
class NestedEvent:
    step_id: str
    payload: BaseModel


@dataclass
class ToolContext:
    run_id: str
    step_id: str
    dataset: str
    language: str
    new_step: Callable[[], str]


ToolEvent = Union[
    ToolSubstep,
    RagSources,
    GuidelineSources,
    LookupChunks,
    NestedEvent,
    ToolResult,
    NeedsUserInput,
]


class Tool(Protocol):
    name: str
    args_model: Type[BaseModel]
    max_calls: int

    def invoke(self, args: BaseModel, ctx: ToolContext) -> AsyncIterator[ToolEvent]: ...


def language_for_dataset(dataset_id: str) -> str:
    return dataset_id.rsplit("_", 1)[-1]


class ToolRegistry:
    def __init__(self, languages: List[str]):
        self._languages = languages
        self._tools: Dict[str, Tool] = {}
        self._specs: Dict[str, Dict[str, ToolSpec]] = {}
        self._file_texts: Dict[str, Dict[str, ToolTexts]] = self._load_texts()

    def _load_texts(self) -> Dict[str, Dict[str, ToolTexts]]:
        file_path = Path("/app/data") / AGENT_TOOLS_FILE
        if not file_path.is_file():
            raise ConfigError(f"Tool texts file does not exist: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in tool texts file {file_path}: {e}")
        try:
            return TypeAdapter(Dict[str, Dict[str, ToolTexts]]).validate_python(data)
        except ValidationError as e:
            raise ConfigError(f"Invalid tool texts file {file_path}: {e}")

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ConfigError(f"Tool '{tool.name}' is already registered.")

        expected_parameters = set(tool.args_model.model_fields)
        for language in self._languages:
            texts = self._file_texts.get(tool.name, {}).get(language)
            if texts is None:
                raise ConfigError(
                    f"Missing tool texts for '{tool.name}' in language '{language}'."
                )
            if not texts.display_name or not texts.description:
                raise ConfigError(
                    f"Empty display name or description for tool '{tool.name}' "
                    f"in language '{language}'."
                )
            text_parameters = set(texts.parameters)
            missing = sorted(expected_parameters - text_parameters)
            unexpected = sorted(text_parameters - expected_parameters)
            if missing or unexpected:
                raise ConfigError(
                    f"Parameter descriptions for tool '{tool.name}' in language "
                    f"'{language}' do not match the tool's arguments: "
                    f"missing {missing}, unexpected {unexpected}."
                )

        schema = tool.args_model.model_json_schema()
        schema.pop("title", None)
        for property_schema in schema.get("properties", {}).values():
            property_schema.pop("title", None)

        self._tools[tool.name] = tool
        self._specs[tool.name] = {}
        for language in self._languages:
            texts = self._file_texts[tool.name][language]
            language_schema = deepcopy(schema)
            for parameter, description in texts.parameters.items():
                language_schema["properties"][parameter]["description"] = description
            self._specs[tool.name][language] = ToolSpec(
                name=tool.name,
                description=texts.description,
                parameters=language_schema,
            )
        LOGGER.info(
            f"Registered agent tool <{tool.name}> for languages <{self._languages}>"
        )

    def specs(self, language: str) -> List[ToolSpec]:
        return [self._specs[name][language] for name in self._tools]

    def display_names(self, language: str) -> Dict[str, str]:
        return {
            name: self._file_texts[name][language].display_name
            for name in self._tools
        }

    def max_calls(self) -> Dict[str, int]:
        return {name: tool.max_calls for name, tool in self._tools.items()}

    async def invoke(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]],
        ctx: ToolContext,
    ) -> AsyncIterator[ToolEvent]:
        tool = self._tools.get(name)
        if tool is None:
            yield ToolResult(
                text=(
                    f"Unknown tool '{name}'. "
                    f"Available tools: {', '.join(self._tools)}."
                ),
                is_error=True,
            )
            return
        if arguments is None:
            yield ToolResult(text="Arguments must be a JSON object.", is_error=True)
            return
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as e:
            yield ToolResult(text=self._invalid_arguments_text(name, e), is_error=True)
            return
        async for event in tool.invoke(args, ctx):
            yield event

    def _invalid_arguments_text(self, name: str, error: ValidationError) -> str:
        entries = []
        for entry in error.errors(include_url=False):
            location = ".".join(str(part) for part in entry["loc"]) or "(root)"
            entries.append(f"{location}: {entry['msg']}")
        return f"Invalid arguments for tool '{name}': {'; '.join(entries)}"
