import time
from typing import Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from api.setting import DEFAULT_MODEL


class Model(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    object: str = "model"
    owned_by: str = "bedrock"
    name: Optional[str] = None
    description: Optional[str] = None


class Models(BaseModel):
    object: str = "list"
    data: List[Model] = []


class ResponseFunction(BaseModel):
    name: Optional[str] = None
    arguments: str


class ToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: ResponseFunction


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image"
    image_url: ImageUrl


class SystemMessage(BaseModel):
    name: Optional[str] = None
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    name: Optional[str] = None
    role: Literal["user"] = "user"
    content: Union[str, List[Union[TextContent, ImageContent]]]


class AssistantMessage(BaseModel):
    name: Optional[str] = None
    role: Literal["assistant"] = "assistant"
    content: Optional[Union[str, List[Union[TextContent, ImageContent]]]] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: object


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class StreamOptions(BaseModel):
    include_usage: bool = True


class ChatRequest(BaseModel):
    messages: List[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]]
    model: str = DEFAULT_MODEL
    frequency_penalty: Optional[float] = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    presence_penalty: Optional[float] = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(default=1.0, le=2.0, ge=0.0)
    top_p: Optional[float] = Field(default=1.0, le=1.0, ge=0.0)
    user: Optional[str] = None  # Not used
    max_tokens: Optional[int] = 2048
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    n: Optional[int] = 1  # Not used
    tools: Optional[List[Tool]] = None
    tool_choice: Union[str, object] = "auto"
    stop: Optional[Union[List[str], str]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseMessage(BaseModel):
    # tool_calls
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None


class BaseChoice(BaseModel):
    index: Optional[int] = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[dict] = None


class Choice(BaseChoice):
    message: ChatResponseMessage


class ChoiceDelta(BaseChoice):
    delta: ChatResponseMessage


class BaseChatResponse(BaseModel):
    # id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid.uuid4())[:8])
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = "fp"


class ChatResponse(BaseChatResponse):
    choices: List[Choice]
    object: Literal["chat.completion"] = "chat.completion"
    usage: Usage


class ChatStreamResponse(BaseChatResponse):
    choices: List[ChoiceDelta]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    usage: Optional[Usage] = None


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str], Iterable[Union[int, Iterable[int]]]]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None  # not used.
    user: Optional[str] = None  # not used.


class Embedding(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[List[float], bytes]
    index: int


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[Embedding]
    model: str
    usage: EmbeddingsUsage


class ErrorMessage(BaseModel):
    message: str


class Error(BaseModel):
    error: ErrorMessage
