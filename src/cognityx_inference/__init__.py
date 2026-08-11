"""Cognityx inference platform and hardware-boundary evaluation."""

from cognityx_inference.capabilities import (
    CertifiedContextProfile,
    CertifiedContextLimitExceeded,
    HardwareDiscoveryRequired,
    ModelContextLimitExceeded,
)
from cognityx_inference.client import CognityxInferenceClient
from cognityx_inference.chat import ChatSettings, CognityxChatSession
from cognityx_inference.contracts import (
    AdapterPurpose,
    DiscoveryPolicy,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    InferenceTimings,
    LoadPolicy,
    ModelCapabilities,
    TokenDetail,
    TokenBudget,
    TokenUsage,
    ThinkingMode,
    ThinkingResolution,
)
from cognityx_inference.research import InferencePairRequest, ResearchContext
from cognityx_inference.lifecycle import (
    ModelIdentity,
    ModelLease,
    ModelManager,
    ModelState,
    ResidentModelStatus,
)
from cognityx_inference.manager import InferenceManager, ServerState, ServerStatus
from cognityx_inference.errors import ContextWindowExceeded
from cognityx_inference.service import InferenceService

InferenceClient = CognityxInferenceClient

__all__ = [
    "AdapterPurpose",
    "FinishReason",
    "CognityxInferenceClient",
    "InferenceClient",
    "CognityxChatSession",
    "ChatSettings",
    "CertifiedContextLimitExceeded",
    "CertifiedContextProfile",
    "ContextWindowExceeded",
    "DiscoveryPolicy",
    "InferenceRequest",
    "InferenceResponse",
    "InferencePairRequest",
    "InferenceService",
    "InferenceManager",
    "InferenceTimings",
    "LoadPolicy",
    "HardwareDiscoveryRequired",
    "ModelContextLimitExceeded",
    "ModelCapabilities",
    "ModelIdentity",
    "ModelLease",
    "ModelManager",
    "ModelState",
    "ResidentModelStatus",
    "ResearchContext",
    "ServerState",
    "ServerStatus",
    "TokenBudget",
    "TokenDetail",
    "TokenUsage",
    "ThinkingMode",
    "ThinkingResolution",
]
