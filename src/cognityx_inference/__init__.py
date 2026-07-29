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
)
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
    "ServerState",
    "ServerStatus",
    "TokenBudget",
    "TokenDetail",
    "TokenUsage",
]
