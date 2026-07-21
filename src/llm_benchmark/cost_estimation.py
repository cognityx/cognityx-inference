from __future__ import annotations

import math
from dataclasses import asdict, dataclass

COST_DISCLAIMER = (
    "Hypothetical hosted API-equivalent estimate; not measured local operating cost."
)


@dataclass
class CostConfig:
    enabled: bool = False
    pricing_label: str | None = None
    input_price_per_million_tokens: float | None = None
    output_price_per_million_tokens: float | None = None
    currency: str = "USD"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_cost_model(
    label: str,
    input_price: float,
    output_price: float,
    currency: str = "USD",
) -> CostConfig:
    for name, value in (("input price", input_price), ("output price", output_price)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    return CostConfig(True, label, input_price, output_price, currency)


def estimate_api_equivalent_cost(
    config: CostConfig,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, object]:
    if not config.enabled:
        return {
            **config.to_dict(),
            "basis": "disabled",
            "disclaimer": COST_DISCLAIMER,
        }

    input_cost = prompt_tokens * float(config.input_price_per_million_tokens) / 1_000_000
    output_cost = output_tokens * float(config.output_price_per_million_tokens) / 1_000_000
    return {
        **config.to_dict(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "estimated_input_cost": round(input_cost, 12),
        "estimated_output_cost": round(output_cost, 12),
        "estimated_total_cost": round(input_cost + output_cost, 12),
        "basis": "user_supplied_token_prices",
        "disclaimer": COST_DISCLAIMER,
    }
