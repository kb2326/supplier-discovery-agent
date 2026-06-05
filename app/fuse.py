"""Stage 3 — reciprocal rank fusion (the recall stage).

Maps to the "fusion" box in the diagram. We merge the two ranked retriever lists
into a single candidate pool using Reciprocal Rank Fusion (RRF):

    score(d) = Σ_retrievers  1 / (k + rank_r(d)),    k = 60

RRF is the right tool here because the structured and semantic arms produce scores
on incompatible scales (overlap counts vs cosine similarity). RRF ignores the raw
scores and fuses on *rank position*, so neither arm can dominate by scale alone.
The fused score is also reused downstream as the normalized "product_match" MCDA
criterion, so a high-recall match flows through to ranking.
"""
from __future__ import annotations

from .schemas import Candidate, Supplier

RRF_K = 60


def reciprocal_rank_fusion(
    suppliers: list[Supplier],
    structured: list[tuple[str, float]],
    semantic: list[tuple[str, float]],
) -> list[Candidate]:
    """Fuse two ranked lists into a single ranked candidate pool."""
    by_id = {s.id: s for s in suppliers}

    structured_rank = {sid: i + 1 for i, (sid, _) in enumerate(structured)}
    semantic_rank = {sid: i + 1 for i, (sid, _) in enumerate(semantic)}

    fused: dict[str, float] = {}
    for sid, rank in structured_rank.items():
        fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank)
    for sid, rank in semantic_rank.items():
        fused[sid] = fused.get(sid, 0.0) + 1.0 / (RRF_K + rank)

    if not fused:
        return []

    # Normalize fused scores to 0-1 so they can feed the MCDA "product_match" criterion.
    max_score = max(fused.values())

    candidates: list[Candidate] = []
    for sid, score in fused.items():
        candidates.append(
            Candidate(
                supplier=by_id[sid],
                structured_rank=structured_rank.get(sid),
                semantic_rank=semantic_rank.get(sid),
                fused_score=score,
                product_match=score / max_score if max_score else 0.0,
            )
        )

    candidates.sort(key=lambda c: c.fused_score, reverse=True)
    return candidates
