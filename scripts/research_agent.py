#!/usr/bin/env python3
"""Enhanced research agent — wide search, connection-drawing, insight generation.

Capabilities:
  1. Wide search: queries multiple sources, prioritises recent + institutional
  2. Connection-drawing: finds links between discovered ideas and existing alphas
  3. Class-level insights: derives patterns that apply across strategy classes
  4. Parameter tuning: sweeps parameters on existing strategies
  5. Composition: blends strategies from complementary classes
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.deployment.registry import StrategyRecord, StrategyRegistry

# Search queries rotated to maintain breadth
SEARCH_TEMPLATES = [
    # Momentum / trend
    "cryptocurrency momentum factor investing 2025 2026 academic paper arXiv SSRN institutional",
    "cross-sectional momentum crypto perpetual futures alpha signal 2025",
    # Mean reversion / stat arb
    "cryptocurrency mean reversion statistical arbitrage trading strategy 2025 preprint",
    "pairs trading cointegration crypto perpetuals 2025 2026",
    # Carry / funding
    "funding rate arbitrage basis trade cryptocurrency perpetual 2025 novel approach",
    "crypto carry trade strategy institutional 2026",
    # Volatility / regime
    "cryptocurrency volatility risk premium trading 2025 regime switching",
    "adaptive portfolio construction crypto regime detection 2026",
    # ML / deep learning
    "deep learning reinforcement learning cryptocurrency trading 2025 backtest results",
    "transformer attention crypto price prediction trading strategy 2025 2026",
    # Institutional / novel
    "Jane Street Two Sigma Citadel crypto trading strategy 2025",
    "top conference NeurIPS ICML crypto finance trading 2025 2026 paper",
]


def search_widely(tools: dict) -> list[dict]:
    """Search across multiple queries, return deduplicated paper list.

    Returns list of {title, url, snippet, source, year, institution} dicts.
    """
    results = []

    # Rotate through query templates
    week = datetime.now(timezone.utc).isocalendar()[1]
    queries = SEARCH_TEMPLATES[week % len(SEARCH_TEMPLATES) :]
    queries += SEARCH_TEMPLATES[: week % len(SEARCH_TEMPLATES)]
    queries = queries[:5]  # Search 5 queries per cycle

    search_fn = tools.get("web_search")
    if not search_fn:
        logger.warning("No web_search tool available")
        return []

    for query in queries:
        try:
            raw = search_fn(query)
            results.append(
                {"query": query, "results": str(raw)[:3000], "timestamp": datetime.now(timezone.utc).isoformat()}
            )
        except Exception as e:
            logger.warning(f"Search failed for '{query[:60]}...': {e}")

    return results


def draw_connections(
    research_findings: list[dict],
    registry: StrategyRegistry,
) -> list[dict]:
    """Draw connections between research findings and existing strategies.

    For each finding, asks:
    - Does this complement an existing strategy?
    - Does it work in regimes where we're weak?
    - Can it be composed with something we already have?
    """
    connections = []
    active = registry.get_active()

    for finding in research_findings:
        finding_text = str(finding.get("results", ""))[:500].lower()

        for strategy in active:
            # Simple keyword matching for connections
            relevance = 0
            keywords = strategy.hypothesis.lower().split()
            for kw in keywords[:10]:
                if kw in finding_text and len(kw) > 3:
                    relevance += 1

            if relevance >= 2:
                connections.append(
                    {
                        "strategy": strategy.name,
                        "class": strategy.class_,
                        "finding_snippet": str(finding.get("results", ""))[:200],
                        "relevance_score": relevance,
                        "connection_type": "keyword_match",
                        "suggested_action": f"Consider composing {strategy.name} with finding above",
                    }
                )

    # Also check complementary classes
    for strategy in active:
        complements = registry.find_connections(strategy)
        for comp in complements:
            connections.append(
                {
                    "strategy": strategy.name,
                    "complement": comp.name,
                    "complement_class": comp.class_,
                    "connection_type": "class_complement",
                    "suggested_action": f"Blend {strategy.name} ({strategy.class_}) with {comp.name} ({comp.class_})",
                }
            )

    return connections[:20]  # Cap at 20 connections


def generate_class_insights(registry: StrategyRegistry) -> list[dict]:
    """Derive insights that apply across entire strategy classes.

    Looks for patterns like:
    - All momentum strategies work in bull, fail in bear
    - All mean reversion strategies have Sharpe < 0 on BTC
    - Funding strategies have low drawdown but tiny returns
    """
    insights = []

    for class_ in ["momentum", "mean_reversion", "carry", "volatility"]:
        strats = registry.get_by_class(class_)
        active = [s for s in strats if s.status != "archived"]

        if len(active) < 2:
            continue

        # Aggregate performance
        sharpes = [s.sharpe_test for s in active if s.sharpe_test != 0]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
        best = max(active, key=lambda s: s.sharpe_test) if active else None

        # Find common failure regimes
        regime_scores = {}
        for s in active:
            for regime, perf in s.regime_performance.items():
                if isinstance(perf, dict):
                    if regime not in regime_scores:
                        regime_scores[regime] = []
                    regime_scores[regime].append(perf.get("sharpe", 0))

        weak_regimes = [
            r for r, scores in regime_scores.items()
            if sum(scores) / len(scores) < 0
        ]
        strong_regimes = [
            r for r, scores in regime_scores.items()
            if sum(scores) / len(scores) > 0.5
        ]

        insight = {
            "class": class_,
            "strategy_count": len(active),
            "avg_test_sharpe": round(avg_sharpe, 3),
            "best_strategy": best.name if best else "none",
            "best_sharpe": best.sharpe_test if best else 0,
            "strong_in": strong_regimes,
            "weak_in": weak_regimes,
            "generalisation": f"{class_} strategies average Sharpe {avg_sharpe:.2f}. "
            f"Best performer: {best.name if best else 'none'} ({best.sharpe_test if best else 0:.2f}). "
            f"Works in: {', '.join(strong_regimes) if strong_regimes else 'no strong regimes'}. "
            f"Fails in: {', '.join(weak_regimes) if weak_regimes else 'no weak regimes'}.",
        }
        insights.append(insight)

    # Cross-class insight: which classes complement each other?
    for class_a in ["momentum", "mean_reversion", "carry", "volatility"]:
        for class_b in ["momentum", "mean_reversion", "carry", "volatility"]:
            if class_a >= class_b:
                continue
            strats_a = registry.get_by_class(class_a)
            strats_b = registry.get_by_class(class_b)
            if not strats_a or not strats_b:
                continue

            # Check if they have complementary regime performance
            a_best = max(strats_a, key=lambda s: s.sharpe_test)
            b_best = max(strats_b, key=lambda s: s.sharpe_test)

            a_strong = set()
            b_strong = set()
            for r, v in a_best.regime_performance.items():
                if isinstance(v, dict) and v.get("sharpe", 0) > 0.5:
                    a_strong.add(r)
            for r, v in b_best.regime_performance.items():
                if isinstance(v, dict) and v.get("sharpe", 0) > 0.5:
                    b_strong.add(r)

            overlap = a_strong & b_strong
            unique_a = a_strong - b_strong
            unique_b = b_strong - a_strong

            if unique_a or unique_b:
                insights.append(
                    {
                        "class": f"{class_a}+{class_b}",
                        "type": "class_complement",
                        "strategy_count": 2,
                        "avg_test_sharpe": round((a_best.sharpe_test + b_best.sharpe_test) / 2, 3),
                        "generalisation": (
                            f"{class_a} works in {unique_a}, {class_b} works in {unique_b}. "
                            f"Blending could cover both sets. Overlap: {overlap}."
                        ),
                    }
                )

    return insights


def dream(
    research: list[dict],
    connections: list[dict],
    insights: list[dict],
    registry: StrategyRegistry,
) -> list[dict]:
    """Generate novel strategy ideas by synthesising research + connections + insights.

    'Dreaming' means:
    - Combining two successful strategies from complementary classes
    - Applying a class-level insight to generate a new variant
    - Taking a research finding and composing it with an existing alpha
    """
    dreams = []

    # Dream 1: Compose complementary strategies
    composition_pairs = [c for c in connections if c.get("connection_type") == "class_complement"]
    for pair in composition_pairs[:3]:
        dreams.append(
            {
                "type": "composition",
                "ingredients": [pair.get("strategy", ""), pair.get("complement", "")],
                "rationale": pair.get("suggested_action", ""),
                "dream": f"Blend {pair.get('strategy','')} and {pair.get('complement','')} into a composite strategy",
                "expected_benefit": "Combined regime coverage — each covers the other's weakness",
            }
        )

    # Dream 2: Apply class insight to generate new variant
    for insight in insights[:3]:
        if insight.get("type") == "class_complement":
            continue
        if insight.get("weak_in"):
            opposite_class = {
                "momentum": "mean_reversion",
                "mean_reversion": "momentum",
                "carry": "volatility",
                "volatility": "carry",
            }.get(insight["class"], "")
            dreams.append(
                {
                    "type": "class_insight",
                    "class": insight["class"],
                    "weakness": insight.get("weak_in", []),
                    "rationale": insight.get("generalisation", ""),
                    "dream": f"Create a {insight['class']} variant that performs well in {insight.get('weak_in', [])} "
                    f"by incorporating {opposite_class} elements",
                }
            )

    # Dream 3: Tune parameters on best strategies
    for class_ in ["momentum", "carry", "volatility"]:
        best = registry.get_best_by_class().get(class_)
        if best and best.sharpe_test > 0:
            dreams.append(
                {
                    "type": "parameter_tuning",
                    "strategy": best.name,
                    "current_params": best.parameters,
                    "dream": f"Parameter sweep on {best.name}: optimise {list(best.parameters.keys())[:3] if best.parameters else 'key params'}",
                    "expected_benefit": "Potential 10-30% Sharpe improvement from parameter optimisation",
                }
            )

    return dreams


def main():
    """Run enhanced research cycle."""
    logger.info("=" * 60)
    logger.info("Enhanced Research Agent — Wide Search + Dreaming")
    logger.info("=" * 60)

    # Mock tools
    tools = {
        "web_search": lambda q: f"[Search: {q}]",
    }

    # 1. Search widely
    logger.info("\n🔍 Searching widely...")
    findings = search_widely(tools)
    logger.info(f"  Queries: {len(findings)}")

    # 2. Load registry
    registry = StrategyRegistry()
    logger.info(f"  Registry: {len(registry.strategies)} strategies")

    # 3. Draw connections
    logger.info("\n🔗 Drawing connections...")
    connections = draw_connections(findings, registry)
    logger.info(f"  Connections: {len(connections)}")
    for c in connections[:3]:
        logger.info(f"    {c.get('connection_type')}: {c.get('suggested_action', '')[:100]}")

    # 4. Generate class insights
    logger.info("\n💡 Class-level insights...")
    insights = generate_class_insights(registry)
    for i in insights:
        logger.info(f"    {i.get('class','?'):<12} {i.get('generalisation', '')[:120]}")

    # 5. Dream
    logger.info("\n🌙 Dreaming...")
    dreams = dream(findings, connections, insights, registry)
    for d in dreams:
        logger.info(f"    [{d['type']}] {d.get('dream', '')[:120]}")

    # Save dreams for agent to act on
    dreams_path = Path("data/processed/dreams.json")
    dreams_path.parent.mkdir(parents=True, exist_ok=True)
    dreams_path.write_text(json.dumps(dreams, indent=2, default=str))

    logger.info(f"\n  {len(dreams)} dreams saved to {dreams_path}")
    logger.info("Enhanced research complete")


if __name__ == "__main__":
    main()
