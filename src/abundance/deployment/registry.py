"""Strategy registry — orderly book of all strategies with metadata.

Hierarchical directory structure:
  strategies/
    momentum/          — trend-following, momentum, breakout
    mean_reversion/    — RSI, bollinger, stat-arb
    carry/             — funding, basis, arbitrage
    volatility/        — vol-targeting, options-like
    ml_based/          — ML-generated, learned parameters
    composite/         — blended, ensemble, meta-strategies
    archived/          — rejected, deprecated, failed experiments
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class StrategyRecord:
    """Metadata for one strategy."""

    name: str
    class_: str  # momentum, mean_reversion, carry, etc.
    file_path: str
    status: str  # active, paper_trading, archived, rejected
    source: str  # paper title + arXiv ID, or "agent-generated"
    created_at: str
    sharpe_full: float = 0.0
    sharpe_test: float = 0.0
    sharpe_2025: float = 0.0
    max_dd_pct: float = 0.0
    regime_performance: dict = field(default_factory=dict)
    hypothesis: str = ""
    parameters: dict = field(default_factory=dict)
    composed_from: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "class": self.class_,
            "file_path": self.file_path,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at,
            "sharpe_full": self.sharpe_full,
            "sharpe_test": self.sharpe_test,
            "sharpe_2025": self.sharpe_2025,
            "max_dd_pct": self.max_dd_pct,
            "regime_performance": self.regime_performance,
            "hypothesis": self.hypothesis[:200],
            "parameters": self.parameters,
            "composed_from": self.composed_from,
            "notes": self.notes,
        }


class StrategyRegistry:
    """Orderly book of all strategies — never overwrite, only archive."""

    REGISTRY_PATH = Path("data/processed/strategy_registry.json")
    STRATEGIES_DIR = Path("src/abundance/strategies")

    def __init__(self):
        self.registry_path = Path("data/processed/strategy_registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.strategies: dict[str, StrategyRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            for name, d in data.items():
                # JSON has "class" but dataclass field is "class_"
                if "class" in d:
                    d["class_"] = d.pop("class")
                self.strategies[name] = StrategyRecord(**d)

    def save(self) -> None:
        data = {name: s.to_dict() for name, s in self.strategies.items()}
        self.registry_path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        name: str,
        class_: str,
        file_path: str,
        source: str,
        hypothesis: str = "",
        sharpe_full: float = 0.0,
        sharpe_test: float = 0.0,
        max_dd_pct: float = 0.0,
        parameters: dict | None = None,
        composed_from: list[str] | None = None,
    ) -> StrategyRecord:
        """Register a new strategy. Never overwrites existing — archives old instead."""
        if name in self.strategies:
            old = self.strategies[name]
            if old.status != "archived":
                old.status = "archived"
                old.notes = f"Replaced by new version on {datetime.now(timezone.utc).isoformat()}"
                self._move_to_archive(name)

        record = StrategyRecord(
            name=name,
            class_=class_,
            file_path=file_path,
            status="paper_trading" if sharpe_test > 0 else "active",
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
            sharpe_full=sharpe_full,
            sharpe_test=sharpe_test,
            max_dd_pct=max_dd_pct,
            hypothesis=hypothesis,
            parameters=parameters or {},
            composed_from=composed_from or [],
        )
        self.strategies[name] = record
        self.save()
        return record

    def _move_to_archive(self, name: str) -> None:
        """Move strategy file to archived/ directory."""
        if name not in self.strategies:
            return
        old_path = Path(self.strategies[name].file_path)
        archive_dir = self.STRATEGIES_DIR / "archived"
        archive_dir.mkdir(parents=True, exist_ok=True)
        if old_path.exists():
            new_path = archive_dir / old_path.name
            old_path.rename(new_path)

    def get_active(self) -> list[StrategyRecord]:
        """Return all non-archived strategies."""
        return [s for s in self.strategies.values() if s.status != "archived"]

    def get_by_class(self, class_: str) -> list[StrategyRecord]:
        """Return strategies by class."""
        return [s for s in self.strategies.values() if s.class_ == class_]

    def get_best_by_class(self) -> dict[str, StrategyRecord]:
        """Return the best strategy per class (by test Sharpe)."""
        best = {}
        for s in self.strategies.values():
            if s.status == "archived":
                continue
            if s.class_ not in best or s.sharpe_test > best[s.class_].sharpe_test:
                best[s.class_] = s
        return best

    def find_connections(self, strategy: StrategyRecord) -> list[StrategyRecord]:
        """Find related strategies for composition/dreaming.

        Looks for:
        - Same class but different parameters
        - Complementary classes (e.g., momentum + mean_reversion)
        - Strategies that worked in regimes where this one failed
        """
        related = []

        # Same class, different approach
        for s in self.get_by_class(strategy.class_):
            if s.name != strategy.name and s.sharpe_test > 0:
                related.append(s)

        # Complementary classes
        complements = {
            "momentum": "mean_reversion",
            "mean_reversion": "momentum",
            "carry": "volatility",
            "volatility": "carry",
        }
        complement_class = complements.get(strategy.class_)
        if complement_class:
            for s in self.get_by_class(complement_class):
                if s.sharpe_test > 0:
                    related.append(s)

        # Strategies that worked in regimes where this failed
        my_bad_regimes = {
            r for r, v in strategy.regime_performance.items()
            if isinstance(v, dict) and v.get("sharpe", 0) < 0
        }
        for s in self.strategies.values():
            if s.name == strategy.name or s.status == "archived":
                continue
            for regime, perf in s.regime_performance.items():
                if isinstance(perf, dict) and perf.get("sharpe", 0) > 0.5 and regime in my_bad_regimes:
                    related.append(s)
                    break

        return related

    def summary(self) -> str:
        """Human-readable summary of the strategy book."""
        lines = ["Strategy Book Summary", "=" * 60]
        for class_ in ["momentum", "mean_reversion", "carry", "volatility", "composite", "ml_based"]:
            strats = self.get_by_class(class_)
            active = [s for s in strats if s.status != "archived"]
            archived = [s for s in strats if s.status == "archived"]
            if active or archived:
                lines.append(f"\n{class_.upper()} ({len(active)} active, {len(archived)} archived):")
                for s in sorted(active, key=lambda x: x.sharpe_test, reverse=True):
                    lines.append(
                        f"  {'🟢' if s.sharpe_test > 0.5 else '🟡' if s.sharpe_test > 0 else '🔴'} "
                        f"{s.name:<30} Test Sharpe {s.sharpe_test:>6.3f}  2025 {s.sharpe_2025:>6.3f}"
                    )
        return "\n".join(lines)
