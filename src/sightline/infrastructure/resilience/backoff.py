"""Exponential backoff with jitter.

Spec S3.5 requires "exponential backoff and jitter". Jitter is not cosmetic here: the
retrieval stage fans out concurrently over every planned query, so without it a single 429
would put every branch on an identical retry schedule and they would collide again, in lockstep,
on each subsequent attempt. Jitter decorrelates them.

The RNG is injectable so that jitter is deterministic under test - a retry policy whose timing
cannot be asserted on is a retry policy nobody can prove works.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sightline.config.settings import ResilienceSettings

#: Module-level generator used when no RNG is injected. Not cryptographic by design:
#: this decorrelates retry timing, it does not protect anything.
_DEFAULT_RNG = random.Random()  # noqa: S311


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Delay schedule for successive retry attempts."""

    initial_seconds: float
    multiplier: float
    max_seconds: float
    jitter_ratio: float

    @classmethod
    def from_settings(cls, settings: ResilienceSettings) -> BackoffPolicy:
        return cls(
            initial_seconds=settings.backoff_initial_seconds,
            multiplier=settings.backoff_multiplier,
            max_seconds=settings.backoff_max_seconds,
            jitter_ratio=settings.jitter_ratio,
        )

    def base_delay(self, attempt: int) -> float:
        """The un-jittered delay after ``attempt`` (1-indexed) has failed.

        Capped at :attr:`max_seconds` so a long retry budget cannot produce an unbounded wait.
        """
        exponent = max(0, attempt - 1)
        return min(self.initial_seconds * (self.multiplier**exponent), self.max_seconds)

    def delay_for(
        self,
        attempt: int,
        *,
        retry_after_seconds: float | None = None,
        rng: random.Random | None = None,
    ) -> float:
        """Delay before the next attempt, in seconds.

        When the server supplied ``Retry-After``, that directive wins: it is honoured (bounded
        by :attr:`max_seconds`) and jittered **upward only**, because jittering it downward
        would mean deliberately retrying sooner than the server asked us to.
        """
        generator = rng if rng is not None else _DEFAULT_RNG

        if retry_after_seconds is not None and retry_after_seconds >= 0:
            honoured = min(retry_after_seconds, self.max_seconds)
            return honoured + honoured * generator.uniform(0.0, self.jitter_ratio)

        base = self.base_delay(attempt)
        jittered = base * (1.0 + generator.uniform(-self.jitter_ratio, self.jitter_ratio))
        return max(0.0, jittered)
