"""Persistence models.

Imported as a package so that ``Base.metadata`` is fully populated before Alembic reads it -
a model that is never imported is a table Alembic will silently propose dropping.
"""

from sightline.infrastructure.persistence.models.discovered_query import DiscoveredQueryModel
from sightline.infrastructure.persistence.models.insight import InsightModel
from sightline.infrastructure.persistence.models.pipeline_run import PipelineRunModel
from sightline.infrastructure.persistence.models.profile import ProfileModel
from sightline.infrastructure.persistence.models.recommendation import RecommendationModel

__all__ = [
    "DiscoveredQueryModel",
    "InsightModel",
    "PipelineRunModel",
    "ProfileModel",
    "RecommendationModel",
]
