from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class GetSettingQuery(Query):
    pass
