from pydantic import BaseModel, field_validator


class BBoxIn(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @field_validator("w", "h")
    @classmethod
    def positive(cls, v):
        if v <= 0:
            raise ValueError("w/h должны быть положительными")
        return v


class SearchResultItem(BaseModel):
    gallery_id: str
    confidence: float


class SearchResponse(BaseModel):
    query_id: str | None = None
    results: list[SearchResultItem]
    rejected: bool
    attributes: dict | None = None
