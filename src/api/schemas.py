from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request body for the /ask endpoint."""
    query: str
    history: list[dict] | None = None
