from pydantic import BaseModel

class HouseRequest(BaseModel):
    zip_code: int
    state: str
    sqft: float
    bedrooms: int
    bathrooms: float
    year_built: int
    
    quality: int