import pandas as pd

def map_to_model_input(req):
    data = {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotArea": req.sqft * 5,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",

        "OverallQual": req.quality,
        "OverallCond": 5,

        "YearBuilt": req.year_built,
        "YearRemodAdd": req.year_built,

        "RoofStyle": "Gable",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",

        "TotalBsmtSF": req.sqft * 0.5,
        "GrLivArea": req.sqft,

        "FullBath": req.bathrooms,
        "HalfBath": 0,

        "BedroomAbvGr": req.bedrooms,
        "KitchenAbvGr": 1,

        "GarageCars": 2,
        "GarageArea": 400,

        "YrSold": 2020,
        "SaleCondition": "Normal"
    }
    return pd.DataFrame([data])