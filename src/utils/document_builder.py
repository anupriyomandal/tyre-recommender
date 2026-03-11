import pandas as pd

def build_document(row: pd.Series) -> str:
    """
    Convert a CSV row into a natural language document for embedding.
    """
    category = row.get("category", "NA")
    brand = row.get("vehicle-brand", "NA")
    model = row.get("vehicle-model", "NA")
    variant = row.get("vehicle-variant", "NA")
    year = row.get("manufacturing-year", "NA")
    
    rec_tyre = row.get("recommended-tyre", row.get("recommended-sku.1", "None"))
    if str(rec_tyre).strip() in ["NA", "None", "", "#N/A"]:
        rec_tyre = row.get("recommended-sku.1", "None")
        
    upsize_tyre = row.get("upsize-tyre", row.get("upsize-sku.1", "None"))
    if str(upsize_tyre).strip() in ["NA", "None", "", "#N/A"]:
        upsize_tyre = row.get("upsize-sku.1", "None")
        
    others_tyre = row.get("others-tyre", row.get("others-sku.1", "None"))
    if str(others_tyre).strip() in ["NA", "None", "", "#N/A"]:
        others_tyre = row.get("others-sku.1", "None")

    document = (
        f"Vehicle Category: {category}\n"
        f"Brand: {brand}\n"
        f"Model: {model}\n"
        f"Variant: {variant}\n"
        f"Manufacturing Year: {year}\n\n"
        f"Recommended Tyre: {rec_tyre}\n"
        f"Upsize Tyre: {upsize_tyre}\n"
        f"Other Tyre: {others_tyre}"
    )
    return document
