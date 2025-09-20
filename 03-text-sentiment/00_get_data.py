"""
Generate synthetic multilingual (EN/DE) review dataset for sentiment analysis
or load an existing CSV at data/processed/reviews.csv if present.

Columns: review_id, text, label  (label: 1=positive, 0=negative)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import random

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT / "reviews.csv"

def generate_synthetic_reviews(n=10000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    english_positive = [
        "I love this product", "Excellent quality and fast delivery", "Very satisfied with purchase",
        "Works perfectly", "Highly recommend to everyone", "Amazing value for money"
    ]
    english_negative = [
        "Terrible, broke in days", "Very disappointed", "Product not as described",
        "Bad quality", "Do not recommend", "Waste of money"
    ]
    german_positive = [
        "Tolles Produkt", "Sehr zufrieden mit dem Kauf", "Funktioniert einwandfrei",
        "Sehr gutes Preis-Leistungs-Verhältnis", "Kann ich nur empfehlen"
    ]
    german_negative = [
        "Sehr enttäuscht", "Schlechte Qualität", "Funktioniert nicht", "Geldverschwendung",
        "Nicht zu empfehlen"
    ]
    products = ["phone", "headphones", "laptop", "coffee maker", "backpack", "running shoes", "kamera", "tasche"]
    rows = []
    for i in range(n):
        # 70% English, 30% German
        if random.random() < 0.7:
            lang = "en"
            if random.random() < 0.5:
                text = f"{random.choice(english_positive)} for my {random.choice(products)}."
                label = 1
            else:
                text = f"{random.choice(english_negative)} about the {random.choice(products)}."
                label = 0
        else:
            lang = "de"
            if random.random() < 0.5:
                text = f"{random.choice(german_positive)} für meine {random.choice(products)}."
                label = 1
            else:
                text = f"{random.choice(german_negative)} mit der {random.choice(products)}."
                label = 0
        rows.append({"review_id": f"R{i+1:06d}", "text": text, "label": label, "lang": lang})
    df = pd.DataFrame(rows)
    return df

def main():
    if OUT_CSV.exists():
        print(f"Found existing dataset at {OUT_CSV}. Using it.")
    else:
        print("Generating synthetic reviews dataset...")
        df = generate_synthetic_reviews(n=10000)
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved synthetic dataset to: {OUT_CSV} (rows: {len(df):,})")
    print("If you have a real reviews CSV, place it at:", OUT_CSV)

if __name__ == "__main__":
    main()
