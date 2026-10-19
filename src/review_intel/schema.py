"""The label space: eight aspects of a car review, each not mentioned / positive / negative / mixed."""

ASPECTS = {
    "price": "purchase price, value for money, running and insurance costs, resale value",
    "performance": "acceleration, power, handling, braking, towing, transmission and engine feel",
    "comfort": "seats, ride quality, cabin noise, space and legroom, climate control",
    "reliability": "breakdowns, repairs, recalls, build-quality faults, warranty and dealer service",
    "design": "exterior and interior styling, materials and finish, how it looks",
    "safety": "safety ratings, airbags, visibility, driver-assistance and collision-avoidance features",
    "fuel_economy": "fuel consumption (mpg), electric range and charging, fuel costs",
    "technology": "infotainment, touchscreen, phone integration, sound system, apps and software",
}
SENTIMENTS = ["none", "positive", "negative", "mixed"]  # index 0 = aspect not mentioned
OVERALL = ["negative", "mixed", "positive"]

LABELS = [f"{a}:{s}" for a in ASPECTS for s in SENTIMENTS] + [f"overall:{o}" for o in OVERALL]
# 35 logits: 8 aspect groups of 4, then 3 for the overall verdict
N_ASPECT_LOGITS = len(ASPECTS) * len(SENTIMENTS)
NICE = {
    "price": "Price & value",
    "performance": "Performance",
    "comfort": "Comfort",
    "reliability": "Reliability",
    "design": "Design",
    "safety": "Safety",
    "fuel_economy": "Fuel economy",
    "technology": "Technology",
}
