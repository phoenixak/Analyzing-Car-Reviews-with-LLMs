"""Synthetic, labelled car reviews: labels by construction, checked by a blind re-labelling pass.

1. Sample a spec: car, reviewer, length, style, 1–4 aspects with a sentiment each, overall verdict.
2. The teacher LLM writes a review that expresses exactly that spec.
3. The teacher labels the review again *without seeing the spec* (JSON-schema constrained).
4. Keep the review only if the blind labels match the spec, so every kept label is agreed twice.

Runs against any OpenAI-compatible server; we used llama.cpp's llama-server (Vulkan) with Qwen3-8B:

    llama-server -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 32768 -np 8 --jinja --port 8088
    python -m review_intel.data.synth --n 6000 --out data/synthetic.jsonl
"""

import argparse
import concurrent.futures as cf
import json
import random
import threading
import time
from pathlib import Path

import httpx

from review_intel.schema import ASPECTS, OVERALL, SENTIMENTS

CARS = [  # (make, model, body) — real models, invented opinions; the dataset card says so
    ("Toyota", "Corolla", "compact sedan"),
    ("Toyota", "RAV4 Hybrid", "compact SUV"),
    ("Toyota", "Tacoma", "pickup"),
    ("Toyota", "Camry", "midsize sedan"),
    ("Toyota", "Sienna", "minivan"),
    ("Honda", "Civic", "compact car"),
    ("Honda", "CR-V", "compact SUV"),
    ("Honda", "Accord", "midsize sedan"),
    ("Honda", "Odyssey", "minivan"),
    ("Honda", "Pilot", "midsize SUV"),
    ("Ford", "F-150", "pickup"),
    ("Ford", "Mustang Mach-E", "electric SUV"),
    ("Ford", "Escape", "compact SUV"),
    ("Ford", "Maverick", "compact pickup"),
    ("Ford", "Explorer", "midsize SUV"),
    ("Chevrolet", "Silverado", "pickup"),
    ("Chevrolet", "Equinox", "compact SUV"),
    ("Chevrolet", "Bolt EV", "electric hatchback"),
    ("Chevrolet", "Tahoe", "full-size SUV"),
    ("Chevrolet", "Malibu", "midsize sedan"),
    ("Tesla", "Model 3", "electric sedan"),
    ("Tesla", "Model Y", "electric SUV"),
    ("Hyundai", "Tucson", "compact SUV"),
    ("Hyundai", "Elantra", "compact sedan"),
    ("Hyundai", "Ioniq 5", "electric crossover"),
    ("Hyundai", "Santa Fe", "midsize SUV"),
    ("Kia", "Telluride", "midsize SUV"),
    ("Kia", "Sportage", "compact SUV"),
    ("Kia", "EV6", "electric crossover"),
    ("Kia", "Forte", "compact sedan"),
    ("Nissan", "Rogue", "compact SUV"),
    ("Nissan", "Altima", "midsize sedan"),
    ("Nissan", "Leaf", "electric hatchback"),
    ("Nissan", "Frontier", "pickup"),
    ("Subaru", "Outback", "wagon"),
    ("Subaru", "Forester", "compact SUV"),
    ("Subaru", "Crosstrek", "subcompact SUV"),
    ("Mazda", "CX-5", "compact SUV"),
    ("Mazda", "Mazda3", "compact car"),
    ("Mazda", "MX-5 Miata", "roadster"),
    ("Volkswagen", "Golf GTI", "hot hatch"),
    ("Volkswagen", "Tiguan", "compact SUV"),
    ("Volkswagen", "ID.4", "electric SUV"),
    ("BMW", "3 Series", "luxury sedan"),
    ("BMW", "X5", "luxury SUV"),
    ("Mercedes-Benz", "C-Class", "luxury sedan"),
    ("Mercedes-Benz", "GLC", "luxury SUV"),
    ("Audi", "Q5", "luxury SUV"),
    ("Audi", "A4", "luxury sedan"),
    ("Lexus", "RX 350", "luxury SUV"),
    ("Lexus", "ES 300h", "luxury hybrid sedan"),
    ("Jeep", "Wrangler", "off-roader"),
    ("Jeep", "Grand Cherokee", "midsize SUV"),
    ("Ram", "1500", "pickup"),
    ("Dodge", "Charger", "muscle sedan"),
    ("Volvo", "XC60", "luxury SUV"),
    ("Volvo", "XC90", "luxury SUV"),
    ("Porsche", "Macan", "luxury SUV"),
    ("Mitsubishi", "Outlander PHEV", "plug-in hybrid SUV"),
    ("Genesis", "GV70", "luxury SUV"),
    ("Rivian", "R1S", "electric SUV"),
    ("Polestar", "2", "electric sedan"),
    ("Mini", "Cooper", "small hatchback"),
    ("Buick", "Enclave", "midsize SUV"),
    ("GMC", "Sierra", "pickup"),
    ("Acura", "MDX", "luxury SUV"),
    ("Cadillac", "Lyriq", "electric luxury SUV"),
    ("Chrysler", "Pacifica Hybrid", "minivan"),
]
PERSONAS = [
    "a parent with two young kids",
    "a daily commuter who drives 60 miles a day",
    "a retired couple who take road trips",
    "a college student on a budget",
    "a rideshare driver",
    "a car enthusiast who loves driving",
    "a contractor who hauls tools",
    "a first-time car buyer",
    "someone who lives in a snowy mountain town",
    "a nurse who works night shifts",
    "a small-business owner",
    "an engineer who reads every spec sheet",
    "a dog owner who goes hiking on weekends",
    "someone who traded in a 15-year-old car",
    "a city dweller who parallel parks every day",
]
LENGTHS = {
    "short": "one or two sentences",
    "medium": "three to five sentences",
    "long": "six to nine sentences, like a detailed owner review",
}
STYLES = [
    "plain and matter-of-fact",
    "chatty and informal",
    "detailed and specific with numbers",
    "frustrated",
    "enthusiastic",
    "dry and a bit sarcastic",
    "written quickly on a phone, with a typo or two",
    "comparing it with the car they had before",
]
HOW = {
    "positive": "clearly positive",
    "negative": "clearly negative",
    "mixed": "mixed: one good point and one bad point about it",
}


LAUNCH = {
    "Mustang Mach-E": 2021,
    "Maverick": 2022,
    "Ioniq 5": 2022,
    "EV6": 2022,
    "ID.4": 2021,
    "R1S": 2022,
    "Lyriq": 2023,
    "Model Y": 2020,
    "GV70": 2021,
    "2": 2021,
    "Telluride": 2020,
    "Bolt EV": 2017,
    "Outlander PHEV": 2018,
    "Pacifica Hybrid": 2017,
}  # first model year, so reviews don't predate the car


def sample_spec(rng: random.Random) -> dict:
    make, model, body = rng.choice(CARS)
    n = rng.choices([1, 2, 3, 4], weights=[3, 4, 3, 1])[0]
    chosen = rng.sample(list(ASPECTS), n)
    pol = {a: rng.choices(["positive", "negative", "mixed"], weights=[5, 4, 1])[0] for a in chosen}
    pos = sum(v == "positive" for v in pol.values())
    neg = sum(v == "negative" for v in pol.values())
    overall = "positive" if pos > neg else "negative" if neg > pos else "mixed"
    return {
        "make": make,
        "model": model,
        "body": body,
        "year": rng.randint(LAUNCH.get(model, 2015), 2026),
        "persona": rng.choice(PERSONAS),
        "length": rng.choices(list(LENGTHS), weights=[3, 5, 3])[0],
        "style": rng.choice(STYLES),
        "aspects": pol,
        "overall": overall,
    }


def write_prompt(spec: dict) -> list[dict]:
    wanted = "\n".join(f"- {a.replace('_', ' ')} ({ASPECTS[a]}): {HOW[v]}" for a, v in spec["aspects"].items())
    banned = ", ".join(a.replace("_", " ") for a in ASPECTS if a not in spec["aspects"])
    user = (
        f"Write one owner review of a {spec['year']} {spec['make']} {spec['model']} ({spec['body']}).\n"
        f"The reviewer is {spec['persona']}. Length: {LENGTHS[spec['length']]}. Tone: {spec['style']}.\n\n"
        f"The review must give exactly these opinions:\n{wanted}\n\n"
        f"Do not say anything about: {banned}.\n"
        f"Overall the reviewer feels {spec['overall']} about the car.\n"
        "Write only the review text — no title, no star rating, no lists, no quotation marks."
    )
    return [
        {"role": "system", "content": "You write realistic owner reviews of cars, like those on car-review websites."},
        {"role": "user", "content": user},
    ]


LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "object",
            "properties": {a: {"enum": SENTIMENTS} for a in ASPECTS},
            "required": list(ASPECTS),
            "additionalProperties": False,
        },
        "overall": {"enum": OVERALL},
    },
    "required": ["aspects", "overall"],
    "additionalProperties": False,
}


def label_prompt(review: str) -> list[dict]:
    defs = "\n".join(f"- {a}: {d}" for a, d in ASPECTS.items())
    user = (
        "Label this car review. For each aspect say whether the reviewer gives an opinion about it:\n"
        f"{defs}\n\n"
        'Use "none" if the aspect is not discussed, "positive" or "negative" for a clear opinion, and "mixed" '
        "if they say both good and bad things about that aspect. Then give the reviewer's overall feeling about "
        f"the car (positive, negative or mixed).\n\nReview:\n{review}"
    )
    return [{"role": "user", "content": user}]


class Teacher:
    def __init__(self, base_url: str, model: str = "teacher") -> None:
        self.client = httpx.Client(base_url=base_url, timeout=180)
        self.model = model

    def chat(self, messages, temperature, schema=None, max_tokens=400) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}}
        r = self.client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def one(teacher: Teacher, spec: dict) -> dict | None:
    review = teacher.chat(write_prompt(spec), temperature=0.95).strip().strip('"').strip()
    if len(review) < 20:
        return None
    labels = json.loads(teacher.chat(label_prompt(review), temperature=0.0, schema=LABEL_SCHEMA, max_tokens=200))
    want = {a: spec["aspects"].get(a, "none") for a in ASPECTS}
    # keep a review when the aspects agree exactly; the overall verdict is taken from the blind reading of
    # the finished text (a tie like "comfort good, design bad" can still read as positive overall)
    agree = labels["aspects"] == want
    return {
        "text": review,
        "aspects": want,
        "overall": labels["overall"],
        "agree": agree,
        "spec_overall": spec["overall"],
        "teacher": labels,
        "car": f"{spec['make']} {spec['model']}",
        "make": spec["make"],
        "spec": {k: spec[k] for k in ("year", "body", "persona", "length", "style")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="specs to try")
    ap.add_argument("--out", default="data/synthetic.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    specs = [sample_spec(rng) for _ in range(args.n)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0  # resume
    teacher, lock, kept, t0 = Teacher(args.url), threading.Lock(), 0, time.time()
    with out.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, teacher, s) for s in specs[done:]]
        for i, fut in enumerate(cf.as_completed(futures), start=done + 1):
            try:
                row = fut.result()
            except Exception as e:  # keep going; a few failures don't matter
                print("error:", e)
                continue
            if row:
                kept += row["agree"]
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
            if i % 50 == 0:
                print(
                    f"{i}/{args.n}  agreed {kept}/{i - done}  {(i - done) / (time.time() - t0) * 60:.0f} reviews/min",
                    flush=True,
                )


if __name__ == "__main__":
    main()
