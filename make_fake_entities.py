"""Generate the fake-entity probe set (Phase 2 of docs/HALLUCINATION_PLAN.md).

Matched pairs: each template is asked about real entities (answerable, aliases
provided) and fabricated ones (unanswerable by construction). The question is
whether the workspace flags fabrication while the output sounds fluent.

Writes probes/fake_entities.json in the format modal_fit.py::uncertainty takes
via --questions-file (extra field `entity_real` is passed through to rows).
"""
import json

REAL_PHYSICISTS = [
    ("Richard Feynman", "1965"), ("Marie Curie", "1903"), ("Albert Einstein", "1921"),
    ("Niels Bohr", "1922"), ("Enrico Fermi", "1938"), ("Paul Dirac", "1933"),
    ("Werner Heisenberg", "1932"), ("Max Planck", "1918"), ("Peter Higgs", "2013"),
    ("Ernest Rutherford", "1908"),
]
FAKE_PHYSICISTS = [
    "Elena Morvath", "Tobias Ravnsborg", "Ichiro Kanemoto", "Beatriz Olande",
    "Viktor Shalenko", "Margarethe Lindqvist-Baur", "Samuel Okonjo-Pryce",
    "Anneliese Vardo", "Dmitri Kolvenko", "Rosalind Achterberg",
]
REAL_NOVELS = [
    ("Moby-Dick", ["Herman Melville", "Melville"]), ("Pride and Prejudice", ["Jane Austen", "Austen"]),
    ("The Great Gatsby", ["F. Scott Fitzgerald", "Fitzgerald"]), ("1984", ["George Orwell", "Orwell"]),
    ("Crime and Punishment", ["Fyodor Dostoevsky", "Dostoevsky", "Dostoyevsky"]),
    ("One Hundred Years of Solitude", ["Gabriel Garcia Marquez", "Garcia Marquez", "Márquez"]),
    ("The Catcher in the Rye", ["J.D. Salinger", "Salinger"]), ("Beloved", ["Toni Morrison", "Morrison"]),
    ("Things Fall Apart", ["Chinua Achebe", "Achebe"]), ("Don Quixote", ["Miguel de Cervantes", "Cervantes"]),
]
FAKE_NOVELS = [
    "The Salt Cartographer", "Winter in the House of Brann", "A Lantern for the Drowned",
    "The Nineteenth Parallel", "Ash and Clockwork", "The Vintner's Daughter of Osteg",
    "Letters to a Quiet Harbor", "The Last Ferryman of Doleu", "Midnight at the Verrick Hotel",
    "The Orchardist's Rebellion",
]
REAL_CITIES = [
    ("Marseille", ["France"]), ("Osaka", ["Japan"]), ("Curitiba", ["Brazil"]),
    ("Krakow", ["Poland"]), ("Adelaide", ["Australia"]), ("Mombasa", ["Kenya"]),
    ("Valparaiso", ["Chile"]), ("Bergen", ["Norway"]), ("Izmir", ["Turkey"]),
    ("Guadalajara", ["Mexico"]),
]
FAKE_CITIES = [
    "Vellemara", "Kostrivny", "Ouadrane", "Sillanpaa Bay", "Terravesca",
    "Nokoyama", "Brindlemere", "Zaltoria", "Pemburu Hilir", "Cascavelle-sur-Mer",
]
REAL_BATTLES = [
    ("Hastings", ["1066"]), ("Waterloo", ["1815"]), ("Gettysburg", ["1863"]),
    ("Agincourt", ["1415"]), ("Stalingrad", ["1942", "1943"]), ("Trafalgar", ["1805"]),
    ("Midway", ["1942"]), ("Thermopylae", ["480"]), ("Lepanto", ["1571"]), ("Yorktown", ["1781"]),
]
FAKE_BATTLES = [
    "Karvenholm", "the Draval Pass", "Osterbruck", "the Meridan Fields", "Quillane Ridge",
    "Tessovar", "the Brumig Delta", "Vantorre", "Ilyev Crossing", "the Sarn Estuary",
]
REAL_ELEMENTS = [
    ("tungsten", ["W"]), ("sodium", ["Na"]), ("potassium", ["K"]), ("iron", ["Fe"]),
    ("gold", ["Au"]), ("mercury", ["Hg"]), ("lead", ["Pb"]), ("tin", ["Sn"]),
    ("antimony", ["Sb"]), ("silver", ["Ag"]),
]
FAKE_ELEMENTS = [
    "voltrium", "casperine", "meridium", "oxalonium", "brenthite",
    "julvarium", "tandrelite", "quorvium", "selbanite", "ravonium",
]

items = []
def add(q, aliases, real, domain):
    items.append({"q": q, "aliases": aliases, "entity_real": int(real), "domain": domain})

for name, yr in REAL_PHYSICISTS:
    add(f"In what year did the physicist {name} win the Nobel Prize?", [yr], 1, "physicist")
for name in FAKE_PHYSICISTS:
    add(f"In what year did the physicist {name} win the Nobel Prize?", [], 0, "physicist")
for title, auth in REAL_NOVELS:
    add(f"Who wrote the novel {title}?", auth, 1, "novel")
for title in FAKE_NOVELS:
    add(f"Who wrote the novel {title}?", [], 0, "novel")
for city, c in REAL_CITIES:
    add(f"In which country is the city of {city}?", c, 1, "city")
for city in FAKE_CITIES:
    add(f"In which country is the city of {city}?", [], 0, "city")
for b, yrs in REAL_BATTLES:
    add(f"In what year was the Battle of {b}?", yrs, 1, "battle")
for b in FAKE_BATTLES:
    add(f"In what year was the Battle of {b}?", [], 0, "battle")
for el, sym in REAL_ELEMENTS:
    add(f"What is the chemical symbol for {el}?", sym, 1, "element")
for el in FAKE_ELEMENTS:
    add(f"What is the chemical symbol for {el}?", [], 0, "element")

with open("probes/fake_entities.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=1)
print(f"wrote probes/fake_entities.json: {len(items)} questions "
      f"({sum(i['entity_real'] for i in items)} real, "
      f"{sum(1-i['entity_real'] for i in items)} fake)")
