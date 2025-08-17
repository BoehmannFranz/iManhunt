# iManhunt
iManhunt simuliert Fluchtbewegungen in einer fiktiven 3D-Stadt. Auf Basis eines Straßengraphen, Topografie und Checkpoints lernen RL-Profile mit unterschiedlichen Traits (Alter, Impulsivität, Gesundheit etc.). Resultat: zeitabhängige Heatmaps, fünf Routen, Antreffwahrscheinlichkeiten und optimierte Kontrollpunkte.

> **Hinweis & Ethik**: Dieses Projekt ist ein **rein fiktionaler** Methoden-Demonstrator für städtische **Bewegungs- und Risiko­simulationen**. Es dient **nicht** der Unterstützung realer Flucht- oder Ausweich­strategien. Keine realen Daten, Personen oder Orte; kein operativer Einsatz. Nutzen: Forschung, Lehre, Analytics-Prototyping (z. B. Krisen-/Evakuierungs­simulation, Crowd Dynamics).

---

## Inhaltsverzeichnis
1. [Zielbild & Scope](#zielbild--scope)  
2. [Systemübersicht](#systemübersicht)  
3. [MVP (Minimum Viable Product)](#mvp-minimum-viable-product)  
4. [Unterprojekte (SP1–SP10)](#unterprojekte-sp1sp10)  
5. [Daten & Annahmen](#daten--annahmen)  
6. [Charakterprofile (5 Standard-Personas)](#charakterprofile-5-standard-personas)  
7. [Formale Modelle (Kurzüberblick)](#formale-modelle-kurzüberblick)  
8. [Visualisierung (2D/3D)](#visualisierung-23d)  
9. [Evaluation & Metriken](#evaluation--metriken)  
10. [Projektstruktur (Repo-Layout)](#projektstruktur-repo-layout)  
11. [Konfiguration (Beispiel)](#konfiguration-beispiel)  
12. [Roadmap & Meilensteine](#roadmap--meilensteine)  
13. [Nicht-Ziele & Grenzen](#nicht-ziele--grenzen)  
14. [Lizenz & Nutzung](#lizenz--nutzung)

---

## Zielbild & Scope

**Fragestellung**  
Ein Akteur (fiktional) entkommt aus einem fixen Startpunkt in einer synthetischen Stadt. Wir prognostizieren:
- **zeitabhängige Aufenthaltswahrscheinlichkeiten** auf dem Straßengraph,
- **fünf parallele Fluchtrouten** basierend auf unterschiedlichen **Charakterprofilen**,
- **Antreff-/Begegnungswahrscheinlichkeiten** an vorgegebenen **Checkpoints**.

**Outputs**
- **Heatmaps** je Zeitschritt und Profil,
- **3D-Trails** der Pfade (Zeitverlauf),
- **Checkpoint-Analysen** (Ranking, Zeitreihen),
- reproducible **Konfigurations- und Ergebnisdateien** (GeoJSON/CSV/MP4/HTML).

**Zielgruppe**: Forschung/Lehre, Data-Science-Prototyping, RL/Graph-Analytics-Demos.  
**Nicht-Ziel**: reale Einsatzplanung, reale Personen, reale Städte.

---

## Systemübersicht

- **Umwelt**: Gerichteter Straßengraph *G(V,E)* (synthetisch), Kantenattribute: Länge, Typ, Steigung, erwartete Geschwindigkeit, Reisedauer.  
- **Akteure**: 5 Profile (Traits → RL-Hyperparameter & Reward-Gewichte).  
- **Dynamik**: RL-Policies wählen Kanten; Monte-Carlo-Rollouts erzeugen zeitliche Positionsverteilungen *pₜ*.  
- **Beobachtung** (optional): bayesianische Updates durch fiktive Sichtungen.  
- **Analyse**: Checkpoints, Top-Korridore, Zeitreihen, Sensitivitätsstudien.  
- **Visualisierung**: 2D-Heatmaps, 3D-Stadt (extrudierte Geometrie), Zeit-Slider.

---

## MVP (Minimum Viable Product)

**Ziel des MVP**  
Ein lauffähiger End-to-End-Pfad: synthetische Stadt → 5 Profile → kurze RL-Trainings → Rollouts → 2D-Heatmaps & Checkpoint-Zeitreihen.

**Umfang**
- Synthetischer **Grid-Graph** (z. B. 50×50 Knoten), konstante Gebäudeflächen.  
- **Kantenlänge**, **Straßentyp**, simple **Steigung** (parametrisch).  
- **Traits → RL-Parameter** (einfaches Mapping, s. unten).  
- **Reward-Composer** (Distanz/ Zeit/ Steigung/ Risiko/ Zielpotenzial/ Ermüdung).  
- **PPO** (kurz trainiert), **N** Rollouts/Profil, Aggregation *pₜ*.  
- 2D-**Heatmap** je *t*, **Checkpoint-Ranking** + **P_enc**-Zeitreihen.  
- CLI: `python main.py --config cfg/demo.json`.

**Akzeptanzkriterien**
- [ ] Reproduzierbare Runs (Seed) erzeugen 5 unterschiedlich geprägte Pfadfelder.  
- [ ] Mind. 3 Zeitstempel visualisiert (t₁, t₂, t₃) als Heatmap.  
- [ ] Checkpoint-Zeitreihen und kumulative **P_enc** pro Profil verfügbar.  
- [ ] Laufzeit < 10 min auf Standard-Laptop (kleine Dimensionen).

---

## Unterprojekte (SP1–SP10)

### SP1 – Datenkern & Graph
- **Ziele**: Generierung eines synthetischen Straßengraphen; Kantenattribute (Länge, Typ, Steigung, Geschwindigkeit, Reisedauer τ).  
- **Deliverables**: `graph_build.py`, Beispiel-Graphen (`data/synth_city_*`).  
- **Akzeptanz**: Konsistenter gerichteter Graph, Basiskosten je Kante.

### SP2 – Trait-Sampler & Mapping
- **Ziele**: Sampling von Traits (Alter, Impulsivität, Weitsicht, Gesundheit, Topografie-Sensitivität, Risiko, Sucht, Familie, Motiv) → Mapping auf (γ, α_ent, w_* , v_max).  
- **Deliverables**: `traits.py`, Tests; 5 Profile per Run.  
- **Akzeptanz**: Parameter-Spread erzeugt sichtbare Routen-Diversität.

### SP3 – Reward-Composer
- **Ziele**: Komponierbares Reward-System mit Caps/Regularisierung; Option für hierarchische Struktur (HRL) vorbereitet.  
- **Deliverables**: `reward.py`, dokumentierte Formeln.  
- **Akzeptanz**: Unit-Tests für Term-Signaturen, numerische Stabilität.

### SP4 – RL-Umgebung & Training
- **Ziele**: `gymnasium`-Env (State, Action, Step), PPO-Training (stable-baselines3), kurze Episoden.  
- **Deliverables**: `env.py`, `train.py`.  
- **Akzeptanz**: Lernkurven, Policy stabilisiert innerhalb kurzer Episoden.

### SP5 – Rollouts & Propagation
- **Ziele**: Monte-Carlo-Trajektorien, Aggregation zu *pₜ*, Sparse-Vektor-Pipelines.  
- **Deliverables**: `rollout.py`, `probmap.py`.  
- **Akzeptanz**: Zeitlich kohärente Wahrscheinlichkeitsfelder.

### SP6 – Checkpoint-Analyse
- **Ziele**: Berechnung `q_i(t)`, kumulatives `P_enc(t)`, Ranking der Checkpoints, Isochronen.  
- **Deliverables**: `checkpoints.py`, CSV-Reports.  
- **Akzeptanz**: Stimmige Zeitreihen & Plausibilitäten.

### SP7 – 2D-Visualisierung & Heatmaps
- **Ziele**: Interaktive 2D-Heatmaps (HTML) und statische PNGs; Kantenfärbung nach Übergangswahrscheinlichkeit.  
- **Deliverables**: `viz/heatmap_2d.py`.  
- **Akzeptanz**: Export je *t* & Profil, lesbare Legenden.

### SP8 – 3D-Stadtvisualisierung
- **Ziele**: Extrudierte Gebäude, Straßen-Splines, Trails (5 Farben, Zeit-Fade), Checkpoint-Sphären mit Tooltips.  
- **Deliverables**: `viz/trails_3d.py`, GLTF/HTML-Export.  
- **Akzeptanz**: Kurzer 3D-Walkthrough/MP4; klare Profil-Unterscheidung.

### SP9 – Konfiguration & CLI/Minimal-UI
- **Ziele**: JSON-Konfigs (Global, Profile, RL), CLI-Kommandos; Param-Formular (optional).  
- **Deliverables**: `cfg/*.json`, `main.py`.  
- **Akzeptanz**: Ein-Kommando-Run (E2E), Validierung der Config.

### SP10 – Evaluation, QA & Doku
- **Ziele**: Metriken, Sensitivitäten, README/How-To, reproduzierbare Beispiele.  
- **Deliverables**: `notebooks/01_mvp_demo.ipynb`, `docs/`.  
- **Akzeptanz**: Report mit Kennzahlen & bekannten Grenzen.

---

## Daten & Annahmen

- **Stadt**: parametrische, fiktive Grid-Stadt (Größe, Dichte, Gebäudeflächen, syn. Steigung).  
- **Startpunkt**: fix (z. B. „Prison_01“) mit Startzeit *t₀*.  
- **Checkpoints**: Kreise mit Radius *rᵢ*, Sensitivität *sᵢ*.  
- **Zeitscheiben**: Δt (z. B. 2 min), Horizont (z. B. 240 min).  
- **Keine realen Daten** (keine OSM/DEM, es sei denn rein synthetisch erzeugt).

---

## Charakterprofile (5 Standard-Personas)

1. **Impulsiv & risikofreudig (jung)** – hohe Exploration, niedrige Risikoaversion, schnelle Korridore, optionaler POI-Trigger.  
2. **Strategin (mittleres Alter)** – hoher Planungshorizont, niedrige Exploration, minimiert Risiko-Fläche & Steigung.  
3. **Suchtgetrieben** – moderates γ, POI-Attraction (gekappt), mögliche Umwege.  
4. **Familiengebunden** – kurzfristiger Schlenker Richtung Bindung, danach Ausbruchskorridor.  
5. **Gesundheitlich eingeschränkt (älter)** – hohe Steigungskosten, geringere v_max, Präferenz flacher Hauptachsen.

---

## Formale Modelle (Kurzüberblick)

**Traits → RL/Reward**  
- Discount `γ`, Entropie-Koeff. `α_ent`, Gewichte `w_*`, Max-Speed aus `age, imp, plan, health, topo, risk, add, fam, mot`.  
- Beispiel: `γ = 0.90 + 0.09 * sigmoid((age−35)/10) * plan`  
- `w_detect = w0 * (1 − risk)`; `w_slope = w0 * (topo + (1 − health))`.

**Reward (pro Schritt)**  
