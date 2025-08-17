# iManhunt — Mathematische & Theoretische Herleitung (SP1–SP3, konzeptuell)

> Umgebung: Anaconda (Dev-Env **iManhunt**).  
> Ziel: Formale Grundlegung für eine Simulation zeitabhängiger Fluchtbewegungen in einer fiktiven 3D-Stadt – **ohne Code**, mit Skizzenhinweisen.

---

## 1) Problemformulierung

Gegeben ist eine synthetische Stadt als gerichteter Graph \(G=(V,E)\) mit Knoten \(v\in V\) (Kreuzungen/POIs) und Kanten \(e=(v\!\to\!u)\in E\) (Straßen/Wege). Ein Akteur startet am Knoten \(v_0\) zur Zeit \(t_0\). Gesucht sind:

1. **Zeitabhängige Aufenthaltswahrscheinlichkeiten** \(p_t(v)\) über den Knoten.  
2. **Routenfelder** (Pfaddichten) für **5 Persönlichkeitsprofile**.  
3. **Antreff-/Begegnungswahrscheinlichkeit** an **Checkpoints** \(C_i\) (Radius \(r_i\), Sensitivität \(s_i\)) je Zeitfenster.  
4. **Heatmaps** (2D/3D) und **Top-Korridore**.

---

## 2) Stadt- & Bewegungsmodell

### 2.1 Geometrie & Topografie
- Knotenkoordinaten \((x_v,y_v,z_v)\); Höhen \(z_v\) stammen aus einem synthetischen DEM.  
- Kantenlänge \(L_e=\| (x_v,y_v)-(x_u,y_u)\|\).  
- **Steigung** (signiert) je Kante:
\[
\text{grade}_e=\frac{z_u - z_v}{L_e},\qquad \text{slope\%}_e=100\cdot \text{grade}_e.
\]

### 2.2 Geschwindigkeit & Reisezeit (Fuß/zu Fuß als Basismodell)
- Basisspeed nach Straßentyp: \(v_0(\text{arterial})\ge v_0(\text{residential})\ge v_0(\text{footpath})\).
- Steigungseinfluss (einfaches Energiemodell):
\[
v_e=\begin{cases}
v_0\cdot e^{-\alpha\,\text{grade}_e}, & \text{grade}_e\ge 0 \\
v_0\cdot(1+\min\{b\,|\text{grade}_e|,\,b_{\max}\}), & \text{grade}_e<0
\end{cases}
\]
- Reisezeit auf \(e\):
\[
\tau_e = \frac{L_e}{v_e}.
\]

**Skizze A (Text):** Gitterstadt mit Höhenlinien; Pfeile zeigen Kanten. Farbcodierung der Kanten nach \(v_e\) (schnell/ langsam).

---

## 3) MDP-Formulierung & RL-Grundlagen

Wir modellieren die Entscheidung als **Markov Decision Process (MDP)** auf \(G\).

- **Zustand** \(s_t=(v,\phi_t)\): aktueller Knoten \(v\) und Zustandsvektor \(\phi_t\) (Uhrzeit, Fatigue, lokale Risikoschätzung, ggf. Wetter).  
- **Aktion** \(a_t\in \mathcal{A}(v)\): Wahl einer ausgehenden Kante \(e=(v\!\to\!u)\).  
- **Übergang**: \(s_{t+\Delta t}=(u,\phi_{t+\Delta t})\) nach Traversalzeit \(\tau_e\).  
- **Politik** \(\pi_\theta(a|s)\) (stochastisch, z. B. softmax/entropiereguliert).  
- **Discount** \(\gamma\in(0,1)\).

### 3.1 Reward-Komposition (geschachtelt)
\[
R_t= w_{\text{dist}}\cdot \Delta \text{RadialDist}
- w_{\text{time}}\cdot \Delta t
- w_{\text{slope}}\cdot \text{EnergyCost}
- w_{\text{detect}}\cdot \text{DetectRisk}
+ w_{\text{goal}}\cdot \text{GoalPot}
+ w_{\text{POI}}\cdot(\text{POI}_{\text{attr}}-\text{POI}_{\text{rep}})
- w_{\text{fatigue}}\cdot \text{Fatigue}.
\]

- \(\Delta \text{RadialDist}\): Zuwachs der Entfernung zum Start/„Grenze“ (Langstrecke).  
- \(\text{EnergyCost}\): Funktion von \(L_e\), \(\text{grade}_e\), Schrittfrequenz (annähernd proportional zu Steigungsarbeit).  
- \(\text{DetectRisk}\): Aufenthaltszeit in Zonen hoher Sichtbarkeit / Checkpoints; s. §5.  
- \(\text{GoalPot}\): Potentialfeld für Motiv (Stadt verlassen, Untertauchen, Gewässernähe).  
- \(\text{POI}\): Sucht-/Familien-Trigger als schwache Attraktoren/Repeller.  
- \(\text{Fatigue}\): kumulative Ermüdung → reduziert effektive \(v_e\) im Zeitverlauf.

### 3.2 Bellman-Gleichung (diskret)
\[
V^\pi(s)=\mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k}\,\big|\,s_t=s\right],\quad
\pi^\*\in\arg\max_\pi V^\pi.
\]
Entropie-Regularisierung (MaxEnt-RL) fügt \(\alpha H(\pi(\cdot|s))\) hinzu → explorativere Policies.

**Skizze B (Text):** Lokale Entscheidungsstern-Grafik am Knoten \(v\): ausgehende Kanten mit Balken \(U_e\) (Nettonutzen); Softmax-Wahrscheinlichkeiten als Pfeildicke.

---

## 4) Zeitmodell: Semi-Markov / kontinuierliche Zeit

Diskrete Kantenwahl mit edge-spezifischer **Verweilzeit** \(\tau_e\) entspricht einem **Semi-Markov-Prozess**. Alternativ: kontinuierliches Modell mit **Generator** \(Q\):

- **Raten** \(r_{v\to u}\propto \kappa\cdot \exp\{\beta\,U_e\}\), normiert über Nachbarn.  
- **Generator**: \(Q_{vu}=r_{v\to u}\) für \(u\neq v\), \(Q_{vv}=-\sum_{u\neq v} r_{v\to u}\).  
- **Mastergleichung**:
\[
\frac{d}{dt}p_t = p_t\,Q,\qquad p_0=\delta_{v_0}.
\]
Die Parameter \(\beta\) (Rationalität) und \(\kappa\) (Zeitskala) werden aus \(\tau_e\) kalibriert.

**Skizze C (Text):** Balkendiagramm der Raten \(r_{v\to u}\); darunter Zeitachsen-Skizze für Semi-Markov-Verweilzeiten.

---

## 5) Detektions- & Begegnungswahrscheinlichkeit

Sei \(C_i\) ein Checkpoint mit Radius \(r_i\) und Sensitivität \(s_i\). Definiere den **graph-basierten Ball** \(B_i(r_i)\subseteq V\) (Knoten in Reichweite).

- Momentane Trefferchance:
\[
q_i(t) = s_i\sum_{v\in B_i(r_i)} p_t(v).
\]
- Aggregiert über ein kleines Fenster \(\Delta t\) (näherungsweise unabhängig):
\[
P_{\text{enc}}(t,\Delta t)=1-\prod_i (1-q_i(t)).
\]
Alternative kontinuierliche Hazard-Formulierung:
\[
\frac{d}{dt}\,P_{\text{hit}}(t) = \lambda(t)\,[1-P_{\text{hit}}(t)],\quad
\lambda(t)=\sum_i \lambda_i(t),\ \lambda_i(t)\approx \frac{q_i(t)}{\Delta t}.
\]

**Skizze D (Text):** Karte mit Checkpoint-Kreisen; Heatmap \(p_t\) darunter; Kurve \(P_{\text{hit}}(t)\) als aufsteigende S-Form.

---

## 6) Bayesianisches Update durch Sichtungen (optional)

Sichtung \(o=(x_o,t_o)\) mit Unsicherheit \(\sigma\). Likelihood am Knoten \(v\):
\[
\mathcal{L}(o|v)\propto \exp\!\left(-\frac{d_G(v,x_o)^2}{2\sigma^2}\right),
\]
wobei \(d_G\) die graphbasierte Distanz ist. Update:
\[
\tilde p_{t_o}(v) \propto p_{t_o}(v)\cdot \mathcal{L}(o|v),\quad
p_{t_o}=\frac{\tilde p_{t_o}}{\sum_w \tilde p_{t_o}(w)}.
\]
Mehrere Meldungen: Produkt der Likelihoods mit ggf. Fehlalarmmodell (gemischte Likelihood).

**Skizze E (Text):** Vor-/Nach-Update-Heatmap; Sichtung als roter Marker; lokale Verstärkung der Masse.

---

## 7) Persönlichkeitsprofile (Traits → RL/Reward-Parameter)

Traits \(\theta=\{\text{age},\text{imp},\text{plan},\text{health},\text{topo},\text{risk},\text{add},\text{fam},\text{mot}\}\).

**Beispielhafte Mappings:**
- **Horizont**: \(\gamma = 0.90 + 0.09\cdot \sigma((\text{age}-35)/10)\cdot \text{plan}\).  
- **Exploration**: \(\alpha_{\text{ent}}=\alpha_0+c_1\cdot \text{imp}\).  
- **Steigungskosten**: \(w_{\text{slope}}=w_0\cdot(\text{topo}+1-\text{health})\).  
- **Risikoaversion**: \(w_{\text{detect}}=w_0\cdot(1-\text{risk})\).  
- **Max-Speed**: skaliert mit **health**, gedämpft durch **topo**.  
- **POI-Term**: \(w_{\text{POI}}\) moduliert durch **add**/**fam** (kappen!).

**Interpretation:**  
- **Impulsiv** → höhere Exploration, geringere Risikoaversion, mehr Kurzstrecken-Sprints.  
- **Strategisch** → höheres \(\gamma\), glättet Pfade, meidet riskante Flächen.  
- **Gesundheitlich eingeschränkt** → hohe Steigungskosten, längere Reisezeiten.

**Skizze F (Text):** Radar-Chart der 5 Profile mit unterschiedlichen Trait-Ausprägungen.

---

## 8) Pfaddichten, Heatmaps & Isochronen

### 8.1 Occupancy Measure (Aufenthaltsmaß)
Aus Monte-Carlo-Rollouts (oder Mastergleichung) ergibt sich \(p_t(v)\). Eine **Kantenflussdichte** kann über Zählmaße \(\rho_t(e)\) (Durchtritte je Zeiteinheit) definiert werden.

### 8.2 Glättung auf Graphen
Heatmap-Rendering: Kernel-Smooting auf \(G\) (z. B. Heat-Kernel \(K_t=\exp(t\Delta_G)\) mit Graph-Laplacian \(\Delta_G\)) oder einfache Nachbarschaftsfaltung.

### 8.3 Isochronen
Erreichbare Mengen \(\mathcal{R}(t)=\{v\mid d_\tau(v_0,v)\le t\}\) auf Basis der reisezeitgewichteten Distanz \(d_\tau\).

**Skizze G (Text):** Drei Zeitstempel \(t_1<t_2<t_3\): wachsende Isochronen; Heatmap-Intensität verschiebt sich radial.

---

## 9) Checkpoint-Optimierung (optional)

Ziel: Wähle \(K\) Checkpoints zur Maximierung der erwarteten **Trefferwahrscheinlichkeit** bis \(T\):
\[
\max_{\{C_1,\dots,C_K\}} \quad \mathbb{E}\big[P_{\text{hit}}(T)\big].
\]
Heuristik: Greedy nach **marginalem Zugewinn** \(\Delta P_{\text{hit}}\) (submodulare Annäherung oft gut). Nebenbedingungen: Deckung, Abstand, Ressourcen.

**Skizze H (Text):** Heatmap + Kandidatenpunkte; iterative Auswahl mit schwindendem Grenznutzen.

---

## 10) Validierung & Sensitivität (Überlegungen)

- **Ablation**: ohne Steigung, ohne Risiko, ohne POI → Vergleich der Hotspots.  
- **Sensitivität**: Variation einzelner Traits → Verschiebung von Routen/Checkpoints.  
- **Diversität**: Jensen–Shannon-Divergenz zwischen Profil-Heatmaps.  
- **Plausibilität**: Höhere \(w_{\text{detect}}\) → stärkere Meidung von Detektionszonen; höhere **topo**/**low health** → Meidung steiler Korridore.  
- **Ethik/Sicherheit**: rein fiktive Stadt; schwache POI-Attraktion; keine realen Daten.

---

## 11) Theoretischer MVP-Fahrplan (ohne Code)

1. **SP1: Graph & Attribute**  
   Festlegen von \(G\), \(L_e\), \(\text{grade}_e\), \(v_e\), \(\tau_e\).  
   *Ergebnis:* konsistenter Semi-Markov-Unterbau.

2. **SP2: Traits & Mapping**  
   Definition der fünf Profile; Parametertabellen \(\to\{\gamma,\alpha_{\text{ent}},w_*,v_{\max}\}\).

3. **SP3: Reward-Composer**  
   Kalibrierte Gewichte; Caps/Regularisierung gegen Reward-Hacking.

4. **SP4: Policy-Entstehung (konzeptuell)**  
   Entweder RL-Training (PPO/MaxEnt) **oder** analytische Softmax-Wahl via Nutzen \(U_e\).

5. **SP5: Propagation**  
   Rollouts \(\Rightarrow\) \(p_t(v)\), \(\rho_t(e)\); alternativ Mastergleichung \(dp/dt=pQ\).

6. **SP6: Checkpoints & Begegnung**  
   \(q_i(t)\), \(P_{\text{enc}}(t,\Delta t)\), Ranking; **Isochronen** als Kontext.

7. **SP7–SP8: Visual-Konzept**  
   2D-Heatmaps (Zeit-Slider), 3D-Trails (farbcodierte Profile), Tooltips.

---

## 12) Skizzenübersicht (verbale Leitfäden)

- **A:** Gitterstadt mit Höhenlinien und gefärbten Kanten (Geschwindigkeit).  
- **B:** Entscheidungsstern am Knoten: Nutzenspalten pro Kante, Softmax-Pfeildicken.  
- **C:** Zeitachse mit Semi-Markov-Verweilzeiten, Ratenpfeile \(r_{v\to u}\).  
- **D:** Checkpoint-Kreise + Heatmap + Kurve \(P_{\text{hit}}(t)\).  
- **E:** Vor-/Nach-Bayes-Update-Heatmap bei Sichtung.  
- **F:** Radar-Chart der fünf Profile.  
- **G:** Isochronen \(t_1,t_2,t_3\) + wachsende Erreichbarkeitsfront.  
- **H:** Greedy-Auswahl von Checkpoints (marginaler Zugewinn skizziert).

---

## 13) Annahmen & Grenzen

- Vereinfachte Dynamik (Fußweg, keine Fahrzeuge/ÖPNV).  
- Stationäre Sichtbarkeitsfelder (keine Gegenspieler-Strategie).  
- Trait-Mapping heuristisch (später kalibrierbar).  
- Synthetische Topografie/POIs; keine Realdaten.

---

## 14) Kernaussagen

- iManhunt basiert auf einem **Semi-Markov-MDP auf Graphen**.  
- **Traits** steuern **Policy-Form** via \(\gamma,\alpha_{\text{ent}},w_*,v_{\max}\).  
- **Heatmaps** entstehen aus **Occupancy-Maßen** \(p_t\) bzw. **Mastergleichung**.  
- **Begegnungswahrscheinlichkeiten** folgen aus örtlicher Überdeckung und Sensitivitäten.  
- Das Konzept ist **modular** (SP1–SP10) und **skalierbar** (mehr Modalitäten, Gegenstrategien).

---
