# Pharma Stock Prediction Using Clinical Trial Data and Stock Data
TODO:
1. [ ] make something with eventstudies like Budenny
2. [ ] changes dataset: reduce the amount of times/events

## General description and goal of the project
This project aims to explore the relationship between pharmaceutical clinical trials and stock market performance. We use clinical trial data from [clinicaltrials.gov](https://clinicaltrials.gov) and stock price data from the [Alpaca API](https://alpaca.markets) to develop a model that can predict the stock performance of pharmaceutical companies based on their clinical trial activities.

### Goal:
The goal of this project is to build a machine learning model that can predict short-term price movements of pharma stocks based on clinical trial outcomes, milestones, and other related data.

## Summary and results across experiments

- **Experiment 1** focused on using past clinical trial success/failure data and basic stock price movement trends to predict short-term price changes.
- **Experiment 2** expanded the dataset by incorporating additional features
---

### Research
This project aims to explore the relationship between pharmaceutical clinical trials and stock market performance. 

The relationship between pharmaceutical clinical trials and stock market performance is significant. Clinical trial announcements, especially those with positive outcomes, can lead to increased stock prices for pharmaceutical companies. For instance, positive trial results often result in a median cumulative abnormal return of 0.8% on the announcement day, while negative results can lead to a decrease of 2.0% [Hwang, 2013].

The impact varies by trial phase and disease type, but generally, the market reacts more strongly to negative news than positive [Singh, 2022]. Additionally, the stock market's response to clinical trial announcements can be influenced by factors such as the company's drug portfolio size and the network effect of related events [Budennyy, 2022].

During the COVID-19 pandemic, vaccine trial announcements significantly boosted stock market performance, reflecting investor optimism [Chan, 2021]. Overall, clinical trial outcomes are critical events that can substantially affect a company's market valuation.

Hwang, T (2013). Stock Market Returns and Clinical Trial Results of Investigational Compounds: An Event Study Analysis of Large Biopharmaceutical Companies. PLoS ONE, 8. https://doi.org/10.1371/journal.pone.0071966

Singh, M et al. (2022). The reaction of sponsor stock prices to clinical trial outcomes: An event study analysis. PLoS ONE, 17. https://doi.org/10.1371/journal.pone.0272851

Budennyy, S et al. (2022). New drugs and stock market: how to predict pharma market reaction to clinical trial announcements. ArXiv, abs/2208.07248. https://doi.org/10.48550/arXiv.2208.07248

Chan, K F et al. (2021). COVID-19 Vaccines: Saving Lives and the Global Stock Markets. SSRN Electronic Journal. https://doi.org/10.2139/SSRN.3785533


---

## Experiment 1
### Data
- **Clinical trial data**: Sourced from [clinicaltrials.gov](https://clinicaltrials.gov). This includes information such as trial phases, enrollment numbers, and outcomes (success/failure).

| NCT Number   | Study Title                                                                                              | Study URL                                              | Acronym | Study Status | Study Results | Interventions                                                                                     | Sponsor | Collaborators | Sex  | Age                   | Phases  | Enrollment | Study Type     | Study Design                                                                                                   | Start Date | Primary Completion Date | Completion Date | First Posted | Results First Posted | Last Update Posted |
|--------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------|---------|--------------|---------------|--------------------------------------------------------------------------------------------------|---------|---------------|-------|-----------------------|---------|------------|----------------|---------------------------------------------------------------------------------------------------------------|------------|--------------------------|-----------------|--------------|----------------------|---------------------|
| NCT06511973  | A Study to Learn About Whether BAYH006689 Causes Skin Irritation When Applied as a Topical Gel in Healthy Participants | https://clinicaltrials.gov/study/NCT06511973          |         | COMPLETED    | NO            | DRUG: Naproxen (BAYH006689)\|DRUG: Placebo Gel\|DRUG: A solution of 0.9% Saline\|DRUG: A solution of 0.2% SLS | Bayer   |               | ALL   | ADULT, OLDER_ADULT    | PHASE1  | 42         | INTERVENTIONAL | Allocation: RANDOMIZED\|Intervention Model: PARALLEL\|Masking: TRIPLE (PARTICIPANT, INVESTIGATOR, OUTCOMES_ASSESSOR)\|Primary Purpose: OTHER | 2024-07-23 | 2024-08-20              | 2024-08-20      | 2024-07-22   |                      | 2024-09-19         |


- **Stock data**: Historical stock price data of Bayer was obtained from [Investing.com](https://www.investing.com/equities/bayer-ag-historical-data) 

| Date       | Price | Open  | High  | Low   | Vol. | Change % |
|-----------|-------|-------|-------|-------|------|-------------|
| 02/04/2009 | 42.21 | 42.91 | 43.28 | 42.21 |      | -1.86%   |
| 02/05/2009 | 42.75 | 41.96 | 42.75 | 41.51 |      | 1.28%    |

### Features
Hier eine Übersicht des Feature Sets nach der Datenaufbereitung

| Price | Open  | High  | Low   | Vol. | Enrollment | Start Date_timestamp | Start Date_days_since_ref | Start Date_year | Start Date_month | Start Date_day | Primary Completion Date_timestamp | Primary Completion Date_days_since_ref | Primary Completion Date_year | Primary Completion Date_month | Primary Completion Date_day | Completion Date_timestamp | Completion Date_days_since_ref | Completion Date_year | Completion Date_month | Completion Date_day | First Posted_timestamp | First Posted_days_since_ref | First Posted_year | First Posted_month | First Posted_day | Last Update Posted_timestamp | Last Update Posted_days_since_ref | Last Update Posted_year | Last Update Posted_month | Last Update Posted_day | Sex_ALL | Sex_FEMALE | Sex_MALE | Phases_PHASE1 | Phases_PHASE1\|PHASE2 | Phases_PHASE2 | Phases_PHASE2\|PHASE3 | Phases_PHASE3 | Age_ADULT | Age_ADULT, OLDER_ADULT | Age_CHILD | Age_CHILD, ADULT | Age_CHILD, ADULT, OLDER_ADULT | Age_OLDER_ADULT | Study Results_NO | Study Results_YES | Study Status_COMPLETED |
|-------|-------|-------|-------|------|------------|-----------------------|---------------------------|-----------------|------------------|---------------|----------------------------------|----------------------------------------|-----------------------------|--------------------------|-----------------------|-------------------------|-----------------------------|------------------|------------------|----------------|-------------------------|------------------------|-----------------|-------------------|----------------|--------------------------------|----------------------------------|---------------------|---------------------|--------------------|---------|------------|----------|---------------|-------------------------|---------------|-----------------------|---------------|-----------|-------------------------|----------|------------------|-----------------------------|-----------------|------------------|-------------------|-------------------------|
| 40.61 | 41.5  | 41.98 | 40.61 | 0.0  | 262.0      | 0.9555648497982487    | 0.32820145602100487       | 0.3478260869565162 | 0.09090909090909091 | 0.7333333333333333 | 0.9643809228577862                | 0.42147066359210866                    | 0.4285714285714306           | 0.4545454545454545         | 0.8666666666666667     | 0.9643809228577862      | 0.42147066359210866          | 0.4285714285714306 | 0.4545454545454545 | 0.8666666666666667 | 0.3279512020093289     | 0.32795120200932903      | 0.3478260869565162 | 0.1818181818181818 | 0.1            | 0.9485117788122444           | 0.9485117788122442              | 0.9473684210526301       | 0.9090909090909091       | 0.6666666666666666      | True    | False      | False    | False         | False                  | False         | False                | True          | False     | True                   | False    | False            | False                       | False           | True             | True              | True                    |

### LSTM Input Parameter
| Parameter                 | Value                                                               |
| ------------------------- | ------------------------------------------------------------------- |
| Input Size                | 148 (Number of features)                                            |
| Hidden Size               | 512 (Number of LSTM units per layer)                                |
| Number of Layers          | 20 (Number of stacked LSTM layers)                                  |
| Output Size               | 1 (Predicting a single value, e.g., stock price)                    |
| Epochs                    | 1000                                                                |
| Learning Rate             | 0.003                                                               |
| Data Filtering Criteria   | Rows where `Date` is between `Start Date` and `Start Date + 7 days` |
| Target Variable           | `Change %`                                                          |
| Data Preparation Function | `predictor.load_and_prepare_data_v2(window=7)`                      |

### Ergebnis

#### Kennwerte
##### **Target Range**
- **Min = -0.1307**, **Max = 0.0724**
    - Dies ist der Wertebereich der Zielvariable (`targets`) in den Trainingsdaten.
    - Es zeigt, dass das Modell versucht, Werte in diesem Bereich vorherzusagen.
    - Negative Werte können beispielsweise auf Kursrückgänge (z. B. prozentuale Änderungen) hinweisen, positive Werte auf Kursanstiege.

##### **Loss Values**
- **Training Loss**: Verlustwert auf den Trainingsdaten.
- **Test Loss**: Verlustwert auf den Testdaten.
- **Beispiele aus den Epochen**:
    - Epoch [10]: Loss: 0.0004 (Training), Test Loss: 0.0003
    - Epoch [20]: Loss: 0.0004 (Training), Test Loss: 0.0028
    - Epoch [100]: Loss: 0.0003 (Training), Test Loss: 0.0003

**Interpretation**:
- Der **Loss** misst, wie gut das Modell Vorhersagen im Vergleich zu den tatsächlichen Werten macht.
- Ein niedriger **Loss** zeigt an, dass das Modell die Daten gut anpasst.
- **Auffälligkeit bei Epoch [20]:** Der Test Loss steigt stark (0.0028), kehrt aber zu einem niedrigen Wert zurück. Das könnte ein vorübergehendes Overfitting sein, das durch die Optimierung abnimmt.
##### **Mean Absolute Error**
- **Wert: 0.0116**
    - Der durchschnittliche absolute Fehler zwischen den vorhergesagten und den tatsächlichen Werten.
    - Ein MAE von 0.0116 bedeutet, dass die durchschnittliche Abweichung der Vorhersagen vom Zielwert etwa **1.16 %** beträgt (bei normierten Daten).

**Interpretation**:
- Je kleiner der MAE, desto präziser sind die Vorhersagen.
- Ein MAE von 0.0116 bei normierten Daten ist ein guter Wert.
Mean Squared Error
- **Wert: 0.000271**
    - Der durchschnittliche quadratische Fehler zwischen den vorhergesagten und tatsächlichen Werten.
    - Quadratische Fehler gewichten größere Abweichungen stärker als kleine Abweichungen.
**Interpretation**:
- Niedrigere MSE-Werte zeigen bessere Modellleistung.
- Im Vergleich zum MAE ist die MSE empfindlicher gegenüber großen Fehlern.
##### **R² Score**
- **Wert: -0.0002**
    - Misst, wie gut die Vorhersagen des Modells die Varianz der Zielvariable erklären.
    - Ein R² von 1 bedeutet perfekte Vorhersagen, während ein R² von 0 bedeutet, dass das Modell keine bessere Vorhersage als der Mittelwert macht.
    - Ein negativer R²-Wert (-0.0002) bedeutet, dass das Modell **leicht schlechter** abschneidet als eine einfache Mittelwertsvorhersage.

**Interpretation**:
- Ein negativer R²-Wert ist ungewöhnlich bei geringen Fehlern wie MAE und MSE. Es könnte darauf hinweisen, dass die Varianz der Zielvariable sehr klein ist (d.h., die Werte sind nah am Mittelwert).
- **Zusammenhang prüfen:** Zielwerte (Change %) haben möglicherweise eine geringe Streuung, was den R²-Wert beeinflusst.

### Fazit



---

## Experiment 2

---
### Data

Wie bei 1.
### Features
Hier eine Übersicht des Feature Sets nach der Datenaufbereitung

| Price | Open  | High  | Low   | Vol. | Enrollment | Start Date_timestamp | Start Date_days_since_ref | Start Date_year | Start Date_month | Start Date_day | Primary Completion Date_timestamp | Primary Completion Date_days_since_ref | Primary Completion Date_year | Primary Completion Date_month | Primary Completion Date_day | Completion Date_timestamp | Completion Date_days_since_ref | Completion Date_year | Completion Date_month | Completion Date_day | First Posted_timestamp | First Posted_days_since_ref | First Posted_year | First Posted_month | First Posted_day | Last Update Posted_timestamp | Last Update Posted_days_since_ref | Last Update Posted_year | Last Update Posted_month | Last Update Posted_day | Sex_ALL | Sex_FEMALE | Sex_MALE | Phases_PHASE1 | Phases_PHASE1\|PHASE2 | Phases_PHASE2 | Phases_PHASE2\|PHASE3 | Phases_PHASE3 | Age_ADULT | Age_ADULT, OLDER_ADULT | Age_CHILD | Age_CHILD, ADULT | Age_CHILD, ADULT, OLDER_ADULT | Age_OLDER_ADULT | Study Results_NO | Study Results_YES | Study Status_COMPLETED |
|-------|-------|-------|-------|------|------------|-----------------------|---------------------------|-----------------|------------------|---------------|----------------------------------|----------------------------------------|-----------------------------|--------------------------|-----------------------|-------------------------|-----------------------------|------------------|------------------|----------------|-------------------------|------------------------|-----------------|-------------------|----------------|--------------------------------|----------------------------------|---------------------|---------------------|--------------------|---------|------------|----------|---------------|-------------------------|---------------|-----------------------|---------------|-----------|-------------------------|----------|------------------|-----------------------------|-----------------|------------------|-------------------|-------------------------|
| 40.61 | 41.5  | 41.98 | 40.61 | 0.0  | 262.0      | 0.9555648497982487    | 0.32820145602100487       | 0.3478260869565162 | 0.09090909090909091 | 0.7333333333333333 | 0.9643809228577862                | 0.42147066359210866                    | 0.4285714285714306           | 0.4545454545454545         | 0.8666666666666667     | 0.9643809228577862      | 0.42147066359210866          | 0.4285714285714306 | 0.4545454545454545 | 0.8666666666666667 | 0.3279512020093289     | 0.32795120200932903      | 0.3478260869565162 | 0.1818181818181818 | 0.1            | 0.9485117788122444           | 0.9485117788122442              | 0.9473684210526301       | 0.9090909090909091       | 0.6666666666666666      | True    | False      | False    | False         | False                  | False         | False                | True          | False     | True                   | False    | False            | False                       | False           | True             | True              | True                    |

### LSTM Input Parameter
| Parameter                 | Value                                                               |
| ------------------------- | ------------------------------------------------------------------- |
| Input Size                | 148 (Number of features)                                            |
| Hidden Size               | 512 (Number of LSTM units per layer)                                |
| Number of Layers          | 20 (Number of stacked LSTM layers)                                  |
| Output Size               | 1 (Predicting a single value, e.g., stock price)                    |
| Epochs                    | 1000                                                                |
| Learning Rate             | 0.003                                                               |
| Data Filtering Criteria   | Rows where `Date` is between `Start Date` and `Start Date + 3 days` |
| Target Variable           | `Change %`                                                          |
| Data Preparation Function | `predictor.load_and_prepare_data_v2(window=3)`                      |

### Ergebnis

![[Pasted image 20241129054645.png]]

#### Kennwerte
### Loss *

- **Trainingsverlust:** Dies ist die Fehlerrate, die das Modell auf den Trainingsdaten erreicht hat. Es zeigt, wie gut das Modell die Muster in den Trainingsdaten gelernt hat.
- **Testverlust:** Zeigt die Fehlerrate des Modells auf den Testdaten (Daten, die nicht während des Trainings verwendet wurden). Ein niedriger Testverlust zeigt, dass das Modell generalisiert und nicht überangepasst ist.

Im Verlauf der Epochen (Trainingsepochen) sinken sowohl Trainings- als auch Testverlust stetig, was ein Zeichen für einen gut funktionierenden Lernprozess ist. Idealerweise sollten sich beide Werte angleichen.

---

### **2. MAE :**

- Der MAE misst die durchschnittliche absolute Abweichung zwischen den vorhergesagten und den tatsächlichen Werten.
- **Interpretation:** Je niedriger der MAE, desto besser sind die Vorhersagen des Modells.
- In diesem Fall beträgt der MAE **0.0121**, was bedeutet, dass die durchschnittliche Abweichung der Vorhersagen vom tatsächlichen Wert etwa 0.0121 (in der Einheit der Zielgröße) beträgt.

---

### **3. MSE :**

- Der MSE ist das Mittel der quadrierten Abweichungen zwischen den vorhergesagten und den tatsächlichen Werten.
- **Interpretation:** Ein niedriger MSE zeigt an, dass die Vorhersagen des Modells nahe an den tatsächlichen Werten liegen.
- Der MSE ist empfindlicher gegenüber Ausreißern, da Abweichungen quadriert werden. In diesem Fall beträgt der MSE **0.000287**, was ebenfalls auf eine gute Modellleistung hinweist.

---

### **4. R² (Bestimmtheitsmaß, R-Squared):**

- Das R² misst, wie gut die Varianz der Zielgröße durch die unabhängigen Variablen erklärt wird. Es liegt im Bereich von 0 bis 1 (oder kann negativ sein, wenn das Modell schlechter ist als ein simpler Durchschnittswert).
    - **R² = 1:** Perfekte Vorhersage.
    - **R² = 0:** Das Modell macht zufällige Vorhersagen, ohne einen Zusammenhang zu erklären.
    - **R² < 0:** Das Modell ist schlechter als eine simple Durchschnittsvorhersage.
- **Interpretation:** In diesem Fall ist das R² negativ (-0.059), was darauf hinweist, dass das Modell die Varianz in den Testdaten schlecht erklärt. Es deutet darauf hin, dass das Modell auf den Testdaten möglicherweise nicht gut generalisiert oder dass das Datenset weitere Anpassungen benötigt.