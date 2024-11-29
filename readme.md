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

---

### Features


| Price | Open  | High  | Low   | Vol. | Enrollment | Completion Date_timestamp | Completion Date_days_since_ref | Completion Date_year | Completion Date_month | Completion Date_day | First Posted_timestamp | First Posted_days_since_ref | First Posted_year   | First Posted_month  | First Posted_day   | Completion Date_timestamp | Completion Date_days_since_ref | Completion Date_year | Completion Date_month | Completion Date_day | First Posted_timestamp | First Posted_days_since_ref | First Posted_year   | First Posted_month  | First Posted_day   | Sex_ALL | Sex_FEMALE | Sex_MALE | Phases_PHASE1 | Phases_PHASE1 | PHASE2 | Phases_PHASE2 | Phases_PHASE2 | PHASE3 | Phases_PHASE3 | Age_ADULT | Age_ADULT, OLDER_ADULT | Age_CHILD | Age_CHILD, ADULT | Age_CHILD, ADULT, OLDER_ADULT | Age_OLDER_ADULT | Study Results_NO | Study Results_YES | Study Status_COMPLETED |
| ----- | ----- | ----- | ----- | ---- | ---------- | ------------------------- | ------------------------------ | -------------------- | --------------------- | ------------------- | ---------------------- | --------------------------- | ------------------- | ------------------- | ------------------ | ------------------------- | ------------------------------ | -------------------- | --------------------- | ------------------- | ---------------------- | --------------------------- | ------------------- | ------------------- | ------------------ | ------- | ---------- | -------- | ------------- | ------------- | ------ | ------------- | ------------- | ------ | ------------- | --------- | ---------------------- | --------- | ---------------- | ----------------------------- | --------------- | ---------------- | ----------------- | ---------------------- |
| 39.08 | 38.53 | 39.08 | 38.34 | 0.0  | 45.0       | 0.9549950278623844        | 0.2690238278247502             | 0.2857142857142776   | 0.1818181818181818    | 0.8                 | 0.20511900490371948    | 0.20511900490371965         | 0.21739130434782794 | 0.36363636363636365 | 0.3666666666666667 | 0.9549950278623844        | 0.2690238278247502             | 0.2857142857142776   | 0.1818181818181818    | 0.8                 | 0.20511900490371948    | 0.20511900490371965         | 0.21739130434782794 | 0.36363636363636365 | 0.3666666666666667 | False   | True       | False    | False         | False         | True   | False         | False         | False  | False         | True      | False                  | False     | True             | False                         | True            | False            | True              |                        |

##### Ergebnis

Epoch [10/100], Loss: 0.0014, Test Loss: 0.0003
Epoch [20/100], Loss: 0.0006, Test Loss: 0.0009
Epoch [30/100], Loss: 0.0005, Test Loss: 0.0005
Epoch [40/100], Loss: 0.0004, Test Loss: 0.0004
Epoch [50/100], Loss: 0.0004, Test Loss: 0.0003
...
Epoch [100/100], Loss: 0.0004, Test Loss: 0.0003
[0.14396969974040985, 0.0020105463918298483, 0.029588710516691208, 0.017405936494469643, 0.0024881656281650066, 0.002606841968372464, 0.008103483356535435, 0.005176508333534002, 0.001384328119456768, 0.00033087277552112937, 0.0011152330553159118, 0.0021656390745192766, 0.0025588292628526688, 0.0021987855434417725, 0.001453573815524578, 0.0007492545410059392, 0.00035318968002684414, 0.0003279401862528175, 0.0005715168663300574, 0.0008945006993599236, 0.0011148841585963964, 0.0011383402161300182, 0.000981713179498911, 0.000735066831111908, 0.0005002822726964951, 0.00034610353759489954, 0.0002935489756055176, 0.0003242186503484845, 0.00039847646257840097, 0.00047392002306878567, 0.0005190691445022821, 0.0005202398169785738, 0.0004815894353669137, 0.00041990887257270515, 0.00035677224514074624, 0.000310826872009784, 0.0002923836000263691, 0.0003014355606865138, 0.00032916184864006937, 0.00036203418858349323, 0.00038694788236171007, 0.0003955465217586607, 0.0003863005549646914, 0.0003638722700998187, 0.0003364150761626661, 0.00031216340721584857, 0.0002966801985166967, 0.0002915835939347744, 0.0002948591427411884, 0.00030230116681195796, 0.00030937048722989857, 0.0003127855889033526, 0.0003113893326371908, 0.0003061449097003788, 0.00029941293178126216, 0.0002938640827778727, 0.00029144674772396684, 0.00029275432461872697, 0.00029695939156226814, 0.00030227028764784336, 0.0003066836216021329, 0.0003087226068601012, 0.00030788881122134626, 0.0003046951605938375, 0.0003003274614457041, 0.00029611808713525534, 0.0002930605842266232, 0.0002915383956860751, 0.0002913296630140394, 0.00029183365404605865, 0.0002923911379184574, 0.0002925607259385288, 0.0002922526327893138, 0.0002916933153755963, 0.0002912638010457158, 0.0002912956988438964, 0.0002919150283560157, 0.0002929915499407798, 0.000294202211080119, 0.0002951693022623658, 0.000295608420856297, 0.00029542381525970995, 0.0002947201719507575, 0.00029373684083111584, 0.0002927430614363402, 0.0002919416583608836, 0.0002914183132816106, 0.00029114741482771933, 0.00029103984707035124, 0.0002910042239818722, 0.00029099182575009763, 0.00029100850224494934, 0.0002910948824137449, 0.00029128947062417865, 0.0002915961085818708, 0.00029197082039900124, 0.0002923336869571358, 0.00029259794973768294, 0.00029270158847793937, 0.00029262801399454474]
Loss Value Training: 0.00035307893995195627
Loss Value Test: 0.00029262801399454474
MAE: 0.012998974948588487
MSE: 0.0002926280175775518
R²: -0.009032370297524794


#### Sigmoid

###### Ergebnis
Epoch [10/100], Loss: 0.0687, Test Loss: 0.0632
Epoch [20/100], Loss: 0.0128, Test Loss: 0.0091
Epoch [30/100], Loss: 0.0047, Test Loss: 0.0039
Epoch [40/100], Loss: 0.0015, Test Loss: 0.0016
Epoch [50/100], Loss: 0.0005, Test Loss: 0.0006
Epoch [60/100], Loss: 0.0004, Test Loss: 0.0004
...
Epoch [100/100], Loss: 0.0004, Test Loss: 0.0003
[3.408458948135376, 0.16542644798755646, 0.13893724977970123, 0.30870988965034485, 0.19226916134357452, 0.05854929983615875, 0.0012520436430349946, 0.027994943782687187, 0.06822532415390015, 0.06318000704050064, 0.03507213294506073, 0.011598581448197365, 0.0012496740091592073, 0.0014568489277735353, 0.006872178521007299, 0.012652027420699596, 0.01584428735077381, 0.015711944550275803, 0.01301178801804781, 0.009079745039343834, 0.005186916328966618, 0.00222479528747499, 0.000620637962128967, 0.00037332094507291913, 0.0011507419403642416, 0.0024285821709781885, 0.003654166590422392, 0.004402045160531998, 0.004475621972233057, 0.003922923933714628, 0.002971345093101263, 0.0019196872599422932, 0.001034483895637095, 0.0004827066441066563, 0.00031012590625323355, 0.0004573999613057822, 0.0007985378615558147, 0.0011863010004162788, 0.0014924644492566586, 0.0016351898666471243, 0.001590109895914793, 0.0013857146259397268, 0.0010868830140680075, 0.0007723816088400781, 0.0005126115283928812, 0.00035290626692585647, 0.0003056743007618934, 0.0003522027691360563, 0.00045248062815517187, 0.0005594851681962609, 0.0006334512145258486, 0.0006520291208289564, 0.0006138579337857664, 0.0005353810847736895, 0.00044287866330705583, 0.0003629028797149658, 0.00031427296926267445, 0.0003037581918761134, 0.00032605999149382114, 0.0003673287865240127, 0.00041056654299609363, 0.0004410377296153456, 0.00045013794442638755, 0.0004368700901977718, 0.00040688810986466706, 0.0003697637584991753, 0.0003355392545927316, 0.00031168709392659366, 0.00030131914536468685, 0.00030300550861284137, 0.00031202530954033136, 0.0003224489919375628, 0.00032926618587225676, 0.0003298638912383467, 0.00032446434488520026, 0.00031554067390970886, 0.00030656345188617706, 0.0003006112528964877, 0.00029933644691482186, 0.00030258105834946036, 0.00030867033638060093, 0.0003151765267830342, 0.00031982490327209234, 0.0003212140582036227, 0.0003191404102835804, 0.0003144793154206127, 0.00030873031937517226, 0.0003034330438822508, 0.00029966916190460324, 0.0002978062257170677, 0.0002975309325847775, 0.0002981111756525934, 0.0002987548359669745, 0.00029891927260905504, 0.0002984667371492833, 0.00029763419297523797, 0.0002968601475004107, 0.0002965591265819967, 0.00029693898977711797, 0.00029792531859129667]
Loss Value Training: 0.00035968245356343687
Loss Value Test: 0.00029792531859129667
MAE: 0.013232369488845543
MSE: 0.00029792534542423944
R²: -0.02729847932438778

