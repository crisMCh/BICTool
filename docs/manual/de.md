<div class="title-page" markdown="1">

# BICTool Benutzerhandbuch

<p class="sub">Objektive, automatisierte Bewertung der Bildqualität von Kopf-CT<br>
Version {{VERSION}}</p>

<p class="sub">Entwickelt für das Bundesamt für Strahlenschutz (BfS)</p>

<div class="note" markdown="1">
**Nur zur Entscheidungsunterstützung.** BICTool ist kein Medizinprodukt und nicht für die Diagnose bestimmt.
Das Studienurteil stammt aus einem vorläufigen Modell und muss immer von einer fachkundigen Person geprüft
werden.
</div>

</div>

<div class="contents" markdown="1">

[TOC]

</div>

## 1. Was BICTool macht {#what-it-does}

BICTool misst die Bildqualität nativer Kopf-CT-Untersuchungen direkt in den Patientenbildern. Eine
Phantommessung ist nicht nötig. Für jede geeignete Schicht

1. erkennt BICTool die Anatomie mit einer KI-Segmentierung (Falx, Septum pellucidum, Ventrikel, Hirnlappen,
   Basalganglien u. a.) und trennt weiße Substanz, graue Substanz und Liquor anhand ihrer CT-Werte;
2. setzt es zwei Paare von Messregionen (ROIs): eine auf einer dünnen Struktur (Falx cerebri oder Septum
   pellucidum) zur Messung der **Schärfe** und eine in homogenem Gewebe daneben zur Messung des
   **Rauschens**;
3. berechnet es Schärfe (MTF), Rauschen (Rauschleistungsspektrum, NPS) und das Verhältnis von Struktur zu
   Rauschen;
4. fasst es die Schichten zu einem **Studienurteil** zusammen - *Acceptable* (akzeptabel), *Not acceptable*
   (nicht akzeptabel), *Uncertain* (unsicher) oder *Not assessable* (nicht beurteilbar) - und weist auf
   Schichten und Befunde hin, die einen Blick wert sind (mögliche Artefakte, Sichtbarkeit der Basalganglien,
   eine Prüfung der Segmentierung).

Alle Ergebnisse lassen sich als Excel-Bericht oder als CSV-Tabellen exportieren. Exporte enthalten nie
Patientennamen oder -IDs.

Die Programmoberfläche ist englisch. In diesem Handbuch stehen die Bezeichnungen der Oberfläche daher
englisch und *kursiv*, so wie sie im Programm erscheinen.

## 2. Installation und Start {#installation}

### 2.1 Voraussetzungen {#requirements}

| Punkt | Voraussetzung |
|---|---|
| Betriebssystem | Windows 10 (Version 1809 oder neuer) oder Windows 11, 64 Bit |
| Arbeitsspeicher | 8 GB RAM empfohlen; BICTool selbst braucht etwa 1,5 GB |
| Speicherplatz | etwa 3 GB für das entpackte Programm |
| Rechte | keine Administratorrechte nötig |

### 2.2 Installation {#install}

BICTool wird als ZIP-Datei geliefert (`BICTool-{{VERSION}}-win64.zip`).

1. Entpacken Sie die ZIP-Datei in einen beliebigen Ordner, z. B. `C:\Users\<Name>\BICTool`. Entpacken Sie das
   ganze Archiv - starten Sie das Programm nicht aus der ZIP-Datei heraus.
2. Starten Sie `BICTool.exe` im entpackten Ordner (eine Desktop-Verknüpfung erhalten Sie mit Rechtsklick >
   *Senden an > Desktop*).
3. Beim ersten Start zeigt Windows eventuell *„Der Computer wurde durch Windows geschützt“*, weil das Programm
   nicht signiert ist. Klicken Sie auf *Weitere Informationen* und dann auf *Trotzdem ausführen*.

Der Ordner muss zusammenbleiben: `BICTool.exe` braucht die Ordner `_internal` (Programmbibliotheken) und
`models` (das Segmentierungsnetz). Zum Entfernen löschen Sie den Ordner.

### 2.3 Installation prüfen {#self-test}

Das Programm kann sich ohne Oberfläche an einer DICOM-Serie selbst prüfen. Öffnen Sie eine Eingabeaufforderung
im BICTool-Ordner und geben Sie ein:

```
start /wait "" BICTool.exe --self-test "D:\Pfad\zu\einer\Kopf-CT-Serie"
```

Das Programm segmentiert die Serie, misst sie und berechnet das Studienurteil; das dauert etwa zwei Minuten.
Das Ergebnis (*SELF-TEST PASSED* oder *SELF-TEST FAILED* mit Grund) steht in der Protokolldatei
(Abschnitt 8.2). Lesen Sie es, bevor Sie BICTool erneut starten, denn jeder Start beginnt ein neues Protokoll.

## 3. Das Hauptfenster {#main-window}

<figure markdown="1">
![Hauptfenster mit geladener Studie](images/01_viewer.png)
<figcaption>Abbildung 1 - Hauptfenster: Bildbetrachter (links) und Seitenleiste mit den drei Arbeitsschritten (rechts).</figcaption>
</figure>

| Bereich | Inhalt |
|---|---|
| Menüleiste | *File* (Öffnen), *Study* (Schließen, neu laden), *View* (DICOM-Tags), *Tools* (Segmentierung, Einstellungen der Methoden, eigenes Fenster), *Settings* (Protokoll, Textgröße), *Export*, *Help* (Handbücher, Info) |
| *Viewer* / *Results* | Die beiden Seiten des Fensters, umgeschaltet oben links |
| *Viewer* | Die CT-Schicht mit Überlagerungen und ROIs; Graustufenfenster (*Brain*, *Subdural*, *Bone*, *Stroke*, *Soft tissue*); Schichtregler. Das Fenster ändert nur die Darstellung, nie eine Messung |
| Seitenleiste | Serie, dann die drei Schritte: *1. Segmentation*, *2. ROI Selection*, *3. Run QA metrics* |

Fahren Sie mit der Maus über ein Bedienelement, um zu sehen, was es tut. Ein ausgegrautes Element nennt im
Tooltip den Grund (z. B. *„Open a study first“* - zuerst eine Studie öffnen).

**Hilfefenster.** Drücken Sie **F1**, um die Hilfe zu dem Bedienelement unter dem Mauszeiger zu öffnen, oder
klicken Sie oben im Fenster auf **? Help**. Das Hilfefenster enthält dieses Handbuch (Deutsch oder Englisch)
und die *Technical Reference* (englisch), die jeden Algorithmus und Parameter erklärt. Es hat ein
Inhaltsverzeichnis, eine Volltextsuche und die Schaltflächen *Zurück* / *Vor* und kann neben BICTool geöffnet
bleiben. Auch *Help > Help Contents* öffnet es; die PDF-Fassungen stehen im Menü *Help* und im Ordner `manual`.

## 4. Arbeiten mit BICTool {#working}

### 4.1 Studie öffnen (Eingabe) {#open-study}

Wählen Sie *File > Open DICOM Folder (Series)…* und den Ordner mit der CT-Serie. BICTool liest alle
DICOM-Dateien darin, ordnet sie Serien zu und wählt die axiale Kopfserie mit etwa 4 mm Schichtdicke. Hat die
Studie mehrere Serien, wählen Sie eine andere in der Serienliste der Seitenleiste.

**Geeignete Eingabe:**

- natives CT des Kopfes, axiale Schichten, DICOM-Format;
- Weichteil-(Hirn-)Rekonstruktionskernel, wie für das Routine-Kopf-CT üblich (z. B. Siemens Hr44/H30s,
  Canon FC26). Scharfe Knochenkernel liegen außerhalb des Messbereichs;
- beliebige Matrixgröße und Pixelabstand.

*File > Open DICOM File (Single Slice)…* öffnet nur ein einzelnes Bild; für das Studienurteil braucht man
mehrere Schichten.

### 4.2 Schritt 1 - Segmentierung {#segmentation}

Die anatomische Segmentierung startet automatisch beim Öffnen einer Studie und dauert etwa 1-2 Sekunden pro
Schicht. Der Fortschritt steht im Kasten *Anatomical*.

<figure markdown="1">
![Anatomische Segmentierung](images/02_segmentation.png){ .image }
<figcaption>Abbildung 2 - Anatomische Segmentierung auf dem Bild (Show Segments).</figcaption>
</figure>

<figure markdown="1">
![Schritt 1 in der Seitenleiste](images/02a_step1_box.png){ .box }
<figcaption>Abbildung 3 - Schritt 1 in der Seitenleiste: anatomische Segmentierung und WM/GM-Trennung.</figcaption>
</figure>

- **Show Segments** zeigt die Strukturen auf dem Bild; **Select Segments…** wählt, welche Strukturen und wie
  deckend sie gezeigt werden.
- Ist die ganze Serie segmentiert, sagt eine Zeile im Kasten, ob die Segmentierung **schlüssig aussieht**
  (*looks consistent*) oder **unzuverlässig sein kann** (*may be unreliable*). Im zweiten Fall sehen Sie sich
  die Überlagerungen auf einigen Schichten an: Läsionen, Artefakte oder starkes Rauschen können die Anatomie
  der KI in verstreute Stücke zerfallen lassen, und dann können die automatischen ROIs falsch liegen. Die
  Prüfung nennt keinen Grund.

**Trennung weiße/graue Substanz (WM/GM separation).** Die ROIs für das Rauschen liegen in der weißen
Substanz. Weiße und graue Substanz werden anhand ihrer CT-Werte mit einer dieser Methoden getrennt (Kasten
*WM/GM separation*, Liste *Method*):

| Methode | Kurz |
|---|---|
| Adaptive Thresholding (Standard) | Lokale Schwelle, leicht geglättet, HU-Bereich aus den Basalganglien dieser Studie. Passt für die meisten Scans |
| Fuzzy C-Means | Teilt das Gehirn in drei Klassen (Liquor, weiße, graue Substanz) |
| HU Thresholding | Feste HU-Bereiche je Gewebe |
| Anchored (prototype) | Nutzt die Basalganglien als Referenz und korrigiert HU-Drift; noch nicht validiert |

<figure markdown="1">
![Weiße und graue Substanz](images/03_wmgm.png){ .image }
<figcaption>Abbildung 4 - Weiße (WM) und graue Substanz (GM), gefunden mit der gewählten Methode.</figcaption>
</figure>

<div class="note" markdown="1">
**Wählen Sie die Methode, die am besten zum Scan passt.** Keine Methode ist auf jedem Scan die beste. Haken Sie
*WM* und *GM* an, sehen Sie sich einige Schichten an und probieren Sie eine andere Methode, wenn die weiße
Substanz nicht stimmt. Die Rausch-ROIs werden in die weiße Substanz gesetzt, die die Methode findet - die Wahl
beeinflusst also die Messung.
</div>

*CSF* zeigt den Liquor, *Brain Mask* die Hirnkontur, innerhalb derer getrennt wird.

Technische Einzelheiten: *Technical Reference*, [Kapitel 4-6](tr:tr-segmentation) (Segmentierung, Hirnmaske, [WM/GM-Methoden](tr:tr-wmgm)).

### 4.3 Schritt 2 - ROI-Auswahl {#roi-selection}

Klicken Sie auf **Start auto ROIs**. BICTool setzt auf jede geeignete Schicht zwei ROI-Paare:

| Paar | ROI1 (Schärfe) | ROI2 (Rauschen) |
|---|---|---|
| Falx | auf der Falx cerebri | weiße Substanz daneben |
| Septum | auf dem Septum pellucidum | der Ventrikel daneben |

Schichten mit zu wenig Hirngewebe (an der Schädelbasis und am Scheitel) erhalten keine automatischen ROIs.

<figure markdown="1">
![Automatische ROIs](images/04_auto_rois.png){ .image }
<figcaption>Abbildung 5 - Automatische ROIs auf einer Schicht: F1 auf der Falx, F2 weiße Substanz, S1 auf dem Septum, S2 Ventrikel.</figcaption>
</figure>

<figure markdown="1">
![ROI-Kasten](images/04a_roi_box.png){ .box }
<figcaption>Abbildung 6 - Schritt 2 in der Seitenleiste; ROI details zeigt Lage und Größe.</figcaption>
</figure>

**Prüfen Sie die ROIs auf einigen Schichten** (Regler oder Mausrad):

- ROI ziehen, um sie zu verschieben;
- Rechtsklick auf eine ROI, um sie zu löschen oder als Falx-, Weiße-Substanz-, Septum- oder Ventrikel-ROI zu
  speichern;
- auf dem Bild ein eigenes Rechteck aufziehen und mit Rechtsklick speichern;
- **Delete ROI** entfernt die gewählte ROI (oder Taste *Entf*); **Restore automatic** stellt die automatischen
  ROIs dieser Schicht wieder her.

Stimmen die ROIs, haken Sie im Kasten *Confirm ROIs* **ROIs checked - lock them for QA** an. Die ROIs sind dann
gesperrt; Haken entfernen, um sie wieder zu bearbeiten.

<figure markdown="1">
![Bestätigte ROIs](images/05_confirmed.png){ .box }
<figcaption>Abbildung 7 - ROIs bestätigt und gesperrt.</figcaption>
</figure>

Technische Einzelheiten: *Technical Reference*, [Kapitel 7](tr:tr-rois) (wie die ROIs platziert werden).

### 4.4 Schritt 3 - Messung {#qa-run}

Klicken Sie auf **▶ Run QA**. BICTool misst jedes bestätigte ROI-Paar auf jeder Schicht (etwa eine
Sekunde pro Paar) und öffnet danach die Seite *Results*. Eine laufende Messung lässt sich abbrechen; fertige
Schichten bleiben erhalten.

<figure markdown="1">
![Messung starten](images/05a_run_box.png){ .box }
<figcaption>Abbildung 8 - Schritt 3: Run QA, mit den Messwerten der gezeigten Schicht.</figcaption>
</figure>

Technische Einzelheiten: *Technical Reference*, [Kapitel 8](tr:tr-measurements) (MTF, NPS, PS/MTF).

### 4.5 Ergebnisse pro Schicht {#results-per-slice}

<figure markdown="1">
![Ergebnisse einer Schicht](images/06_results_curves.png)
<figcaption>Abbildung 9 - Seite Results, Reiter Per slice: die vier Kurven einer Schicht.</figcaption>
</figure>

Der Reiter *Per slice* zeigt eine Schicht nach der anderen (Regler, Pfeile oder Mausrad); *Falx* / *Septum*
schaltet zwischen den beiden ROI-Paaren um. Die vier Kurven:

| Kurve | Bedeutung |
|---|---|
| MTF | Schärfe, gemessen an der Kante der Struktur: je höher die Kurve, desto schärfer |
| NPS | Rauschleistungsspektrum im homogenen Gewebe: wie stark und wie fein oder grob das Rauschen ist |
| PS | Leistungsspektrum an der Struktur: Bilddetail plus Rauschen |
| PS/MTF | Leistungsspektrum geteilt durch die Schärfe: hohe Werte bei niedrigen Frequenzen bedeuten Struktur, nicht Rauschen |

Die Zeile *„This slice resembles: …“* (diese Schicht ähnelt …) ist nur ein Hinweis: Das Modell beurteilt
ganze Studien, nicht einzelne Schichten.

*Show: Intermediate steps (advanced)* ersetzt die Kurven durch die Zwischenschritte der Messung - die für die
MTF genutzte Kante, die Kantenprofile, beide ROIs, die 2D-Spektren und die Merkmalswerte. Das dient der
genauen Prüfung einer einzelnen Messung.

<figure markdown="1">
![Zwischenschritte](images/07_results_steps.png)
<figcaption>Abbildung 10 - Zwischenschritte einer Messung.</figcaption>
</figure>

### 4.6 Studienübersicht und Urteil {#study-summary}

<figure markdown="1">
![Studienübersicht](images/08_study_summary.png)
<figcaption>Abbildung 11 - Study summary: Urteil, Befunde und alle gemessenen Schichten in einer Tabelle.</figcaption>
</figure>

Das **Studienurteil** (*Study verdict*) gilt für die ganze Studie. Es vergleicht fünf Messwerte (Mediane über
die Falx-Schichten) mit Studien, die ein Radiologe bewertet hat:

| Urteil | Bedeutung |
|---|---|
| **Acceptable** (grün) | Die Messwerte entsprechen klar den als akzeptabel bewerteten Studien |
| **Not acceptable** (rot) | Die Messwerte entsprechen klar den als nicht akzeptabel bewerteten Studien |
| **Uncertain, leaning acceptable / not acceptable** | Keine klare Antwort; die Messwerte neigen leicht zu einer Seite. Prüfen Sie die Bilder selbst |
| **Not assessable - metal artefacts in the brain** | Metallartefakte reichen ins Gehirn. Sie stören die Messung, daher kann BICTool die Studie nicht beurteilen. Ob die Bilder ausreichen, entscheidet die befundende Person - bei Metallimplantaten ist das CT oft die einzige Möglichkeit |
| *Not available* | Weniger als 5 Falx-Schichten konnten gemessen werden |

Die Prozentzahl (*„90 % likely acceptable“*) ist die Schätzung des Modells. Sie wird viele Male mit leicht
veränderter Auswahl der bewerteten Studien und der Schichten dieser Studie wiederholt; das Urteil lautet nur
dann *Acceptable* oder *Not acceptable*, wenn die mittleren 95 % dieser Wiederholungen alle auf derselben Seite
von 50 % liegen. **Details** zeigt
jeden Messwert mit seinem Intervall und dem Bereich guter Studien.

<figure markdown="1">
![Details zum Urteil](images/09_summary_details.png)
<figcaption>Abbildung 12 - Details zum Urteil.</figcaption>
</figure>

Unter dem Urteil nennen einfache Zeilen, worauf zu achten ist:

| Zeile | Bedeutung |
|---|---|
| *Sharpness / Noise … is higher or lower than in good studies* | Die Messwerte, die das Urteil Richtung *nicht akzeptabel* schieben |
| *Artefacts (noise across slices)* | Das Rauschen schwankt stark von Schicht zu Schicht (Bewegung, Implantate) |
| *Artefacts (metal)* | Metall auf diesen Schichten; ob seine Streifen ins Gehirn reichen |
| *Artefacts (posterior fossa)* | Ein mögliches dunkles Band quer durch den Hirnstamm (Strahlaufhärtung zwischen den Felsenbeinen) |
| *Basal ganglia (provisional)* | Auf welchen Schichten die Basalganglien gegen die Capsula interna sichtbar sind |
| *Segmentation (provisional)* | Ob die anatomische Segmentierung schlüssig aussieht |
| *Unusually noisy slices* | Schichten mit deutlich stärkerem Rauschen als der Rest der Studie |
| *Check the ROIs* | Falx- und Septum-Paar widersprechen sich - ROIs ansehen |

Die als *provisional* (vorläufig) markierten Befunde zeigen, wo man hinsehen sollte; sie ändern das Urteil
nicht (außer Metallartefakten im Gehirn). Die Tabelle darunter listet jede gemessene Schicht; ein Klick auf
eine Zeile öffnet diese Schicht.

Technische Einzelheiten: *Technical Reference*, [Kapitel 9](tr:tr-verdict) (Urteilsmodell) und [Kapitel 10](tr:tr-findings) (Befunde).

### 4.7 Export (Ausgabe) {#export}

Wählen Sie *Export > Export Results…* (nach einer Messung verfügbar).

<figure markdown="1">
![Export-Dialog](images/10_export_dialog.png){ .box }
<figcaption>Abbildung 13 - Export: Umfang und Formate; der Dateiname wird im nächsten Schritt gewählt.</figcaption>
</figure>

1. **Umfang** (*Scope*): die *aktuelle Schicht* (*Current slice*) oder die *ganze Studie* (*Whole study*).
2. **Formate** (beliebig kombinierbar):
    - *Report - Excel workbook (.xlsx)*: ein lesbarer Bericht. Blatt *Summary*: Studie, Urteil und Befunde wie
       im Programm, Messwerte mit Einheiten, technische Details darunter. Blatt *Per slice*: eine Zeile je
       Schicht und ROI-Paar, filterbar.
    - *All numbers - CSV tables (.csv)*: die vollständigen Tabellen zur Auswertung in anderen Programmen
       (reiner Text).
    - *Charts - JPG images* (nur aktuelle Schicht): die vier Kurven.
3. Klicken Sie auf **Save as…** und wählen Sie Dateiname und Ordner. BICTool schlägt einen Namen vor, z. B.
   *BICTool study 2026-03-12 Head native 4.0 Hr40*; alle Dateien eines Exports erhalten diesen Namen (z. B.
   *… - summary.csv*, *… - per slice.csv*, *… - MTF.jpg*). Der zuletzt genutzte Ordner wird gemerkt.

Exporte enthalten nur de-identifizierte Angaben: keinen Patientennamen, keine ID, kein Geburtsdatum.

## 5. Die Messgrößen {#measurements}

| Messgröße | Einheit | Aussage |
|---|---|---|
| Schärfe (*Sharpness*, MTF-Fläche) | 1/mm | Fläche unter der Modulationsübertragungsfunktion an der Kante von Falx oder Septum. Höher = schärfer |
| Rauschen (*Noise*, NPS-Varianz) | HU²·mm² | Stärke des Rauschens im homogenen Gewebe (ROI2), aus dem Rauschleistungsspektrum |
| Rauschtextur (f_av, f_peak) | 1/mm | Mittlere und Spitzenfrequenz des Rauschens: feines oder grobes Korn |
| Struktur zu Rauschen (*PS/MTF ratio, LF/HF*) | - | Anteil niedriger gegenüber hoher Frequenzen im Strukturspektrum, korrigiert um die Schärfe |
| Kontrast (*Contrast*, RMS) | - | Standardabweichung geteilt durch den Mittelwert von ROI1: wie stark sich die Struktur abhebt |

Das Studienurteil nutzt fünf Werte auf Studienebene: MTF-Fläche, Rauschen im homogenen Gewebe, Rauschen an der
Falx, Varianz an der Falx und das PS/MTF-Verhältnis. Wie jeder Wert berechnet wird - Formeln, Parameter,
Grenzen und Literatur - beschreibt die *Technical Reference* (Hilfefenster, F1; englisch).

Technische Einzelheiten: *Technical Reference*, [Kapitel 8](tr:tr-measurements) und die [Parameterliste](tr:tr-parameters).

## 6. Einstellungen und Parameter {#settings}

### 6.1 Methoden der WM/GM-Trennung {#wmgm-methods}

Jede Methode hat eigene Einstellungen (*Tools > … Settings* oder die Schaltfläche ⚙ neben der Methode).
*Restore Defaults* stellt die erprobten Werte wieder her; ändern Sie sie nur, wenn die Überlagerung auf einem
Scan deutlich falsch ist.

| Methode | Parameter |
|---|---|
| Adaptive Thresholding | Größe des lokalen Fensters (px); plausibler HU-Bereich (unten, oben) - gilt, wenn der Bereich aus der Studie abgeschaltet oder nicht verfügbar ist; minimale Regionsgröße (px); Glättung (mm, 0 = aus); HU-Bereich aus der Studie verwenden |
| Fuzzy C-Means | Unschärfe m; maximale Iterationen; räumliches Mischgewicht; minimale Regionsgröße (px) |
| HU Thresholding | Schwellen für Luft und Knochen beim Entfernen des Schädels (HU); HU-Bereiche für Liquor, weiße und graue Substanz; Glättung (Medianfiltergröße, px); minimale Regionsgröße (px) |

Technische Einzelheiten: *Technical Reference*, [Kapitel 6](tr:tr-wmgm).

### 6.2 Darstellung {#viewing}

- **Fenster** (*Window*, Werkzeugleiste des Bildbetrachters): *Brain* (Standard), *Subdural*, *Bone*,
  *Stroke*, *Soft tissue*.
- **Tools > Custom Window…**: eigene Fensterbreite und -lage (HU). Nur Darstellung.
- **View > DICOM Tags…**: der DICOM-Header des aktuellen Bildes.

### 6.3 Programmeinstellungen {#program-settings}

- **Settings > Text Size…**: Größe aller Texte in BICTool (80-200 %, Standard 125 %). Wirkt nach einem
  Neustart.

<figure markdown="1">
![Textgröße](images/11_text_size.png){ .box }
<figcaption>Abbildung 14 - Textgröße.</figcaption>
</figure>

- **Settings > Show Trace Log**: ein technisches Protokoll unten im Fenster, zur Fehlersuche.
- Fensteraufteilung, geöffnete Abschnitte, Textgröße und der letzte Exportordner werden je Windows-Benutzer
  gespeichert (Datei `%APPDATA%\BfS\BICTool.ini`).
- *Tools > Advanced > QA Recipe* enthält eine alte Berechnung, die nur Ergebnisse der früheren Octave-Version
  nachbildet. Behalten Sie die Voreinstellung *Physical*.

## 7. Grenzen {#limitations}

- **Nur zur Entscheidungsunterstützung.** Das Urteil stammt aus einem vorläufigen Modell, trainiert auf
  Studien, die ein Radiologe bewertet hat; es ist noch nicht an unabhängigen Daten bestätigt. Sehen Sie sich
  immer die Bilder an.
- Das Urteil ist für das routinemäßige native Kopf-CT mit Weichteilkernel gedacht. Andere Protokolle,
  Kontrastmitteluntersuchungen, Kinder und scharfe Kernel liegen außerhalb seines Bereichs.
- Pathologie an oder neben der Falx (Blutung, Tumor, Operation) kann die Schärfemessung stören. Prüfen Sie
  die ROIs auf solchen Schichten.
- Die KI-Segmentierung wurde an Gehirnen Erwachsener trainiert. Große Läsionen, starke Artefakte oder
  ungewöhnliche Anatomie können sie fehlleiten; die Segmentierungsprüfung weist auf solche Studien hin.
- Die vorläufigen Befunde (Hirnstammband, Basalganglien, Segmentierungsprüfung, Metallartefakte) nutzen
  vorläufige Grenzwerte und werden mit mehr Daten verfeinert.

## 8. Fehlerbehebung {#troubleshooting}

### 8.1 Probleme und Lösungen {#problems}

| Problem | Ursache und Lösung |
|---|---|
| *„Der Computer wurde durch Windows geschützt“* beim ersten Start | Das Programm ist nicht signiert. *Weitere Informationen > Trotzdem ausführen* |
| Das Programm startet nicht oder schließt sich sofort | Prüfen Sie, ob die ganze ZIP-Datei entpackt wurde und die Ordner `_internal` und `models` neben `BICTool.exe` liegen. Sehen Sie in die Protokolldatei (8.2) |
| *„Anatomy segmentation weights not found“* | Die Datei `models\weights_mayoiNPH.hdf5` fehlt. Entpacken Sie die ZIP-Datei erneut |
| Die Segmentierung ist sehr langsam | Etwa 1-2 s pro Schicht sind normal; sie läuft einmal pro Studie. Die Dauer hängt vom Prozessor ab. Schließen Sie andere Programme, die viel Speicher oder Rechenzeit brauchen |
| Keine oder die falsche Serie gewählt | Wählen Sie die Serie in der Serienliste der Seitenleiste. Gelesen werden nur DICOM-Dateien |
| *Start auto ROIs* ist ausgegraut | Die Segmentierung ist noch nicht fertig - warten Sie, bis der Kasten *Anatomical* die Serie als segmentiert meldet |
| Wenige oder keine automatischen ROIs | Falx oder Septum sind auf diesen Schichten nicht sichtbar oder nicht segmentiert, oder die Schichten enthalten zu wenig Hirngewebe. Zeichnen Sie ROIs von Hand |
| ROI2 liegt in grauer Substanz oder Liquor | Wählen Sie eine andere WM/GM-Methode (4.2) und starten Sie die automatischen ROIs neu, oder verschieben Sie ROI2 von Hand |
| *Run QA* ist ausgegraut | Haken Sie zuerst *ROIs checked - lock them for QA* an |
| Einige Schichten *failed* (fehlgeschlagen) | Die Schärfekante ließ sich dort nicht messen (z. B. keine klare Falx). Die übrigen Schichten werden genutzt; das Urteil braucht mindestens 5 Falx-Schichten |
| *„Segmentation may be unreliable“* | Sehen Sie sich die Überlagerungen an; korrigieren Sie die ROIs bei Bedarf von Hand |
| Export schlägt fehl | Der Ordner ist nicht beschreibbar oder eine gleichnamige Datei ist in Excel geöffnet. Schließen Sie sie oder wählen Sie einen anderen Ordner |
| Text zu klein oder zu groß | *Settings > Text Size…* und Neustart |

### 8.2 Protokolldatei {#log-file}

BICTool schreibt ein Protokoll der letzten Sitzung nach

```
%LOCALAPPDATA%\BICTool\logs\bictool.log
```

(in die Adresszeile des Windows-Explorers einfügen). Bitte senden Sie diese Datei mit, wenn Sie ein Problem
melden. Sie enthält keine Patientennamen oder -IDs, kann aber Ordnernamen enthalten - prüfen Sie diese, bevor Sie die
Datei außerhalb Ihrer Einrichtung weitergeben. *Settings > Show Trace Log* zeigt dieselben Angaben im
Programm.

## 9. Lizenz und Danksagung {#license}

BICTool ist freie Software unter der GNU General Public License Version 3 oder später (`LICENSE.txt`). Die
Fremdkomponenten und ihre Lizenzen stehen in `THIRD_PARTY_NOTICES.txt` und im Ordner `licenses`.

Die anatomische Segmentierung nutzt Netz und Gewichte des Mayo Clinic Radiology Informatics Lab
(github.com/jasonccai/HeadCTSegmentation):
Cai JC, Akkus Z, Philbrick KA, et al. *Fully Automated Segmentation of Neuroanatomy on Head CT Using Deep
Learning.* Radiol Artif Intell. 2020;2(5):e190183.
