# Das System

Der AI-Act-Chatbot der KI-Servicestelle (RTR) beantwortet Fragen zum EU AI Act. Sein eigentlicher Zweck geht aber über das Beantworten hinaus: Er macht
  erfahrbar, wie ein KI-Auskunftssystem intern arbeitet. Er ist gleichzeitig Auskunftswerkzeug und didaktisches Schaufenster.

Der AI Act Chatbot geht aber weiter als das: Er zeigt zusätzlich an, die ein RAG-System funktioniert. Hintergrund ist es, dass zum Entwicklungszeitpunkt RAGs der neue "heiße Scheiß" waren, aber niemand unter den Entscheidern und Anwendern so recht wusste, wie sie funktionieren. Der AI Act Chatbot zeigt deshalb im Frontend:

Dieser doppelte Anspruch ist kein Beiwerk, sondern die Identität des Projekts und muss in jeder Weiterentwicklung erhalten bleiben. Konkret zeigt das
  System heute offen:

- Das vollständige Prompt, das an das Modell geht (Retrieve -> Augment  -> Generate werden im UI als sichtbare Schritte durchlaufen);
- alle gefundenen Quellen mit ihrer semantischen Ähnlichkeit (Cosine Similarity in %), inklusive der manuell gepflegten Querverweise zwischen Bestimmungen. Es wird derzeit die Kosinus-Ähnlichkeit verwendet, auf 25 Quellen beschränkt (oder 15000 Token, was früher erreicht ist)
- und auch der nicht verwendeten Quellen samt Grund (Kontextfenster voll, Duplikat);
- den tatsächlich gemessenen Energieverbrauch (CPU/GPU/RAM) je Phase und in Summe pro Frage;
- eine umfassende technische und rechtliche Dokumentation der Komponenten.
  
# Warum jetzt eine Weiterentwicklung nötig ist (veränderter Kontext)

Das System entstand 2024, als RAG das neue, kaum verstandene Thema war. Seither hat sich der Kontext verschoben:

- RAG ist Mainstream geworden — das didaktische „So funktioniert ein RAG" zieht weniger; das neue erklärungsbedürftige Thema sind Agents und Tool Calling. Auch wenn "RAG" weiterhin oft unklar ist, und deshalb nicht ganz verloren gehen soll.
- Die EU-Kommission hat umfangreiche Leitlinien zum AI Act veröffentlicht. Diese sind hochrelevant, behandeln aber eng umrissene Spezialthemen — sie
  passen nicht ohne Weiteres in den bestehenden semantischen Suchpool.
- Die zugrunde liegende Technik (Modelle, Retrieval-Qualität) hat einen Sprung gemacht und soll mitgezogen werden.

# Zielbild der Weiterentwicklung

Aus der heutigen linearen Pipeline (eine Frage → ein fixer Retrieval-Schritt → ein LLM-Aufruf) soll eine agentische Oberfläche mit Tool Calling werden:
Das Modell entscheidet mit agentischer Steuerung innerhalb bestimmter Grenzen selbst, welche Werkzeuge es wann nutzt (z. B. semantische Suche, gezielte Leitlinien-Recherche, Nachschlagen eines konkreten  Artikels, ggf. mehrere Suchrunden).

Bedingung: Der didaktische Anspruch wird auf die Agentik übertragen, nicht aufgegeben. So wie das UI heute Retrieve/Augment/Generate sichtbar macht, soll es künftig jeden Agentenschritt nachvollziehbar zeigen — welches Tool mit welcher Eingabe aufgerufen wurde, was es zurückgab, warum das Modell den   nächsten Schritt wählte. Ziel ist, „Agents" für Entscheider:innen und Anwender:innen genauso zu entmystifizieren, wie es das System bisher für RAG getan hat. Energie- und Quellentransparenz bleiben dabei vollständig erhalten (Energiemessung dann ggf. über mehrere Tool-/LLM-Aufrufe pro Frage).

In einer Weiterentwicklung soll deshalb einiges adaptiert werden:
1. Aus dem reinen RAG soll eine agentische Oberfläche inklusive Tool calling werden. Aber weiterhin mit dem Ansinnen, das alles einfach erklärt werden soll.
2. Zusätzlich zur semantischen Suche aus AI Act, Erwägungsgründen und KI-Servicestellen-Texte sollen auch die Leitlinien der EU-Kommission aufgenommen werden. Wir gehen davon aus, dass eine reine Aufnahme in den Dqtenpool nicht zielführend ist, weil sie spezielle Themen behandeln und bei einer reinen Semantic Similarity oft die eigentlich relevanten Bestimmungen verdrängen werden.
3. Die Technik soll ein Update erfahren. Es bleibt bei self-hosting mit ollama und einem Hetzner-Rechner mit RTX 4000 ADA (20 GB VRAM). Aktuell wird mistral-small für LLM und snowflake für Embedding verwendet. Aktuell wird noch kein Reranker verwendet. Künftig soll ein Reranker verwendet werden.

# Leitprinzipien für alle Maßnahmen

1. Transparenz vor Magie — jede neue Fähigkeit muss im UI erklärbar/sichtbar gemacht werden.
2. Strikte Quellenbindung — Belegpflicht und Halluzinationsvermeidung bleiben unverhandelbar.
3. Self-Hosting & Datensouveränität — keine externen Cloud-LLM-Abhängigkeiten; Betrieb auf der vorhandenen GPU.
4. Energietransparenz — der gemessene Verbrauch pro Frage bleibt ein sichtbares Feature, auch über mehrere Agentenschritte hinweg.
5. Normhierarchie erhalten — verbindliche Bestimmungen dürfen durch erläuternde Materialien nicht verdrängt, sondern nur ergänzt werden.

# Infos zu Leitlinien

Es gibt folgende Leitlinien der Europäische Kommission
* LL zu verbotener KI
* LL zur Definition von "KI-System"
* LL zu Hochrisiko-KI-Systeme nach Artikel 6
* LL zum Umfang der Verpflichtungen für GPAI-Systeme
* LL zu Transparenzpflichten

Zusätzlich gibt es zwei Codes of Practice
* CoP zu Transparenzpflichten
* CoP zu GPAI-Systemen


# Verfügbare Tools

* suche_ai_act(suchbegriff: string) - die bisherige semantische Suche in AI Act, Erwägungsgründen und Materialien der KI-Servicestelle. Inklusive Querverweisen.
* heute() - gibt das heutige Datum zurück
* definition(begriff: string) - Gibt die Definition zu einem juristischen Betriff zurück. Findet sich im AI Act üblicherweise in Artikel 3. Sofern es einfache Ergänzungen zur Definition aus den Leitlinien gibt, werden diese ebenfalls zurückgegeben. Es gibt Definitionen für KI-System, [...], nachgelagerter Anbieter - wird das Tool mit einem nicht enthaltenen Begriff aufgerufen, gibt es diese Liste zurück.
* anwendbarkeit(artikel_nummer: string) Gibt Informationen zum Inkrafttreten bzw der Anwendbarkeit von einzelnen Artikeln zurück. Kein Reines Datum, sondern als Prosa beschrieben, da auch Ausnahmebestimmungen (z.B. Grandfather-Clauses) gelten können. 
* artikel_nachschlagen(artikel_nummer: string): Gibt den Reintext eines Artikels zurück. Erlaubt eine granulare Abfrage bis auf die erste Ebene (z.B. Artikel 5-2 für Absatz 2 von Artikel 5). Tiefere Abfragen geben die höhere Ebene zurück.
* suche_leitlinien_praxisleitfaeden(leitlinie: string[]|optional, bezugsartikel: string|optional, frage: string). Suche in den Leitlinien und Praxisleitfäden. Bezugsartikel bezeichnet den Artikel, um den eine Frage besteht (z.B. Artikel 5). "Frage" ist die Frage in Klartext. Suche_leitlinien führt anschließend eine Suche in den verschiedenen Leitlinien durch. Rückgabe ist eine Liste der gefundenen Quellen, in einer lexikalischen und semantisch hybriden Suche. Parameter "Leitlinie" schränkt die Suche auf eine Anzahl von Leitlinien aus [hochrisiko, definition ki-system, leitlinie_transparenz, verbote, gpai-anwendbarkeit, praxisleitfaden_transparenz] ein.
  * Inhalte der Leitlinien:
    * hochrisiko: XXX
    * definition KI-System: Erläuterungen zu Art 3 Z 1 AI Act ("KI-System")
    * leitinlinie_transparenz: lorem ipsum..
    * ...
* rueckfrage_nutzer(string[] antwortvorschlaege): STelle eine Rückfrage an den Nutzer. Vorgegebene Antwortmöglichkeiten als optionale Parameter. Der Nutzer hat danach die Möglichkeit, aus den vorgegebenen Antwortmöglichkeiten auszuwählen, oder aber, eine eigene Antwort zu tippen. Rückgabe des Tools ist ein STring mit dem entweder ausgewählten Antwortvorschlag, oder der freien Rückgabe des Nutzers.


# Agentischer Modus

* Pro Anfrage maximal 3 Calls von jedem der aufgelisteten Tools
* Das UI zeigt die rohe Chain-of-Thought an.
* Jede Antwort enthält Quellenverweise
* Bei Fragen zur zeitlichen Geltung wird anwendbarkeit() verwendet
* Das UI ist maximal transparent. Etwa auch hinsichtlich Tool-Parameter, der Liste der Tools, dem Einfluss des Rerankers.
* Das UI stellt den Agenten als "Timeline" dar. Werden Sub-Tools eingesetzt, sind auch dort alle Schritte transparent.

