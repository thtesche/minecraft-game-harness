# Migrationsplan: Minecraft LLM-Agent zu Laya (Hybrid-Architektur)

Dieser Plan skizziert die konkreten Schritte, um den aktuellen Minecraft-Agenten (LLM-basiert) teilweise auf das Laya-Entscheidungsmodell zu migrieren, orchestriert durch LangGraph.

## 🎯 1. Zielsetzung & Scope
**Anwendungsfall:** Laya wird genutzt, um deterministische Werkzeugauswahlen (z. B. `collect_drop`, `mine_block`) und die Auswahl des nächsten Teilschritts für ein übergeordnetes Ziel (z. B. "Spitzhacke herstellen") in Echtzeit zu übernehmen.
**Fragetyp in Laya:** `choice`-Fragen (Klassifizierung aus einer Liste von Optionen).

## 📊 2. State-Repräsentation
**Format:** Großer, standardisierter JSON-Dump (Volles Inventar, Gesundheit, Blöcke im 5x5 Radius, aktuelles Ziel).
**Modell-Wahl:** Da der JSON-Dump groß wird, verwenden wir das Modell `laya-multilingual` mit dem Parameter `max_len=8192` (das Basis-Modell fasst nur 512 Token).

## 🔄 3. Datengenerierung & Training (Behavior Cloning)
1. **Logging aktivieren:** Der aktuelle LLM-Agent speichert bei jedem erfolgreichen Schritt den JSON-State und das gewählte Werkzeug.
2. **Fine-Tuning:** Mit dem [offiziellen Kaggle-Notebook von Laya](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) wird das Modell auf diesen Datensatz trainiert.

## 🔌 4. Architektur & Orchestrierung (LangGraph)
Anstatt ein eigenes Harness zu schreiben, nutzen wir **LangGraph** (`laya[langchain]`), um den "Main Loop" zu steuern. Dies ermöglicht eine saubere Integration von Laya, dem großen LLM und den MCP-Tools.

**Der Ablauf im Graphen:**
1. **Start:** Der Graph erhält den aktuellen Minecraft-State.
2. **Laya-Router-Node:** Laya analysiert den State in einem Forward-Pass (ca. 33ms).
3. **Konfidenz-Check (Fallback):** Laya wird mit `min_confidence=0.85` konfiguriert.
   - **Fast-Path (System 1):** Ist sich Laya sicher (>= 85%), routet der Graph direkt zum entsprechenden `ToolNode`.
   - **Slow-Path (System 2):** Fällt die Konfidenz unter 85% (Laya gibt `None` zurück), routet der Graph als Fallback an den **LLM-Node**.
4. **LLM-Node:** Das große LLM (z.B. Claude/Gemini) hat über `.bind_tools()` Zugriff auf exakt dieselben LangChain-Tools. Es übernimmt die komplexe Planung und ruft das Tool auf.
5. **Tool-Execution:** Der `ToolNode` (der den MCP-Client wrappt) führt die Aktion in Minecraft aus, aktualisiert den State, und der Loop beginnt von vorn.

---

## 🛠️ Nächste Schritte zur Umsetzung

1. **Abhängigkeiten installieren:**
   ```bash
   pip install "laya[langchain]" langgraph
   ```
2. **Tools vorbereiten:**
   Wandle deine bestehenden MCP-Aufrufe in LangChain-Tools um (z.B. via `@tool` Decorator).
3. **LangGraph aufbauen:**
   Definiere den Workflow mit `StateGraph`. Binde Laya als Router-Knoten ein, der basierend auf dem `min_confidence` Parameter entweder eine direkte Tool-Ausführung oder den Umweg über das LLM anstößt.
4. **Datensatz-Generierung starten:** 
   Lass deinen bisherigen (reinen LLM-)Agenten laufen und sammle erfolgreiche `(State, Tool)` Paare als JSONL-Datei für das spätere Fine-Tuning von Laya.

---

## 📚 Quellen
- [Laya GitHub Repository](https://github.com/NandhaKishorM/laya)
- [Laya Online-Dokumentation](https://nandhakishorm.github.io/laya/)
- [Laya Fine-Tuning Notebook (Kaggle)](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
