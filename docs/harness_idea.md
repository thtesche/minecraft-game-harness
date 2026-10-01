# Migration Plan: Minecraft LLM Agent to Laya (Hybrid Architecture)

This plan sketches the concrete steps to partially migrate the current Minecraft agent (LLM-based) to the Laya decision model, orchestrated by LangGraph.

## 🎯 1. Goal & Scope
**Use case:** Laya is used to take over deterministic tool selections (e.g. `collect_drop`, `mine_block`) and the selection of the next sub-step for a higher-level goal (e.g. "craft a stone pickaxe") in real time.
**Question type in Laya:** `choice` questions (classification from a list of options).

## 📊 2. State Representation
**Format:** Large, standardized JSON dump (full inventory, health, blocks within a 5x5 radius, current goal).
**Model choice:** Since the JSON dump gets large, we use the `laya-multilingual` model with the parameter `max_len=8192` (the base model only fits 512 tokens).

## 🔄 3. Data Generation & Training (Behavior Cloning)
1. **Enable logging:** The current LLM agent stores the JSON state and the chosen tool on every successful step.
2. **Fine-tuning:** Using the [official Laya Kaggle notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), the model is trained on this dataset.

## 🔌 4. Architecture & Orchestration (LangGraph)
Instead of writing our own harness, we use **LangGraph** (`laya[langchain]`) to control the "main loop". This enables a clean integration of Laya, the large LLM, and the MCP tools.

**The flow inside the graph:**
1. **Start:** The graph receives the current Minecraft state.
2. **Laya router node:** Laya analyzes the state in a forward pass (approx. 33ms).
3. **Confidence check (fallback):** Laya is configured with `min_confidence=0.85`.
   - **Fast path (System 1):** If Laya is confident (>= 85%), the graph routes directly to the corresponding `ToolNode`.
   - **Slow path (System 2):** If confidence drops below 85% (Laya returns `None`), the graph falls back and routes to the **LLM node**.
4. **LLM node:** The large LLM (e.g. Claude/Gemini) has access to exactly the same LangChain tools via `.bind_tools()`. It takes over the complex planning and calls the tool.
5. **Tool execution:** The `ToolNode` (which wraps the MCP client) executes the action in Minecraft, updates the state, and the loop starts over.

---

## 🛠️ Next Steps for Implementation

1. **Install dependencies:**
   ```bash
   pip install "laya[langchain]" langgraph
   ```
2. **Prepare the tools:**
   Convert your existing MCP calls into LangChain tools (e.g. via the `@tool` decorator).
3. **Build the LangGraph:**
   Define the workflow with `StateGraph`. Bind Laya as a router node that, based on the `min_confidence` parameter, either triggers direct tool execution or takes the detour via the LLM.
4. **Start dataset generation:** 
   Run your previous (pure LLM) agent and collect successful `(State, Tool)` pairs as a JSONL file for Laya's later fine-tuning.

---

## 📚 Sources
- [Laya GitHub repository](https://github.com/NandhaKishorM/laya)
- [Laya online documentation](https://nandhakishorm.github.io/laya/)
- [Laya fine-tuning notebook (Kaggle)](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)