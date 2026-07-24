# Finance Agent

An **autonomous AI agent** that answers finance questions by deciding which tools to call — `get_stock_price` or `get_ratios` — based on the user's question. It maintains conversation memory across turns.

## What Makes This an Agent (Not Just a Chatbot)

| Chatbot | This Agent |
|---------|------------|
| Single-turn, no tools | **Multi-turn with tool calling** |
| Pre-defined responses | **Decides which tool(s) to call** based on question |
| No memory | **Full conversation history** carried forward |
| Can't combine data | **Calls multiple tools** in one turn if needed |

## Tools Available

| Tool | Description | When It's Called |
|------|-------------|------------------|
| `get_stock_price(ticker)` | Returns current market price only | User asks "What's AAPL trading at?" |
| `get_ratios(ticker)` | Returns P/E, P/B, ROE, ROA, NPM | User asks "Is AAPL cheap?" or "Show me AAPL ratios" |

## Example Queries

### Single-Tool Queries
```
You: What is the price of AAPL?
→ Agent calls get_stock_price("AAPL") → Returns $321.45

You: What are the key financial ratios for HDFCBANK.NS?
→ Agent calls get_ratios("HDFCBANK.NS") → Returns P/E, P/B, ROE, ROA, NPM
```

### Multi-Tool Queries (Agent Decides to Call Both)
```
You: Give me price and ratios for AAPL
→ Agent calls get_stock_price("AAPL") AND get_ratios("AAPL")
```

### Valuation/Comparison Queries (Agent Chooses get_ratios)
```
You: Compare HDFC Bank and Axis Bank valuation — which looks cheaper?
→ Agent calls get_ratios("HDFCBANK.NS") AND get_ratios("AXISBANK.NS")
→ Compares P/E, P/B, ROE, ROA, NPM → "Axis Bank is cheaper on P/E (13.77x vs 16.68x)"
```

### Follow-Up Questions (Uses Conversation Memory)
```
You: What is the P/E ratio of HDFCBANK.NS?
→ Agent calls get_ratios("HDFCBANK.NS") → P/E = 16.68x

You: What about Axis Bank?
→ Agent calls get_ratios("AXISBANK.NS") → P/E = 13.77x (remembers context)

You: Which one is cheaper based on P/E?
→ Agent answers from history: "Axis Bank (13.77x vs 16.68x)" — NO new tool calls
```

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│  System Prompt + Conversation History  │
│  (system + user1 + assistant1 + ...)   │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  OpenRouter LLM (openrouter/free)      │
│  Sees: messages + tool schemas         │
│  Decides: respond OR call tool(s)      │
└─────────────────────────────────────┘
     │
     ├─→ No tool_calls → Final answer
     │
     └─→ tool_calls → Execute tool(s)
         │
         ▼
    Append tool results as role="tool" messages
         │
         ▼
    Loop back to LLM with updated history
```

## Setup

```bash
pip install yfinance openai
export OPENROUTER_API_KEY="sk-or-..."
python agent.py
```

## Run

```bash
python agent.py
```

Then ask questions interactively. Type `exit` to quit.

## Conversation History

The global `messages` list accumulates across turns:
- `system` prompt (once)
- Each `user` question
- Each `assistant` response (including `tool_calls`)
- Each `tool` result (with `tool_call_id`)

This means follow-ups like "What about X?" or "Which is cheaper?" work naturally — the agent sees the full context.

## Requirements

- Python 3.10+
- `yfinance` for market data
- `openai` client (works with OpenRouter's OpenAI-compatible API)
- OpenRouter API key (free tier: 50 requests/day)

## Model

Default: `openrouter/free` — auto-routes to an available free model that supports tool calling (e.g., Nemotron 3 Ultra).

Change in `agent.py`:
```python
MODEL = "openrouter/free"  # or specific model like "nvidia/nemotron-3-ultra-550b-a55b:free"
```