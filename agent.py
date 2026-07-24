#!/usr/bin/env python3
"""
Finance Agent with Tool Calling via OpenRouter

Tool-calling loop:
1. Send user question + tool schema to model
2. Model responds with either:
   a) Final answer (text) -> DONE
   b) Tool call request (name + args) -> execute tool, append result, repeat from step 1
"""

import os
import json
import yfinance as yf
import traceback
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter OpenRouter API key: ").strip()

MODEL = "openrouter/free"  # auto-routes to available free model
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ─── Tool Definition (OpenAI function-calling schema) ─────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current raw stock price for a given ticker symbol using yfinance. Returns only the current market price — does NOT provide valuation ratios, ratios, or analysis. Use this when the user ONLY wants the raw price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA', 'HDFCBANK.NS')"
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ratios",
            "description": "Get key financial ratios (P/E, P/B, ROE, ROA, NPM) for a ticker using yfinance. Use this tool whenever the user asks about valuation, whether a stock is cheap or expensive, or wants ratios like P/E, P/B, ROE, or ROA — not just raw stock price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA', 'HDFCBANK.NS')"
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False
            }
        }
    }
]

# ─── Tool Implementation ──────────────────────────────────────────────────
def get_stock_price(ticker: str) -> str:
    """Fetch live price via yfinance."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            return f"Could not fetch price for {ticker.upper()}"
        return f"${price:.2f}"
    except Exception as e:
        return f"Error fetching {ticker.upper()}: {e}"


def get_ratios(ticker: str) -> str:
    """Calculate key financial ratios from yfinance data."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Direct ratios from yfinance
        roe = info.get("returnOnEquity")       # decimal, e.g., 1.41 = 141%
        roa = info.get("returnOnAssets")       # decimal
        npm = info.get("profitMargins")        # decimal
        pe = info.get("trailingPE")            # multiple
        pb = info.get("priceToBook")           # multiple

        # Format as percentages or multiples
        def fmt_pct(x):
            return f"{x*100:.2f}%" if x is not None else "N/A"

        def fmt_mult(x):
            return f"{x:.2f}x" if x is not None else "N/A"

        lines = [
            f"ROE (Return on Equity): {fmt_pct(roe)}",
            f"ROA (Return on Assets): {fmt_pct(roa)}",
            f"NPM (Net Profit Margin): {fmt_pct(npm)}",
            f"P/E (Price to Earnings): {fmt_mult(pe)}",
            f"P/B (Price to Book): {fmt_mult(pb)}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error calculating ratios for {ticker.upper()}: {e}"

# ─── Conversation History ────────────────────────────────────────────────
# Global list that accumulates: system → user → assistant (tool calls/results) → user → assistant...
# Persists across multiple calls to run_agent() so the model remembers context.
messages = [
    {"role": "system", "content": "You are a finance assistant. Use get_stock_price for price queries, get_ratios for financial ratio queries, or both if the user asks for both."}
]

# ─── Tool-Calling Loop ────────────────────────────────────────────────────
def run_agent(question: str) -> str:
    global messages
    # Append current user question to history
    messages.append({"role": "user", "content": question})

    while True:
        try:
            print(f"[DEBUG] Sending {len(messages)} messages to model...")
response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1000
        )
        except Exception as e:
            print(f"\n[ERROR] API call failed!")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"\nMessages list sent ({len(messages)} messages):")
            for i, m in enumerate(messages):
                if isinstance(m, dict):
                    role = m.get('role', '?')
                    content = m.get('content', '')
                    tc = m.get('tool_calls')
                else:
                    role = getattr(m, 'role', '?')
                    content = getattr(m, 'content', '') or ''
                    tc = getattr(m, 'tool_calls', None)
                preview = content[:100] if content else (f"tool_calls={len(tc)}" if tc else "(empty)")
                print(f"  [{i}] {role}: {preview}")
            raise

        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            # Model wants to call a tool
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"[TOOL] {name}({args})")

                if name == "get_stock_price":
                    result = get_stock_price(args["ticker"])
                elif name == "get_ratios":
                    result = get_ratios(args["ticker"])
                else:
                    result = f"Unknown tool: {name}"

                print(f"[RESULT] {result}")
                # Feed result back to model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result
                })
            # Loop continues → model sees tool result and decides next step
        else:
            # Model gave final answer
            answer = msg.content or "(no response)"
            return answer

# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Finance Agent (model: {MODEL})")
    print("Ask about stock prices. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        answer = run_agent(q)
        print(f"\nAgent: {answer}\n")