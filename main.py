"""
FastAPI backend for the inventory bot.
Wraps inventory.db (SQLite) with simple HTTP endpoints.

Run locally with:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API documentation
(FastAPI auto-generates this -- try it, it's the easiest way to test endpoints
by hand before wiring up a frontend).
"""
import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq
import json

load_dotenv()  # reads the .env file and loads API keys into the environment

app = FastAPI(title="Inventory Bot API", version="1.0")

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local dev; tighten this before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "inventory.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us return rows as dicts, not raw tuples
    return conn


@app.get("/")
def root():
    return {"message": "Inventory Bot API is running. Visit /docs for usage."}


@app.get("/plants")
def list_plants():
    """Returns every distinct plant name in the database."""
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT plant FROM inventory ORDER BY plant").fetchall()
    conn.close()
    return {"plants": [r["plant"] for r in rows]}


@app.get("/inventory")
def get_inventory(plant: Optional[str] = None, material: Optional[str] = None):
    """
    Returns inventory rows, optionally filtered by plant and/or material.
    Examples:
      /inventory                          -> everything (106 rows)
      /inventory?plant=Ron Wind Farm      -> just that plant
      /inventory?material=Generator       -> that material across all plants
    """
    conn = get_connection()
    query = "SELECT plant, material, quantity, basis FROM inventory WHERE 1=1"
    params = []
    if plant:
        query += " AND plant LIKE ?"
        params.append(f"%{plant}%")
    if material:
        query += " AND material LIKE ?"
        params.append(f"%{material}%")

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No matching inventory found.")

    return {"count": len(rows), "results": [dict(r) for r in rows]}


@app.get("/websearch")
def web_search(query: str):
    """
    Answers a general question by searching the live web via Tavily.
    Use this for anything NOT in the plant inventory database --
    e.g. "what does SCADA stand for", "current price of steel".
    """
    if tavily_client is None:
        raise HTTPException(
            status_code=500,
            detail="Tavily API key not configured. Check your .env file."
        )

    result = tavily_client.search(query=query, max_results=3, include_answer=True)

    return {
        "query": query,
        "answer": result.get("answer"),          # Tavily's direct AI-generated answer
        "sources": [
            {"title": r["title"], "url": r["url"], "snippet": r["content"][:200]}
            for r in result.get("results", [])
        ]
    }


def _normalize(word: str) -> str:
    """Strip common plural endings so 'gearboxes' matches 'Gearbox', 'bearings' matches 'Bearing', etc."""
    w = word.lower().strip()
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    return w


def query_inventory(plant: str = "", material: str = "") -> str:
    """Query the real plant inventory database."""
    conn = get_connection()
    query = "SELECT plant, material, quantity FROM inventory WHERE 1=1"
    params = []
    if plant:
        query += " AND plant LIKE ?"
        params.append(f"%{plant}%")
    if material:
        query += " AND material LIKE ?"
        params.append(f"%{_normalize(material)}%")
    rows = conn.execute(query, params).fetchall()
    conn.close()
    if not rows:
        return "No matching inventory found."
    return "\n".join(f"{r['plant']}: {r['material']} = {r['quantity']}" for r in rows)


def web_search(query: str) -> str:
    """Search the live internet for general questions."""
    if tavily_client is None:
        return "Web search is not configured."
    result = tavily_client.search(query=query, max_results=3, include_answer=True)
    return result.get("answer") or "No answer found."


def update_inventory(plant: str, material: str, new_quantity: int) -> str:
    """Update the quantity of a specific material at a specific plant. Requires an exact plant+material match."""
    conn = get_connection()
    # find the matching row first, so we can confirm exactly what's being changed
    rows = conn.execute(
        "SELECT id, plant, material, quantity FROM inventory WHERE plant LIKE ? AND material LIKE ?",
        (f"%{plant}%", f"%{_normalize(material)}%"),
    ).fetchall()

    if not rows:
        conn.close()
        return f"No matching row found for plant='{plant}', material='{material}'. Nothing was updated."
    if len(rows) > 1:
        conn.close()
        matches = "; ".join(f"{r['plant']} - {r['material']}" for r in rows)
        return f"Multiple matches found, update aborted to avoid changing the wrong row: {matches}. Please be more specific."

    row = rows[0]
    old_quantity = row["quantity"]
    conn.execute("UPDATE inventory SET quantity = ? WHERE id = ?", (new_quantity, row["id"]))
    conn.commit()
    conn.close()
    return f"Updated: {row['plant']} - {row['material']} changed from {old_quantity} to {new_quantity}."


# Groq (OpenAI-compatible) tool schema -- describes each function so the model can pick one
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": (
                "Query the real plant inventory database. Use this for ANY question about "
                "specific wind/solar plants, their materials (turbines, panels, gearboxes, "
                "inverters, etc.), or quantities held."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant": {"type": "string", "description": "Plant name to filter by, or empty for all plants"},
                    "material": {"type": "string", "description": "Material name to filter by, or empty for all materials"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live internet for general questions NOT related to our specific "
                "plant inventory database -- e.g. definitions, current events, prices, general facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_inventory",
            "description": (
                "Update (overwrite) the quantity of a specific material at a specific plant in the "
                "real database. Only call this after the user has clearly confirmed they want the "
                "change made -- e.g. they said 'yes, update it' or similar. You must pass the FINAL "
                "absolute new_quantity (not a delta) -- if the user says 'add 5', first find the "
                "current quantity via query_inventory, then compute and pass current+5 as new_quantity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant": {"type": "string", "description": "Exact or partial plant name"},
                    "material": {"type": "string", "description": "Exact or partial material name"},
                    "new_quantity": {"type": "integer", "description": "The final absolute quantity to set"},
                },
                "required": ["plant", "material", "new_quantity"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "query_inventory": query_inventory,
    "web_search": web_search,
    "update_inventory": update_inventory,
}


SYSTEM_PROMPT = (
    "You are an assistant for a renewable energy plant inventory system. "
    "You have three tools: query_inventory (for reading questions about specific plants, "
    "materials, or quantities), web_search (for general questions), and update_inventory "
    "(for changing a quantity in the real database). "
    "If query_inventory returns 'No matching inventory found', try web_search "
    "before giving up. Never write a function call out as plain text -- always "
    "use the proper tool-calling mechanism. "
    "CRITICAL: once a tool returns a result, your final answer MUST directly state "
    "the actual information found (the number, the definition, the fact) -- never "
    "reply with only a vague acknowledgment like 'let me know if you need anything else' "
    "without first stating what was actually found. "
    "You have access to the earlier messages in this conversation -- use them for context "
    "(e.g. if the user says 'what about bearings', check what plant was discussed earlier). "
    "IMPORTANT for updates: this is a real, permanent change to the database, so only call "
    "update_inventory after the user has explicitly confirmed (e.g. they said 'yes', 'do it', "
    "'confirm', or similar) in response to you asking them to confirm the exact change. If they "
    "ask to add/remove/change a quantity for the first time without having already confirmed, "
    "first look up the current value with query_inventory, state the exact change you're about "
    "to make (old value -> new value) and ask them to confirm before calling update_inventory. "
    "NEVER reply with a mid-process filler phrase like 'let me check', 'let me look into that', "
    "'let me search', or 'checking now' as your FINAL answer -- these are only acceptable as "
    "internal reasoning before you actually call a tool, never as the message shown to the user. "
    "Every final answer must contain the actual requested information (a number, a name, a fact, "
    "or an explicit question asking for confirmation/clarification) -- not a promise to look something up."
)


# Signs the model leaked its internal tool-call syntax as plain text instead of
# using the actual tool-calling mechanism (a known quirk of some Llama models on Groq)
LEAKED_TOOLCALL_MARKERS = [
    "<|python_tag|>", "python_tag", "query_inventory(", "web_search(", "update_inventory(",
]

# Broader filler detection: catches "I will/I'll/let me/going to" + an action verb
# (check/query/search/look/find), regardless of the exact phrasing used -- this covers
# variants like "I will check", "let me query", "I'll search", "going to look up", etc.
INTENT_WORDS = ["i will", "i'll", "let me", "i am going to", "i'm going to", "going to"]
ACTION_WORDS = ["check", "query", "search", "look", "find", "verify", "confirm", "get the", "pull the", "fetch"]


def _looks_like_filler(text: str) -> bool:
    """Detect an unfinished 'I'll go look this up' reply, OR a leaked raw tool-call, instead of a real answer."""
    if not text:
        return True
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in LEAKED_TOOLCALL_MARKERS):
        return True
    if len(text) < 200:
        has_intent = any(word in lowered for word in INTENT_WORDS)
        has_action = any(word in lowered for word in ACTION_WORDS)
        if has_intent and has_action:
            return True
    return False


def run_chat(messages: list) -> str:
    """Core logic: runs the tool-calling loop given a full message list (with history), returns final answer text."""
    max_rounds = 4  # one extra round of headroom for the filler-retry below
    force_next_round = True  # always force a tool on round 0
    for round_num in range(max_rounds):
        force_tool = force_next_round
        force_next_round = False
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="required" if force_tool else "auto",
                temperature=0,
            )
        except Exception as e:
            if "tool_use_failed" in str(e):
                # First, give it one more real attempt WITH tools (the failure may be transient)
                try:
                    retry_response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="required",
                        temperature=0,
                    )
                    retry_message = retry_response.choices[0].message
                    if retry_message.tool_calls:
                        # succeeded this time -- process it as a normal tool-calling round
                        response_message = retry_message
                        tool_calls = retry_message.tool_calls
                        messages.append(response_message)
                        for tool_call in tool_calls:
                            fn_name = tool_call.function.name
                            fn_args = json.loads(tool_call.function.arguments)
                            fn = AVAILABLE_FUNCTIONS.get(fn_name)
                            try:
                                result = fn(**fn_args) if fn else "Unknown tool requested."
                            except TypeError as te:
                                result = f"Tool call had invalid arguments ({te}). Please try again."
                            messages.append({
                                "role": "tool", "tool_call_id": tool_call.id,
                                "name": fn_name, "content": result,
                            })
                        continue  # go round the main loop again to get the final answer
                except Exception:
                    pass  # fall through to the no-tools fallback below

                fallback = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0,
                )
                fallback_text = fallback.choices[0].message.content
                if _looks_like_filler(fallback_text):
                    return (
                        "Sorry, I had trouble processing that request. Could you try rephrasing it "
                        "(e.g. using the exact plant name) or asking again?"
                    )
                return fallback_text
            raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # Caught an unfinished "let me check" reply, or a leaked raw tool-call syntax,
            # with rounds left -- force it to actually call a tool and finish the job.
            if _looks_like_filler(response_message.content) and round_num < max_rounds - 1:
                messages.append({"role": "user", "content": (
                    "You did not actually provide an answer -- either you only said you would "
                    "check, or you wrote out tool-call syntax as plain text instead of using the "
                    "real tool-calling mechanism. Call the appropriate tool now, properly, and "
                    "give the real answer."
                )})
                force_next_round = True  # make the retry round force a proper tool call too
                continue
            # Last chance and it's STILL broken -- never show raw/leaked syntax to the user.
            if _looks_like_filler(response_message.content):
                return (
                    "Sorry, I had trouble processing that request. Could you try rephrasing it "
                    "(e.g. using the exact plant name) or asking again?"
                )
            return response_message.content

        messages.append(response_message)
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = AVAILABLE_FUNCTIONS.get(fn_name)
            try:
                result = fn(**fn_args) if fn else "Unknown tool requested."
            except TypeError as e:
                result = f"Tool call had invalid arguments ({e}). Please try again with correct parameters."
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": result,
            })

    final = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    final_text = final.choices[0].message.content
    if _looks_like_filler(final_text):
        return (
            "Sorry, I had trouble processing that request. Could you try rephrasing it "
            "(e.g. using the exact plant name) or asking again?"
        )
    return final_text


@app.get("/chat")
def chat(message: str):
    """Simple version, no memory -- kept for backward-compatible testing via /docs."""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured. Check your .env file.")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    answer = run_chat(messages)
    return {"message": message, "answer": answer}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]


@app.post("/chat")
def chat_with_memory(req: ChatRequest):
    """
    Memory-aware version: pass prior conversation turns in `history` so the bot
    can reference earlier questions (e.g. "what about bearings" after asking about a plant).
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured. Check your .env file.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})
    answer = run_chat(messages)
    return {"message": req.message, "answer": answer}


@app.get("/inventory/summary")
def inventory_summary():
    """Returns total item-types and total units per plant -- a quick overview."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT plant, COUNT(*) AS material_types, SUM(quantity) AS total_units
        FROM inventory
        GROUP BY plant
        ORDER BY plant
    """).fetchall()
    conn.close()
    return {"summary": [dict(r) for r in rows]}
