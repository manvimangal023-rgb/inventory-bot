"""
FastAPI backend for the inventory bot.
Wraps inventory.db (SQLite) with simple HTTP endpoints.
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

load_dotenv()

app = FastAPI(title="Inventory Bot API", version="1.0")

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "inventory.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/")
def root():
    return {"message": "Inventory Bot API is running. Visit /docs for usage."}


@app.get("/plants")
def list_plants():
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT plant FROM inventory ORDER BY plant").fetchall()
    conn.close()
    return {"plants": [r["plant"] for r in rows]}


@app.get("/inventory")
def get_inventory(plant: Optional[str] = None, material: Optional[str] = None):
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
    if tavily_client is None:
        raise HTTPException(status_code=500, detail="Tavily API key not configured. Check your .env file.")

    result = tavily_client.search(query=query, max_results=3, include_answer=True)

    return {
        "query": query,
        "answer": result.get("answer"),
        "sources": [
            {"title": r["title"], "url": r["url"], "snippet": r["content"][:200]}
            for r in result.get("results", [])
        ]
    }


def _normalize(word: str) -> str:
    w = word.lower().strip()
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    return w


def query_inventory(plant: str = "", material: str = "") -> str:
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
    if tavily_client is None:
        return "Web search is not configured."
    result = tavily_client.search(query=query, max_results=3, include_answer=True)
    return result.get("answer") or "No answer found."


def update_inventory(plant: str, material: str, new_quantity: int) -> str:
    conn = get_connection()
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
    "You are an inventory assistant for a renewable energy company. You have 3 tools: "
    "query_inventory (look up plant/material data), web_search (general questions), and "
    "update_inventory (change a quantity -- only call this after the user has clearly said "
    "yes/confirm to a specific change you proposed). "
    "Rules: always use a tool to answer, never guess from memory. Your final reply must state "
    "the actual answer (a number, name, or fact) -- never just say you'll check or look something up. "
    "Use earlier messages in this conversation for context, e.g. remembering which plant was just discussed."
)


LEAKED_TOOLCALL_MARKERS = [
    "<|python_tag|>", "python_tag", "query_inventory(", "web_search(", "update_inventory(",
    '"material":', '"plant":', '"new_quantity":', '"query":',
]

INTENT_WORDS = ["i will", "i'll", "let me", "i am going to", "i'm going to", "going to"]
ACTION_WORDS = ["check", "query", "search", "look", "find", "verify", "confirm", "get the", "pull the", "fetch"]


def _looks_like_filler(text: str) -> bool:
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
    max_rounds = 4
    force_next_round = True
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
                        continue
                except Exception:
                    pass

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
            if _looks_like_filler(response_message.content) and round_num < max_rounds - 1:
                messages.append({"role": "user", "content": (
                    "You did not actually provide an answer -- either you only said you would "
                    "check, or you wrote out tool-call syntax as plain text instead of using the "
                    "real tool-calling mechanism. Call the appropriate tool now, properly, and "
                    "give the real answer."
                )})
                force_next_round = True
                continue
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
    history: list[dict] = []


@app.post("/chat")
def chat_with_memory(req: ChatRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured. Check your .env file.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})
    answer = run_chat(messages)
    return {"message": req.message, "answer": answer}


@app.get("/inventory/summary")
def inventory_summary():
    conn = get_connection()
    rows = conn.execute("""
        SELECT plant, COUNT(*) AS material_types, SUM(quantity) AS total_units
        FROM inventory
        GROUP BY plant
        ORDER BY plant
    """).fetchall()
    conn.close()
    return {"summary": [dict(r) for r in rows]}
