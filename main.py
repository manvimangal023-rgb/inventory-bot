"""
FastAPI backend for the inventory bot.
Wraps inventory.db (SQLite) with simple HTTP endpoints.

ARCHITECTURE (merged from two parallel fixes):
1. DETERMINISTIC ROUTER (primary path): our domain is small and known (13 real
   plants, 16 materials) -- we match plant/material names and detect intent with
   plain Python BEFORE ever involving the LLM. This makes the "wrong tool /
   malformed args / hallucinated answer" bug class structurally impossible for
   anything in our own database, since the LLM never decides whether/how to
   call anything for these questions.
2. FIXED AGENTIC FALLBACK: for genuinely open-ended questions the router can't
   match, we still fall back to LLM tool-calling (web_search) -- but with the
   real root-cause bug fixed: the old code appended the raw SDK response object
   back into the conversation, which carries invalid extra fields that made
   Groq reject subsequent requests. That got misread as `tool_use_failed`,
   triggering a no-tools fallback where the model would narrate fake tool
   calls with invented data. Now we serialize only the valid fields, and every
   tool-call failure mode is fed back to the model as a visible, recoverable
   result instead of crashing or hallucinating.

Run locally with:
    uvicorn main:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""
import os
import re
import json
import sqlite3
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq

load_dotenv()

app = FastAPI(title="Inventory Bot API", version="2.0")

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
def web_search_endpoint(query: str):
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


@app.get("/inventory/summary")
def inventory_summary():
    conn = get_connection()
    rows = conn.execute("""
        SELECT plant, COUNT(*) AS material_types, SUM(quantity) AS total_units
        FROM inventory GROUP BY plant ORDER BY plant
    """).fetchall()
    conn.close()
    return {"summary": [dict(r) for r in rows]}


# ============================================================================
# CORE DATA FUNCTIONS -- used by both the deterministic router and the LLM tools
# ============================================================================

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


def get_plant_info(plant: str = "") -> str:
    """Real location, type, capacity, and commissioning year for a plant -- backed by the plant_info table."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT plant, state, plant_type, capacity_mw, approx_year FROM plant_info WHERE plant LIKE ?",
        (f"%{plant}%",),
    ).fetchall()
    if not rows and plant:
        first_word = plant.strip().split()[0]
        rows = conn.execute(
            "SELECT plant, state, plant_type, capacity_mw, approx_year FROM plant_info WHERE plant LIKE ?",
            (f"%{first_word}%",),
        ).fetchall()
    conn.close()
    if not rows:
        return "No matching plant info found."
    return "\n".join(
        f"{r['plant']}: located in {r['state']}, India. Type: {r['plant_type']}. "
        f"Capacity: {r['capacity_mw']}MW. Approx. commissioned: {r['approx_year']}."
        for r in rows
    )


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


# ============================================================================
# DETERMINISTIC ROUTER -- primary path, no LLM involved at all
# ============================================================================

GENERIC_PLANT_WORDS = {"wind", "solar", "park", "project", "farm", "phase", "site", "the", "of", "in", "renew"}
CONFIRMATION_PHRASES = {"yes", "yeah", "yep", "confirm", "do it", "go ahead", "yes please",
                         "yes do it", "sure", "okay", "ok", "yes confirm", "confirmed"}


def _get_known_plants() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT plant FROM inventory").fetchall()
    conn.close()
    return [r["plant"] for r in rows]


def find_plant(text: str) -> Optional[str]:
    """Whole-word matching against distinctive plant-name tokens (ignoring generic
    words like 'wind'/'solar' and single-letter tokens that could false-match)."""
    text_words = set(re.findall(r"[A-Za-z]+", text.lower()))
    best_plant, best_score = None, 0
    for plant in _get_known_plants():
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", plant)
                  if t.lower() not in GENERIC_PLANT_WORDS and len(t) > 1]
        score = sum(1 for t in tokens if t in text_words)
        if score > best_score:
            best_score, best_plant = score, plant
    return best_plant


def find_material(text: str, plant: Optional[str] = None) -> Optional[str]:
    conn = get_connection()
    if plant:
        rows = conn.execute("SELECT DISTINCT material FROM inventory WHERE plant LIKE ?", (f"%{plant}%",)).fetchall()
    else:
        rows = conn.execute("SELECT DISTINCT material FROM inventory").fetchall()
    conn.close()
    materials = [r["material"] for r in rows]

    text_words = {_normalize(w) for w in re.findall(r"[A-Za-z]+", text)}
    best_material, best_score = None, 0
    for m in materials:
        tokens = [_normalize(t) for t in re.findall(r"[A-Za-z]+", m)
                  if len(t) > 3 and t.lower() not in GENERIC_PLANT_WORDS]
        score = sum(1 for t in tokens if t in text_words)
        if score > best_score:
            best_score, best_material = score, m
    return best_material


def detect_intent(text: str) -> str:
    t = text.lower()
    if (re.search(r"\b(update|change|set)\b.*\d+", t) or re.search(r"\badd\b.*\d+", t)
            or re.search(r"\b(remove|reduce|decrease|subtract)\b.*\d+", t)):
        return "update"
    if any(w in t for w in ["where", "location", "situated", "which state", "what state",
                             "capacity of", "type of plant", "when was", "commissioned"]):
        return "location"
    return "inventory"


def parse_new_quantity(text: str, current_qty: int) -> Optional[int]:
    t = text.lower()
    m = re.search(r"\bto\s+(\d+)\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(add|increase)\b.*?(\d+)", t)
    if m:
        return current_qty + int(m.group(2))
    m = re.search(r"\b(remove|reduce|decrease|subtract)\b.*?(\d+)", t)
    if m:
        return current_qty - int(m.group(2))
    return None


def is_confirmation(text: str) -> bool:
    return text.strip().lower().rstrip(".! ") in CONFIRMATION_PHRASES


def extract_current_quantity(query_result_text: str) -> Optional[int]:
    m = re.search(r"=\s*(\d+)\s*$", query_result_text.strip())
    return int(m.group(1)) if m else None


def route_message(message: str, history: list) -> Optional[str]:
    """Returns a final answer string, or None if nothing in our known domain matched
    (caller should fall back to the agentic LLM path in that case)."""
    if is_confirmation(message):
        for turn in reversed(history):
            if turn.get("role") != "user":
                continue
            prior_text = turn.get("content", "") or ""
            prior_plant = find_plant(prior_text)
            if not prior_plant:
                continue
            prior_material = find_material(prior_text, prior_plant)
            if not prior_material or detect_intent(prior_text) != "update":
                continue
            current = query_inventory(plant=prior_plant, material=prior_material)
            current_qty = extract_current_quantity(current)
            if current_qty is None:
                continue
            new_qty = parse_new_quantity(prior_text, current_qty)
            if new_qty is None:
                continue
            return update_inventory(plant=prior_plant, material=prior_material, new_quantity=new_qty)
        return None

    plant = find_plant(message)
    if not plant:
        return None

    intent = detect_intent(message)

    if intent == "location":
        return get_plant_info(plant=plant)

    if intent == "update":
        material = find_material(message, plant)
        if not material:
            return f"Which material at {plant} would you like to update?"
        current = query_inventory(plant=plant, material=material)
        current_qty = extract_current_quantity(current)
        if current_qty is None:
            return f"Couldn't find a current quantity for {material} at {plant} to base the update on."
        new_qty = parse_new_quantity(message, current_qty)
        if new_qty is None:
            return f"{plant} currently has {current_qty} {material}. What should the new quantity be?"
        return f"{plant} currently has {current_qty} {material}. I'll update it to {new_qty} -- reply 'yes' to confirm."

    material = find_material(message, plant)
    return query_inventory(plant=plant, material=material or "")


# ============================================================================
# AGENTIC FALLBACK -- only reached when route_message() returns None, i.e. the
# question isn't about anything in our known plant/material domain.
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live internet for general questions -- e.g. definitions, current events, prices, general facts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {"web_search": web_search}

SYSTEM_PROMPT = (
    "You are a general-knowledge assistant. You have EXACTLY one tool, web_search, for "
    "anything requiring current or factual information you're not certain of. No other "
    "tools exist -- never reference, invent, or narrate a call to any other tool name. "
    "Never write a function call out as plain text -- either you actually invoked "
    "web_search via the real function-calling mechanism and got a real result, or you "
    "say you don't know. Never describe having used a tool in prose (e.g. never say "
    "'Using X, I found...') unless you actually made that exact tool call. "
    "Your final answer must directly state the actual information found -- never a vague "
    "'let me check' or 'I'll look into that' as a final answer."
)

FILLER_PHRASES = ["let me check", "let me look", "let me search", "let me find",
                   "checking now", "i'll check", "i will check", "let me see"]


def _looks_like_filler(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in FILLER_PHRASES) and len(text) < 150


def _serialize_assistant_message(response_message) -> dict:
    """Only the fields valid in an outgoing request -- NOT the raw SDK object,
    which carries extra response-only fields that can make the next API call fail."""
    msg = {"role": "assistant", "content": response_message.content}
    if response_message.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": tc.type,
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in response_message.tool_calls
        ]
    return msg


def run_agentic_fallback(messages: list) -> str:
    max_rounds = 4
    for round_num in range(max_rounds):
        force_tool = round_num == 0
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="required" if force_tool else "auto",
                temperature=0,
            )
        except Exception as e:
            if "tool_use_failed" in str(e) and round_num < max_rounds - 1:
                messages.append({"role": "user", "content": (
                    "Your last tool call attempt was malformed and was not executed. "
                    "Do not answer from memory -- call web_search again using the "
                    "correct function-calling format."
                )})
                continue
            raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            retry_reason = None
            if force_tool:
                retry_reason = (
                    "You did not actually call web_search, even though it was required. "
                    "Do not narrate a tool call in plain text -- call it for real now."
                )
            elif _looks_like_filler(response_message.content):
                retry_reason = "You only said you would check. Call web_search now and give the real answer."
            if retry_reason and round_num < max_rounds - 1:
                messages.append({"role": "user", "content": retry_reason})
                continue
            return response_message.content or "Sorry, I couldn't find an answer to that."

        messages.append(_serialize_assistant_message(response_message))
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                fn_args = None
                result = f"Tool call failed: arguments were not valid JSON ({e})."
            if fn_args is not None:
                fn = AVAILABLE_FUNCTIONS.get(fn_name)
                if not fn:
                    result = f"Unknown tool requested: '{fn_name}'."
                else:
                    try:
                        result = fn(**fn_args)
                    except TypeError as e:
                        result = f"Tool call failed: invalid arguments {fn_args!r} ({e})."
                    except Exception as e:
                        result = f"Tool call failed: {e}."
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn_name, "content": result})

    try:
        final = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
        return final.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")


def handle_chat(message: str, history: list) -> dict:
    """Try the deterministic router first; only fall back to the LLM if nothing matched."""
    routed = route_message(message, history)
    if routed is not None:
        return {"message": message, "answer": routed, "routed": "deterministic"}

    if not groq_client:
        return {"message": message, "answer": web_search(message), "routed": "web_search_direct"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    try:
        answer = run_agentic_fallback(messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
    return {"message": message, "answer": answer, "routed": "agentic_llm"}


@app.get("/chat")
def chat(message: str):
    """Simple version, no memory -- kept for backward-compatible testing via /docs."""
    return handle_chat(message, history=[])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/chat")
def chat_with_memory(req: ChatRequest):
    """Memory-aware version: pass prior conversation turns in `history`."""
    return handle_chat(req.message, history=req.history)
