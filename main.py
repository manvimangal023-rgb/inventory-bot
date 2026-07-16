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
def websearch_endpoint(query: str):
    """
    Manual-testing endpoint: hit /websearch?query=... directly via /docs to see
    Tavily's raw answer + sources, separate from the internal web_search() used
    by the chat tool-calling loop below.
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
    """Search the live internet for general questions (NOT for inventory data -- that always
    comes from query_inventory). The returned string is explicitly labeled as a web result so
    it's never mistaken for verified database data."""
    if tavily_client is None:
        return "Web search is not configured."
    result = tavily_client.search(query=query, max_results=3, include_answer=True)
    answer = result.get("answer")
    if not answer:
        return "No answer found from web search."
    sources = result.get("results", [])
    source_line = f" (source: {sources[0]['url']})" if sources else ""
    return f"[Web search result, not from the inventory database] {answer}{source_line}"


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


# ---------------------------------------------------------------------------
# Deterministic routing layer.
#
# Every bug so far (fake tool narrations, wrong tool names, malformed arguments,
# crashes) came from the same root cause: relying on the LLM to *decide* whether
# and how to call a tool for a task that's actually fully deterministic -- we
# have a fixed, known list of 13 plants and 16 materials in a 106-row table.
#
# So for ordinary lookup questions, we skip the LLM entirely: match the message
# against the known plant/material names in plain Python and query the DB
# directly. The LLM only gets involved for genuinely open-ended things it's
# actually needed for: general-knowledge questions (web_search) and inventory
# updates (which require the confirm-before-write conversation flow).
# ---------------------------------------------------------------------------

import difflib

# Casual ways people refer to each plant, so "ron farm" or "nizamabad" still
# matches even though they don't type the full official name.
PLANT_ALIASES = {
    "Ron Wind Farm": ["ron wind farm", "ron farm", "ron"],
    "Limbwas-I Wind Park": ["limbwas-i wind park", "limbwas-i", "limbwas 1", "limbwas i"],
    "Limbwas-III Wind Park": ["limbwas-iii wind park", "limbwas-iii", "limbwas 3", "limbwas iii"],
    "Code Phase Wind Park": ["code phase wind park", "code phase"],
    "Jasdan Wind Site": ["jasdan wind site", "jasdan"],
    "Karnataka RTC Wind Project": ["karnataka rtc wind", "karnataka wind project", "karnataka wind"],
    "Nizamabad Solar Farm": ["nizamabad solar farm", "nizamabad"],
    "Madhya Pradesh Solar Project": ["madhya pradesh solar", "madhya pradesh"],
    "Rajasthan RTC Solar Project": ["rajasthan rtc solar", "rajasthan rtc"],
    "Karnataka RTC Solar Project": ["karnataka rtc solar"],
    "ReNew Peak Power Project - Wind (Andhra Pradesh)": [
        "peak power project wind", "peak power wind", "andhra pradesh wind",
    ],
    "ReNew Peak Power Project - Solar (Andhra Pradesh)": [
        "peak power project solar", "peak power solar", "andhra pradesh solar",
    ],
    "Jaisalmer Solar Project (Rajasthan)": ["jaisalmer solar project", "jaisalmer"],
}

# ---------------------------------------------------------------------------
# Plant locations, kept right here in code (no database change needed).
#
# WHY THIS EXISTS: location questions used to be routed to a live web search,
# because the inventory table stores no location. But searching a plant *name*
# on the open web is unreliable -- "Ron Wind Farm" matched a fictional wind farm
# from a video game, and the bot returned that. Since we have a fixed list of 13
# real plants, their locations are fixed facts we can just store once, right here,
# and answer instantly and correctly from our own system -- never the web.
#
# ⚠️ VERIFY THESE BEFORE YOUR DEMO. The values below are best-effort based on the
# plant names; several are inferred and may be wrong (especially Ron, Limbwas, and
# Code Phase). Replace any that are inaccurate with the real site/state from your
# own ReNew Power records. Editing is easy -- just change the text on the right.
# Any plant left out of this dict will make the bot honestly say it has no
# recorded location for that plant, rather than guessing.
# ---------------------------------------------------------------------------
PLANT_LOCATIONS = {
    "Ron Wind Farm": "Gadag district, Karnataka, India",            # ⚠️ verify
    "Limbwas-I Wind Park": "Ratlam district, Madhya Pradesh, India",    # ⚠️ verify
    "Limbwas-III Wind Park": "Ratlam district, Madhya Pradesh, India",  # ⚠️ verify
    "Code Phase Wind Park": "Madhya Pradesh, India",                # ⚠️ verify
    "Jasdan Wind Site": "Jasdan, Rajkot district, Gujarat, India",  # ⚠️ verify
    "Karnataka RTC Wind Project": "Karnataka, India",
    "Nizamabad Solar Farm": "Nizamabad district, Telangana, India",
    "Madhya Pradesh Solar Project": "Madhya Pradesh, India",
    "Rajasthan RTC Solar Project": "Rajasthan, India",
    "Karnataka RTC Solar Project": "Karnataka, India",
    "ReNew Peak Power Project - Wind (Andhra Pradesh)": "Andhra Pradesh, India",
    "ReNew Peak Power Project - Solar (Andhra Pradesh)": "Andhra Pradesh, India",
    "Jaisalmer Solar Project (Rajasthan)": "Jaisalmer district, Rajasthan, India",
}

# Casual/synonym ways people refer to each material. Order matters -- more
# specific aliases are listed first so e.g. "yaw bearing" beats a generic "bearing".
MATERIAL_ALIASES = {
    "Blade (pitch) Bearing": ["blade bearing", "pitch bearing", "blade"],
    "Yaw Bearing (slewing ring)": ["yaw bearing", "slewing ring"],
    "Main Shaft Bearing": ["main shaft bearing", "shaft bearing"],
    "Brake Disc": ["brake disc", "brake discs"],
    "Brake Pad Set": ["brake pad", "brake pads"],
    "Central Inverter": ["central inverter", "inverter"],
    "DC Combiner Box": ["combiner box", "dc combiner"],
    "Distribution Transformer": ["transformer"],
    "Gearbox (multi-stage planetary)": ["gearbox", "gear box"],
    "Generator": ["generator"],
    "IGBT Power Module": ["igbt", "power module"],
    "Mounting Structure Set": ["mounting structure", "mounting"],
    "PV Panel (540W module)": ["pv panel", "solar panel", "panel", "panels"],
    "SCADA/Monitoring Unit": ["scada unit", "monitoring unit"],
    "Slip Ring Assembly": ["slip ring"],
    "Wind Turbine (complete unit)": ["wind turbine", "turbine"],
}

# "what does SCADA mean" should never be treated as an inventory lookup, even
# though "SCADA" is also a material name -- route these to the LLM/web_search path.
_DEFINITION_PATTERNS = (
    "what does", "what is the meaning", "meaning of", "define ",
    "explain what", "what's the definition",
)

# Location questions get their own category. The inventory TABLE has no location
# column, but PLANT_LOCATIONS above holds the real location of each of our 13
# plants -- so these are answered from our own records, in plain Python, with NO
# LLM and NO web search. That's what stops a plant name from being searched on the
# open web and matching something unrelated (the GTA "wind farm" incident).
_LOCATION_PATTERNS = ("where is", "where's", "located", "location of")

# Write/confirmation language should always go through the LLM's confirm-before-write
# flow, never the deterministic read-only path.
_UPDATE_PATTERNS = (
    "update", "set the", "set ", "change the", "increase", "decrease",
    "confirm", "yes, do it", "add ", "remove ",
)

# Greetings / small talk. These used to break the bot: a message like "hello"
# names no plant, material, or location, so it fell through to the LLM path --
# where tool_choice="required" FORCED the model to call a tool anyway, so it just
# grabbed a plant's inventory and dumped it (that's why "hello" returned Ron Wind
# Farm's full stock list). We now catch these first and reply conversationally,
# touching no tool and no database at all.
_GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "helloo", "hellow", "hey", "heyy", "heya",
    "yo", "hola", "sup", "howdy", "greetings", "morning", "afternoon",
    "evening", "namaste", "thanks", "thank", "thankyou", "thanku", "thx", "ty",
    "cheers", "ok", "okay", "cool", "nice", "great", "bye", "goodbye",
}
_GREETING_PHRASES = (
    "how are you", "what can you do", "what do you do", "who are you",
    "what are you", "can you help", "need help", "how do you work",
    "what is this", "help me", "what can i ask",
)

_GREETING_REPLY = (
    "Hello! I'm the inventory assistant for our renewable energy plants. I can help you with:\n"
    "- Stock levels — e.g. \"How many turbines are at Ron Wind Farm?\"\n"
    "- Plant locations — e.g. \"Where is Nizamabad Solar Farm located?\"\n"
    "- Updating quantities — e.g. \"Set the generators at Jasdan Wind Site to 25\"\n"
    "- General questions — e.g. \"What is SCADA?\"\n"
    "What would you like to know?"
)


def _looks_like_definition(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in _DEFINITION_PATTERNS)


# Broader "general knowledge / definition" phrasing. A message like "what is RTC
# solar project" or "why is SCADA useful" is a question about a CONCEPT, not a
# request for stock levels -- so when it doesn't name a specific plant, it should
# be answered by web search, never by dumping inventory. This is the check that
# lets the bot finally tell "what IS X" apart from "how many X do we have".
_GENERAL_QUESTION_PATTERNS = (
    "what is", "what are", "what's", "whats", "what're", "what does",
    "what do you mean", "who is", "who are", "who's", "explain",
    "tell me about", "why is", "why are", "why does", "why do",
    "how does", "how do ", "how is", "difference between",
    "define", "definition of", "meaning of",
)


def _looks_like_general_question(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in _GENERAL_QUESTION_PATTERNS)


def _looks_like_location_question(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in _LOCATION_PATTERNS)


def _looks_like_greeting(message: str) -> bool:
    """
    True for greetings / small talk ("hello", "i said hello", "hey there",
    "thanks", "how are you", "what can you do") -- but ONLY when the message
    doesn't actually reference a known plant or material. That guard means a real
    question like "hey, how many turbines at Ron?" is NOT treated as a greeting
    (it names a plant and a material), so it still gets a proper data answer.
    """
    m = message.lower().strip()
    for ch in ".,!?'\"-":
        m = m.replace(ch, " ")
    tokens = [t for t in m.split() if t]
    if not tokens:
        return False

    # If the message clearly names a plant or material, it's a real query, not chit-chat.
    if _match_alias(message, PLANT_ALIASES) or _match_alias(message, MATERIAL_ALIASES):
        return False

    joined = " ".join(tokens)
    if any(p in joined for p in _GREETING_PHRASES):
        return True
    # A short message dominated by greeting words (e.g. "hello", "i said hello", "hey there").
    if len(tokens) <= 6 and any(t in _GREETING_WORDS for t in tokens):
        return True
    return False


def _resolve_plant(message: str, history: Optional[list] = None) -> Optional[str]:
    """
    Find a known plant referenced by this message. If the message itself doesn't
    name one (e.g. a follow-up like "where is it situated" or "what about that one"),
    look backward through recent conversation turns for the last plant mentioned,
    so pronouns resolve correctly instead of being searched/queried literally.
    """
    plant = _match_alias(message, PLANT_ALIASES)
    if plant or not history:
        return plant
    for turn in reversed(history):
        content = turn.get("content", "") if isinstance(turn, dict) else ""
        p = _match_alias(content, PLANT_ALIASES)
        if p:
            return p
    return None


def _try_deterministic_location(message: str, history: Optional[list] = None) -> str:
    """
    Answer a location question from our own PLANT_LOCATIONS records -- never from
    the inventory table (it has no location) and never from a live web search
    (which matched a plant name to an unrelated video-game location before).

    - If we can tell which plant is meant and we have its location -> state it.
    - If we know the plant but haven't recorded its location -> say so honestly.
    - If we can't tell which plant is meant -> ask which plant, rather than
      guessing or searching the web.
    """
    plant = _resolve_plant(message, history)
    if plant is None:
        return (
            "I'm not sure which plant you mean. Could you tell me the plant name? "
            "For example: 'Where is Ron Wind Farm located?'"
        )
    location = PLANT_LOCATIONS.get(plant)
    if location:
        return f"{plant} is located in {location}. (from the inventory system's plant records)"
    return (
        f"I don't have a recorded location for {plant} in the inventory system yet. "
        f"You can add it to the plant records so I can answer this next time."
    )


def _looks_like_update(message: str) -> bool:
    m = message.lower().strip()
    if m in ("yes", "confirm", "do it", "yes do it", "yep", "go ahead"):
        return True
    return any(p in m for p in _UPDATE_PATTERNS)


def _is_list_plants_request(message: str) -> bool:
    """
    True when the user wants the full list of plant NAMES (e.g. "list all plants",
    "enlist all the plants", "what plants do you have", "show me every plant").

    The old code matched only a handful of exact phrases, so "enlist all the plants"
    slipped through -- it fell to the forced-tool LLM path, which just dumped one
    plant's inventory instead. This checks the intent generally: the word "plant"
    plus any listing cue, as long as the message doesn't name a SPECIFIC plant
    (in which case it's a question about that one plant, not a list-them-all).
    """
    m = message.lower()
    if "plant" not in m:
        return False
    if _match_alias(message, PLANT_ALIASES):  # names a specific plant -> not a list-all
        return False
    cues = (
        "list", "enlist", "enumerate", "all plant", "all the plant",
        "which plant", "what plant", "name the plant", "name all",
        "plant names", "every plant", "show", "display", "how many plant",
        "the plants", "give me the plant", "tell me the plant",
    )
    return any(c in m for c in cues)


def _match_alias(text: str, alias_map: dict) -> Optional[str]:
    """Return the canonical name whose longest alias appears in `text`, or None."""
    text_l = text.lower()
    best_name, best_len = None, 0
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in text_l and len(alias) > best_len:
                best_name, best_len = canonical, len(alias)
    if best_name:
        return best_name
    # Fuzzy fallback to tolerate typos (e.g. "nizambad" instead of "nizamabad")
    words = [w for w in text_l.replace("?", " ").replace(",", " ").split() if len(w) > 3]
    best_name, best_score = None, 0.0
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            for w in words:
                score = difflib.SequenceMatcher(None, w, alias).ratio()
                if score > best_score:
                    best_name, best_score = canonical, score
    return best_name if best_score > 0.82 else None


def _try_deterministic_lookup(message: str, history: Optional[list] = None) -> Optional[str]:
    """
    Attempt to answer directly from the database with zero LLM involvement.
    Returns a formatted answer if the message clearly matches a known plant
    and/or material, or None if it doesn't -- in which case the caller falls
    through to the LLM path (general questions, updates, ambiguous phrasing).
    """
    lower = message.lower()

    if _is_list_plants_request(message):
        conn = get_connection()
        rows = conn.execute("SELECT DISTINCT plant FROM inventory ORDER BY plant").fetchall()
        conn.close()
        names = [r["plant"] for r in rows]
        return "The plants in the inventory system are:\n" + "\n".join(f"- {n}" for n in names)

    plant = _resolve_plant(message, history)
    material = _match_alias(lower, MATERIAL_ALIASES)

    if not plant and not material:
        return None  # nothing recognizable -- let the LLM/web_search path handle it

    result = query_inventory(plant=plant or "", material=material or "")
    if result == "No matching inventory found.":
        return None  # let it fall through instead of a dead end

    return result


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
    "You have EXACTLY three tools, with these EXACT names: query_inventory (for reading "
    "questions about specific plants, materials, or quantities), web_search (for general "
    "questions), and update_inventory (for changing a quantity in the real database). No "
    "other tools exist -- never reference, invent, or narrate a call to any tool by any "
    "other name (e.g. there is no 'get_plant_info' or similar). "
    "If query_inventory returns 'No matching inventory found', try web_search "
    "before giving up. Never write a function call out as plain text -- always "
    "use the proper tool-calling mechanism, never describe one in prose (e.g. never say "
    "'Using X, I found...') -- either you actually invoked the tool via the real "
    "function-calling mechanism and got a real result, or you say you don't know. "
    "Any specific plant location, quantity, or fact you state MUST have come from an "
    "actual tool result present earlier in this conversation -- never state such a fact "
    "from memory or assumption. "
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


FILLER_PHRASES = [
    "let me check", "let me look", "let me search", "let me find",
    "checking now", "i'll check", "i will check", "let me see",
]


def _looks_like_filler(text: str) -> bool:
    """Detect an unfinished 'I'll go look this up' reply instead of an actual answer."""
    if not text:
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in FILLER_PHRASES) and len(text) < 150


def _serialize_assistant_message(response_message) -> dict:
    """
    Convert the Groq SDK's response message object into a plain dict containing
    ONLY the fields that are valid in an outgoing request (role, content, tool_calls).

    BUG THIS FIXES: the old code did `messages.append(response_message)`, appending the
    raw pydantic object straight back into the message list. response_message.model_dump()
    actually contains several response-only fields (`annotations`, `executed_tools`,
    `function_call`, `reasoning`) that aren't part of a valid request message. When that
    object got serialized into the *next* API call, Groq would sometimes reject it --
    which the code misread as a `tool_use_failed` error and "fixed" by dropping tools
    entirely and letting the model free-associate an answer. That's the root cause of the
    hallucinated, fake tool-call narrations: the model was being asked to answer as if it
    had tool access (because the system prompt still describes the tools) with no actual
    tool attached, so it made something up that *looked* like a real tool result.
    """
    msg = {"role": "assistant", "content": response_message.content}
    if response_message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in response_message.tool_calls
        ]
    return msg


def run_chat(messages: list) -> str:
    """Core logic: runs the tool-calling loop given a full message list (with history), returns final answer text."""
    max_rounds = 5  # one extra round of headroom for the filler-retry / tool-failure-retry below
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
            err = str(e)
            if "tool_use_failed" in err and round_num < max_rounds - 1:
                # The model emitted a malformed tool call. Do NOT drop the tools and let it
                # free-associate an answer -- that's what produced hallucinated "fake tool
                # narrations" before. Instead, tell it plainly what happened and make it
                # retry the tool call for real, with tools still attached.
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last tool call attempt was malformed and was not executed. "
                        "No information was retrieved. Do not answer from memory and do not "
                        "describe a tool call in plain text -- call the tool again using the "
                        "correct function-calling format."
                    ),
                })
                continue
            raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            retry_reason = None
            if force_tool:
                # tool_choice was "required" this round, so a real tool call should have
                # happened. Groq doesn't always enforce tool_choice="required" for this
                # model -- the model can still just write plain text instead, sometimes
                # narrating a completely invented tool name (e.g. "Using get_plant_info, I
                # found...") with fabricated data. That text is NOT a real tool result and
                # must never be trusted or returned as-is.
                retry_reason = (
                    "You did not actually call a tool, even though one was required for this "
                    "turn, and no information was retrieved. Do not narrate a tool call in "
                    "plain text and do not invent tool or function names -- the only tools "
                    "that exist are query_inventory, web_search, and update_inventory. Call "
                    "one of them for real now using the proper function-calling mechanism."
                )
            elif _looks_like_filler(response_message.content):
                # Caught an unfinished "let me check" reply with rounds left -- force it to
                # actually call a tool and finish the job instead of returning filler.
                retry_reason = (
                    "You did not actually provide an answer -- you only said you would check. "
                    "Call the appropriate tool now and give the real answer."
                )

            if retry_reason and round_num < max_rounds - 1:
                messages.append({"role": "user", "content": retry_reason})
                continue
            return response_message.content

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
                        # The model called the tool with the wrong argument names/shape
                        # (e.g. {"params": ...} instead of {"query": ...}). This used to
                        # raise an uncaught TypeError that crashed the whole request with
                        # an unhandled 500 -- which the frontend couldn't even parse as
                        # JSON, so it looked like "backend unreachable" even though the
                        # backend was fine. Now we feed the error back to the model as a
                        # real tool result so it can see what went wrong and retry with
                        # the correct arguments, instead of crashing the request.
                        result = (
                            f"Tool call failed: '{fn_name}' was called with invalid "
                            f"arguments {fn_args!r} ({e}). Retry using exactly the "
                            f"parameter names defined in the tool schema."
                        )
                    except Exception as e:
                        # Catch-all so any other runtime error (bad SQL, etc.) also
                        # becomes a recoverable tool result instead of a crash.
                        result = f"Tool call failed: '{fn_name}' raised an error: {e}."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": result,
            })

    try:
        final = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
        return final.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")


def _answer_read(message: str, history: Optional[list] = None) -> str:
    """
    Router for READ questions (no database writes). This is the fix for the whole
    class of "it dumped a random plant's inventory" bugs: reads NEVER go through the
    LLM's forced tool-choice anymore, so the model can't be pushed into calling
    query_inventory for a message that wasn't actually about inventory.

    Order:
      1. A general/knowledge question ("what is RTC solar project", "why is SCADA
         useful") that does NOT name a specific plant -> web search.
      2. A concrete inventory question (names a plant and/or material, or asks to
         list all plants) -> answer straight from the database.
      3. Anything left over -> web search (clearly labeled), rather than inventing
         an inventory answer.
    """
    plant_ref = _resolve_plant(message, history)
    if _looks_like_general_question(message) and not plant_ref:
        return web_search(message)

    direct = _try_deterministic_lookup(message, history)
    if direct is not None:
        return direct

    return web_search(message)


def _run_update(message: str, history: Optional[list] = None) -> str:
    """
    Updates are the ONLY thing that still uses the tool-calling LLM loop, because
    they need the confirm-before-write conversation flow and the update_inventory tool.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured. Check your .env file.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})
    return run_chat(messages)


# ===========================================================================
# PERMANENT ROUTING: LLM intent classifier (understands ANY phrasing)
#
# The old routing guessed intent from hardcoded keyword lists, so every new way
# of phrasing a question ("enlist all the plants", "what is RTC solar project")
# broke it and needed another keyword added. That never ends.
#
# Instead we now let the LLM do the ONE thing it's great at -- understanding what
# the user wants, in any wording -- but ONLY to CLASSIFY and EXTRACT, never to
# produce the answer. It returns strict JSON: an intent plus the canonical plant/
# material names. Our code then executes deterministically and pulls every actual
# fact from the database or web search. The model never states an inventory number
# or a location itself, so it cannot hallucinate data -- it only decides routing.
#
# If the classifier is unavailable or returns junk, we fall back to the old
# keyword router (_fallback_route), so the bot degrades gracefully, never crashes.
# ===========================================================================

_VALID_PLANTS = list(PLANT_ALIASES.keys())
_VALID_MATERIALS = list(MATERIAL_ALIASES.keys())
_VALID_INTENTS = {
    "greeting", "list_plants", "location",
    "inventory_lookup", "update", "general_question",
}

_CLASSIFIER_SYSTEM = (
    "You are an intent classifier for a renewable-energy plant inventory assistant. "
    "Read the user's latest message (using the earlier conversation for context) and "
    "output ONLY a single JSON object -- no prose, no markdown, no code fences.\n\n"
    "The JSON must have exactly these keys: intent, plant, material, new_quantity.\n\n"
    "intent must be exactly one of:\n"
    "- \"greeting\": greetings, thanks, small talk, or asking what the bot can do.\n"
    "- \"list_plants\": the user wants the list of ALL plant names.\n"
    "- \"location\": the user asks WHERE a plant is located.\n"
    "- \"inventory_lookup\": the user asks about stock / quantity / which materials a "
    "plant holds, or how many of a material exist.\n"
    "- \"update\": the user wants to change/set/add/remove a stored quantity, OR is "
    "confirming a pending change (e.g. \"yes\", \"confirm\", \"go ahead\").\n"
    "- \"general_question\": a general, definitional, or knowledge question that is NOT "
    "about our specific stored stock (e.g. \"what is SCADA\", \"what is an RTC solar project\", "
    "\"why are gearboxes used\").\n\n"
    "plant: the single best match from this EXACT list, copied verbatim, or null:\n"
    f"{_VALID_PLANTS}\n\n"
    "material: the single best match from this EXACT list, copied verbatim, or null:\n"
    f"{_VALID_MATERIALS}\n\n"
    "new_quantity: an integer for an update (the final absolute amount), else null.\n\n"
    "Rules:\n"
    "- plant and material MUST be copied verbatim from the lists above, or be null. Map "
    "nicknames and typos to the correct entry (e.g. \"ron\" -> \"Ron Wind Farm\", "
    "\"nizambad\" -> \"Nizamabad Solar Farm\", \"gearboxes\" -> \"Gearbox (multi-stage planetary)\").\n"
    "- If the user says \"it\", \"that plant\", or similar, resolve which plant from the "
    "earlier conversation.\n"
    "- A question that merely mentions a material type in a definitional way "
    "(\"what is a generator\") is general_question, NOT inventory_lookup.\n"
    "- Output JSON only."
)


def classify_intent(message: str, history: Optional[list] = None) -> Optional[dict]:
    """
    Ask the LLM to classify the message into a structured intent. Returns a dict
    {intent, plant, material, new_quantity} on success, or None to signal that the
    caller should fall back to the keyword router.
    """
    if not groq_client:
        return None
    msgs = [{"role": "system", "content": _CLASSIFIER_SYSTEM}]
    if history:
        msgs.extend(history[-6:])  # a little context for pronouns / follow-ups
    msgs.append({"role": "user", "content": message})
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            temperature=0,
            response_format={"type": "json_object"},  # force valid JSON
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return None  # any failure -> fall back to keyword routing

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        return None

    plant = data.get("plant")
    if plant not in _VALID_PLANTS:
        plant = None
    material = data.get("material")
    if material not in _VALID_MATERIALS:
        material = None
    nq = data.get("new_quantity")
    if not isinstance(nq, int):
        nq = None
    return {"intent": intent, "plant": plant, "material": material, "new_quantity": nq}


def _all_plant_names_reply() -> str:
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT plant FROM inventory ORDER BY plant").fetchall()
    conn.close()
    return "The plants in the inventory system are:\n" + "\n".join(f"- {r['plant']}" for r in rows)


def _location_reply(plant: Optional[str]) -> str:
    if plant is None:
        return (
            "I'm not sure which plant you mean. Could you tell me the plant name? "
            "For example: 'Where is Ron Wind Farm located?'"
        )
    location = PLANT_LOCATIONS.get(plant)
    if location:
        return f"{plant} is located in {location}. (from the inventory system's plant records)"
    return (
        f"I don't have a recorded location for {plant} in the inventory system yet. "
        f"You can add it to the plant records so I can answer this next time."
    )


def _dispatch(data: dict, message: str, history: Optional[list] = None) -> str:
    """
    Execute the classified intent deterministically. Every fact comes from the
    database or web search -- the classifier only chose the route and the names.
    """
    intent = data["intent"]
    plant = data["plant"]
    material = data["material"]

    if intent == "greeting":
        return _GREETING_REPLY

    if intent == "list_plants":
        return _all_plant_names_reply()

    if intent == "location":
        return _location_reply(plant or _resolve_plant(message, history))

    if intent == "general_question":
        return web_search(message)

    if intent == "update":
        return _run_update(message, history)

    # inventory_lookup
    if not plant and not material:
        return (
            "There are records for 13 plants and their materials. Which plant or material "
            "would you like? For example: \"inventory at Ron Wind Farm\" or \"how many generators\"."
        )
    result = query_inventory(plant=plant or "", material=material or "")
    if result == "No matching inventory found.":
        return (
            "I couldn't find any matching inventory for that. Could you name the plant "
            "or material more specifically?"
        )
    return result


def _fallback_route(message: str, history: Optional[list] = None) -> str:
    """
    Keyword-based router, used only when the LLM classifier is unavailable or
    returns something invalid. This is the previous behavior, kept as a safety net.
    """
    if _looks_like_greeting(message):
        return _GREETING_REPLY
    if _looks_like_location_question(message):
        return _try_deterministic_location(message, history)
    if _looks_like_update(message):
        return _run_update(message, history)
    return _answer_read(message, history)


def _handle(message: str, history: Optional[list] = None) -> str:
    """Single entry point: classify with the LLM, dispatch; fall back to keywords."""
    data = classify_intent(message, history)
    if data is None:
        return _fallback_route(message, history)
    return _dispatch(data, message, history)


@app.get("/chat")
def chat(message: str):
    """Simple version, no memory -- kept for backward-compatible testing via /docs."""
    try:
        answer = _handle(message)
    except HTTPException:
        raise
    except Exception as e:
        # Never let an unhandled exception produce a non-JSON response -- that's what
        # made a real crash look like "backend unreachable" to the frontend.
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
    return {"message": message, "answer": answer}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]


@app.post("/chat")
def chat_with_memory(req: ChatRequest):
    """
    Memory-aware version: pass prior conversation turns in `history` so the bot
    can reference earlier questions (e.g. "what about bearings" after asking about a plant).

    Routing is now handled by the LLM intent classifier (classify_intent), which
    understands any phrasing, with the old keyword router kept as an automatic
    fallback. Every actual fact still comes from the database or web search.
    """
    try:
        answer = _handle(req.message, req.history)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
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
