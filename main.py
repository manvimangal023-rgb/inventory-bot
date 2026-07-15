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

# Location questions get their own category: the DB has no location data at all,
# so these should ALWAYS go to a real web search, clearly labeled as such --
# never answered from the inventory table (which would be a wrong non-answer)
# and never left to the LLM to decide whether/how to call a tool (which is what
# produced the fabricated "Alberta, Canada" answer in the first place).
_LOCATION_PATTERNS = ("where is", "where's", "located", "location of")

# Write/confirmation language should always go through the LLM's confirm-before-write
# flow, never the deterministic read-only path.
_UPDATE_PATTERNS = (
    "update", "set the", "set ", "change the", "increase", "decrease",
    "confirm", "yes, do it", "add ", "remove ",
)


def _looks_like_definition(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in _DEFINITION_PATTERNS)


def _looks_like_location_question(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in _LOCATION_PATTERNS)


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


def _try_deterministic_location(message: str, history: Optional[list] = None) -> Optional[str]:
    """
    Location questions never touch the inventory DB (it has no location data) and
    never go through the LLM's tool-choice mechanism (which is what let the model
    invent a fake tool and a fake Canadian location earlier in this conversation).
    Instead, call the real web_search function directly ourselves. The result is
    always explicitly labeled "[Web search result, not from the inventory database]"
    so there's no ambiguity about where the fact came from.
    """
    plant = _resolve_plant(message, history)
    query = f"{plant} location" if plant else message
    return web_search(query)


def _looks_like_update(message: str) -> bool:
    m = message.lower().strip()
    if m in ("yes", "confirm", "do it", "yes do it", "yep", "go ahead"):
        return True
    return any(p in m for p in _UPDATE_PATTERNS)


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

    if any(p in lower for p in ("list all plants", "which plants", "what plants", "name all the plants", "all plant names")):
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


@app.get("/chat")
def chat(message: str):
    """Simple version, no memory -- kept for backward-compatible testing via /docs."""
    if _looks_like_location_question(message):
        return {"message": message, "answer": _try_deterministic_location(message)}

    if not _looks_like_update(message) and not _looks_like_definition(message):
        direct = _try_deterministic_lookup(message)
        if direct is not None:
            return {"message": message, "answer": direct}

    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured. Check your .env file.")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    try:
        answer = run_chat(messages)
    except HTTPException:
        raise
    except Exception as e:
        # Any unexpected crash here used to return a non-JSON 500 page, which the
        # frontend can't parse -- it looked like "backend unreachable" even though
        # the server was up. Always return real JSON so the frontend can show the
        # actual error instead of a misleading network-failure message.
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
    """
    # Deterministic path first: if this clearly names a known plant/material and
    # isn't an update or a definition question, answer straight from the DB with
    # no LLM involved at all -- so the answer can never be hallucinated or mixed
    # up with the wrong tool/plant.
    if _looks_like_location_question(req.message):
        return {"message": req.message, "answer": _try_deterministic_location(req.message, req.history)}

    if not _looks_like_update(req.message) and not _looks_like_definition(req.message):
        direct = _try_deterministic_lookup(req.message, req.history)
        if direct is not None:
            return {"message": req.message, "answer": direct}

    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured. Check your .env file.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})
    try:
        answer = run_chat(messages)
    except HTTPException:
        raise
    except Exception as e:
        # Same reasoning as the /chat GET endpoint above: never let an unhandled
        # exception produce a non-JSON response, since that's what made a real
        # crash (e.g. the web_search argument-mismatch bug) look like "backend
        # unreachable" to the frontend instead of a visible, debuggable error.
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
