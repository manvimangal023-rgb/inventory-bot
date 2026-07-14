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
                    "plant": {"type": "string",
