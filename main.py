"""
News Agent Backend - FastAPI Implementation with Enhanced Caching
"""
import os
import json
import sqlite3
import feedparser
import requests
import asyncio
import datetime
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import contextmanager

# Ensure all required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI(title="NewsAgent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
DATABASE_PATH = "news_agent.db"

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS saved_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            url TEXT NOT NULL,
            source TEXT,
            category TEXT,
            published_date TEXT,
            saved_date TEXT NOT NULL
        )
        ''')
        conn.commit()

# Initialize database on startup
init_db()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Enhanced News Cache with expiration and persistence
class NewsCache:
    def __init__(self, expiration_minutes=30, max_size=100, persist_file="news_cache.pkl"):
        self.data = defaultdict(list)
        self.last_updated = {}
        self.expiration = timedelta(minutes=expiration_minutes)
        self.max_size = max_size
        self.persist_file = persist_file
        self.load()

    def is_valid(self, category):
        return (category in self.last_updated and 
                datetime.datetime.now() - self.last_updated[category] < self.expiration)

    def update(self, category, items):
        # Enforce max size
        if len(self.data) >= self.max_size:
            oldest = min(self.last_updated.items(), key=lambda x: x[1])[0]
            del self.data[oldest]
            del self.last_updated[oldest]
        
        self.data[category] = items
        self.last_updated[category] = datetime.datetime.now()
        self.save()

    def get(self, category):
        return self.data.get(category, [])

    def save(self):
        with open(self.persist_file, 'wb') as f:
            pickle.dump({
                'data': dict(self.data),
                'last_updated': self.last_updated
            }, f)

    def load(self):
        try:
            with open(self.persist_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.data = defaultdict(list, cache_data['data'])
                self.last_updated = cache_data['last_updated']
        except (FileNotFoundError, EOFError, pickle.PickleError):
            self.data = defaultdict(list)
            self.last_updated = {}

# Initialize cache
news_cache = NewsCache(expiration_minutes=30)

# Updated and verified news sources
NEWS_SOURCES = {
    "tech": [
        {"type": "rss", "url": "https://www.theverge.com/rss/index.xml", "name": "The Verge"},
        {"type": "rss", "url": "https://feeds.feedburner.com/TechCrunch/", "name": "TechCrunch"},
        {"type": "rss", "url": "https://www.wired.com/feed/rss", "name": "Wired"}
    ],
    "india": [
        {"type": "rss", "url": "https://feeds.feedburner.com/ndtvnews-india-news", "name": "NDTV India"},
        {"type": "rss", "url": "https://www.thehindu.com/news/national/feeder/default.rss", "name": "The Hindu"},
        {"type": "rss", "url": "https://www.indiatoday.in/rss/1206514", "name": "India Today"}
    ],
    "global": [
        {"type": "rss", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "name": "BBC World"},
        {"type": "rss", "url": "https://www.aljazeera.com/xml/rss/all.xml", "name": "Al Jazeera"},
        {"type": "rss", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "name": "NY Times World"}
    ],
    "stocks": [
        {"type": "rss", "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "name": "CNBC Markets"},
        {"type": "rss", "url": "https://www.marketwatch.com/rss/topstories", "name": "MarketWatch"},
        {"type": "rss", "url": "https://www.bloomberg.com/markets2.rss", "name": "Bloomberg"}
    ],
    "innovation": [
        {"type": "rss", "url": "https://www.popsci.com/feed/", "name": "Popular Science"},
        {"type": "rss", "url": "https://www.technologyreview.com/feed/", "name": "MIT Tech Review"},
        {"type": "rss", "url": "https://www.nature.com/subjects/technology.rss", "name": "Nature Technology"}
    ]
}

# News importance scoring
def score_news_importance(title, description):
    if not title or not description:
        return 0
    
    text = f"{title} {description}"
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words]
    
    important_keywords = {
        'launch': 3, 'announce': 3, 'new': 2, 'revolutionary': 4, 'breakthrough': 4,
        'update': 2, 'release': 2, 'major': 2, 'critical': 3, 'important': 3,
        'discover': 3, 'invention': 4, 'crisis': 4, 'emergency': 4, 'urgent': 4,
        'significant': 3, 'milestone': 3, 'achievement': 3, 'innovation': 3
    }
    
    score = 1
    for word in filtered_words:
        if word in important_keywords:
            score += important_keywords[word]
    
    if len(filtered_words) > 100:
        score += 2
    elif len(filtered_words) > 50:
        score += 1
    
    return score

# Robust RSS fetching function
async def fetch_rss_news(source):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml'
        }
        
        try:
            response = requests.get(source["url"], headers=headers, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except:
            feed = feedparser.parse(source["url"])
        
        news_items = []
        
        for entry in feed.entries[:15]:
            try:
                title = entry.get('title', 'No title').strip()
                link = entry.get('link', '')
                
                if not title or not link:
                    continue
                    
                description = entry.get('summary', '')
                if description:
                    soup = BeautifulSoup(description, 'html.parser')
                    description = soup.get_text().strip()
                
                importance_score = score_news_importance(title, description)
                if importance_score >= 3:
                    news_items.append({
                        'title': title,
                        'description': description[:200] + '...' if description else '',
                        'url': link,
                        'source': source["name"],
                        'published_date': entry.get('published', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                        'importance': importance_score,
                        'category': None
                    })
            except Exception as e:
                print(f"Error processing entry from {source['name']}: {str(e)}")
                continue
                
        return news_items
        
    except Exception as e:
        print(f"Critical error with {source['name']}: {str(e)}")
        return []

async def fetch_news_by_category(category):
    if news_cache.is_valid(category):
        return news_cache.get(category)
    
    news_items = []
    for source in NEWS_SOURCES.get(category, []):
        if source["type"] == "rss":
            items = await fetch_rss_news(source)
            for item in items:
                item['category'] = category
            news_items.extend(items)
    
    news_items.sort(key=lambda x: x.get('importance', 0), reverse=True)
    top_items = news_items[:5]
    news_cache.update(category, top_items)
    return top_items

async def fetch_all_news():
    all_news = {}
    for category in NEWS_SOURCES.keys():
        all_news[category] = await fetch_news_by_category(category)
    
    update_time = datetime.datetime.now().strftime('%H:%M:%S')
    await manager.broadcast(json.dumps({
        "event": "news_update",
        "data": {
            "message": f"News updated at {update_time}",
            "categories": list(all_news.keys())
        }
    }))
    
    return all_news

# Scheduler for automated news fetching
scheduler = AsyncIOScheduler()

def setup_scheduler():
    scheduler.add_job(fetch_all_news, CronTrigger(minute=0))  # Hourly updates
    scheduler.start()

# Models
class NewsItem(BaseModel):
    title: str
    description: Optional[str] = None
    url: str
    source: str
    category: str
    published_date: Optional[str] = None
    importance: Optional[int] = None

class SavedNews(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    url: str
    source: str
    category: str
    published_date: Optional[str] = None
    saved_date: str

# API Routes
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/api/news")
async def get_all_news():
    if not any(news_cache.data.values()):
        await fetch_all_news()
    return news_cache.data

@app.get("/api/news/{category}")
async def get_category_news(category: str, force_refresh: bool = False):
    if category not in NEWS_SOURCES:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    if force_refresh or not news_cache.is_valid(category):
        await fetch_news_by_category(category)
    
    return news_cache.get(category)

@app.post("/api/news/save")
async def save_news(news_item: NewsItem):
    with get_db_connection() as conn:
        saved_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute('''
        INSERT INTO saved_news (title, description, url, source, category, published_date, saved_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            news_item.title,
            news_item.description,
            news_item.url,
            news_item.source,
            news_item.category,
            news_item.published_date,
            saved_date
        ))
        conn.commit()
    return {"message": "News saved successfully"}

@app.get("/api/news/saved")
async def get_saved_news():
    with get_db_connection() as conn:
        result = conn.execute('SELECT * FROM saved_news ORDER BY saved_date DESC').fetchall()
        return [dict(row) for row in result]

@app.delete("/api/news/saved/{news_id}")
async def delete_saved_news(news_id: int):
    with get_db_connection() as conn:
        conn.execute('DELETE FROM saved_news WHERE id = ?', (news_id,))
        conn.commit()
    return {"message": f"News item {news_id} deleted"}

@app.get("/api/news/refresh")
async def refresh_news(
    background_tasks: BackgroundTasks,
    force_all: bool = False,
    category: Optional[str] = None
):
    async def do_refresh():
        if category:
            await fetch_news_by_category(category)
        elif force_all:
            await fetch_all_news()
        else:
            for cat in NEWS_SOURCES.keys():
                if not news_cache.is_valid(cat):
                    await fetch_news_by_category(cat)
    
    background_tasks.add_task(do_refresh)
    return {"message": "Refresh initiated"}

@app.get("/api/cache/status")
async def cache_status():
    status = {}
    for category in NEWS_SOURCES.keys():
        status[category] = {
            "last_updated": news_cache.last_updated.get(category),
            "is_valid": news_cache.is_valid(category),
            "item_count": len(news_cache.get(category)),
            "expires_in": (
                (news_cache.last_updated[category] + news_cache.expiration - datetime.datetime.now()).total_seconds() / 60
                if category in news_cache.last_updated else None
            )
        }
    return status

@app.get("/api/debug/feeds")
async def debug_feeds():
    results = {}
    for category, sources in NEWS_SOURCES.items():
        results[category] = []
        for source in sources:
            try:
                response = requests.get(source["url"], timeout=5)
                status = f"HTTP {response.status_code}"
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    status += f" ({len(feed.entries)} entries)"
            except Exception as e:
                try:
                    feed = feedparser.parse(source["url"])
                    status = f"Direct parse: {len(feed.entries)} entries"
                except:
                    status = f"Error: {str(e)}"
            
            results[category].append({
                "name": source["name"],
                "url": source["url"],
                "status": status
            })
    return results

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            if request.get("action") == "refresh_news":
                await fetch_all_news()
                await websocket.send_text(json.dumps({"event": "news_refreshed", "data": news_cache.data}))
            elif request.get("action") == "get_category":
                category = request.get("category")
                if category in news_cache.data:
                    await websocket.send_text(json.dumps({
                        "event": "category_news",
                        "data": {
                            "category": category,
                            "news": news_cache.get(category)
                        }
                    }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Catch-all route for client-side routing
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    return FileResponse('static/index.html')

@app.on_event("startup")
async def startup_event():
    await fetch_all_news()
    setup_scheduler()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
