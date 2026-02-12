"""
ResearchTools – Tool calling infrastructure for enhanced research.

Provides a registry of research tools (Brave Search, Firecrawl, DuckDuckGo,
LinkedIn scraping) with unified interface, rate limiting, credit tracking,
and graceful fallbacks.
"""

import os
import time
import json
import logging
import hashlib
from typing import Callable, Optional
from dataclasses import dataclass, field

import requests
import httpx

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Tool Definition & Registry
# ------------------------------------------------------------------ #

@dataclass
class Tool:
    """A research tool that can be called by the orchestrator."""
    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    execute: Callable  # Function to call
    cost: float = 0.0  # Estimated cost per call (for budget tracking)
    rate_limit_seconds: float = 1.0  # Minimum seconds between calls


class ToolRegistry:
    """Registry of available research tools with rate limiting."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._last_call: dict[str, float] = {}
        self._call_counts: dict[str, int] = {}
        self._cache: dict[str, str] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        self._call_counts[tool.name] = 0
        logger.info("Registered tool: %s", tool.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """List all available tools with their schemas (for LLM function calling)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, **kwargs) -> str:
        """
        Call a tool by name with rate limiting and caching.

        Returns:
            Tool output as string
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        # Rate limiting
        now = time.time()
        last = self._last_call.get(name, 0)
        wait = tool.rate_limit_seconds - (now - last)
        if wait > 0:
            time.sleep(wait)

        # Cache check
        cache_key = hashlib.md5(f"{name}:{json.dumps(kwargs, sort_keys=True)}".encode()).hexdigest()
        if cache_key in self._cache:
            logger.info("Cache hit for %s", name)
            return self._cache[cache_key]

        # Execute
        try:
            result = tool.execute(**kwargs)
            self._last_call[name] = time.time()
            self._call_counts[name] = self._call_counts.get(name, 0) + 1
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return f"Error: {e}"

    def get_usage_stats(self) -> dict:
        """Get usage statistics for all tools."""
        return {name: count for name, count in self._call_counts.items()}


# ------------------------------------------------------------------ #
#  Brave Search Tool
# ------------------------------------------------------------------ #

class BraveSearchTool:
    """Brave Search API integration (2,000 free requests/month)."""

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY", "")
        self.available = bool(self.api_key)

    def search(self, query: str, count: int = 5, freshness: str = "") -> str:
        """
        Search the web using Brave Search API.

        Args:
            query: Search query string
            count: Number of results (max 20)
            freshness: Time filter (pd=past day, pw=past week, pm=past month)

        Returns:
            Formatted string of search results
        """
        if not self.available:
            return ""

        try:
            headers = {
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            }
            params = {"q": query, "count": min(count, 20)}
            if freshness:
                params["freshness"] = freshness

            response = requests.get(
                self.API_URL, headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                description = item.get("description", "")
                results.append(f"- {title}: {description}\n  URL: {url}")

            return "\n".join(results) if results else ""

        except Exception as e:
            logger.warning("Brave Search failed: %s", e)
            return ""

    def get_tool(self) -> Optional[Tool]:
        """Create a Tool instance for the registry."""
        if not self.available:
            return None
        return Tool(
            name="brave_search",
            description="Search the web using Brave Search API. Best for finding recent information, news, and general web content.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "description": "Number of results (1-20)", "default": 5},
                    "freshness": {"type": "string", "description": "Time filter: pd=past day, pw=past week, pm=past month", "default": ""},
                },
                "required": ["query"],
            },
            execute=self.search,
            rate_limit_seconds=1.0,
        )


# ------------------------------------------------------------------ #
#  Firecrawl Tool
# ------------------------------------------------------------------ #

class FirecrawlTool:
    """Firecrawl API integration for deep web scraping (500 free credits)."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.available = bool(self.api_key)
        self._credits_used = 0

    def scrape(self, url: str = "", **kwargs) -> str:
        """
        Scrape a single URL and extract clean content.

        Args:
            url: URL to scrape

        Returns:
            Markdown content of the page
        """
        # Handle LLM parameter aliasing
        url = url or kwargs.get("website", "") or kwargs.get("page_url", "") or kwargs.get("link", "")
        if not self.available or not url:
            return ""

        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=self.api_key)
            result = app.scrape(url, formats=["markdown"])
            self._credits_used += 1
            content = ""
            if hasattr(result, "markdown"):
                content = result.markdown or ""
            elif isinstance(result, dict):
                content = result.get("markdown", "") or result.get("content", "")
            return content[:5000]  # Limit output size

        except ImportError:
            logger.warning("firecrawl-py not installed, using fallback HTTP scraping")
            return self._http_scrape(url)
        except Exception as e:
            logger.warning("Firecrawl scrape failed for %s: %s", url, e)
            return self._http_scrape(url)

    def search(self, query: str = "", limit: int = 5, **kwargs) -> str:
        """
        Search the web and scrape results using Firecrawl.

        Args:
            query: Search query
            limit: Number of results

        Returns:
            Formatted search results with scraped content
        """
        # Handle LLM parameter aliasing
        query = query or kwargs.get("search_query", "") or kwargs.get("q", "")
        if not self.available or not query:
            return ""

        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=self.api_key)
            results = app.search(query, limit=min(limit, 5))
            self._credits_used += 2  # 2 credits per 10 results

            output = []
            # v4 SDK returns SearchData with .web list of SearchResultWeb objects
            items = []
            if hasattr(results, "web") and results.web:
                items = results.web
            elif hasattr(results, "data"):
                items = results.data or []
            elif isinstance(results, list):
                items = results
            elif isinstance(results, dict):
                items = results.get("data", [])

            for r in items:
                if hasattr(r, "title"):
                    title = r.title or ""
                    url = r.url or ""
                    desc = getattr(r, "description", "") or ""
                    content = (getattr(r, "markdown", "") or desc)[:500]
                else:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    content = r.get("markdown", r.get("content", ""))[:500]
                output.append(f"- {title}\n  URL: {url}\n  Content: {content}")

            return "\n\n".join(output) if output else ""

        except ImportError:
            return ""
        except Exception as e:
            logger.warning("Firecrawl search failed: %s", e)
            return ""

    def map_site(self, url: str = "", limit: int = 20, **kwargs) -> str:
        """
        Discover URLs on a website.

        Args:
            url: Base URL to map
            limit: Max URLs to return

        Returns:
            Newline-separated list of discovered URLs
        """
        # Handle LLM parameter aliasing (website, site, base_url, etc.)
        url = url or kwargs.get("website", "") or kwargs.get("site", "") or kwargs.get("base_url", "")
        if not self.available or not url:
            return ""

        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=self.api_key)
            result = app.map(url, limit=min(limit, 50))
            self._credits_used += 1

            urls = []
            if hasattr(result, "links"):
                raw = (result.links or [])[:limit]
                urls = [str(getattr(u, "url", u)) for u in raw]
            elif isinstance(result, list):
                urls = [str(getattr(u, "url", u)) for u in result[:limit]]
            elif isinstance(result, dict):
                urls = [str(u) for u in result.get("links", result.get("urls", []))[:limit]]

            return "\n".join(urls) if urls else ""

        except ImportError:
            return ""
        except Exception as e:
            logger.warning("Firecrawl map failed: %s", e)
            return ""

    def _http_scrape(self, url: str) -> str:
        """Fallback HTTP scraper using requests + basic HTML parsing."""
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResumeOptimizer/1.0)"
            })
            response.raise_for_status()
            # Basic content extraction
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "footer", "header"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "footer", "header"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        stripped = data.strip()
                        if stripped:
                            self.text.append(stripped)

            parser = TextExtractor()
            parser.feed(response.text)
            return "\n".join(parser.text)[:5000]

        except Exception as e:
            logger.warning("HTTP scrape failed for %s: %s", url, e)
            return ""

    @property
    def credits_used(self) -> int:
        return self._credits_used

    def get_tools(self) -> list[Tool]:
        """Create Tool instances for the registry."""
        if not self.available:
            return []
        return [
            Tool(
                name="firecrawl_scrape",
                description="Deep scrape a single URL to extract clean content. Best for company websites, job postings, and articles.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to scrape"},
                    },
                    "required": ["url"],
                },
                execute=self.scrape,
                cost=1.0,
                rate_limit_seconds=2.0,
            ),
            Tool(
                name="firecrawl_search",
                description="Search the web and scrape results. Combines search + content extraction.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Number of results", "default": 5},
                    },
                    "required": ["query"],
                },
                execute=self.search,
                cost=2.0,
                rate_limit_seconds=3.0,
            ),
            Tool(
                name="firecrawl_map",
                description="Discover all URLs on a website. Useful for finding career pages, about pages, etc.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Base URL to map"},
                        "limit": {"type": "integer", "description": "Max URLs to return", "default": 20},
                    },
                    "required": ["url"],
                },
                execute=self.map_site,
                cost=1.0,
                rate_limit_seconds=2.0,
            ),
        ]


# ------------------------------------------------------------------ #
#  LinkedIn Scraper Tool (Public Data Only)
# ------------------------------------------------------------------ #

class LinkedInTool:
    """
    LinkedIn public data extraction using web search proxying.

    Instead of directly scraping LinkedIn (ToS issues), this:
    1. Uses Brave/DuckDuckGo to find LinkedIn profile data via search
    2. Uses Firecrawl to extract content from cached/public pages
    3. Extracts structured company/role information
    """

    def __init__(self, brave_tool: BraveSearchTool = None, firecrawl_tool: FirecrawlTool = None):
        self.brave = brave_tool
        self.firecrawl = firecrawl_tool

    def search_company(self, company_name: str) -> str:
        """
        Research a company via LinkedIn-relevant search queries.

        Args:
            company_name: Company name to research

        Returns:
            Aggregated company information
        """
        results = []

        # Search for company LinkedIn page content via web
        queries = [
            f"{company_name} LinkedIn company about values culture",
            f"{company_name} company size employees technology stack",
            f"site:linkedin.com/company {company_name}",
        ]

        for query in queries:
            if self.brave and self.brave.available:
                result = self.brave.search(query, count=3)
                if result:
                    results.append(result)
                    time.sleep(1)

        return "\n\n".join(results) if results else ""

    def search_role(self, job_title: str, company_name: str = "") -> str:
        """
        Research what people in a role typically have on their profiles.

        Args:
            job_title: Target role
            company_name: Optional company for specificity

        Returns:
            Information about typical skills and backgrounds
        """
        results = []
        company_clause = f"at {company_name}" if company_name else ""

        queries = [
            f"{job_title} {company_clause} LinkedIn profile skills background",
            f"{job_title} common skills certifications requirements 2025 2026",
        ]

        for query in queries:
            if self.brave and self.brave.available:
                result = self.brave.search(query, count=3)
                if result:
                    results.append(result)
                    time.sleep(1)

        return "\n\n".join(results) if results else ""

    def get_tools(self) -> list[Tool]:
        """Create Tool instances for the registry."""
        tools = []
        if self.brave and self.brave.available:
            tools.append(Tool(
                name="linkedin_company_research",
                description="Research a company's LinkedIn presence, culture, size, and values using web search.",
                parameters={
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string", "description": "Company name to research"},
                    },
                    "required": ["company_name"],
                },
                execute=self.search_company,
                rate_limit_seconds=3.0,
            ))
            tools.append(Tool(
                name="linkedin_role_research",
                description="Research typical profiles, skills, and backgrounds for a specific role.",
                parameters={
                    "type": "object",
                    "properties": {
                        "job_title": {"type": "string", "description": "Job title to research"},
                        "company_name": {"type": "string", "description": "Optional company name", "default": ""},
                    },
                    "required": ["job_title"],
                },
                execute=self.search_role,
                rate_limit_seconds=3.0,
            ))
        return tools


# ------------------------------------------------------------------ #
#  URL Content Extractor Tool
# ------------------------------------------------------------------ #

class URLExtractorTool:
    """Extract content from any URL using Firecrawl or HTTP fallback."""

    def __init__(self, firecrawl_tool: FirecrawlTool = None):
        self.firecrawl = firecrawl_tool

    def extract(self, url: str) -> str:
        """
        Extract clean text content from a URL.

        Args:
            url: URL to extract content from

        Returns:
            Extracted text content
        """
        if self.firecrawl and self.firecrawl.available:
            return self.firecrawl.scrape(url)
        return self.firecrawl._http_scrape(url) if self.firecrawl else ""

    def get_tool(self) -> Tool:
        """Create a Tool instance for the registry."""
        return Tool(
            name="extract_url",
            description="Extract clean text content from any URL. Works with web pages, articles, job postings.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract content from"},
                },
                "required": ["url"],
            },
            execute=self.extract,
            rate_limit_seconds=2.0,
        )


# ------------------------------------------------------------------ #
#  Factory: Build complete tool registry
# ------------------------------------------------------------------ #

def create_tool_registry() -> ToolRegistry:
    """
    Create and populate a ToolRegistry with all available tools.

    Automatically detects which APIs are configured and only registers
    available tools.

    Returns:
        Populated ToolRegistry
    """
    registry = ToolRegistry()

    # Brave Search
    brave = BraveSearchTool()
    brave_tool = brave.get_tool()
    if brave_tool:
        registry.register(brave_tool)
        logger.info("Brave Search API available (2,000 free requests/month)")

    # Firecrawl
    firecrawl = FirecrawlTool()
    for tool in firecrawl.get_tools():
        registry.register(tool)
    if firecrawl.available:
        logger.info("Firecrawl API available (500 free credits)")

    # LinkedIn (via web search)
    linkedin = LinkedInTool(brave_tool=brave, firecrawl_tool=firecrawl)
    for tool in linkedin.get_tools():
        registry.register(tool)

    # URL Extractor
    extractor = URLExtractorTool(firecrawl_tool=firecrawl)
    registry.register(extractor.get_tool())

    available = registry.list_tools()
    logger.info("Tool registry ready: %d tools available", len(available))
    return registry
