"""
ResearchEngine – Performs live web research to build a "Success Profile"
for a given job title, description, and optionally a target company.

Uses a layered approach:
1. Deep Research (Firecrawl + Brave Search + tool calling) — if APIs configured
2. DuckDuckGo Search — always available as fallback (no API key needed)

The deep research layer provides:
- Company website scraping for real values/culture
- Industry trend analysis from multiple sources
- LinkedIn-style profile data for shadow requirements
- Tool-calling orchestration for multi-step research
"""

import os
import time
import logging
from typing import Optional

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class ResearchEngine:
    """Performs targeted web searches and aggregates results into a SuccessProfile."""

    def __init__(self):
        self.ddgs = DDGS()
        self._delay = 1.5  # seconds between searches to avoid rate limits
        self._deep_research_available = self._check_deep_research()

    def _check_deep_research(self) -> bool:
        """Check if any deep research APIs are configured."""
        has_brave = bool(os.getenv("BRAVE_API_KEY", ""))
        has_firecrawl = bool(os.getenv("FIRECRAWL_API_KEY", ""))
        if has_brave or has_firecrawl:
            logger.info("Deep research APIs available: Brave=%s, Firecrawl=%s", has_brave, has_firecrawl)
            return True
        logger.info("No deep research APIs configured, using DuckDuckGo only")
        return False

    def research(
        self,
        job_title: str,
        job_description: str,
        company_name: Optional[str] = None,
    ) -> dict:
        """
        Run all research searches and return a SuccessProfile dict.

        Uses deep research tools (Firecrawl, Brave, LinkedIn) if available,
        falls back to DuckDuckGo for everything.

        Args:
            job_title: The target job title (e.g. "Senior Backend Engineer")
            job_description: The full job description text
            company_name: Optional target company name

        Returns:
            SuccessProfile dict with all research findings
        """
        logger.info("Starting research for: %s", job_title)
        results = {}

        # --- Deep Research Layer (if APIs configured) ---
        deep_findings = {}
        if self._deep_research_available:
            deep_findings = self._run_deep_research(job_title, job_description, company_name)

        # --- DuckDuckGo Layer (always runs as base/fallback) ---
        results["role_responsibilities"] = self._search_role_responsibilities(job_title)
        results["tech_trends"] = self._search_tech_trends(job_title)

        if company_name:
            results["company_values"] = self._search_company_values(company_name)
            results["recent_news"] = self._search_company_news(company_name)
            results["competitors"] = self._search_competitors(company_name)
            results["shadow_skills"] = self._search_employee_skills(job_title, company_name)
        else:
            results["company_values"] = ""
            results["recent_news"] = ""
            results["competitors"] = ""
            results["shadow_skills"] = ""

        # --- Cultural Tone Analysis ---
        results["cultural_tone"] = self._analyze_cultural_tone(job_description)

        # --- Merge Deep Research into results ---
        if deep_findings:
            # Enhance results with deep research data
            if deep_findings.get("company_insights"):
                results["company_values"] = (
                    results["company_values"] + "\n\n=== Deep Research Findings ===\n" +
                    deep_findings["company_insights"]
                )
            if deep_findings.get("required_skills"):
                skills_text = ", ".join(deep_findings["required_skills"])
                results["shadow_skills"] = (
                    results["shadow_skills"] + "\n\n=== Key Skills Identified ===\n" + skills_text
                )
            if deep_findings.get("industry_trends"):
                results["tech_trends"] = (
                    results["tech_trends"] + "\n\n=== Deep Research Trends ===\n" +
                    deep_findings["industry_trends"]
                )
            if deep_findings.get("competitive_landscape"):
                results["competitors"] = (
                    results["competitors"] + "\n\n=== Market Intelligence ===\n" +
                    deep_findings["competitive_landscape"]
                )

        # Build final profile
        profile = self._build_success_profile(results, job_title, company_name)

        # Add deep research extras to profile
        if deep_findings:
            profile["deep_research"] = deep_findings
            profile["key_technologies"] = deep_findings.get("key_technologies", [])
            profile["insider_tips"] = deep_findings.get("insider_tips", "")
            # Override cultural tone if deep research has stronger signal
            if deep_findings.get("cultural_tone") in ("formal", "casual", "balanced"):
                tone_map = {
                    "formal": "fortune_500_corporate",
                    "casual": "silicon_valley_casual",
                    "balanced": "balanced",
                }
                profile["cultural_tone"] = tone_map.get(
                    deep_findings["cultural_tone"],
                    profile["cultural_tone"]
                )

        logger.info("Research complete. Profile keys: %s", list(profile.keys()))
        return profile

    def _run_deep_research(
        self,
        job_title: str,
        job_description: str,
        company_name: Optional[str],
    ) -> dict:
        """
        Run deep research using the orchestrator with tool calling.

        Returns:
            Dict with deep research findings, or empty dict on failure
        """
        try:
            from config import Config
            from src.research_orchestrator import ResearchOrchestrator

            if not Config.GROQ_API_KEY:
                return {}

            orchestrator = ResearchOrchestrator(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                gateway_api_key=Config.AI_GATEWAY_API_KEY,
            )
            findings = orchestrator.deep_research(
                job_title=job_title,
                job_description=job_description,
                company_name=company_name,
            )
            logger.info("Deep research completed: %d findings", len(findings))
            return findings

        except Exception as e:
            logger.warning("Deep research failed (falling back to DuckDuckGo): %s", e)
            return {}

    # ------------------------------------------------------------------ #
    #  Search Methods
    # ------------------------------------------------------------------ #

    def _safe_search(self, query: str, max_results: int = 5) -> str:
        """Execute a DuckDuckGo text search with retry logic."""
        for attempt in range(3):
            try:
                time.sleep(self._delay)
                raw_results = self.ddgs.text(query, max_results=max_results)
                if raw_results:
                    snippets = []
                    for r in raw_results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        snippets.append(f"- {title}: {body}")
                    return "\n".join(snippets)
                return ""
            except Exception as e:
                logger.warning(
                    "Search attempt %d failed for '%s': %s", attempt + 1, query, e
                )
                time.sleep(2 ** (attempt + 1))
        return ""

    def _search_role_responsibilities(self, job_title: str) -> str:
        """Search for key responsibilities and required skills for the role."""
        query = f"{job_title} key responsibilities skills required qualifications 2025"
        logger.info("Searching: role responsibilities")
        return self._safe_search(query)

    def _search_tech_trends(self, job_title: str) -> str:
        """Search for technology stack and industry trends."""
        query = f"{job_title} technology stack tools trends 2025 2026"
        logger.info("Searching: tech trends")
        return self._safe_search(query)

    def _search_company_values(self, company_name: str) -> str:
        """Search for company core values, mission, and culture."""
        query = f"{company_name} company core values mission culture about us"
        logger.info("Searching: company values for %s", company_name)
        return self._safe_search(query)

    def _search_company_news(self, company_name: str) -> str:
        """Search for recent company news (last 6 months)."""
        query = f"{company_name} news announcements latest 2025 2026"
        logger.info("Searching: recent news for %s", company_name)
        return self._safe_search(query)

    def _search_competitors(self, company_name: str) -> str:
        """Search for company's competitive landscape."""
        query = f"{company_name} competitors market landscape industry rivals"
        logger.info("Searching: competitors of %s", company_name)
        return self._safe_search(query)

    def _search_employee_skills(self, job_title: str, company_name: str) -> str:
        """Search for skills of current employees in similar roles (shadow requirements)."""
        query = (
            f"{job_title} at {company_name} LinkedIn skills tools software commonly used"
        )
        logger.info("Searching: employee skills for %s at %s", job_title, company_name)
        return self._safe_search(query)

    def _analyze_cultural_tone(self, job_description: str) -> dict:
        """
        Analyze the job description's language to determine cultural tone.

        Returns a dict with tone classification and detected keywords.
        """
        jd_lower = job_description.lower()

        casual_keywords = [
            "hustle", "build things", "move fast", "break things", "scrappy",
            "wear many hats", "passionate", "rockstar", "ninja", "guru",
            "disrupt", "startup", "agile", "iterate", "ship it", "hack",
            "collaborative", "fun", "dynamic", "fast-paced", "innovative",
        ]
        corporate_keywords = [
            "synergy", "stakeholder", "governance", "compliance", "enterprise",
            "strategic", "cross-functional", "alignment", "deliverable",
            "value proposition", "roi", "kpi", "metrics-driven", "best practices",
            "scalable", "leverage", "bandwidth", "paradigm", "holistic",
            "thought leadership", "executive", "c-suite",
        ]

        casual_hits = [kw for kw in casual_keywords if kw in jd_lower]
        corporate_hits = [kw for kw in corporate_keywords if kw in jd_lower]

        casual_score = len(casual_hits)
        corporate_score = len(corporate_hits)

        if casual_score > corporate_score + 2:
            tone = "silicon_valley_casual"
        elif corporate_score > casual_score + 2:
            tone = "fortune_500_corporate"
        else:
            tone = "balanced"

        return {
            "tone": tone,
            "casual_keywords": casual_hits,
            "corporate_keywords": corporate_hits,
            "casual_score": casual_score,
            "corporate_score": corporate_score,
        }

    # ------------------------------------------------------------------ #
    #  Profile Builder
    # ------------------------------------------------------------------ #

    def _build_success_profile(
        self, results: dict, job_title: str, company_name: Optional[str]
    ) -> dict:
        """Aggregate all search results into a structured SuccessProfile."""
        profile = {
            "job_title": job_title,
            "company_name": company_name or "Not specified",
            "role_responsibilities": results["role_responsibilities"],
            "tech_trends": results["tech_trends"],
            "company_values": results["company_values"],
            "recent_news": results["recent_news"],
            "competitors": results["competitors"],
            "shadow_skills": results["shadow_skills"],
            "cultural_tone": results["cultural_tone"]["tone"],
            "cultural_tone_details": results["cultural_tone"],
        }
        return profile
