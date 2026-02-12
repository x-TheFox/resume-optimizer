"""
ResearchOrchestrator – Coordinates multi-step research using tool calling.

Implements a simplified Deep Orchestrator pattern:
1. Planner: LLM generates research plan with subtasks
2. Executor: Runs subtasks using tool registry
3. Memory: Stores intermediate results for context
4. Synthesizer: Combines results into actionable intelligence

This gives the resume optimizer deep research capabilities similar to
AI research agents, while staying within free-tier API limits.
"""

import json
import time
import logging
from typing import Optional

from groq import Groq

from src.research_tools import ToolRegistry, create_tool_registry

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """
    Orchestrates multi-step research workflows with tool calling.

    Uses LLM to plan research, execute tools, and synthesize findings.
    """

    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.registry = create_tool_registry()
        self.memory: list[dict] = []  # Accumulated research findings
        self.max_tool_calls = 8  # Budget: max tools to call per research session

    def deep_research(
        self,
        job_title: str,
        job_description: str,
        company_name: Optional[str] = None,
    ) -> dict:
        """
        Perform deep research on a job opportunity using multi-step tool calling.

        This goes beyond basic web search by:
        - Scraping the actual company website for values/culture
        - Analyzing recent news and announcements
        - Extracting real skill requirements from similar job postings
        - Cross-referencing LinkedIn-style profile data

        Args:
            job_title: Target role
            job_description: Full JD text
            company_name: Optional company name

        Returns:
            Dict with deep research findings to enhance analysis
        """
        logger.info("Starting deep research for: %s at %s", job_title, company_name or "unknown")
        available_tools = self.registry.list_tools()

        if not available_tools:
            logger.info("No research tools available, skipping deep research")
            return {}

        # Step 1: Plan research
        plan = self._plan_research(job_title, job_description, company_name, available_tools)
        if not plan:
            return {}

        # Step 2: Execute plan
        tool_calls_made = 0
        for step in plan:
            if tool_calls_made >= self.max_tool_calls:
                logger.info("Tool call budget exhausted (%d calls)", tool_calls_made)
                break

            tool_name = step.get("tool", "")
            params = step.get("params", {})
            purpose = step.get("purpose", "")

            logger.info("Executing: %s — %s", tool_name, purpose)
            result = self.registry.call_tool(tool_name, **params)

            if result and not result.startswith("Error:"):
                self.memory.append({
                    "tool": tool_name,
                    "purpose": purpose,
                    "result": result[:2000],  # Limit memory size
                })
                tool_calls_made += 1

        # Step 3: Synthesize findings
        if self.memory:
            synthesis = self._synthesize_findings(job_title, company_name)
            return synthesis

        return {}

    def _plan_research(
        self,
        job_title: str,
        job_description: str,
        company_name: Optional[str],
        available_tools: list[dict],
    ) -> list[dict]:
        """
        Use LLM to create a research plan based on available tools.

        Returns:
            List of step dicts: [{"tool": "...", "params": {...}, "purpose": "..."}]
        """
        tools_description = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in available_tools
        ])

        system_prompt = (
            "You are a research planning agent. You create concise, "
            "targeted research plans using available tools. "
            "Respond with ONLY a valid JSON array, no other text."
        )

        company_context = f"\nTarget Company: {company_name}" if company_name else ""

        user_prompt = f"""Plan research for this job opportunity:

Job Title: {job_title}{company_context}

Job Description (first 500 chars):
{job_description[:500]}

Available Research Tools:
{tools_description}

Create a research plan with 3-6 steps. Each step uses one tool.
Focus on the MOST VALUABLE research:
1. Company deep dive (if company provided) — scrape their website, find values
2. Role requirements — what skills and experience are actually needed
3. Industry context — recent trends and market data

Respond with ONLY this JSON array:
[
  {{"tool": "tool_name", "params": {{"param1": "value1"}}, "purpose": "What this will tell us"}}
]

Rules:
- Only use tools from the available list
- Keep queries specific and targeted
- Prioritize company research if a company is provided
- Maximum 6 steps"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content or ""
            plan = json.loads(self._extract_json(content))

            if isinstance(plan, list):
                # Validate tools exist
                valid_tool_names = {t["name"] for t in available_tools}
                validated_plan = [
                    step for step in plan
                    if step.get("tool") in valid_tool_names
                ]
                logger.info("Research plan: %d steps (validated from %d)", len(validated_plan), len(plan))
                return validated_plan[:6]

            return []

        except Exception as e:
            logger.error("Research planning failed: %s", e)
            return self._fallback_plan(job_title, company_name, available_tools)

    def _fallback_plan(
        self,
        job_title: str,
        company_name: Optional[str],
        available_tools: list[dict],
    ) -> list[dict]:
        """Generate a deterministic fallback plan when LLM planning fails."""
        tool_names = {t["name"] for t in available_tools}
        plan = []

        # Always search for role info
        if "brave_search" in tool_names:
            plan.append({
                "tool": "brave_search",
                "params": {"query": f"{job_title} key skills requirements 2025 2026", "count": 5},
                "purpose": "Find key skills and requirements for the role",
            })

        if company_name:
            if "brave_search" in tool_names:
                plan.append({
                    "tool": "brave_search",
                    "params": {"query": f"{company_name} company values culture mission", "count": 5},
                    "purpose": "Research company values and culture",
                })
            if "firecrawl_scrape" in tool_names:
                # Try to scrape company website
                plan.append({
                    "tool": "brave_search",
                    "params": {"query": f"{company_name} official website about us", "count": 2},
                    "purpose": "Find company website URL",
                })
            if "linkedin_company_research" in tool_names:
                plan.append({
                    "tool": "linkedin_company_research",
                    "params": {"company_name": company_name},
                    "purpose": "Research company LinkedIn presence",
                })

        if "brave_search" in tool_names:
            plan.append({
                "tool": "brave_search",
                "params": {"query": f"{job_title} technology stack trends 2025 2026", "count": 5},
                "purpose": "Find technology trends for the role",
            })

        return plan[:6]

    def _synthesize_findings(self, job_title: str, company_name: Optional[str]) -> dict:
        """
        Use LLM to synthesize all research findings into structured intelligence.

        Returns:
            Dict with synthesized research data
        """
        findings_text = "\n\n".join([
            f"=== {m['purpose']} (via {m['tool']}) ===\n{m['result']}"
            for m in self.memory
        ])

        system_prompt = (
            "You are a research synthesizer. Analyze the research findings and "
            "extract the most important intelligence for resume optimization. "
            "Respond with ONLY valid JSON, no other text."
        )

        user_prompt = f"""Synthesize these research findings for a {job_title} role{f' at {company_name}' if company_name else ''}:

{findings_text}

Extract and return ONLY this JSON:
{{
  "company_insights": "2-3 sentences about the company's culture, values, and recent initiatives",
  "required_skills": ["list", "of", "key", "skills", "found"],
  "industry_trends": "2-3 sentences about relevant industry trends",
  "cultural_tone": "formal|casual|balanced — based on company's communication style",
  "key_technologies": ["list", "of", "technologies", "and", "tools"],
  "competitive_landscape": "1-2 sentences about competitors or market position",
  "insider_tips": "1-2 specific tips for tailoring a resume to this role/company"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content or ""
            result = json.loads(self._extract_json(content))
            if isinstance(result, dict):
                logger.info("Research synthesis complete: %s", list(result.keys()))
                return result
            return {}

        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            # Return raw findings as fallback
            return {
                "raw_findings": findings_text[:3000],
                "company_insights": "",
                "required_skills": [],
                "industry_trends": "",
            }

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response that might contain markdown fences."""
        text = text.strip()

        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()

        # Find first { or [
        for i, char in enumerate(text):
            if char in "{[":
                end_char = "}" if char == "{" else "]"
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == char:
                        depth += 1
                    elif text[j] == end_char:
                        depth -= 1
                        if depth == 0:
                            return text[i:j + 1]
                break

        return text
