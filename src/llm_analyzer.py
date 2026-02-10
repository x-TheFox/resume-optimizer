"""
LLMAnalyzer – Uses Groq's Llama-4-Maverick to perform deep resume analysis.

Handles gap analysis, section scoring, ATS simulation, suggestion generation,
interview prep, cover letter drafting, and talking points.
"""

import json
import time
import logging
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Orchestrates all LLM-powered analysis using the Groq API."""

    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def analyze(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        success_profile: dict,
    ) -> dict:
        """
        Run the full analysis pipeline and return an AnalysisResult dict.
        """
        logger.info("Starting LLM analysis...")
        profile_text = self._profile_to_text(success_profile)

        # 1) Gap Analysis
        gap_analysis = self._gap_analysis(
            resume_text, job_title, job_description, profile_text
        )

        # 2) Section Scores
        scores = self._score_sections(
            resume_text, job_title, job_description, profile_text
        )

        # 3) Match Scores (Technical + Cultural)
        match_scores = self._compute_match_scores(
            resume_text, job_description, success_profile
        )
        scores.update(match_scores)

        # 4) ATS Simulation
        ats_result = self._ats_simulation(resume_text)

        # 5) Optimization Suggestions
        suggestions = self._generate_suggestions(
            resume_text, job_title, job_description, gap_analysis, profile_text,
            success_profile.get("cultural_tone", "balanced"),
        )

        # 6) Interview Questions
        interview_questions = self._generate_interview_questions(
            gap_analysis, profile_text, job_title
        )

        # 7) Cover Letter
        cover_letter = self._generate_cover_letter(
            resume_text, job_title, job_description, profile_text
        )

        # 8) Talking Points for each suggestion
        talking_points = self._generate_talking_points(suggestions)

        # Merge talking points into suggestions
        for i, s in enumerate(suggestions):
            if i < len(talking_points):
                s["talking_point"] = talking_points[i]

        result = {
            "gap_analysis": gap_analysis,
            "scores": scores,
            "ats_score": ats_result.get("score", 0),
            "ats_warnings": ats_result.get("warnings", []),
            "suggestions": suggestions,
            "interview_questions": interview_questions,
            "cover_letter": cover_letter,
            "overall_summary": gap_analysis[:500] if gap_analysis else "",
        }

        logger.info("LLM analysis complete. Scores: %s", scores)
        return result

    # ------------------------------------------------------------------ #
    #  Analysis Methods
    # ------------------------------------------------------------------ #

    def _gap_analysis(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        profile_text: str,
    ) -> str:
        """Perform a detailed gap analysis between resume and success profile."""
        system_prompt = (
            "You are an expert career strategist and resume analyst. "
            "Analyze the gap between a candidate's resume and the target role's requirements. "
            "Be specific about what's missing, what's strong, and what needs improvement. "
            "Reference specific parts of the resume and specific requirements."
        )
        user_prompt = f"""## Target Role: {job_title}

## Job Description:
{job_description}

## Research-Based Success Profile:
{profile_text}

## Candidate's Resume:
{resume_text}

---

Provide a detailed gap analysis covering:
1. **Strengths** — What aligns well with the target role
2. **Gaps** — Skills, experience, or keywords that are missing
3. **Opportunities** — How to bridge the gaps with existing experience
4. **Priority Actions** — The top 5 most impactful changes to make

Be specific and reference actual content from the resume."""

        return self._call_groq(system_prompt, user_prompt)

    def _score_sections(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        profile_text: str,
    ) -> dict:
        """Score Skills, Experience, and Impact sections 0-100."""
        system_prompt = (
            "You are a resume scoring engine. You must respond with ONLY valid JSON, "
            "no other text before or after. "
            "Score the resume on three dimensions against the job requirements."
        )
        user_prompt = f"""## Target Role: {job_title}

## Job Description:
{job_description}

## Success Profile:
{profile_text}

## Resume:
{resume_text}

---

Score this resume on exactly three dimensions, each 0-100.
Respond with ONLY this JSON format, nothing else:
{{"skills": <int>, "experience": <int>, "impact": <int>}}

- skills: How well the candidate's skills match the role requirements
- experience: How relevant and sufficient the candidate's experience is
- impact: How well the resume demonstrates measurable impact and results"""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            scores = json.loads(self._extract_json(response))
            return {
                "skills": max(0, min(100, scores.get("skills", 50))),
                "experience": max(0, min(100, scores.get("experience", 50))),
                "impact": max(0, min(100, scores.get("impact", 50))),
            }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse scores: %s | Response: %s", e, response[:200])
            return {"skills": 50, "experience": 50, "impact": 50}

    def _compute_match_scores(
        self, resume_text: str, job_description: str, success_profile: dict
    ) -> dict:
        """Compute Technical Match and Cultural Match scores."""
        system_prompt = (
            "You are a hiring match evaluator. You must respond with ONLY valid JSON, "
            "no other text before or after. "
            "Evaluate the candidate's fit on two dimensions."
        )
        cultural_info = (
            f"Company values: {success_profile.get('company_values', 'N/A')}\n"
            f"Cultural tone: {success_profile.get('cultural_tone', 'balanced')}\n"
            f"Recent news: {success_profile.get('recent_news', 'N/A')}"
        )
        user_prompt = f"""## Job Description:
{job_description}

## Company Culture & Research:
{cultural_info}

## Resume:
{resume_text}

---

Evaluate and score these two dimensions, each 0-100:
Respond with ONLY this JSON, nothing else:
{{"technical_match": <int>, "cultural_match": <int>}}

- technical_match: How well the candidate's technical skills align with the role
- cultural_match: How well the resume's language and presentation match the company's culture and values"""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            scores = json.loads(self._extract_json(response))
            return {
                "technical_match": max(0, min(100, scores.get("technical_match", 50))),
                "cultural_match": max(0, min(100, scores.get("cultural_match", 50))),
            }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse match scores: %s", e)
            return {"technical_match": 50, "cultural_match": 50}

    def _ats_simulation(self, resume_text: str) -> dict:
        """Simulate a rigid ATS parser and evaluate readability."""
        system_prompt = (
            "You are a strict Applicant Tracking System (ATS) parser. "
            "You must respond with ONLY valid JSON, no other text. "
            "Evaluate whether a resume can be parsed correctly by automated systems."
        )
        user_prompt = f"""## Resume Content:
{resume_text}

---

Analyze this resume for ATS compatibility. Check for:
1. Standard section headers (Education, Experience, Skills, etc.)
2. Parseable date formats
3. Contact information present
4. No complex formatting issues (tables, columns, graphics mentions)
5. Keyword density for searchability
6. Consistent formatting patterns

Respond with ONLY this JSON, nothing else:
{{"score": <int 0-100>, "warnings": ["warning1", "warning2", ...]}}

The score should reflect how well an ATS can parse and categorize this resume.
Warnings should be specific, actionable items."""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            result = json.loads(self._extract_json(response))
            return {
                "score": max(0, min(100, result.get("score", 50))),
                "warnings": result.get("warnings", []),
            }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse ATS result: %s", e)
            return {"score": 50, "warnings": ["Unable to complete ATS analysis"]}

    def _generate_suggestions(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        gap_analysis: str,
        profile_text: str,
        cultural_tone: str,
    ) -> list:
        """Generate specific, per-paragraph optimization suggestions."""
        tone_instruction = ""
        if cultural_tone == "silicon_valley_casual":
            tone_instruction = (
                "The target company uses casual, startup-style language. "
                "Mirror this tone: use action verbs like 'shipped', 'built', 'drove'. "
                "Keep language dynamic and concise."
            )
        elif cultural_tone == "fortune_500_corporate":
            tone_instruction = (
                "The target company uses formal, corporate language. "
                "Mirror this tone: use phrases like 'spearheaded', 'stakeholder management', "
                "'strategic initiative'. Keep language polished and professional."
            )
        else:
            tone_instruction = (
                "Use a balanced professional tone — confident but not overly formal."
            )

        system_prompt = (
            "You are an expert resume optimizer. You must respond with ONLY a valid JSON array, "
            "no other text before or after. "
            "Generate specific text replacement suggestions that improve ATS compatibility, "
            "demonstrate impact, and align with the target role. "
            f"{tone_instruction} "
            "CRITICAL: Each 'original_text' MUST be an EXACT substring from the resume. "
            "Copy it character-for-character. Do not paraphrase or approximate."
        )
        user_prompt = f"""## Target Role: {job_title}

## Job Description:
{job_description}

## Gap Analysis:
{gap_analysis}

## Success Profile:
{profile_text}

## Resume:
{resume_text}

---

Generate 5-10 specific text replacement suggestions. Each suggestion must replace an EXACT
piece of text from the resume with an improved version.

Respond with ONLY a JSON array in this format, nothing else:
[
  {{
    "section": "Experience|Skills|Summary|Education",
    "original_text": "exact text from resume to find and replace",
    "replacement_text": "improved version of the text",
    "reason": "brief explanation of why this change improves the resume"
  }}
]

Rules:
- original_text must be a VERBATIM substring from the resume
- replacement_text should be similar length (±30%) to preserve document layout
- Focus on: quantifying impact, adding keywords, improving action verbs, aligning with role
- Do NOT change names, dates, company names, or educational institutions"""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            suggestions = json.loads(self._extract_json(response))
            if isinstance(suggestions, list):
                return suggestions
            return []
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse suggestions: %s", e)
            return []

    def _generate_interview_questions(
        self, gap_analysis: str, profile_text: str, job_title: str
    ) -> list:
        """Generate interview questions based on resume weaknesses."""
        system_prompt = (
            "You are an interview preparation coach. You must respond with ONLY a valid JSON array "
            "of strings, no other text. Generate interview questions that the candidate is "
            "likely to face based on the gaps in their resume."
        )
        user_prompt = f"""## Target Role: {job_title}

## Gap Analysis (weaknesses identified):
{gap_analysis}

## Success Profile:
{profile_text}

---

Generate 10-15 likely interview questions that specifically target the WEAKNESSES
and GAPS found in this candidate's resume. These should be questions the candidate
needs to prepare for.

Respond with ONLY a JSON array of strings, nothing else:
["Question 1?", "Question 2?", ...]

Include a mix of:
- Technical questions about missing skills
- Behavioral questions about experience gaps
- Situational questions about unfamiliar scenarios
- Questions about career trajectory and motivation"""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            questions = json.loads(self._extract_json(response))
            if isinstance(questions, list):
                return questions
            return []
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse interview questions: %s", e)
            return []

    def _generate_cover_letter(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        profile_text: str,
    ) -> str:
        """Auto-generate a cover letter referencing research findings."""
        system_prompt = (
            "You are an expert cover letter writer. Write a compelling, personalized "
            "cover letter that references specific company research findings. "
            "The letter should feel authentic, not templated."
        )
        user_prompt = f"""## Target Role: {job_title}

## Job Description:
{job_description}

## Company Research:
{profile_text}

## Candidate's Resume:
{resume_text}

---

Write a professional cover letter (300-400 words) that:
1. Opens with a hook referencing a specific company news item or initiative from the research
2. Highlights the candidate's most relevant experience for THIS specific role
3. Demonstrates understanding of the company's values and culture
4. Addresses 1-2 potential gaps proactively with transferable skills
5. Closes with enthusiasm and a specific call to action

Do NOT use placeholder brackets like [Company Name] — use the actual company name from the research.
Write the content only — no JSON wrapping needed."""

        return self._call_groq(system_prompt, user_prompt)

    def _generate_talking_points(self, suggestions: list) -> list:
        """Generate interview talking points to defend each auto-applied edit."""
        if not suggestions:
            return []

        system_prompt = (
            "You are an interview coach. You must respond with ONLY a valid JSON array "
            "of strings, no other text. "
            "For each resume edit, create a brief talking point the candidate can use "
            "in an interview to naturally discuss the updated claim."
        )
        edits_text = ""
        for i, s in enumerate(suggestions):
            edits_text += (
                f"\n{i+1}. Changed: \"{s.get('original_text', '')}\"\n"
                f"   To: \"{s.get('replacement_text', '')}\"\n"
                f"   Reason: {s.get('reason', '')}\n"
            )

        user_prompt = f"""Here are the edits made to the resume:
{edits_text}

For each edit (numbered), generate a 2-3 sentence "talking point" the candidate
can memorize to naturally discuss this experience in an interview, defending
the new phrasing with specific details they should prepare.

Respond with ONLY a JSON array of strings (one per edit), nothing else:
["Talking point 1...", "Talking point 2...", ...]"""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            points = json.loads(self._extract_json(response))
            if isinstance(points, list):
                return points
            return []
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse talking points: %s", e)
            return []

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def _profile_to_text(self, profile: dict) -> str:
        """Convert a SuccessProfile dict to a readable text block."""
        parts = [
            f"Job Title: {profile.get('job_title', 'N/A')}",
            f"Company: {profile.get('company_name', 'N/A')}",
            "",
            "=== Role Responsibilities ===",
            profile.get("role_responsibilities", "No data"),
            "",
            "=== Technology & Industry Trends ===",
            profile.get("tech_trends", "No data"),
            "",
            "=== Company Values & Culture ===",
            profile.get("company_values", "No data"),
            "",
            "=== Recent Company News ===",
            profile.get("recent_news", "No data"),
            "",
            "=== Competitive Landscape ===",
            profile.get("competitors", "No data"),
            "",
            "=== Shadow Skills (Employee Skill DNA) ===",
            profile.get("shadow_skills", "No data"),
            "",
            f"=== Cultural Tone: {profile.get('cultural_tone', 'balanced')} ===",
        ]
        return "\n".join(parts)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from a response that might contain markdown fences or extra text."""
        text = text.strip()
        
        # Try to find JSON in code fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()

        # Find the earliest starting character
        idx_open_brace = text.find("{")
        idx_open_bracket = text.find("[")

        if idx_open_brace == -1 and idx_open_bracket == -1:
            return text

        # Determine which comes first
        if idx_open_brace != -1 and (idx_open_bracket == -1 or idx_open_brace < idx_open_bracket):
            # It's an object
            start = idx_open_brace
            char = "{"
            end_char = "}"
        else:
            # It's an array
            start = idx_open_bracket
            char = "["
            end_char = "]"
            
        # Extract balanced block
        depth = 0
        for i in range(start, len(text)):
            if text[i] == char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return text.strip()

    def _call_groq(
        self, system_prompt: str, user_prompt: str, max_retries: int = 3
    ) -> str:
        """Call the Groq API with retry and rate-limit handling."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=8000,
                )
                content = response.choices[0].message.content
                return content if content else ""
            except Exception as e:
                logger.warning(
                    "Groq API attempt %d failed: %s", attempt + 1, str(e)
                )
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 10 * (attempt + 1)
                    logger.info("Rate limited. Waiting %ds...", wait_time)
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.error("All Groq API retries failed: %s", e)
                    return ""
        return ""
