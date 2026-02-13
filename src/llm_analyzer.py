"""
LLMAnalyzer – Uses a round-robin LLM provider (Groq + Vercel AI Gateway)
to perform deep resume analysis with automatic failover.

Handles gap analysis, section scoring, ATS simulation, suggestion generation,
interview prep, cover letter drafting, and talking points.
"""

import json
import time
import logging
from typing import Optional

from src.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Orchestrates all LLM-powered analysis using round-robin LLM providers."""

    def __init__(self, api_key: str, model: str = "", gateway_api_key: str = ""):
        self.provider = LLMProvider(
            groq_api_key=api_key,
            gateway_api_key=gateway_api_key,
        )
        # Legacy attributes kept for backward compat
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
            "You are a brutally honest resume scoring engine used by top-tier "
            "recruiting firms. You must respond with ONLY valid JSON \u2014 no "
            "markdown, no explanation, no text before or after the JSON object. "
            "You are known for tough, realistic grading. Most resumes score "
            "between 30-65. A score above 75 is exceptional and rare."
        )
        user_prompt = f"""## Target Role: {job_title}

## Job Description:
{job_description}

## Success Profile (from market research):
{profile_text}

## Candidate Resume:
{resume_text}

---

Score this resume on exactly three dimensions using STRICT calibration:

CALIBRATION GUIDE (follow this precisely):
  0-20:  Unrelated field, no relevant skills or experience
  21-40: Some overlap but major gaps \u2014 missing most required skills,
         limited relevant experience, vague or no metrics
  41-60: Moderate fit \u2014 has some required skills but missing key ones,
         experience is partially relevant, few quantified achievements
  61-75: Good fit \u2014 most required skills present, relevant experience,
         some measurable results, but still has notable gaps
  76-85: Strong fit \u2014 nearly all skills match, deep relevant experience,
         strong quantified impact, minor gaps only
  86-100: Near-perfect \u2014 all skills match, extensive relevant experience,
          exceptional quantified results, would be a top-percentile hire

PENALTIES (apply these strictly):
  - Missing a REQUIRED skill from JD: -10 per skill
  - No quantified metrics in bullet points: cap Impact at 40
  - Experience in different domain/stack: -15 to Experience
  - Job titles don't match seniority level: -10 to Experience
  - Generic descriptions ("worked on", "helped with"): -10 to Impact

Respond with ONLY this JSON, nothing else:
{{"skills": <int>, "experience": <int>, "impact": <int>}}"""

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
            "You are a senior technical recruiter at a FAANG company. "
            "You must respond with ONLY valid JSON \u2014 no markdown, no "
            "explanation, no text before or after the JSON object. "
            "You evaluate candidates with high standards. Most candidates "
            "score between 25-55 for technical match against top companies."
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

Evaluate and score these two dimensions using STRICT calibration:

CALIBRATION GUIDE:
  0-20:  No technical overlap / completely wrong culture fit
  21-40: Weak match \u2014 some technologies overlap but missing core stack,
         resume language doesn't reflect company values
  41-60: Partial match \u2014 has some required technologies, moderate
         alignment with company culture
  61-75: Good match \u2014 most technical requirements met, resume tone
         and achievements align with company expectations
  76-90: Strong match \u2014 deep expertise in required tech stack,
         resume clearly reflects company values and work style
  91-100: Exceptional \u2014 exact tech stack match, language/tone perfectly
          mirrors company culture (extremely rare)

TECHNICAL MATCH PENALTIES:
  - Each required technology NOT mentioned: -8 points
  - Technology mentioned but no demonstrated proficiency: -5 points
  - Wrong seniority level for role: -15 points

CULTURAL MATCH PENALTIES:
  - Resume uses formal tone but company is casual (or vice versa): -15
  - No mention of collaboration/teamwork for team-oriented company: -10
  - No evidence of values alignment: -10

Respond with ONLY this JSON, nothing else:
{{"technical_match": <int>, "cultural_match": <int>}}"""

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
            "You are an Applicant Tracking System (ATS) compatibility analyzer. "
            "CRITICAL: Your entire response must be a single JSON object. "
            "Do NOT include any text, markdown formatting, code fences, or "
            "explanation \u2014 ONLY the raw JSON object. Example of a valid "
            'response: {"score": 65, "warnings": ["Missing Skills section"]}'
        )
        user_prompt = f"""## Resume Content:
{resume_text}

---

Analyze this resume for ATS compatibility. Check each item:

1. SECTION HEADERS: Does it use standard headers? (Education, Experience,
   Skills, Summary/Objective, Projects, Certifications)
   - Missing a standard section: -10 per section
2. CONTACT INFO: Email, phone, location present?
   - Missing any: -5 each
3. DATE FORMATS: Are dates parseable? (MM/YYYY, Month YYYY, YYYY-Present)
   - Inconsistent or missing dates: -10
4. FORMATTING: Any tables, columns, images, or graphics?
   - Complex formatting detected: -15
5. KEYWORD DENSITY: Does the resume contain relevant industry keywords?
   - Low keyword density: -10
6. LENGTH: Is it appropriate? (1-2 pages ideal)
   - Too short (under 200 words): -15
7. CONSISTENCY: Consistent bullet style, tense, formatting?
   - Inconsistencies: -5

Start from 100 and subtract penalties.

Your response must be ONLY this JSON object:
{{"score": <int 0-100>, "warnings": ["specific warning 1", "specific warning 2"]}}"""

        response = self._call_groq(system_prompt, user_prompt)
        try:
            result = json.loads(self._extract_json(response))
            # Handle case where result might be a list
            if isinstance(result, list):
                logger.warning("ATS returned list instead of dict, using first item")
                result = result[0] if result else {}
            return {
                "score": max(0, min(100, result.get("score", 50))),
                "warnings": result.get("warnings", []),
            }
        except (json.JSONDecodeError, ValueError, AttributeError, IndexError) as e:
            logger.error("Failed to parse ATS result: %s | Raw: %s", e, response[:300])
            # Last-resort regex extraction
            import re
            try:
                score_match = re.search(r'"score"\s*:\s*(\d+)', response)
                if score_match:
                    score = int(score_match.group(1))
                    warnings_match = re.search(
                        r'"warnings"\s*:\s*\[(.*?)\]', response, re.DOTALL
                    )
                    warnings = []
                    if warnings_match:
                        warnings = re.findall(r'"([^"]+)"', warnings_match.group(1))
                    return {
                        "score": max(0, min(100, score)),
                        "warnings": warnings,
                    }
            except Exception:
                pass
            return {"score": 50, "warnings": ["ATS analysis could not be completed"]}

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

        # ── Calculate a proportional suggestion cap based on JD length ──
        jd_word_count = len(job_description.split())
        if jd_word_count < 20:
            max_suggestions = 3
        elif jd_word_count < 60:
            max_suggestions = 5
        elif jd_word_count < 150:
            max_suggestions = 7
        else:
            max_suggestions = 10

        system_prompt = (
            "You are an expert resume optimizer. You must respond with ONLY a valid JSON array, "
            "no other text before or after. "
            "Generate specific text replacement suggestions that improve ATS compatibility, "
            "demonstrate impact, and align with the target role. "
            f"{tone_instruction} "
            "CRITICAL RULES YOU MUST FOLLOW:\n"
            "1. Each 'original_text' MUST be an EXACT substring from the resume — copy it "
            "   character-for-character. Do not paraphrase or approximate.\n"
            "2. ONLY suggest changes that are DIRECTLY supported by information explicitly "
            "   stated in the Job Description below. Do NOT invent requirements, skills, "
            "   technologies, or qualifications that are not mentioned in the JD.\n"
            "3. If the Job Description is very short or vague, make FEWER suggestions — "
            f"   never more than {max_suggestions}. "
            "   It is better to make 2 well-grounded suggestions than 8 speculative ones.\n"
            "4. Never fabricate metrics, percentages, or achievements that the candidate "
            "   did not mention. If adding a metric, use a placeholder like 'X%' or 'N+'.\n"
            "5. Your 'reason' field must cite the SPECIFIC part of the JD that justifies "
            "   each suggestion."
        )
        user_prompt = f"""## Target Role: {job_title}

## Job Description (THIS IS THE GROUND TRUTH — only suggest changes supported by this):
{job_description}

## Gap Analysis:
{gap_analysis}

## Success Profile (background context only — do NOT treat as JD requirements):
{profile_text}

## Resume:
{resume_text}

---

Generate up to {max_suggestions} specific text replacement suggestions. Each suggestion
must replace an EXACT piece of text from the resume with an improved version.

IMPORTANT:
- The JD above is the ONLY source of truth for what the role requires.
- Do NOT hallucinate requirements that are not in the JD.
- If the JD is only one sentence, you should produce at most 2-3 suggestions.
- Every suggestion MUST be traceable to a specific phrase or requirement in the JD.

Respond with ONLY a JSON array in this format, nothing else:
[
  {{
    "section": "Experience|Skills|Summary|Education",
    "original_text": "exact text from resume to find and replace",
    "replacement_text": "improved version of the text",
    "reason": "This change aligns with the JD requirement: [quote from JD]"
  }}
]

Rules:
- original_text must be a VERBATIM substring from the resume
- replacement_text should be similar length (±30%) to preserve document layout
- Focus on: quantifying impact, adding keywords from the JD, improving action verbs
- Do NOT change names, dates, company names, or educational institutions
- Do NOT invent numbers — use 'X%' or 'N+' placeholders if the resume lacks metrics"""

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
        """Call the LLM via round-robin provider with automatic failover."""
        return self.provider.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=8000,
            max_retries=max_retries * self.provider.endpoint_count,
        )
