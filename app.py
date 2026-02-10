"""
Resume Optimizer – Flask Application

Main entry point for the web application. Handles file uploads,
orchestrates the analysis pipeline, and serves results.
"""

import os
import uuid
import logging
import traceback

from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename

from config import Config
from src.research_engine import ResearchEngine
from src.llm_analyzer import LLMAnalyzer
from src.resume_editor import ResumeEditor
from src.pdf_generator import PDFGenerator

# ------------------------------------------------------------------ #
#  Setup
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = Config.SECRET_KEY

ALLOWED_EXTENSIONS = {"docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Full analysis pipeline:
    1. Save uploaded .docx
    2. Extract text
    3. Run web research
    4. Run LLM analysis
    5. Optionally apply suggestions
    6. Generate PDFs
    7. Return results JSON
    """
    try:
        # --- Validate inputs ---
        if "resume" not in request.files:
            return jsonify({"error": "No resume file uploaded"}), 400

        file = request.files["resume"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Please upload a .docx file"}), 400

        job_title = request.form.get("job_title", "").strip()
        job_description = request.form.get("job_description", "").strip()
        company_name = request.form.get("company_name", "").strip() or None
        auto_apply = request.form.get("auto_apply", "false").lower() == "true"

        if not job_title:
            return jsonify({"error": "Job title is required"}), 400
        if not job_description:
            return jsonify({"error": "Job description is required"}), 400

        # --- Validate API key ---
        if not Config.GROQ_API_KEY:
            return jsonify({
                "error": "GROQ_API_KEY not configured. Add it to your .env file."
            }), 500

        # Ensure directories exist
        Config.validate()

        # --- Save uploaded file ---
        session_id = str(uuid.uuid4())[:8]
        original_name = secure_filename(file.filename)
        upload_path = os.path.join(Config.UPLOAD_FOLDER, f"{session_id}_{original_name}")
        file.save(upload_path)
        logger.info("File saved: %s", upload_path)

        # --- Step 1: Extract resume text ---
        editor = ResumeEditor(upload_path)
        resume_text = editor.extract_text()
        logger.info("Extracted %d characters from resume", len(resume_text))

        if len(resume_text) < 50:
            return jsonify({
                "error": "The uploaded document appears to be empty or too short."
            }), 400

        # --- Step 2: Web Research ---
        research_engine = ResearchEngine()
        success_profile = research_engine.research(
            job_title=job_title,
            job_description=job_description,
            company_name=company_name,
        )

        # --- Step 3: LLM Analysis ---
        analyzer = LLMAnalyzer(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
        )
        analysis = analyzer.analyze(
            resume_text=resume_text,
            job_title=job_title,
            job_description=job_description,
            success_profile=success_profile,
        )

        # --- Step 4: Apply suggestions (if auto-apply) ---
        optimized_filename = None
        edit_results = None
        if auto_apply and analysis.get("suggestions"):
            edit_results = editor.apply_suggestions(analysis["suggestions"])

            # Add talking points page
            editor.add_talking_points_page(analysis["suggestions"])

            optimized_filename = f"{session_id}_optimized_{original_name}"
            optimized_path = os.path.join(Config.OUTPUT_FOLDER, optimized_filename)
            editor.save(optimized_path)
            logger.info("Optimized resume saved: %s", optimized_path)

        # --- Step 5: Generate PDFs ---
        interview_pdf_filename = None
        cover_letter_pdf_filename = None
        pdf_gen = PDFGenerator()

        if analysis.get("interview_questions"):
            interview_pdf_filename = f"{session_id}_interview_prep.pdf"
            interview_path = os.path.join(Config.OUTPUT_FOLDER, interview_pdf_filename)
            pdf_gen.generate_interview_prep(
                questions=analysis["interview_questions"],
                job_title=job_title,
                output_path=interview_path,
            )

        if analysis.get("cover_letter"):
            cover_letter_pdf_filename = f"{session_id}_cover_letter.pdf"
            cover_letter_path = os.path.join(Config.OUTPUT_FOLDER, cover_letter_pdf_filename)
            pdf_gen.generate_cover_letter(
                cover_letter_text=analysis["cover_letter"],
                job_title=job_title,
                company_name=company_name or "Target Company",
                output_path=cover_letter_path,
            )

        # --- Build response ---
        response = {
            "success": True,
            "scores": analysis.get("scores", {}),
            "ats_score": analysis.get("ats_score", 0),
            "ats_warnings": analysis.get("ats_warnings", []),
            "gap_analysis": analysis.get("gap_analysis", ""),
            "suggestions": analysis.get("suggestions", []),
            "interview_questions": analysis.get("interview_questions", []),
            "cover_letter": analysis.get("cover_letter", ""),
            "overall_summary": analysis.get("overall_summary", ""),
            "auto_applied": auto_apply,
            "edit_results": edit_results,
            "downloads": {
                "optimized_resume": f"/download/{optimized_filename}" if optimized_filename else None,
                "interview_prep": f"/download/{interview_pdf_filename}" if interview_pdf_filename else None,
                "cover_letter": f"/download/{cover_letter_pdf_filename}" if cover_letter_pdf_filename else None,
            },
            "research_summary": {
                "company": company_name or "Not specified",
                "cultural_tone": success_profile.get("cultural_tone", "balanced"),
            },
        }

        return jsonify(response)

    except Exception as e:
        logger.error("Analysis failed: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/download/<filename>")
def download(filename):
    """Serve generated output files for download."""
    safe_name = secure_filename(filename)
    file_path = os.path.join(Config.OUTPUT_FOLDER, safe_name)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True, download_name=safe_name)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    Config.validate()
    logger.info("Starting Resume Optimizer on http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
