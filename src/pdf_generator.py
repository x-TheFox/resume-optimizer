"""
PDFGenerator – Creates professionally styled PDFs for interview prep
and cover letters using ReportLab.

Supports both file-path output and in-memory BytesIO output for
serverless environments like Vercel.
"""

import io
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    ListFlowable,
    ListItem,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generate professional PDF documents for interview prep and cover letters."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Create custom paragraph styles for a polished look."""
        self.styles.add(ParagraphStyle(
            name="DocTitle",
            parent=self.styles["Heading1"],
            fontSize=22,
            textColor=HexColor("#1a1a2e"),
            spaceAfter=12,
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name="DocSubtitle",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=HexColor("#4a4a6a"),
            alignment=TA_CENTER,
            spaceAfter=24,
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading2"],
            fontSize=14,
            textColor=HexColor("#16213e"),
            spaceBefore=18,
            spaceAfter=8,
            borderWidth=0,
            borderPadding=0,
        ))
        self.styles.add(ParagraphStyle(
            name="QuestionText",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=HexColor("#2d2d44"),
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=12,
            leading=16,
        ))
        self.styles.add(ParagraphStyle(
            name="BodyText2",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=HexColor("#333333"),
            alignment=TA_JUSTIFY,
            leading=16,
            spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            name="CoverLetterBody",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=HexColor("#2d2d44"),
            alignment=TA_JUSTIFY,
            leading=17,
            spaceAfter=10,
            firstLineIndent=0,
        ))

    def generate_interview_prep(
        self, questions: list, job_title: str, output_path: str = None
    ) -> io.BytesIO:
        """
        Generate a 1-page Interview Prep PDF with likely questions.

        Args:
            questions: List of interview question strings
            job_title: Target job title for the header
            output_path: Path to save the PDF (None = return BytesIO)

        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = io.BytesIO()
        target = output_path if output_path else buffer
        doc = SimpleDocTemplate(
            target,
            pagesize=letter,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        elements = []

        # Title
        elements.append(Paragraph("Interview Preparation Guide", self.styles["DocTitle"]))
        elements.append(Paragraph(
            f"Targeted questions for <b>{self._escape(job_title)}</b> role — "
            f"based on identified resume gaps",
            self.styles["DocSubtitle"],
        ))

        # Divider
        elements.append(HRFlowable(
            width="100%", thickness=1, color=HexColor("#4f8cff"),
            spaceAfter=12, spaceBefore=4,
        ))

        # Categorize questions
        technical = []
        behavioral = []
        situational = []

        for q in questions:
            q_lower = q.lower()
            if any(kw in q_lower for kw in ["how would you", "what would you do", "imagine", "scenario"]):
                situational.append(q)
            elif any(kw in q_lower for kw in ["tell me about", "describe a time", "give an example", "have you ever"]):
                behavioral.append(q)
            else:
                technical.append(q)

        # If categorization didn't split well, just use all
        if not technical and not behavioral and not situational:
            technical = questions

        if technical:
            elements.append(Paragraph("Technical & Skills Questions", self.styles["SectionHeader"]))
            for i, q in enumerate(technical, 1):
                elements.append(Paragraph(
                    f"<b>{i}.</b> {self._escape(q)}", self.styles["QuestionText"]
                ))

        if behavioral:
            elements.append(Paragraph("Behavioral Questions", self.styles["SectionHeader"]))
            for i, q in enumerate(behavioral, 1):
                elements.append(Paragraph(
                    f"<b>{i}.</b> {self._escape(q)}", self.styles["QuestionText"]
                ))

        if situational:
            elements.append(Paragraph("Situational Questions", self.styles["SectionHeader"]))
            for i, q in enumerate(situational, 1):
                elements.append(Paragraph(
                    f"<b>{i}.</b> {self._escape(q)}", self.styles["QuestionText"]
                ))

        # Footer tip
        elements.append(Spacer(1, 16))
        elements.append(HRFlowable(
            width="100%", thickness=0.5, color=HexColor("#cccccc"),
            spaceAfter=8, spaceBefore=4,
        ))
        elements.append(Paragraph(
            "<i>💡 Tip: For each question, prepare a STAR-format answer "
            "(Situation, Task, Action, Result) with specific metrics where possible.</i>",
            self.styles["QuestionText"],
        ))

        doc.build(elements)
        if output_path:
            logger.info("Interview prep PDF saved to: %s", output_path)
        buffer.seek(0)
        return buffer

    def generate_cover_letter(
        self, cover_letter_text: str, job_title: str, company_name: str, output_path: str = None
    ) -> io.BytesIO:
        """
        Generate a professionally formatted cover letter PDF.

        Args:
            cover_letter_text: The cover letter content
            job_title: Target job title
            company_name: Target company name
            output_path: Path to save the PDF (None = return BytesIO)

        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = io.BytesIO()
        target = output_path if output_path else buffer
        doc = SimpleDocTemplate(
            target,
            pagesize=letter,
            topMargin=1 * inch,
            bottomMargin=1 * inch,
            leftMargin=1 * inch,
            rightMargin=1 * inch,
        )

        elements = []

        # Header
        elements.append(Paragraph("Cover Letter", self.styles["DocTitle"]))
        elements.append(Paragraph(
            f"Application for <b>{self._escape(job_title)}</b> at <b>{self._escape(company_name)}</b>",
            self.styles["DocSubtitle"],
        ))

        elements.append(HRFlowable(
            width="100%", thickness=1, color=HexColor("#4f8cff"),
            spaceAfter=20, spaceBefore=8,
        ))

        # Body paragraphs
        paragraphs = cover_letter_text.strip().split("\n\n")
        for para_text in paragraphs:
            para_text = para_text.strip()
            if para_text:
                # Handle single newlines within paragraphs
                para_text = para_text.replace("\n", "<br/>")
                elements.append(Paragraph(
                    self._escape_preserve_br(para_text),
                    self.styles["CoverLetterBody"],
                ))

        doc.build(elements)
        if output_path:
            logger.info("Cover letter PDF saved to: %s", output_path)
        buffer.seek(0)
        return buffer

    def generate_talking_points_pdf(
        self, suggestions: list, job_title: str, output_path: str = None
    ) -> io.BytesIO:
        """
        Generate a Talking Points PDF that documents every resume edit
        so the candidate can explain changes in interviews.

        Args:
            suggestions: List of suggestion dicts with original_text,
                         replacement_text, reason, talking_point, section
            job_title: Target job title for the header
            output_path: Path to save the PDF (None = return BytesIO)

        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = io.BytesIO()
        target = output_path if output_path else buffer
        doc = SimpleDocTemplate(
            target,
            pagesize=letter,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        elements = []

        # Title
        elements.append(Paragraph(
            "Interview Talking Points", self.styles["DocTitle"]
        ))
        elements.append(Paragraph(
            f"Prepared responses for every resume optimization — "
            f"<b>{self._escape(job_title)}</b> role",
            self.styles["DocSubtitle"],
        ))
        elements.append(HRFlowable(
            width="100%", thickness=1, color=HexColor("#4f8cff"),
            spaceAfter=14, spaceBefore=4,
        ))

        # Custom small styles for diffs
        diff_old_style = ParagraphStyle(
            name="DiffOld",
            parent=self.styles["Normal"],
            fontSize=9.5,
            textColor=HexColor("#888888"),
            leftIndent=16,
            leading=14,
            spaceAfter=2,
        )
        diff_new_style = ParagraphStyle(
            name="DiffNew",
            parent=self.styles["Normal"],
            fontSize=9.5,
            textColor=HexColor("#1a1a2e"),
            leftIndent=16,
            leading=14,
            spaceAfter=6,
        )
        reason_style = ParagraphStyle(
            name="Reason",
            parent=self.styles["Normal"],
            fontSize=10,
            textColor=HexColor("#4a4a6a"),
            leftIndent=16,
            leading=14,
            spaceAfter=4,
        )
        tp_style = ParagraphStyle(
            name="TalkingPt",
            parent=self.styles["Normal"],
            fontSize=10.5,
            textColor=HexColor("#2d2d44"),
            leftIndent=16,
            leading=15,
            spaceAfter=8,
        )

        for i, s in enumerate(suggestions, 1):
            section = s.get("section", "General")
            original = s.get("original_text", "")
            replacement = s.get("replacement_text", "")
            reason = s.get("reason", "")
            point = s.get("talking_point", "")

            # Section header
            elements.append(Paragraph(
                f"Edit {i}: {self._escape(section)}",
                self.styles["SectionHeader"],
            ))

            # Before / after
            if original:
                elements.append(Paragraph(
                    f"<b>Before:</b> <strike>{self._escape(original)}</strike>",
                    diff_old_style,
                ))
            if replacement:
                elements.append(Paragraph(
                    f"<b>After:</b> {self._escape(replacement)}",
                    diff_new_style,
                ))

            # Reason
            if reason:
                elements.append(Paragraph(
                    f"<i>Why: {self._escape(reason)}</i>",
                    reason_style,
                ))

            # Talking point
            if point:
                elements.append(Paragraph(
                    f"🎤 <b>Say in interview:</b> {self._escape(point)}",
                    tp_style,
                ))

            # Divider between edits
            if i < len(suggestions):
                elements.append(HRFlowable(
                    width="80%", thickness=0.4, color=HexColor("#dddddd"),
                    spaceAfter=6, spaceBefore=6,
                ))

        # Footer tip
        elements.append(Spacer(1, 16))
        elements.append(HRFlowable(
            width="100%", thickness=0.5, color=HexColor("#cccccc"),
            spaceAfter=8, spaceBefore=4,
        ))
        elements.append(Paragraph(
            "<i>💡 Review each talking point before your interview. "
            "Practice saying them aloud in 30-60 seconds each.</i>",
            self.styles["QuestionText"],
        ))

        doc.build(elements)
        if output_path:
            logger.info("Talking points PDF saved to: %s", output_path)
        buffer.seek(0)
        return buffer

    def _escape(self, text: str) -> str:
        """Escape special XML characters for ReportLab paragraphs."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _escape_preserve_br(self, text: str) -> str:
        """Escape XML characters but preserve <br/> tags."""
        text = text.replace("&", "&amp;")
        # Temporarily protect <br/> tags
        text = text.replace("<br/>", "%%BR%%")
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("%%BR%%", "<br/>")
        return text

