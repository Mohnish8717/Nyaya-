import pymupdf
from app.pdf_generator import generate_legal_pdf

def test_generate_legal_pdf_produces_valid_pdf():
    title = "Application under Section 156(3) CrPC before the Judicial Magistrate"
    content = """IN THE COURT OF THE LEARNED JUDICIAL MAGISTRATE FIRST CLASS AT NEW DELHI
CRIMINAL MISC. APPLICATION NO. ________ OF 2026

IN THE MATTER OF:
Ramesh Kumar ... APPLICANT

VERSUS

1. State through SHO
2. Suresh Verma ... RESPONDENTS

MOST RESPECTFULLY SHOWETH:
1. That the applicant is a citizen of India residing at New Delhi.
2. That on 10.05.2026 the respondent committed a cognizable offence.

PRAYER:
Wherefore, it is prayed that this Hon'ble Court direct the police to register an FIR.

APPLICANT
VERIFICATION:
Verified at New Delhi that the contents are true.
DEPONENT
"""
    pdf_bytes = generate_legal_pdf(title, content, "application_156_3", warning="Review with an advocate.")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    
    # Verify PDF structure using PyMuPDF
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1
    page1_text = doc[0].get_text()
    assert "SECTION 156(3)" in page1_text.upper()
    assert "STATUTORY NOTICE" in page1_text
    assert "Nyaya Legal Information System" in page1_text
    assert "Page 1" in page1_text


def test_generate_universal_legal_pdfs_across_domains():
    from app import tools, facts
    ctx = tools.ToolContext(text="draft", jurisdiction="IN-DL", ledger=facts.Ledger())

    doc_cases = [
        ("bail_application", "Bail Application under Section 439 CrPC", "Accused was falsely implicated in theft FIR.", "Release applicant on regular bail."),
        ("anticipatory_bail", "Application for Anticipatory Bail under Section 438 CrPC", "Apprehension of arrest in business dispute.", "Direct release on bail in the event of arrest."),
        ("civil_suit", "Plaint in Suit for Permanent Injunction under Order VII Rule 1 CPC", "Defendant is trying to dispossess plaintiff from suit property.", "Grant permanent prohibitory injunction."),
        ("legal_notice_106_tpa", "Notice of Termination of Tenancy under Section 106 TPA", "Tenant has defaulted on rent for 6 months.", "Vacate premises within 15 days."),
        ("writ_petition", "Writ Petition under Article 226 of the Constitution", "Statutory authority arbitrarily cancelled license without notice.", "Issue Writ of Mandamus quashing cancellation."),
        ("maintenance_application", "Application for Maintenance under Section 125 CrPC", "Spouse refused to provide maintenance despite sufficient means.", "Grant monthly maintenance of Rs 25,000."),
        ("general_petition", "Petition before National Company Law Tribunal", "Oppression and mismanagement of private limited company.", "Appoint independent administrator."),
    ]

    for doc_type, title, grounds, prayer in doc_cases:
        res = tools.execute(
            "draft_legal_document",
            {
                "document_type": doc_type,
                "title": title,
                "complainant_or_applicant": "Sunil Sharma, Age 38, S/o Ramesh Sharma, R/o New Delhi",
                "respondent_or_opposite_party": "Vikram Malhotra, R/o New Delhi",
                "facts_and_grounds": grounds,
                "prayer": prayer,
                "jurisdiction": "IN-DL",
            },
            ctx,
        )
        assert res.ok is True, f"Failed drafting {doc_type}: {res.error}"
        artifact = res.data["artifact"]
        assert "content" in artifact
        assert len(artifact["content"]) > 200

        pdf_bytes = generate_legal_pdf(
            title=artifact["title"],
            content=artifact["content"],
            document_type=artifact["type"],
            warning=artifact.get("warning", ""),
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        first_page = doc[0].get_text()
        assert "NYAYA" in first_page.upper()
        assert "PAGE 1" in first_page.upper()


def test_normalize_and_format_squashed_legal_notice():
    from app.pdf_generator import normalize_legal_draft_text, generate_legal_pdf

    raw_notice = (
        "LEGAL NOTICE To, Ramesh Kumar, S/o Shri Suresh, New Delhi, Date: 10.05.2026\n"
        "Subject: Legal Notice for recovery of Rs. 1,00,000/- due under Promissory Note.\n"
        "Sir/Madam, Under instructions from my client, I serve you notice:\n"
        "1. My client states you took a loan of Rs. 1,00,000/-. "
        "2. You executed a Promissory Note. "
        "3. You failed to repay.\n"
        "Copy retained for record.\n"
        "Advocate for Client"
    )

    normalized = normalize_legal_draft_text(raw_notice)
    assert "LEGAL NOTICE\n\nTo," in normalized
    assert "1. My client states" in normalized
    assert "2. You executed" in normalized
    assert "3. You failed" in normalized

    # PDF generation should produce valid pages without turning the whole document bold
    pdf_bytes = generate_legal_pdf("Legal Notice", raw_notice, "legal_notice", warning="General draft.")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1500
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1
    page_text = doc[0].get_text()
    assert "LEGAL NOTICE" in page_text
    assert "Ramesh Kumar" in page_text
    assert "1. My client states" in page_text

