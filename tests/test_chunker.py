from pathlib import Path

from sungrid_copilot.ingestion.chunker import chunk_markdown

KB = Path(__file__).resolve().parents[1] / "data" / "kb"


def test_one_chunk_per_section_with_contextual_header():
    chunks = chunk_markdown("02_incentive_rebate_programs", (KB / "02_incentive_rebate_programs.md").read_text())
    sections = [c.section for c in chunks]
    assert sections == [
        "Rooftop Rebate Program",
        "Commercial Incentive Program",
        "Appeals Process",
        "Program Funding Status",
        "Interaction with Other Incentive Programs",
        "Documentation Requirements",
    ]
    assert chunks[2].embed_text.startswith("Incentive & Rebate Programs > Appeals Process\n")
    assert chunks[2].citation == "02_incentive_rebate_programs § Appeals Process"


def test_preamble_becomes_overview_chunk():
    chunks = chunk_markdown("06_x", (KB / "06_ambiguous_rebate_billing_adjustments.md").read_text())
    assert chunks[0].section == "Overview"
    assert "previously approved rooftop rebate is later reversed" in chunks[0].text


def test_tables_are_linearised_into_quotable_rows():
    chunks = chunk_markdown("03_billing_faqs", (KB / "03_billing_faqs.md").read_text())
    fee = next(c for c in chunks if c.section == "How is my monthly bill calculated?")
    assert "Household tier — Monthly Membership Fee: $8; Net Metering Rate: $0.07/kWh credit." in fee.text
    assert "|" not in fee.text


def test_markdown_emphasis_removed_and_ids_deterministic():
    md = (KB / "04_technical_installation_guidance.md").read_text()
    a, b = chunk_markdown("04", md), chunk_markdown("04", md)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all("**" not in c.text for c in a)


def test_cross_document_references_are_captured():
    chunks = chunk_markdown("01_program_policies", (KB / "01_program_policies.md").read_text())
    grievance = next(c for c in chunks if c.section == "Grievance Procedure")
    assert "Incentive & Rebate Programs" in grievance.references


def test_long_sections_are_split_with_overlap():
    body = " ".join(f"Sentence number {i} talks about solar panels." for i in range(120))
    chunks = chunk_markdown("long", f"# T — Long\n\n## Big\n{body}", max_words=100)
    assert len(chunks) > 1
    assert chunks[1].section == "Big (part 2)"
    last_of_first = chunks[0].text.split(". ")[-1]
    assert last_of_first.rstrip(".") in chunks[1].text  # one-sentence overlap
