from src.ingestion.input_loader import load_entities


def test_load_entities_from_sample_csv():
    entities = load_entities("data/input/companies.csv")
    assert len(entities) == 2

    first = entities[0]
    assert first.company == "Godrej Industries"
    assert first.leader_name == "Nadir Godrej"
    assert first.seed_urls == []  # empty 'sources' cell must not become ["nan"]


def test_load_entities_parses_pipe_separated_sources(tmp_path):
    csv_path = tmp_path / "entities.csv"
    csv_path.write_text(
        "company,leader_name,sources\n"
        "Acme Ltd,Jane Doe,https://a.com|https://b.com\n"
    )
    entities = load_entities(csv_path)
    assert entities[0].seed_urls == ["https://a.com", "https://b.com"]
