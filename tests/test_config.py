from src.config import load_schema


def test_load_schema():
    schema = load_schema("config/schema.yaml")
    assert schema.project_name
    assert 0 < schema.confidence_threshold <= 1
    names = schema.field_names()
    assert "company_name" in names
    assert "revenue" in names
    assert "csr_activities" in names

    revenue = next(f for f in schema.fields if f.name == "revenue")
    assert revenue.type == "number"
    assert revenue.unit == "INR crore"
