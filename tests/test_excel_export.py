from src.config import FieldSpec, SchemaConfig
from src.export.excel_export import profiles_to_dataframe
from src.models import EntityProfile, FieldExtraction


def _schema() -> SchemaConfig:
    return SchemaConfig(
        project_name="Test",
        confidence_threshold=0.6,
        fields=[
            FieldSpec(name="company_name", type="string"),
            FieldSpec(name="revenue", type="number", unit="INR crore"),
        ],
    )


def test_profiles_to_dataframe_includes_audit_columns():
    profile = EntityProfile(
        entity_id="acme",
        fields={
            "company_name": FieldExtraction(
                value="Acme Ltd", confidence=0.9, citation="https://x", needs_review=False
            ),
            "revenue": FieldExtraction(
                value=None, confidence=0.1, citation="", needs_review=True
            ),
        },
    )

    df = profiles_to_dataframe([profile], _schema())

    assert df.loc[0, "company_name"] == "Acme Ltd"
    assert df.loc[0, "company_name_confidence"] == 0.9
    assert bool(df.loc[0, "revenue_needs_review"]) is True


def test_profiles_to_dataframe_handles_missing_field():
    profile = EntityProfile(entity_id="acme", fields={})
    df = profiles_to_dataframe([profile], _schema())
    assert df.loc[0, "company_name"] is None
    assert bool(df.loc[0, "company_name_needs_review"]) is True
