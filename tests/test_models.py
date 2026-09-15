from src.models import make_entity_id


def test_make_entity_id_is_deterministic_and_safe():
    id1 = make_entity_id("Godrej Industries", "Nadir Godrej")
    id2 = make_entity_id("Godrej Industries", "Nadir Godrej")
    assert id1 == id2
    assert " " not in id1
    assert id1 == id1.lower()


def test_make_entity_id_handles_punctuation():
    entity_id = make_entity_id("Tata & Sons, Pvt. Ltd.", "")
    assert "&" not in entity_id
    assert "," not in entity_id
    assert entity_id
