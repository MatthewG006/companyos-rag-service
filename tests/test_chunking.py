from app.config import Settings
from app.ingest import chunk_text


def test_short_text_is_one_chunk():
    settings = Settings(chunk_size=900, chunk_overlap=150)
    assert chunk_text("Water the ferns twice a week.", settings) == ["Water the ferns twice a week."]


def test_long_text_splits_into_bounded_chunks():
    settings = Settings(chunk_size=200, chunk_overlap=40)
    text = "\n\n".join(
        f"Paragraph {i} discusses greenhouse humidity control, watering schedules, "
        "soil composition, light exposure, and seasonal plant care routines in detail."
        for i in range(12)
    )
    chunks = chunk_text(text, settings)
    assert len(chunks) > 3
    assert all(chunks), "chunks must be non-empty"
    assert all(len(c) <= 200 for c in chunks), "chunks must respect the configured size"


def test_empty_text_yields_no_chunks():
    settings = Settings()
    assert chunk_text("   \n  ", settings) == []
