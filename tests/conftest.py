import pytest

from ragbench.data import Corpus, Document

DOCS = [
    Document(
        "d1",
        "Vitamin D and bones",
        "Vitamin D supplementation reduces fracture risk in the elderly.",
    ),
    Document("d2", "Aspirin and cancer", "Regular aspirin use lowers colorectal cancer incidence."),
    Document("d3", "Exercise and cognition", "Aerobic exercise improves memory in older adults."),
    Document(
        "d4", "Sleep deprivation", "Chronic sleep loss impairs attention and immune function."
    ),
    Document("d5", "BRCA1 mutations", "BRCA1 mutations raise the lifetime risk of breast cancer."),
]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        documents=list(DOCS),
        queries={"q1": "does vitamin D prevent fractures", "q2": "aspirin colorectal cancer"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
