"""News pipeline: RSS parsing, PIT-safe corpus ingestion, headlines, event tags."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from leantrade.core.types import InstrumentId, Observation
from leantrade.data.corpus.news import (
    ingest_news, recent_headlines, tag_events, evidence_summary_for,
)
from leantrade.data.corpus.store import CorpusStore
from leantrade.data.providers.base import FetchRequest
from leantrade.data.providers.news import RSSNewsProvider, parse_rss
from leantrade.data.store.pit import PITStore
from leantrade.llm.extraction.schemas import ExtractedEvidence

UTC = timezone.utc

RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Reliance posts record quarterly results</title>
<description>&lt;p&gt;Profit up 12%&lt;/p&gt;</description>
<link>http://x/1</link><pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate></item>
<item><title>No timestamp article</title><link>http://x/2</link></item>
<item><title>Too old</title><link>http://x/3</link>
<pubDate>Mon, 01 Jan 2023 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def _obs(iid: str, title: str, published: datetime, url: str) -> Observation:
    return Observation(
        instrument_id=InstrumentId(iid), source_id="rss", kind="news",
        event_time=published, available_at=published,
        payload={"title": title, "summary": "", "url": url}, pit_level="partial",
    )


def test_parse_rss_strips_html_and_parses_dates():
    items = parse_rss(RSS)
    assert items[0]["summary"] == "Profit up 12%"
    assert items[0]["published"] == datetime(2024, 1, 15, 10, tzinfo=UTC)
    assert items[1]["published"] is None


def test_provider_drops_undated_and_out_of_window(monkeypatch):
    p = RSSNewsProvider(templates=["http://feed/{ticker}"])
    monkeypatch.setattr(p, "_get", lambda url: RSS)
    req = FetchRequest(
        "news", since=datetime(2024, 1, 1, tzinfo=UTC), until=datetime(2024, 2, 1, tzinfo=UTC),
        instruments={InstrumentId("XNSE:RELIANCE")}, market="XNSE",
    )
    out = list(p.fetch(req))
    assert [o.payload["title"] for o in out] == ["Reliance posts record quarterly results"]
    assert out[0].available_at == out[0].event_time  # never earlier than publication


def test_same_timestamp_articles_not_collapsed_in_pit_store(tmp_path):
    t = datetime(2024, 1, 15, 10, tzinfo=UTC)
    with PITStore(tmp_path) as store:
        store.ingest([_obs("XNSE:TCS", "A", t, "u1"), _obs("XNSE:TCS", "B", t, "u2")])
        df = store.query_observations(as_of=t + timedelta(days=1), since=t - timedelta(days=1),
                                      kind="news")
    assert df.height == 2


def test_ingest_merges_instruments_and_respects_pit(tmp_path):
    t = datetime(2024, 1, 15, 10, tzinfo=UTC)
    corpus = CorpusStore(tmp_path / "c.duckdb")
    added = ingest_news(
        [_obs("XNSE:TCS", "Sector news", t, "u"), _obs("XNSE:INFY", "Sector news", t, "u")],
        corpus,
    )
    assert added == 1  # one article, two instruments
    day = t + timedelta(days=1)
    assert recent_headlines(corpus, "XNSE:INFY", day)[0].endswith("Sector news")
    # Before publication: invisible (no lookahead).
    assert recent_headlines(corpus, "XNSE:INFY", t - timedelta(hours=1)) == []
    # Older than window: excluded.
    assert recent_headlines(corpus, "XNSE:INFY", t + timedelta(days=30), days=7) == []
    corpus.close()


def test_event_tags():
    assert tag_events(["Q3 results beat", "Board approves dividend"]) == {
        "earnings", "corporate_action"}
    assert tag_events(["Stock moves sideways"]) == set()


def test_evidence_roundtrip_and_unextracted(tmp_path):
    t = datetime(2024, 1, 15, 10, tzinfo=UTC)
    corpus = CorpusStore(tmp_path / "c.duckdb")
    ingest_news([_obs("XNSE:TCS", "Guidance raised", t, "u")], corpus)
    now = datetime(2024, 1, 16)
    docs = corpus.unextracted_documents(now)
    assert len(docs) == 1
    corpus.save_evidence(ExtractedEvidence(
        doc_id=docs[0]["doc_id"], instrument_ids=["XNSE:TCS"],
        available_at=datetime(2024, 1, 15, 10), event_type="guidance",
        guidance_change="raised", confidence=0.8,
    ))
    assert corpus.unextracted_documents(now) == []
    feats = evidence_summary_for(corpus, "XNSE:TCS", datetime(2024, 1, 16, tzinfo=UTC))
    assert feats["guidance_change_score"] == 1.0
    # PIT: evidence published after as_of is not visible.
    assert evidence_summary_for(corpus, "XNSE:TCS", datetime(2024, 1, 14, tzinfo=UTC)) == {}
    corpus.close()
