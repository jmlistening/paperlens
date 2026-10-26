import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile
import fitz
import collect_dataset as c


def feed(ids, total=None):
    entries = ''.join(f'<entry><id>http://arxiv.org/abs/{i}</id><title>Title {i}</title><summary>Abstract</summary></entry>' for i in ids)
    return (f'<feed xmlns="{c.NS["a"]}" xmlns:o="{c.NS["o"]}"><o:totalResults>{len(ids) if total is None else total}</o:totalResults>{entries}</feed>').encode()


class FakeHTTP:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def get(self, url, maximum):
        self.calls.append(url)
        result = next(self.values)
        if isinstance(result, Exception):
            raise result
        return result


class CollectorTests(unittest.TestCase):
    def test_larger_pdf_limit_and_explicit_failed_retry(self):
        self.prepare_paper()
        row = self.state['papers']['2401.00001']
        row.update(status='failed', error='Response exceeds configured size limit')
        with fitz.open() as doc:
            doc.new_page().insert_text((72, 72), 'Retry test')
            raw = doc.tobytes()
        class SizeCheckingHTTP:
            def get(inner, url, maximum):
                self.assertEqual(maximum, 200 * 1024 * 1024)
                return raw
        c.download(self.state, 1, SizeCheckingHTTP(), self.base, lambda: None,
                   retry_failed=True, max_pdf_mb=200)
        self.assertEqual(row['status'], 'downloaded')
        self.assertNotIn('error', row)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.config = {"queries": [{"name": "rag", "query": "ti:rag"}],
                       "date_start": "202001010000", "date_end": "202610202359",
                       "target_unique_papers": 1000, "development_families": ["2401.00001"]}
        self.state = c.initial_state(self.config)

    def test_page_buffer_resume_and_version_dedup(self):
        http = FakeHTTP([feed(["2401.00001v1", "2401.00001v2", "2401.00002v1", "2401.00003v1"])])
        c.gather(self.state, 1, http, self.base, lambda: None)
        recovered = json.loads(json.dumps(self.state))
        c.gather(recovered, 3, http, self.base, lambda: None)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(len(recovered["papers"]), 3)
        self.assertEqual(recovered["duplicate_hits"], 1)
        self.assertTrue(recovered["papers"]["2401.00001"]["exclude_from_test"])
        self.assertEqual(recovered["papers"]["2401.00001"]["arxiv_id"], "2401.00001v1")

    def test_api_error_and_html_are_not_empty_success(self):
        for data in [b'<html>Error</html>', feed(['errors#bad_query'])]:
            with self.assertRaises(ValueError):
                c.parse_feed(data)

    def test_repeated_page_stops(self):
        data = feed(["2401.00001v1"], total=99)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            c.gather(self.state, 2, FakeHTTP([data, data]), self.base, lambda: None)

    def test_exhausted_queries_report_shortfall(self):
        c.gather(self.state, 100, FakeHTTP([feed([])]), self.base, lambda: None)
        self.assertEqual(self.state["papers"], {})
        self.assertTrue(self.state["queries"][0]["done"])

    def prepare_paper(self):
        c.gather(self.state, 1, FakeHTTP([feed(["2401.00001v1"])]), self.base, lambda: None)

    def test_pdf_resume_without_network_and_hash_tampering(self):
        self.prepare_paper()
        with fitz.open() as doc:
            doc.new_page().insert_text((72, 72), "Example scientific document")
            raw = doc.tobytes()
        c.download(self.state, 1, FakeHTTP([raw]), self.base, lambda: None)
        c.download(self.state, 1, FakeHTTP([]), self.base, lambda: None)
        row = self.state['papers']['2401.00001']
        self.assertEqual(row['status'], 'downloaded')
        self.assertTrue(row['pdf']['parse_review_required'])
        (self.base / 'pdfs' / '2401.00001v1.pdf').write_bytes(raw + b'changed')
        c.download(self.state, 1, FakeHTTP([]), self.base, lambda: None)
        self.assertEqual(row['status'], 'failed')
        self.assertIn('hash changed', row['error'])

    def test_html_pdf_not_written_and_retry_explicit(self):
        self.prepare_paper()
        c.download(self.state, 1, FakeHTTP([b'<html>rate limit</html>']), self.base, lambda: None)
        self.assertEqual(self.state['papers']['2401.00001']['status'], 'failed')
        self.assertFalse(list((self.base / 'pdfs').glob('*.pdf')))
        c.download(self.state, 1, FakeHTTP([]), self.base, lambda: None)

    def test_cooldown_persisted_and_blocks_next_request(self):
        http = c.HTTP(self.base)
        error = urllib.error.HTTPError('https://export.arxiv.org/api/query', 429, 'Limited', {'Retry-After': '1800'}, None)
        with patch.object(http.opener, 'open', side_effect=error) as opener:
            with self.assertRaises(c.Paused):
                http.get('https://export.arxiv.org/api/query', 1024)
            with self.assertRaises(c.Paused):
                http.get('https://export.arxiv.org/api/query', 1024)
            self.assertEqual(opener.call_count, 1)
        self.assertTrue(http.cooldown.exists())

    def test_partial_transfer_is_not_success(self):
        class Response(io.BytesIO):
            headers = {'Content-Length': '100'}
            def geturl(self):
                return 'https://export.arxiv.org/pdf/2401.00001v1'
        http = c.HTTP(self.base)
        with patch.object(http.opener, 'open', return_value=Response(b'%PDF-short')):
            with self.assertRaisesRegex(OSError, 'Incomplete'):
                http.get('https://export.arxiv.org/pdf/2401.00001v1', 1024)

    def test_report_contains_no_pdf_and_does_not_invent_qa(self):
        out = c.report(self.state, self.base, 'paused', 'test', 100)
        with zipfile.ZipFile(out) as z:
            self.assertFalse(any(name.endswith('.pdf') for name in z.namelist()))
            summary = json.loads(z.read('summary.json'))
            self.assertEqual(summary['verified_qa'], 0)
            self.assertEqual(summary['unique_metadata'], 0)

    def test_redirect_allowlist(self):
        for url in ['http://arxiv.org/pdf/x', 'https://evil.example/file', 'https://arxiv.org:9000/file']:
            with self.assertRaises(ValueError):
                c.checked_url(url)


if __name__ == '__main__':
    unittest.main()
