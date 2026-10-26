import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_history as h


def row(pid, date='2023-05-01T00:00:00Z'):
    return dict(arxiv_id=pid, family_id=pid.split('v')[0], title='Test', abstract='Evidence',
                published=date, status='downloaded')


def feed(rows, total=None):
    entries = ''.join('<entry><id>https://arxiv.org/abs/' + r['arxiv_id'] + '</id><title>' +
                      escape(r['title']) + '</title><summary>' + escape(r['abstract']) +
                      '</summary><published>' + r['published'] + '</published></entry>' for r in rows)
    return ('<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/">'
            f'<o:totalResults>{len(rows) if total is None else total}</o:totalResults>{entries}</feed>').encode()


class FakeHTTP:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def get(self, url, max_bytes):
        self.calls.append(url)
        if not self.pages:
            raise AssertionError('Unexpected network request')
        return self.pages.pop(0)


def config(quotas=(2,)):
    return dict(target_unique_papers=sum(quotas), development_families=['2305.00002'],
                queries=[dict(name=f'q{i}', query='ti:retrieval', date_start='202301010000',
                              date_end='202312312359', quota=n) for i, n in enumerate(quotas)])


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = {'papers': {'2305.00001': row('2305.00001v1')}}

    def gather(self, state, limit, http):
        with contextlib.redirect_stdout(io.StringIO()):
            h.gather(state, limit, http, self.base, lambda: None)

    def test_exclude_all_versions_and_resume_keeps_buffer(self):
        s = h.initialize(config(), self.source)
        http = FakeHTTP([feed([row('2305.00001v2'), row('2305.00002v1'), row('2305.00003v1')])])
        self.gather(s, 1, http)
        self.assertEqual(s['excluded_hits'], 1)
        self.assertEqual(len(s['queries'][0]['buffer']), 1)
        # A JSON round trip approximates restart after persisted checkpoint.
        s = json.loads(json.dumps(s))
        self.gather(s, 2, http)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(len(s['papers']), 2)
        self.assertTrue(s['papers']['2305.00002']['exclude_from_test'])
        self.assertEqual(s['papers']['2305.00003']['split'], 'unassigned')

    def test_overlapping_queries_do_not_double_count_quota(self):
        s = h.initialize(config((1, 1)), self.source)
        http = FakeHTTP([feed([row('2305.00002v1')]), feed([row('2305.00002v2'), row('2305.00003v1')])])
        self.gather(s, 2, http)
        self.assertEqual([q['selected'] for q in s['queries']], [1, 1])
        self.assertEqual(s['duplicate_hits'], 1)
        self.assertEqual(s['papers']['2305.00002']['query_matches'], ['q0', 'q1'])

    def test_shortfall_does_not_fill_from_other_cell(self):
        s = h.initialize(config((1, 2)), self.source)
        http = FakeHTTP([feed([row('2305.00002v1'), row('2305.00003v1')]), feed([])])
        self.gather(s, 3, http)
        self.assertEqual(len(s['papers']), 1)
        self.assertEqual(len(s['queries'][0]['buffer']), 1)
        summary = h.make_summary(s, 'checkpoint_incomplete', None, 3, 200)
        self.assertEqual(summary['quota_status'][1]['remaining'], 2)
        self.assertTrue(summary['quota_status'][1]['exhausted'])

    def test_date_mismatch_does_not_advance_pagination(self):
        s = h.initialize(config(), self.source)
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            self.gather(s, 2, FakeHTTP([feed([row('2305.00002v1', '2024-01-01T00:00:00Z')])]))
        self.assertEqual(s['queries'][0]['start'], 0)
        self.assertEqual(s['papers'], {})

    def test_repeated_page_stops(self):
        s = h.initialize(config((5,)), self.source)
        page = feed([row('2305.00002v1')], total=100)
        with self.assertRaisesRegex(ValueError, 'Repeated API page'):
            self.gather(s, 5, FakeHTTP([page, page]))
        self.assertEqual(len(s['papers']), 1)

    def test_main_resume_and_original_preservation(self):
        root = self.base
        (root / 'dataset_config').mkdir()
        (root / 'dataset_config/history_v1.json').write_text(json.dumps(config((1,))))
        source_path = root / 'data/dataset_v1/state.json'
        source_path.parent.mkdir(parents=True)
        source_path.write_text(json.dumps(self.source))
        original = source_path.read_bytes()
        fake = FakeHTTP([feed([row('2305.00002v1')])])
        module = types.SimpleNamespace(pipeline_lock=lambda root: contextlib.nullcontext())

        def download(state, *args):
            for r in state['papers'].values():
                r['status'] = 'downloaded'

        with patch.object(h, 'ROOT', root), patch.dict(sys.modules, {'run_pipeline': module, 'fitz': types.ModuleType('fitz')}), \
             patch.object(h.transport, 'HTTP', return_value=fake), patch.object(h.transport, 'download', side_effect=download), \
             patch.object(sys, 'argv', ['collect_history.py', '--run', '--limit', '1']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h.main(), 0)
            self.assertEqual(h.main(), 0)
        self.assertEqual(source_path.read_bytes(), original)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(list((root / 'data/dataset_history_v1/reports').glob('*.zip'))), 2)

    def test_production_quotas(self):
        cfg = h.transport.read(Path(h.__file__).parent / 'dataset_config/history_v1.json')
        self.assertEqual(sum(q['quota'] for q in cfg['queries']), 1200)
        self.assertEqual(len({q['name'] for q in cfg['queries']}), len(cfg['queries']))
        self.assertEqual({q['date_start'][:4] for q in cfg['queries']}, set(map(str, range(2020, 2026))))


if __name__ == '__main__':
    unittest.main()
