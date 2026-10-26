import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
import zipfile
import screen_dataset as s


def response(value):
    content = json.dumps(value)
    data = {'model': 'mock', 'usage': {'total_tokens': 123}, 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
    return N(model_dump=lambda: data, choices=[N(finish_reason='stop', message=N(content=content))])


class Client:
    def __init__(self, values):
        self.values, self.calls = iter(values), 0
        self.chat = N(completions=self)
    def create(self, **kwargs):
        self.calls += 1
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return response(value)


class ScreenTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.row = {'arxiv_id': '2401.00001v1', 'status': 'prepared', 'title': 'Retrieval methods for scientific evidence',
                    'abstract': 'We study retrieval augmented generation.', 'exclude_from_test': True, 'flags': ['some_low_text_pages']}
        self.valid = {'relevance': 'core', 'topics': ['rag'], 'reason': '研究检索增强生成',
                      'evidence': {'field': 'abstract', 'quote': self.row['abstract']}}
        self.state = {'settings': {'model': 'mock'}, 'results': {}}
        self.save = lambda: s.atomic(self.out / 'state.json', s.json_bytes(self.state))

    def test_quote_validation_rejects_invented_evidence(self):
        value = dict(self.valid, evidence={'field': 'abstract', 'quote': 'An entirely fabricated quote.'})
        with self.assertRaisesRegex(ValueError, 'exact source'):
            s.validate(value, s.input_for(self.row))

    def test_success_persists_raw_usage_and_resumes_without_rebilling(self):
        client = Client([self.valid])
        s.run_batch([self.row], 100, self.state, self.out, client, self.save)
        s.run_batch([self.row], 100, self.state, self.out, client, self.save)
        self.assertEqual(client.calls, 1)
        record = self.state['results'][self.row['arxiv_id']]
        self.assertEqual(record['status'], 'screened_pending_review')
        self.assertTrue(record['exclude_from_test'])
        self.assertFalse(record['human_reviewed'])
        self.assertEqual(s.usage_total(self.state['results'])['known_total_tokens'], 123)
        self.assertTrue((self.out / 'raw' / (self.row['arxiv_id'] + '.json')).exists())

    def test_invalid_response_is_review_item_with_usage_preserved(self):
        s.run_batch([self.row], 1, self.state, self.out, Client([{}]), self.save)
        self.assertEqual(self.state['results'][self.row['arxiv_id']]['status'], 'invalid_response_pending_review')
        self.assertEqual(s.usage_total(self.state['results'])['known_total_tokens'], 123)

    def test_unknown_request_stops_and_is_not_automatically_retried(self):
        client = Client([TimeoutError()])
        with self.assertRaises(TimeoutError):
            s.run_batch([self.row], 1, self.state, self.out, client, self.save)
        s.run_batch([self.row], 1, self.state, self.out, client, self.save)
        self.assertEqual(client.calls, 1)
        self.assertEqual(s.usage_total(self.state['results'])['usage_unknown_requests'], 1)

    def test_report_excludes_secrets_and_fulltext(self):
        (self.out / '.env').write_text('SECRET', encoding='utf-8')
        s.run_batch([self.row], 1, self.state, self.out, Client([self.valid]), self.save)
        path = s.make_report(self.state, [self.row], self.out, 1, 'checkpoint_finished')
        with zipfile.ZipFile(path) as z:
            self.assertFalse(any(n.endswith(('.env', '.pdf', '.jsonl')) for n in z.namelist()))
            summary = json.loads(z.read('summary.json'))
            self.assertEqual(summary['human_verified_papers'], 0)
            self.assertEqual(summary['candidate_qa'], 0)

    def test_stable_selection_is_cumulative(self):
        rows = [dict(self.row, arxiv_id=f'2401.{i:05d}v1') for i in range(50)]
        first = s.ordered_rows({'papers': rows})
        second = s.ordered_rows({'papers': list(reversed(rows))})
        self.assertEqual(first, second)
        self.assertEqual(s.pending_rows(first, 10, {first[0]['arxiv_id']: {}}), first[1:10])


if __name__ == '__main__':
    unittest.main()
