import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import fitz
import prepare_dataset as p


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.out = self.base / 'prepared_v1'
        (self.base / 'pdfs').mkdir()
        with fitz.open() as doc:
            doc.new_page().insert_text((72, 72), 'A careful study of retrieval and evidence')
            doc.new_page().insert_text((72, 72), 'Second physical page contains the evidence.')
            raw = doc.tobytes()
        self.pdf = self.base / 'pdfs' / '2401.00001v1.pdf'
        self.pdf.write_bytes(raw)
        self.row = {'arxiv_id': '2401.00001v1', 'family_id': '2401.00001',
                    'title': 'A careful study of retrieval and evidence', 'abstract': 'Example',
                    'published': '2024-01-01', 'exclude_from_test': True,
                    'status': 'downloaded', 'pdf': {'sha256': p.digest(raw)}}

    def test_pages_and_resume_and_corrupted_output_rebuild(self):
        record, reused = p.extract(self.row, self.base, self.out)
        path = self.out / record['pages_file']
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([r['pdf_page'] for r in rows], [1, 2])
        self.assertIn('Second physical', rows[1]['extracted_text'])
        self.assertFalse(record['evidence_verified'])
        self.assertFalse(reused)
        self.assertTrue(p.extract(self.row, self.base, self.out)[1])
        path.write_text('broken', encoding='utf-8')
        self.assertFalse(p.extract(self.row, self.base, self.out)[1])

    def test_changed_pdf_is_blocked(self):
        self.pdf.write_bytes(self.pdf.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            p.extract(self.row, self.base, self.out)

    def test_duplicate_candidates_do_not_delete_records(self):
        other = dict(self.row, arxiv_id='2401.00002v1')
        rows = [self.row, other]
        candidates = p.duplicate_candidates(rows)
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(c['reason'] == 'same_normalized_title' for c in candidates))
        self.assertTrue(all(c['decision'] == 'pending_review' for c in candidates))

    def test_unavailable_source_preserved_and_report_has_no_fulltext(self):
        missing = dict(self.row, arxiv_id='2401.00002v1', family_id='2401.00002',
                       status='failed', error='PDF HTTP 404', exclude_from_test=False)
        state = {'papers': {self.row['family_id']: self.row, missing['family_id']: missing}, 'config': {}}
        (self.base / 'state.json').write_bytes(p.json_bytes(state))
        summary, report = p.prepare(self.base, self.out)
        self.assertEqual(summary['status_counts'], {'prepared': 1, 'source_unavailable': 1})
        self.assertEqual(summary['candidate_qa'], 0)
        self.assertEqual(summary['development_excluded_papers'], 1)
        with zipfile.ZipFile(report) as z:
            self.assertFalse(any(n.endswith(('.pdf', '.jsonl')) for n in z.namelist()))
        self.assertTrue(self.pdf.exists())


if __name__ == '__main__':
    unittest.main()
