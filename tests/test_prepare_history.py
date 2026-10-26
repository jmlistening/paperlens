import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fitz
import prepare_history as h


class HistoryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.base = self.root/'data/dataset_history_v1'
        self.pdfs = self.base/'pdfs'
        self.pdfs.mkdir(parents=True)
        (self.root/'dataset_config').mkdir()
        self.config = {'version':'test','target_unique_papers':2,'queries':[]}
        h.save(self.root/'dataset_config/history_v1.json',self.config)
        for name in ('collect_history.py','history_transport_v1.py'):
            shutil.copy2(h.ROOT/name,self.root/name)
        self.state = {'version':'paperlens_history_collector_v1','config':self.config,
            'settings':{'code_sha256':h.sha(self.root/'collect_history.py'),
                        'transport_sha256':h.sha(self.root/'history_transport_v1.py')},
            'excluded_families_sha256':'test','papers':{}}
        # Sentinels represent original batch, frozen code, old review and application index.
        self.sentinels = {}
        for name in ('data/dataset_v1/state.json','data/dataset_v1/prepared_v1/manifest.json',
                     'data/dataset_v1/prepared_v1/screening_v1/state.json','data/rag/vector_store.npz',
                     'prepare_dataset.py','screen_dataset.py','review_notes/review.json'):
            p=self.root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'preserve-me')
            self.sentinels[p]=p.read_bytes()

    def add_pdf(self, pid='2301.00001v1', pages=2, bad=False):
        p=self.pdfs/(pid+'.pdf')
        if bad:
            p.write_bytes(b'%PDF-invalid, not a complete PDF')
        else:
            with fitz.open() as doc:
                for i in range(pages):
                    page=doc.new_page()
                    if i != 1:
                        page.insert_text((72,72),f'A study of retrieval evidence. Physical page {i+1}.')
                doc.save(p)
        row={'arxiv_id':pid,'family_id':pid.split('v')[0], 'title':'A study of retrieval evidence',
             'abstract':'Example abstract','published':'2023-01-01T00:00:00Z',
             'query_matches':['test'], 'quota_owner':'test','status':'downloaded',
             'exclude_from_test':False,'pdf':{'sha256':h.sha(p)}}
        self.state['papers'][row['family_id']]=row
        self.persist()
        return row

    def persist(self):
        h.save(self.base/'state.json',self.state)

    def run_prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return h.run(self.root)

    def manifest(self):
        return h.original.load(self.base/'prepared_v1/manifest.json')

    def test_pages_text_quality_and_original_preservation(self):
        self.add_pdf()
        summary,report=self.run_prepare()
        self.assertEqual(summary['status_counts'],{'prepared':1})
        self.assertEqual(summary['flagged_prepared_papers'],1)
        self.assertTrue(summary['collection']['partial_batch'])
        row=self.manifest()['papers'][0]
        lines=[json.loads(x) for x in (self.base/'prepared_v1'/row['pages_file']).read_text(encoding='utf-8').split('\n') if x]
        self.assertEqual([x['pdf_page'] for x in lines],[1,2])
        self.assertIn('Physical page 1',lines[0]['extracted_text'])
        self.assertEqual(lines[1]['normalized_text'],'')
        self.assertFalse(row['human_reviewed'])
        self.assertTrue(h.valid_cached(row,self.state['papers']['2301.00001'],self.base/'prepared_v1'))
        for p,raw in self.sentinels.items(): self.assertEqual(p.read_bytes(),raw)
        import zipfile
        with zipfile.ZipFile(report) as z:
            self.assertFalse(any(x.endswith(('.pdf','.jsonl','.env','.sqlite3')) for x in z.namelist()))

    def test_resume_reuses_cache_without_extractor(self):
        self.add_pdf(); self.run_prepare()
        row=self.manifest()['papers'][0]
        page=self.base/'prepared_v1'/row['pages_file']; before=page.stat().st_mtime_ns
        with patch.object(h.original,'extract',side_effect=AssertionError('must reuse')):
            summary,_=self.run_prepare()
        self.assertEqual(summary['reused_papers'],1)
        self.assertEqual(page.stat().st_mtime_ns,before)

    def test_corrupt_pages_and_record_rebuild_without_overwrite(self):
        self.add_pdf(); self.run_prepare()
        old=self.manifest()['papers'][0]; out=self.base/'prepared_v1'
        page=out/old['pages_file']; page.write_bytes(b'broken cache')
        summary,_=self.run_prepare()
        new=self.manifest()['papers'][0]
        self.assertEqual(summary['newly_extracted_papers'],1)
        self.assertNotEqual(old['pages_file'],new['pages_file'])
        self.assertEqual(page.read_bytes(),b'broken cache')
        record=out/new['record_file']; record.write_bytes(b'broken record')
        self.run_prepare()
        self.assertEqual(record.read_bytes(),b'broken record')
        self.assertNotEqual(new['record_file'],self.manifest()['papers'][0]['record_file'])

    def test_interruption_checkpoint_and_resume(self):
        self.add_pdf(); self.add_pdf('2301.00002v1')
        real=h.original.extract; calls=[]
        def interrupted(row,*args):
            calls.append(row['arxiv_id'])
            if len(calls)==2: raise KeyboardInterrupt()
            return real(row,*args)
        with patch.object(h.original,'extract',side_effect=interrupted):
            summary,_=self.run_prepare()
        self.assertEqual(summary['run_status'],'interrupted')
        self.assertEqual(summary['unprocessed_records'],1)
        summary,_=self.run_prepare()
        self.assertEqual(summary['status_counts'],{'prepared':2})
        self.assertEqual(summary['reused_papers'],1)

    def test_source_pdf_changed_stops_without_touching_outputs(self):
        self.add_pdf(); self.run_prepare()
        p=self.base/'prepared_v1/manifest.json'; before=p.read_bytes()
        with (self.pdfs/'2301.00001v1.pdf').open('ab') as f: f.write(b'changed')
        with self.assertRaises(h.SourceChanged): self.run_prepare()
        self.assertEqual(p.read_bytes(),before)

    def test_source_metadata_and_config_changes_stop(self):
        self.add_pdf(); self.run_prepare()
        original=copy.deepcopy(self.state)
        self.state['papers']['2301.00001']['title']='Changed title'; self.persist()
        with self.assertRaises(h.SourceChanged): self.run_prepare()
        self.state=original; self.state['config']['target_unique_papers']=3; self.persist()
        h.save(self.root/'dataset_config/history_v1.json',self.state['config'])
        with self.assertRaises(h.SourceChanged): self.run_prepare()

    def test_download_status_downgrade_cannot_hide_changed_pdf(self):
        self.add_pdf(); self.run_prepare()
        with (self.pdfs/'2301.00001v1.pdf').open('ab') as f: f.write(b'changed')
        self.state['papers']['2301.00001']['status']='failed'; self.persist()
        with self.assertRaises(h.SourceChanged): self.run_prepare()

    def test_bad_pdf_download_failure_and_long_document_are_distinct(self):
        self.add_pdf(bad=True)
        self.add_pdf('2301.00002v1',pages=101)
        missing=copy.deepcopy(self.state['papers']['2301.00001'])
        missing.update(arxiv_id='2301.00003v1',family_id='2301.00003',status='failed',error='PDF HTTP 404')
        self.state['papers']['2301.00003']=missing; self.persist()
        summary,_=self.run_prepare()
        self.assertEqual(summary['status_counts'],{'preparation_failed':1,'prepared':1,'source_unavailable':1})
        good=next(x for x in self.manifest()['papers'] if x['status']=='prepared')
        self.assertIn('long_document_over_100_pages',good['flags'])
        self.assertEqual(good['page_count'],101)

    def test_additive_collection_and_failed_to_downloaded_resume(self):
        row=self.add_pdf(); row.update(status='failed',error='test'); row.pop('pdf'); self.persist()
        self.run_prepare()
        self.add_pdf(); self.add_pdf('2301.00002v1')
        summary,_=self.run_prepare()
        self.assertEqual(summary['status_counts'],{'prepared':2})

    def test_existing_lock_blocks_without_output(self):
        self.add_pdf()
        with h.pipeline_lock(self.root):
            with self.assertRaises(RuntimeError): self.run_prepare()
        self.assertFalse((self.base/'prepared_v1').exists())

    def test_source_changes_during_extraction_stop(self):
        self.add_pdf()
        real=h.original.extract
        def change(row,*args):
            result=real(row,*args)
            with (self.pdfs/(row['arxiv_id']+'.pdf')).open('ab') as f: f.write(b'changed')
            return result
        with patch.object(h.original,'extract',side_effect=change):
            summary,_=self.run_prepare()
        self.assertEqual(summary['run_status'],'stopped')
        self.assertIn('SourceChanged',summary['error'])


if __name__ == '__main__':
    unittest.main()
