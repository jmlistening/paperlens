"""Offline, resumable history-only fulltext preparation. Never changes original corpus.

Uses prepare_dataset.extract unchanged, but publishes immutable per-paper generations
and per-run reports. Source/processing changes fail closed; additive collection is OK.
All real work holds the same pipeline_lock as the collector. No network or models.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid
import zipfile

import prepare_dataset as original
from run_pipeline import pipeline_lock

ROOT = Path(__file__).resolve().parent
VERSION = 'history_prepare_v1'


class SourceChanged(ValueError):
    pass


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def save(path, value):
    original.atomic(path, original.json_bytes(value))


def paths(root):
    root = Path(root).resolve()
    base = root/'data/dataset_history_v1'
    out = base/'prepared_v1'
    for path in (root/'data', base, base/'pdfs', out):
        if path.resolve() != path:
            raise ValueError('History path redirects outside its dedicated location: ' + str(path))
    return base, out


def safe_child(base, relative):
    path = (base/relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Artifact path escapes dedicated output')
    return path


def load_source(root):
    base, out = paths(root)
    raw = (base/'state.json').read_bytes()
    source = json.loads(raw.decode('utf-8-sig'))
    config = original.load(Path(root)/'dataset_config/history_v1.json')
    if source.get('version') != 'paperlens_history_collector_v1' or source.get('config') != config:
        raise SourceChanged('Collector version/config does not match frozen history configuration')
    import fitz
    contract = {'processing_version': VERSION, 'prepare_history_sha256': sha(Path(__file__)),
        'extractor_sha256':sha(Path(original.__file__)), 'pymupdf':fitz.VersionBind,
        'source_config_sha256':fingerprint(config), 'source_version':source['version'],
        'source_settings':source['settings'], 'excluded_families_sha256':source['excluded_families_sha256']}
    for name, key in [('collect_history.py','code_sha256'), ('history_transport_v1.py','transport_sha256')]:
        if sha(Path(root)/name) != source['settings'][key]:
            raise SourceChanged('Collector code differs from its frozen state: ' + name)
    rows = source['papers']
    if not isinstance(rows, dict):
        raise ValueError('Source papers must be a mapping')
    seen = set()
    for family, row in rows.items():
        pid = row.get('arxiv_id','')
        if not re.fullmatch(r'\d{4}\.\d{4,5}v[1-9]\d*',pid) or re.sub(r'v\d+$','',pid) != family:
            raise ValueError('Invalid paper identity')
        if pid in seen:
            raise ValueError('Duplicate source ID')
        seen.add(pid)
        if row.get('status') not in ('pending','failed','downloaded'):
            raise ValueError('Unknown download status: ' + pid)
    return base, out, raw, source, contract


def ledger_row(row):
    return {'arxiv_id':row['arxiv_id'], 'metadata_sha256':fingerprint(original.base_record(row)),
            'pdf_sha256':row.get('pdf',{}).get('sha256')}


def preflight(base, source, checkpoint, contract):
    if checkpoint and checkpoint['contract'] != contract:
        raise SourceChanged('Frozen source config or processing code/runtime changed; use a reviewed new output version')
    rows = source['papers']
    ledger = checkpoint.get('source_ledger',{}) if checkpoint else {}
    for family, previous in ledger.items():
        if family not in rows:
            raise SourceChanged('Previously observed source paper disappeared: ' + family)
        current = ledger_row(rows[family])
        for key in ('arxiv_id','metadata_sha256'):
            if current[key] != previous[key]:
                raise SourceChanged('Previously observed source metadata changed: ' + family)
        if previous['pdf_sha256'] and current['pdf_sha256'] != previous['pdf_sha256']:
            raise SourceChanged('Previously observed source PDF hash changed: ' + family)
        if previous['pdf_sha256']:
            p = safe_child(base/'pdfs',previous['arxiv_id']+'.pdf')
            if not p.is_file() or sha(p) != previous['pdf_sha256']:
                raise SourceChanged('Previously observed PDF bytes changed or disappeared: ' + family)
    for row in rows.values():
        if row['status'] == 'downloaded':
            p = safe_child(base/'pdfs',row['arxiv_id']+'.pdf')
            expected = row.get('pdf',{}).get('sha256')
            if not expected or not p.is_file() or sha(p) != expected:
                raise SourceChanged('Downloaded PDF missing or differs from source hash: ' + row['arxiv_id'])


def valid_cached(record, row, out):
    """Treat damaged outputs as cache misses; retain old bytes in immutable generations."""
    if not record or record.get('status') != 'prepared':
        return False
    try:
        if record['fingerprint']['pdf_sha256'] != row['pdf']['sha256']:
            return False
        if record['fingerprint']['metadata_sha256'] != original.digest(original.json_bytes(original.base_record(row))):
            return False
        pages = safe_child(out,record['pages_file'])
        meta = safe_child(out,record['record_file'])
        if sha(meta) != record['record_sha256'] or sha(pages) != record['pages_sha256']:
            return False
        saved = original.load(meta)
        if any(saved.get(k) != record.get(k) for k in ('page_count','pages_sha256','text_chars','fingerprint')):
            return False
        lines = [json.loads(x) for x in pages.read_text(encoding='utf-8').split('\n') if x.strip()]
        if len(lines) != record['page_count']:
            return False
        return all(x['paper_id'] == row['arxiv_id'] and x['pdf_page'] == i+1 and
                   x['pdf_sha256'] == row['pdf']['sha256'] and
                   x['normalized_text'] == original.normalized(x['extracted_text']) and
                   original.digest(x['normalized_text'].encode('utf-8')) == x['text_sha256']
                   for i,x in enumerate(lines))
    except (OSError,ValueError,KeyError,TypeError):
        return False


def collection_status(base, source):
    reports = sorted((base/'reports').glob('*.zip'))
    info = {'report':None, 'report_matches_current_state':False, 'run_status':'unknown'}
    if reports:
        with zipfile.ZipFile(reports[-1]) as z:
            summary = json.loads(z.read('summary.json'))
            matching = json.loads(z.read('state.json')) == source
        info.update(report=reports[-1].relative_to(base).as_posix(),
                    report_matches_current_state=matching,
                    run_status=summary['run_status'] if matching else 'state_newer_than_report')
    info['target_reached'] = len(source['papers']) >= source['config']['target_unique_papers']
    info['partial_batch'] = not (info['report_matches_current_state'] and
                                info['run_status'] == 'completed_checkpoint' and info['target_reached'])
    return info


def run(root=ROOT):
    root = Path(root).resolve()
    # A busy collector rejects preparation here, before source/output access.
    with pipeline_lock(root):
        base, out, raw, source, contract = load_source(root)
        checkpoint_path = out/'checkpoint.json'
        checkpoint = original.load(checkpoint_path) if checkpoint_path.exists() else {}
        preflight(base, source, checkpoint, contract)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder = out/'runs'/run_id
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'source_state_snapshot.json').write_bytes(raw)
        save(folder/'source_config.json',source['config'])
        previous = checkpoint.get('results',{})
        checkpoint = {'contract':contract, 'source_ledger':{k:ledger_row(v) for k,v in source['papers'].items()},
                      'results':previous, 'source_state_sha256':original.digest(raw), 'last_run':run_id}
        save(checkpoint_path,checkpoint)
        records, attempts, events = [], [], []
        reused_count = new_count = 0
        run_status, fatal = 'completed', None
        try:
            for family, row in source['papers'].items():
                if (base/'state.json').read_bytes() != raw:
                    raise SourceChanged('Collection state changed during preparation')
                pid = row['arxiv_id']
                record = {**original.base_record(row), 'status':'source_unavailable',
                    'split':'unassigned','human_reviewed':False,'evidence_verified':False,
                    'flags':[], 'error':row.get('error') or row['status'],
                    'source_download_status':row['status'], 'processing_version':VERSION}
                if row['status'] == 'downloaded':
                    source_pdf = safe_child(base/'pdfs',pid+'.pdf')
                    if sha(source_pdf) != row['pdf']['sha256']:
                        raise SourceChanged('Source PDF changed during preparation: ' + pid)
                    cached = checkpoint['results'].get(pid)
                    if valid_cached(cached,row,out):
                        record = cached
                        reused_count += 1
                        events.append({'paper_id':pid,'action':'reused'})
                    else:
                        generation = out/'artifacts'/pid/uuid.uuid4().hex
                        generation.mkdir(parents=True,exist_ok=False)
                        try:
                            record, _ = original.extract(row,base,generation)
                            pages = generation/record['pages_file']
                            meta = generation/'records'/(pid+'.json')
                            record.update(pages_file=pages.relative_to(out).as_posix(),
                                record_file=meta.relative_to(out).as_posix(), record_sha256=sha(meta),
                                processing_version=VERSION, source_config_sha256=contract['source_config_sha256'],
                                quota_owner=row.get('quota_owner'), human_reviewed=False)
                            if not valid_cached(record,row,out):
                                raise ValueError('Fresh extraction failed page/text integrity check')
                            if sha(source_pdf) != row['pdf']['sha256']:
                                raise SourceChanged('Source PDF changed during extraction: ' + pid)
                            new_count += 1
                            events.append({'paper_id':pid,'action':'rebuilt_invalid_cache' if cached else 'extracted'})
                        except KeyboardInterrupt:
                            raise
                        except SourceChanged:
                            raise
                        except Exception as exc:
                            record = {**original.base_record(row), 'status':'preparation_failed',
                                'split':'unassigned','human_reviewed':False,'evidence_verified':False,
                                'flags':[], 'error_type':type(exc).__name__, 'processing_version':VERSION,
                                'failed_artifact_dir':generation.relative_to(out).as_posix()}
                            attempts.append(record)
                checkpoint['results'][pid] = record
                records.append(record)
                save(checkpoint_path,checkpoint)
                print(f'[{len(records)}/{len(source["papers"])}] {pid}: {record["status"]}',flush=True)
            if (base/'state.json').read_bytes() != raw:
                raise SourceChanged('Collection state changed before finalization')
            if load_source(root)[4] != contract:
                raise SourceChanged('Source config or processing code changed during preparation')
        except KeyboardInterrupt:
            run_status = 'interrupted'
        except Exception as exc:
            run_status, fatal = 'stopped', type(exc).__name__ + ': ' + str(exc)
        prepared = [r for r in records if r['status'] == 'prepared']
        status_counts = dict(Counter(r['status'] for r in records))
        if run_status == 'completed' and any(r['status'] != 'prepared' for r in records):
            run_status = 'completed_with_issues'
        if run_status == 'completed' and not records:
            run_status = 'no_available_papers'
        summary = {'version':VERSION,'created_at':original.stamp(),'run_status':run_status,'error':fatal,
            'source_metadata_count':len(source['papers']), 'source_status_counts':dict(Counter(r['status'] for r in source['papers'].values())),
            'processed_records':len(records),'unprocessed_records':len(source['papers'])-len(records),
            'status_counts':status_counts,'prepared_pages':sum(r['page_count'] for r in prepared),
            'prepared_text_chars':sum(r['text_chars'] for r in prepared),
            'flagged_prepared_papers':sum(bool(r['flags']) for r in prepared),
            'flag_counts':dict(Counter(f for r in prepared for f in r['flags'])),
            'reused_papers':reused_count,'newly_extracted_papers':new_count,
            'source_state_sha256':original.digest(raw),'processing_contract':contract,
            'collection':collection_status(base,source),'human_verified_papers':0,
            'candidate_qa':0,'verified_qa':0,'frozen_test_qa':0,'llm_requests':0,
            'limitations':['Program checks only; no human verification, OCR, table/formula reconstruction.',
                          'No cross-batch deduplication or relevance screening; no index updated.']}
        manifest = {'summary':summary,'papers':records}
        save(folder/'manifest.json',manifest)
        save(folder/'exceptions.json',[r for r in records if r['status'] != 'prepared'])
        save(folder/'events.json',events)
        save(folder/'failure_attempts.json',attempts)
        save(out/'manifest.json',manifest)
        report = out/'reports'/(run_id+'.zip')
        report.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(report,'x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('summary.json',original.json_bytes(summary))
            z.writestr('papers_review.json',original.json_bytes(records))
            z.writestr('source_config.json',original.json_bytes(source['config']))
            z.writestr('exceptions.json',original.json_bytes([r for r in records if r['status'] != 'prepared']))
            z.writestr('events.json',original.json_bytes(events))
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        print('REPORT ZIP: ' + str(report),flush=True)
        return summary, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    if not args.run:
        base, out = paths(ROOT)
        print(f'Offline preview: input={base}; output={out}; --run holds the shared pipeline lock.')
        return 0
    summary, _ = run()
    return 0 if summary['run_status'] == 'completed' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc),file=sys.stderr)
        raise SystemExit(2)
