"""Year/topic-stratified supplement. No LLM calls; original corpus stays frozen."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import urllib.parse
import zipfile
import json
import history_transport_v1 as transport

ROOT = Path(__file__).resolve().parent
VERSION = 'paperlens_history_collector_v1'


def fingerprint(value):
    return transport.sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8'))


def initialize(config, source):
    papers = source['papers']
    if not isinstance(papers, dict) or not papers:
        raise ValueError('Original corpus state has no papers')
    families = sorted(papers)
    for family, row in papers.items():
        if transport.identity(row['arxiv_id']) != family:
            raise ValueError('Original corpus family/ID mismatch')
    state = transport.initial_state(config)
    state.update(version=VERSION, excluded_families=families,
                 excluded_families_sha256=fingerprint(families), excluded_hits=0,
                 source_metadata_count=len(papers),
                 source_downloaded_count=sum(r['status'] == 'downloaded' for r in papers.values()))
    for q, spec in zip(state['queries'], config['queries']):
        q.update(date_start=spec['date_start'], date_end=spec['date_end'],
                 quota=spec['quota'], selected=0)
    return state


def gather(state, limit, http, base, save):
    excluded = set(state['excluded_families'])
    queries = state['queries']
    while len(state['papers']) < limit:
        if all(q['selected'] >= q['quota'] or (q['done'] and not q['buffer']) for q in queries):
            return
        q = queries[state['cursor'] % len(queries)]
        state['cursor'] += 1
        if q['selected'] >= q['quota'] or (q['done'] and not q['buffer']):
            continue
        if not q['buffer']:
            query = f"({q['query']}) AND submittedDate:[{q['date_start']} TO {q['date_end']}]"
            url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({
                'search_query': query, 'start': q['start'], 'max_results': 25,
                'sortBy': 'submittedDate', 'sortOrder': 'descending'})
            raw = http.get(url, 5 * 1024 * 1024)
            rows, total = transport.parse_feed(raw)
            digest = fingerprint([r['arxiv_id'] for r in rows])
            if rows and digest in q['page_hashes']:
                raise ValueError('Repeated API page: ' + q['name'])
            # Validate before committing pagination, so an unexpected response loses no rows.
            for row in rows:
                date = datetime.fromisoformat(row['published'].replace('Z', '+00:00'))
                if date.tzinfo is None:
                    raise ValueError('Missing publication timezone')
                key = date.astimezone(timezone.utc).strftime('%Y%m%d%H%M')
                if not q['date_start'] <= key <= q['date_end']:
                    raise ValueError('API returned publication outside frozen date window')
            feeds = base / 'feeds'
            feeds.mkdir(parents=True, exist_ok=True)
            (feeds / f"{q['name']}_{q['start']:06d}.xml").write_bytes(raw)
            q['page_hashes'].append(digest)
            q.update(total_reported=total, start=q['start'] + len(rows), buffer=rows)
            q['done'] = not rows or q['start'] >= total
            save()
            print(f"Metadata {q['name']}: fetched {q['start']}/{total}; new unique {len(state['papers'])}", flush=True)
        if q['buffer']:
            row = q['buffer'].pop(0)
            family = row['family_id']
            if family in excluded:
                state['excluded_hits'] += 1
            elif family in state['papers']:
                state['duplicate_hits'] += 1
                old = state['papers'][family]
                if q['name'] not in old['query_matches']:
                    old['query_matches'].append(q['name'])
            else:
                row.update(query_matches=[q['name']], quota_owner=q['name'], status='pending',
                           split='unassigned', human_reviewed=False,
                           exclude_from_test=family in state['config']['development_families'])
                state['papers'][family] = row
                q['selected'] += 1
            save()


def make_summary(state, status, error, limit, max_pdf_mb):
    papers = list(state['papers'].values())
    counts = dict(Counter(r['status'] for r in papers))
    return dict(version=VERSION, created_at=transport.now(), run_status=status, error=error,
                cumulative_new_metadata_checkpoint=limit,
                eventual_new_metadata_target=state['config']['target_unique_papers'],
                new_unique_metadata=len(papers), new_status_counts=counts,
                source_metadata_count=state['source_metadata_count'],
                source_downloaded_count_at_start=state['source_downloaded_count'],
                combined_unique_metadata=state['source_metadata_count'] + len(papers),
                combined_downloaded_at_source_snapshot=state['source_downloaded_count'] + counts.get('downloaded', 0),
                max_pdf_mib=max_pdf_mb, excluded_existing_hits=state['excluded_hits'],
                duplicate_new_query_hits=state['duplicate_hits'],
                year_counts=dict(Counter(r['published'][:4] for r in papers)),
                quota_status=[dict(name=q['name'], target=q['quota'], selected=q['selected'],
                                   remaining=q['quota']-q['selected'],
                                   exhausted=q['done'] and not q['buffer']) for q in state['queries']],
                parse_review_required=sum(r.get('pdf', {}).get('parse_review_required', False) for r in papers),
                human_verified_papers=0, candidate_qa=0, verified_qa=0, frozen_test_qa=0,
                llm_requests=0, llm_tokens=0,
                notes=['New metadata target is not a guarantee of downloadable or relevant papers.',
                       'Quota ownership is first selected query, not a validated topic label.',
                       'Selected by year and keyword; recent-first within each cell; not representative.',
                       'Latest returned version may be newer than the initial publication year.',
                       'Missing quota is reported, never silently reassigned. Original corpus is not modified.'])


def report(state, base, status, error, limit, max_pdf_mb):
    summary = make_summary(state, status, error, limit, max_pdf_mb)
    out = base / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.zip')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in [('summary.json', summary), ('state.json', state),
                            ('papers.json', list(state['papers'].values()))]:
            z.writestr(name, json.dumps(value, ensure_ascii=False, indent=2))
        for p in sorted((base / 'feeds').glob('*.xml')):
            z.write(p, 'feeds/' + p.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('REPORT ZIP: ' + str(out.resolve()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int, default=1200, help='Cumulative new metadata count, not added on each run')
    parser.add_argument('--max-pdf-mb', type=int, default=200)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    config = transport.read(ROOT / 'dataset_config' / 'history_v1.json')
    if not 1 <= args.limit <= config['target_unique_papers'] or not 1 <= args.max_pdf_mb <= 512:
        parser.error('limit must be 1..1200 and max-pdf-mb 1..512')
    if sum(q['quota'] for q in config['queries']) != config['target_unique_papers']:
        raise ValueError('Quota sum differs from target')
    if not args.run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print('Offline preview only. Run: python collect_history.py --run --limit 1200 --max-pdf-mb 200')
        return 0
    import fitz  # Check dependency before creating state or network calls.
    from run_pipeline import pipeline_lock
    with pipeline_lock(ROOT):
        source_path = ROOT / 'data' / 'dataset_v1' / 'state.json'
        source = transport.read(source_path)
        base = ROOT / 'data' / 'dataset_history_v1'
        base.mkdir(parents=True, exist_ok=True)
        path = base / 'state.json'
        settings = dict(code_sha256=transport.sha(Path(__file__).read_bytes()),
                        transport_sha256=transport.sha(Path(transport.__file__).read_bytes()))
        if path.exists():
            state = transport.read(path)
            if state['version'] != VERSION or state['config'] != config or state['settings'] != settings:
                raise ValueError('Frozen config/code changed; restore original supplement files before resume')
            if fingerprint(sorted(source['papers'])) != state['excluded_families_sha256']:
                raise ValueError('Original corpus family set changed; review before resume')
            if args.limit < len(state['papers']):
                raise ValueError('Checkpoint limit cannot be lower than already selected count')
        else:
            state = initialize(config, source)
            state['settings'] = settings
            transport.atomic(base / 'source_inventory_snapshot.json', {
                'source_state_sha256': transport.sha(source_path.read_bytes()),
                'families': state['excluded_families'], 'created_at': transport.now()})
        save = lambda: transport.atomic(path, state)
        save()
        status, error = 'completed_checkpoint', None
        try:
            http = transport.HTTP(ROOT / 'data')
            gather(state, args.limit, http, base, save)
            transport.download(state, args.limit, http, base, save, args.retry_failed, args.max_pdf_mb)
            if len(state['papers']) < args.limit or any(r['status'] != 'downloaded' for r in state['papers'].values()):
                status = 'checkpoint_incomplete'
        except KeyboardInterrupt:
            status, error = 'interrupted', 'Rerun same command to resume'
        except Exception as exc:
            status, error = 'paused', type(exc).__name__ + ': ' + str(exc)[:600]
        finally:
            save()
            report(state, base, status, error, args.limit, args.max_pdf_mb)
        return 0 if status == 'completed_checkpoint' else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
