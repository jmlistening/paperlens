"""PaperLens 统一入口：默认预览；--send 才实际发送邮件。

请与 fetch_papers.py、rank_papers.py、download_selected.py、
summarize_selected.py、新版 send_digest.py 放在同一目录。
使用已有 .env、当前兴趣 profile 和 data/delivery.sqlite3。
本程序不会设置定时任务，也不会执行全文翻译。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ('fetch_papers', 'rank_papers', 'download_selected', 'summarize_selected', 'send_digest')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        self.terminal.write(text)
        self.terminal.flush()
        return len(text)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


@contextmanager
def pipeline_lock(root=ROOT):
    """OS file lock releases when the process exits, including a crash."""
    path = Path(root) / 'data' / 'pipeline.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('另一个统一流程正在运行。请等它结束，不要删除锁文件。') from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_modules():
    modules = {}
    for name in STAGES:
        path = ROOT / (name + '.py')
        if not path.is_file():
            raise ValueError(f'缺少 {name}.py，请确认文件没有带 (1) 或 .txt 后缀。')
        spec = importlib.util.spec_from_file_location('_paperlens_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, 'main', None)):
            raise ValueError(f'{name}.py 缺少 main()。')
        modules[name] = module
    sender = modules['send_digest']
    required = ('open_db', 'eligible_papers', 'checked_address', 'paper_key', 'load_papers')
    if any(not callable(getattr(sender, name, None)) for name in required):
        raise ValueError('send_digest.py 不是支持 SQLite 去重的新版，请先替换。')
    return modules


def validate_candidates(papers, sender):
    if not isinstance(papers, list):
        raise ValueError('抓取结果必须是论文列表。')
    keys = set()
    checked = []
    for p in papers:
        for name in ('arxiv_id', 'title', 'abstract', 'url', 'pdf_url'):
            if not isinstance(p.get(name), str) or not p[name].strip():
                raise ValueError(f'抓取结果缺少有效字段：{name}')
        key = sender.paper_key(p['arxiv_id'])
        if key in keys:
            continue
        keys.add(key)
        checked.append(p)
    return checked


def validate_recommendations(path, papers):
    data = read_json(path)
    ranked = data['ranked_papers']
    selected_ids = data['selected_ids']
    original = {p['arxiv_id']: p for p in papers}
    ids = [p['arxiv_id'] for p in ranked]
    if len(ids) != len(set(ids)) or set(ids) != set(original):
        raise ValueError('本轮评分结果与候选列表不对应。')
    if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= set(ids):
        raise ValueError('本轮入选论文编号不正确。')
    for p in ranked:
        if any(p.get(k) != v for k, v in original[p['arxiv_id']].items()):
            raise ValueError('评分过程改变了原论文元数据。')
        a = p['assessment']
        if type(a['score']) is not int or not 0 <= a['score'] <= 100:
            raise ValueError('评分不是 0—100 的整数。')
    lookup = {p['arxiv_id']: p for p in ranked}
    return [lookup[key] for key in selected_ids]


def validate_pdfs(papers, downloader):
    import re
    for p in papers:
        name = re.sub(r'[^A-Za-z0-9._-]', '_', p['arxiv_id']) + '.pdf'
        path = ROOT / 'data' / 'pdfs' / name
        if not downloader.is_pdf(path):
            raise ValueError(f'{name} 未下载成功。本轮停止，不生成速览、不发送邮件。')


def empty_outputs(reason, run_id):
    data = ROOT / 'data'
    # Leave valid, mutually consistent empty results so standalone send_digest
    # cannot accidentally send a previous run after a normal empty result.
    recommendation = data / 'recommendations.json'
    write_json(recommendation, {
        'created_at': utc_now(), 'run_id': run_id, 'candidate_count': 0,
        'selected_ids': [], 'ranked_papers': [], 'scope': reason,
    })
    write_json(data / 'selected_summaries.json', {
        'recommendations_sha256': hashlib.sha256(recommendation.read_bytes()).hexdigest(),
        'papers': [], 'scope': reason,
    })
    (data / 'digest_preview.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>PaperLens</title>'
        '<h1>本轮没有可发送的论文</h1><p>' + reason + '</p>', encoding='utf-8')


def execute(modules, run_dir, state, send):
    sender = modules['send_digest']
    config = sender.dotenv_values(ROOT / '.env', encoding='utf-8-sig')
    recipient = sender.checked_address(config.get('MAIL_TO'))
    for key in ('DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'DEEPSEEK_MODEL'):
        if not (config.get(key) or '').strip():
            raise ValueError(f'.env 缺少 {key}。')
    if send:
        for key in ('SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD'):
            if not (config.get(key) or '').strip():
                raise ValueError(f'.env 缺少 {key}。')
        sender.checked_address(config.get('SMTP_USER'))
        if config['SMTP_PORT'].strip() != '465':
            raise ValueError('SMTP_PORT 需为 465。')
    # Lazy import: profile_config depends on other stage modules.
    from profile_config import load_active_profile

    profile = load_active_profile(ROOT)
    if type(profile.get('top_k', 3)) is not int or not 1 <= profile.get('top_k', 3) <= 20:
        raise ValueError('兴趣配置的 top_k 必须是 1—20 的整数。')
    if type(profile.get('min_score', 60)) is not int or not 0 <= profile.get('min_score', 60) <= 100:
        raise ValueError('兴趣配置的 min_score 必须是 0—100 的整数。')

    data = ROOT / 'data'
    previous = run_dir / 'previous'
    previous.mkdir()
    for name in ('papers.json', 'recommendations.json', 'selected_summaries.json', 'digest_preview.html'):
        path = data / name
        if path.exists():
            path.replace(previous / name)
    (data / 'digest_preview.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>PaperLens</title>'
        '<h1>本轮流程尚未完成</h1><p>请查看运行日志，勿使用上一轮邮件预览。</p>', encoding='utf-8')

    def stage(name, label, action):
        state['stage'] = name
        write_json(run_dir / 'run.json', state)
        print(f'\n=== {label} ===', flush=True)
        result = action()
        state['completed_stages'].append(name)
        write_json(run_dir / 'run.json', state)
        return result

    def fetch():
        modules['fetch_papers'].main()
        path = data / 'papers.json'
        # Existing fetch script returns without creating a file on zero results.
        # Previous file was moved away; failure exceptions still propagate.
        result = validate_candidates(read_json(path) if path.exists() else [], sender)
        write_json(run_dir / 'fetched_papers.json', result)
        return result

    papers = stage('fetch', '1/5 获取 arXiv 候选论文', fetch)
    state['fetched_count'] = len(papers)
    db = sender.open_db()
    try:
        fresh = sender.eligible_papers(db, recipient, papers)
    finally:
        db.close()
    state['eligible_count'] = len(fresh)
    print(f'获取 {len(papers)} 篇；去重后剩余 {len(fresh)} 篇。')
    write_json(data / 'papers.json', fresh)
    if not fresh:
        reason = '没有查询结果，或候选论文已经推送／处于待核实状态。'
        empty_outputs(reason, state['run_id'])
        print(reason + ' 本轮不调用评分和摘要模型，不发送邮件。')
        return 'no_new_papers'

    def rank():
        modules['rank_papers'].main()
        return validate_recommendations(data / 'recommendations.json', fresh)

    selected = stage('rank', '2/5 根据兴趣筛选', rank)
    state['selected_count'] = len(selected)
    if not selected:
        # Preserve rankings; create an empty summary matching this recommendation.
        write_json(data / 'selected_summaries.json', {
            'recommendations_sha256': hashlib.sha256((data / 'recommendations.json').read_bytes()).hexdigest(),
            'papers': [], 'scope': '本轮没有论文达到兴趣阈值',
        })
        sender.main([])
        print('本轮没有论文达到阈值，无需下载、生成速览或发送邮件。')
        return 'no_matches'

    def download():
        modules['download_selected'].main()
        validate_pdfs(selected, modules['download_selected'])

    stage('download', '3/5 下载入选论文原文 PDF', download)

    def summarize():
        modules['summarize_selected'].main()
        result = sender.load_papers()  # Verifies hash and selected ID order.
        if any(not isinstance(p.get('summary'), str) or not p['summary'].strip() for p in result):
            raise ValueError('中文速览有空内容，本轮停止。')

    stage('summarize', '4/5 生成中文速览（优先复用缓存）', summarize)
    stage('digest', '5/5 去重并' + ('发送邮件' if send else '生成邮件预览'),
          lambda: sender.main(['--send'] if send else []))
    return 'send_step_completed' if send else 'preview_ready'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--send', action='store_true', help='执行完整流程并发送新论文邮件')
    args = parser.parse_args(argv)
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    with pipeline_lock():
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        run_dir = ROOT / 'data' / 'runs' / run_id
        run_dir.mkdir(parents=True)
        state = {'run_id': run_id, 'started_at': utc_now(), 'mode': 'send' if args.send else 'preview',
                 'status': 'running', 'stage': 'preflight', 'completed_stages': []}
        write_json(run_dir / 'run.json', state)
        old_cwd = Path.cwd()
        with (run_dir / 'pipeline.log').open('w', encoding='utf-8') as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    os.chdir(ROOT)
                    print('PaperLens：' + ('运行并发送' if args.send else '运行并预览，不发送邮件'))
                    print('有新候选时，评分与中文速览可能消耗 API 额度。')
                    print(f'本轮日志：{run_dir / "pipeline.log"}')
                    modules = load_modules()
                    state['status'] = execute(modules, run_dir, state, args.send)
                    print('\n本轮流程已结束。')
                    if not args.send and state['status'] == 'preview_ready':
                        print(f'打开预览：{ROOT / "data" / "digest_preview.html"}')
                        print('确认当前预览后发送：python send_digest.py --send')
                        print('不要同时单独运行其他阶段脚本。')
                    return 0
                except KeyboardInterrupt:
                    state['status'] = 'interrupted'
                    print('\n已中断。若已开始发送，请先查看 python send_digest.py --history。')
                    return 130
                except Exception as exc:
                    state['status'] = 'failed'
                    # Record type/code only, avoiding SDK response bodies and credentials.
                    state['error_type'] = type(exc).__name__
                    code = getattr(exc, 'status_code', None)
                    if code is not None:
                        state['http_status'] = code
                    print(f'\n流程停止于 {state["stage"]}：{type(exc).__name__}'
                          + (f'，HTTP {code}' if code is not None else ''))
                    if isinstance(exc, (ValueError, FileNotFoundError, ModuleNotFoundError, RuntimeError)):
                        print(str(exc))
                    print('后续步骤未继续执行。若失败发生在发送阶段，先查 --history 和邮箱。')
                    return 1
                finally:
                    os.chdir(old_cwd)
                    state['finished_at'] = utc_now()
                    write_json(run_dir / 'run.json', state)
                    print(f'运行记录：{run_dir / "run.json"}')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f'无法启动流程：{exc}')
        raise SystemExit(1)
