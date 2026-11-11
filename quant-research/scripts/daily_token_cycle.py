#!/usr/bin/env python3
"""After-close refresh, incremental token training, forecasting and watchlist delivery."""
import argparse
import fcntl
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import torch
from daily_refresh import capture
from publish_daily_start_stage import publish as publish_start_stage
from publish_daily_token_view import publish
from review_daily_early_stage import review as review_early_stage
from review_daily_token import review

from quant_research.daily_input import normalized, refresh
from quant_research.daily_loop import read, verify
from quant_research.daily_token import cycle
from quant_research.publication import OperationJournal, launch_plan
from quant_research.research_review import freeze_decision, refresh_research
from quant_research.storage import file_hash, utc_now, write_json

BASE = Path(__file__).resolve().parents[1]


def select_signal(calendar, now, after, explicit=None):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('Timezone-aware time required')
    local = now.tz_convert('Asia/Shanghai')
    dates = sorted(calendar.loc[calendar.is_open, 'date'].tolist())
    if explicit:
        if explicit not in dates or pd.Timestamp(explicit+'T15:00:00', tz='Asia/Shanghai') > now:
            raise ValueError('Requested close has not completed')
        return explicit
    eligible = [d for d in dates if pd.Timestamp(d+'T'+after+':00', tz='Asia/Shanghai') <= local]
    return max(eligible) if eligible else None


def freeze_published(run_root, protocol, journal):
    with journal.stage('freeze_research_decision'):
        bundle = freeze_decision(run_root, protocol)
        write_json(Path(run_root).parent.parent/'decision-publications'/(Path(run_root).name+'.json'),
            dict(bundle=str(bundle.resolve()), manifest_sha256=file_hash(bundle/'manifest.json')))
    return bundle


def preflight(config, signal_date=None):
    cfg = read(config)
    evidence = read(cfg['calendar_evidence'])
    now = pd.Timestamp(utc_now()).tz_convert('Asia/Shanghai')
    if not evidence['valid_from'] <= now.strftime('%Y-%m-%d') <= evidence['valid_through']:
        raise ValueError('Calendar verification needs renewal for the current date')
    cache = Path(cfg['store'])/'calendar'/(evidence['valid_from']+'_'+evidence['valid_through'])
    calendar, _ = normalized(cache/'normalized/calendar')
    if not read(cache/'normalized/calendar/normalized.json')['coverage']['calendar_range_complete']:
        raise ValueError('Preflight needs a complete cached calendar')
    closed = set()
    for start, end in evidence.get('closed_ranges', []):
        closed.update(pd.date_range(start, end).strftime('%Y-%m-%d'))
    expected = pd.to_datetime(calendar.date).dt.weekday.lt(5) & ~calendar.date.isin(closed)
    if not expected.equals(calendar.is_open.astype(bool)):
        raise ValueError('Cached calendar differs from verified exchange notices')
    signal = signal_date or select_signal(calendar, now, cfg['run_after_shanghai'])
    if signal is None:
        signal = calendar.loc[calendar.is_open.astype(bool), 'date'].min()
    result = launch_plan(calendar, signal, cfg['store'], now, cfg['run_after_shanghai'])
    candidates = calendar.loc[calendar.is_open.astype(bool) & calendar.date.ge(now.strftime('%Y-%m-%d')), 'date']
    if len(candidates):
        next_signal = candidates.iloc[0]
        if next_signal != signal:
            result['next_window'] = launch_plan(calendar, next_signal, cfg['store'], now, cfg['run_after_shanghai'])
    return result


def _run(config, signal_date, journal, research_protocol=None):
    cfg = read(config)
    store = Path(cfg['store'])
    store.mkdir(parents=True, exist_ok=True)
    research_protocol = research_protocol or read(BASE/'configs/research-review-v1.json')
    with (store / 'cycle.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='already_running')
        torch.set_num_threads(4)
        evidence = read(cfg['calendar_evidence'])
        now = pd.Timestamp(utc_now()).tz_convert('Asia/Shanghai')
        if not evidence['valid_from'] <= now.strftime('%Y-%m-%d') <= evidence['valid_through']:
            raise ValueError('Calendar verification needs renewal for the current date')
        cache = store / 'calendar' / (evidence['valid_from']+'_'+evidence['valid_through'])
        with journal.stage('calendar_capture'):
            cal_path = capture(Path(cfg['rust_repo']), Path(cfg['env_file']), cache, 'trade_cal', 'calendar',
                dict(exchange='SSE', start_date=evidence['valid_from'].replace('-', ''), end_date=evidence['valid_through'].replace('-', '')))
        calendar, _ = normalized(cal_path)
        if 'closed_ranges' in evidence:
            closed = set()
            for start, end in evidence['closed_ranges']:
                closed.update(pd.date_range(start, end).strftime('%Y-%m-%d'))
            expected = pd.to_datetime(calendar.date).dt.weekday.lt(5) & ~calendar.date.isin(closed)
            if not expected.equals(calendar.is_open.astype(bool)):
                raise ValueError('Provider calendar differs from verified exchange notices')
        signal = select_signal(calendar, now, cfg['run_after_shanghai'], signal_date)
        if signal is None:
            return dict(status='before_first_close')
        journal.set_signal(signal)
        if (store / f'runs/{signal}/manifest.json').exists():
            verify(store / f'runs/{signal}')
            source = read(store / f'runs/{signal}/binding.json')['source']
            # A ranking-only revision reuses completed forecasts without reapplying training.
            frozen_cfg = read(store / f'runs/{signal}/binding.json')['config']
            ignored = {'ranking', 'ranking_method', 'start_stage_filter'}
            if {k: v for k, v in cfg.items() if k not in ignored} != {k: v for k, v in frozen_cfg.items() if k not in ignored}:
                raise ValueError('Completed close differs beyond the ranking configuration')
            cycle(source, store, frozen_cfg)
            with journal.stage('publish_reference_ranking'):
                report = publish(store / f'runs/{signal}', cfg.get('ranking_method'))
            with journal.stage('publish_start_stage'):
                start_stage_report = (publish_start_stage(store / f'runs/{signal}', cfg['start_stage_filter'])
                                      if cfg.get('start_stage_filter') else None)
            freeze_published(store / f'runs/{signal}', research_protocol, journal)
            with journal.stage('mature_outcome_review'):
                review(store, source)
                review_early_stage(store, source)
            write_json(store / 'last-status.json', dict(status='already_completed', signal=signal, at=utc_now(), run=str(store / f'runs/{signal}')))
            result = dict(status='already_completed', signal=signal, run=str(store / f'runs/{signal}'), report=str(report))
            if start_stage_report:
                result['start_stage_report'] = str(start_stage_report)
            return result
        days = sorted(calendar.loc[calendar.is_open, 'date'])
        offset = days.index(signal)
        future = days[offset+1:offset+6]
        if len(future) < 5:
            raise ValueError('Calendar evidence does not cover five future trading sessions; renew it')
        deadline = pd.Timestamp(future[0]+'T09:15:00', tz='Asia/Shanghai')
        journal.value['publication_deadline'] = deadline.isoformat()
        journal.flush()
        budget = launch_plan(calendar, signal, store, utc_now(), cfg['run_after_shanghai'])
        journal.value['launch_plan'] = budget
        journal.flush()
        if pd.Timestamp(utc_now()) >= deadline and signal_date is None:
            result = dict(status='missed_publication_deadline', signal=signal, at=utc_now(),
                deadline=deadline.isoformat(),
                message='No new training started. An explicit --signal-date permits a labeled retrospective run.')
            write_json(store / 'last-status.json', result)
            return result
        if budget['status'] == 'insufficient_measured_budget' and signal_date is None:
            result = dict(status='insufficient_measured_budget', signal=signal, at=utc_now(),
                          launch_plan=budget, message='No new training started; measured duration exceeds the remaining publication budget.')
            write_json(store/'last-status.json', result)
            return result
        current = read(store / 'current.json') if (store / 'current.json').exists() else None
        base = Path(current['source'] if current else cfg['initial_source'])
        anchor = read(base / 'panel-h5/manifest.json')['price_data_through']
        workspace = store / 'data' / signal
        source = workspace / 'input'
        if not source.exists():
            # Reuse the already verified calendar; a second provider request can hit its quota.
            cached_calendar = workspace / 'normalized/calendar'
            if not cached_calendar.exists():
                shutil.copytree(cal_path, cached_calendar)
                write_json(workspace / 'calendar-reuse.json', dict(source=str(cal_path), reused_at=utc_now()))
            with journal.stage('market_data_capture'):
                capture(Path(cfg['rust_repo']), Path(cfg['env_file']), workspace, 'stock_basic', 'current-master', dict(list_status='L'))
                for date in calendar.loc[calendar.is_open & calendar.date.ge(anchor) & calendar.date.le(signal), 'date']:
                    capture(Path(cfg['rust_repo']), Path(cfg['env_file']), workspace, 'daily', f'daily-{date.replace("-", "")}-reference',
                        dict(trade_date=date.replace('-', ''), limit=6000))
            with journal.stage('data_validation'):
                refresh(base, workspace / 'normalized', source, future[0], Path(cfg['calendar_evidence']))
        else:
            verify(source / 'snapshot')
            verify(source / 'panel-h5')
        result = cycle(source, store, cfg, journal=journal)
        with journal.stage('publish_reference_ranking'):
            report = publish(result, cfg.get('ranking_method'))
        with journal.stage('publish_start_stage'):
            start_stage_report = (publish_start_stage(result, cfg['start_stage_filter'])
                                  if cfg.get('start_stage_filter') else None)
        freeze_published(result, research_protocol, journal)
        with journal.stage('mature_outcome_review'):
            review(store, source)
            review_early_stage(store, source)
        write_json(store / 'last-status.json', dict(status='completed', signal=signal, at=utc_now(), run=str(result)))
        response = dict(status='completed', signal=signal, run=str(result), report=str(report))
        if start_stage_report:
            response['start_stage_report'] = str(start_stage_report)
        return response


def run(config, signal_date=None, research_protocol=None):
    journal = OperationJournal(read(config)['store'])
    try:
        result = _run(config, signal_date, journal, research_protocol)
    except BaseException:
        journal.finish('failed')
        raise
    journal.finish(result['status'])
    result['operation_journal'] = str(journal.path)
    return result


def daily_retention_sources(result, config):
    run_root = Path(result['run']).resolve()
    binding = read(run_root / 'binding.json')
    cfg = read(config)
    source_workspace = Path(binding['source']).resolve().parent
    paths = [run_root, source_workspace, Path(config).resolve(),
             BASE / 'configs/r2-artifact-retention-v1.json', BASE / 'src',
             BASE / 'scripts/daily_token_cycle.py', BASE / 'scripts/archive_research_artifact.py']
    for key in ['calendar_evidence', 'initial_profiles']:
        if cfg.get(key):
            paths.append(Path(cfg[key]).resolve())
    decoder = binding.get('decoder')
    if decoder and decoder.get('path'):
        paths.append(Path(decoder['path']).resolve())
    for name, value in result.items():
        if name.endswith('report') or name == 'report':
            report = Path(value).resolve()
            paths.append(report.parent if report.is_file() else report)
    for value in result.get('research_evidence', []):
        paths.append(Path(value))
    for name in ('research_summary', 'operation_journal'):
        if result.get(name):
            paths.append(Path(result[name]))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def retain_daily(result, config, retention_config, archive_cli, cloud_config):
    from archive_research_artifact import archive

    run_root = Path(result['run']).resolve()
    signal = result['signal']
    code_files = sorted((BASE/'src').rglob('*.py')) + [Path(config),
        BASE/'scripts/daily_token_cycle.py', BASE/'scripts/archive_research_artifact.py']
    code_identity = hashlib.sha256(b''.join(path.read_bytes() for path in code_files)).digest()
    identity = hashlib.sha256((run_root / 'manifest.json').read_bytes()+code_identity).hexdigest()[:12]
    archive_id = f'daily-token-{signal}-{identity}'
    output = BASE / 'artifacts/r2-retention' / archive_id
    core_result = {key: value for key, value in result.items()
                   if key not in {'research_evidence', 'research_summary', 'operation_journal'}}
    core = archive(Path(retention_config), archive_id,
                   daily_retention_sources(core_result, config), output,
                   Path(archive_cli), Path(cloud_config))
    compact = [Path(path) for path in result.get('research_evidence', [])]
    compact += [Path(result[key]) for key in ('research_summary', 'operation_journal') if result.get(key)]
    if not compact:
        return core
    files = sorted({file for path in compact
                    for file in (path.rglob('*') if path.is_dir() else [path]) if file.is_file()})
    compact_id = hashlib.sha256(b''.join(file.read_bytes() for file in files)).hexdigest()[:16]
    compact_archive_id = f'daily-research-{signal}-{compact_id}'
    evidence = archive(Path(retention_config), compact_archive_id, compact,
                       BASE/'artifacts/r2-retention'/compact_archive_id,
                       Path(archive_cli), Path(cloud_config))
    receipt = run_root.parent.parent/'retention-publications'/f'{signal}-{compact_id}.json'
    write_json(receipt, dict(core=str(core), compact_evidence=str(evidence), at=utc_now()))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/daily-token-v1.json'))
    parser.add_argument('--signal-date', help='Manually select an already-completed trading close (YYYY-MM-DD); late reports remain labeled late')
    parser.add_argument('--shadow-config', type=Path,
        help='Also record the frozen research candidate beside this completed daily run')
    parser.add_argument('--matched-config', type=Path,
        help='Continue the matched baseline/candidate chain after the main daily cycle')
    parser.add_argument('--calibration-config', type=Path,
        help='Also publish the learned-return candidate after the matched daily update')
    parser.add_argument('--early-stage-validation', type=Path,
        help='Also publish the independently validated supplemental early-stage watchlist')
    parser.add_argument('--r2-retain', action='store_true',
        help='Archive all important daily inputs and outputs to R2 and verify a full restore')
    parser.add_argument('--research-review-config', type=Path,
        default=BASE/'configs/research-review-v1.json')
    parser.add_argument('--research-export', type=Path,
        default=BASE.parent/'stock-indicator-dashboard/public/research/latest.json')
    parser.add_argument('--event-overlay', type=Path, action='append', default=[],
        help='Include an explicitly frozen event-overlay bundle in unified research review')
    parser.add_argument('--execution-evidence', type=Path,
        help='Import verified canonical execution tables and archived coverage proofs')
    parser.add_argument('--preflight', action='store_true',
        help='Inspect cached calendar and measured publication budget without requests or training')
    parser.add_argument('--r2-config', type=Path,
        default=BASE/'configs/r2-artifact-retention-v1.json')
    parser.add_argument('--r2-cli', type=Path,
        default=Path('/Users/guo/github/mootdx-cf/target/debug/mootdx-cf-rs'))
    parser.add_argument('--r2-cloud-config', type=Path,
        default=Path('/Users/guo/github/mootdx-cf/workers/research/operator.env'))
    args = parser.parse_args()
    if args.calibration_config and not args.matched_config:
        parser.error('--calibration-config requires --matched-config')
    if args.preflight:
        print(preflight(args.config, args.signal_date), flush=True)
        raise SystemExit(0)
    try:
        result = run(args.config, args.signal_date, read(args.research_review_config))
        if args.shadow_config and result.get('run'):
            from token_ranking_shadow import run_shadow
            result['shadow_report'] = str(run_shadow(Path(result['run']), args.shadow_config))
        if args.matched_config and result.get('run'):
            from token_ranking_daily import run_daily_comparison
            result['matched_report'] = str(run_daily_comparison(args.matched_config, Path(result['run'])))
            if args.calibration_config:
                from token_return_calibration_daily import publish as publish_calibrated
                result['calibrated_report'] = str(publish_calibrated(
                    Path(result['matched_report']).parent, args.calibration_config))
        if args.early_stage_validation and result.get('run'):
            from publish_daily_early_stage import publish as publish_early_stage
            result['early_stage_report'] = str(publish_early_stage(
                Path(result['run']), args.early_stage_validation))
        if result.get('run'):
            source = read(Path(result['run'])/'binding.json')['source']
            research = refresh_research(read(args.config)['store'], source,
                read(args.research_review_config), args.event_overlay, args.research_export,
                args.execution_evidence, analysis_only=not args.execution_evidence)
            result['research_summary'] = research['summary']
            result['research_evidence'] = research['evidence']
        elif result['status'] in {'missed_publication_deadline', 'insufficient_measured_budget'}:
            store = Path(read(args.config)['store'])
            if (store/'current.json').exists():
                research = refresh_research(store, read(store/'current.json')['source'],
                    read(args.research_review_config), args.event_overlay, args.research_export,
                    args.execution_evidence, analysis_only=not args.execution_evidence)
                result['research_summary'] = research['summary']
        if args.r2_retain and result.get('run'):
            result['r2_retention'] = str(retain_daily(
                result, args.config, args.r2_config, args.r2_cli, args.r2_cloud_config))
        print(result, flush=True)
    except Exception as error:
        cfg = read(args.config)
        write_json(Path(cfg['store']) / 'last-error.json', dict(at=utc_now(), error_type=type(error).__name__, message=str(error)))
        raise
