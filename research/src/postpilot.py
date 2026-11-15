"""Independent validation replay, paired analysis, and an immutable pre-test seal.

All evaluation and numerical analysis must be executed in Google Colab. No learn
call exists here. Existing source runs and the Stage 1 snapshot remain read-only.
"""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .data import sha256_file, write_json
from .paired_analysis import paired_analysis
from .pilot import read_pilot_batch, _checked_run, _run_path, _verified_test_evidence
from .pilot_plan import canonical_digest
from .ppo_training import _contract, configure_torch_runtime, torch_runtime_info
from .replay_audit import _teacher_replay, _closed_loop, _exact_tables
from .validation import verify_file_map


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_evaluator_checkout(project_root, evaluator_sha):
    root = Path(project_root)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', '.'], text=True)
    if actual != evaluator_sha or dirty.strip():
        raise ValueError('Evaluator must be the exact published, unmodified Git checkout')


def seal_report(root, name, report):
    root = Path(root)
    report['files_sha256'] = {p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {name, name[:-5] + '.sha256'}}
    write_json(root / name, report)
    (root / (name[:-5] + '.sha256')).write_text(sha256_file(root / name) + '  ' + name + '\n', encoding='utf-8')
    verify_file_map(root, report['files_sha256'])


def verified_report(root, name):
    root = Path(root)
    if (root / (name[:-5] + '.sha256')).read_text().split() != [sha256_file(root / name), name]:
        raise ValueError('Report sidecar mismatch: ' + name)
    report = read_json(root / name)
    verify_file_map(root, report['files_sha256'])
    return report


def verified_batch(root, protocol):
    root = Path(root)
    if (protocol.get('metrics') != ['net_sharpe', 'cagr', 'maximum_drawdown',
            'portfolio_turnover_mean_daily', 'total_commissions']
            or protocol.get('volatility_rule') != 'high_if_sigma_gt_train_sigma_ref_else_low'
            or protocol.get('primary_comparison') != 'adaptive_minus_fixed'
            or protocol.get('primary_metric') != 'net_sharpe'
            or protocol.get('selection_rule') != 'mean_across_all_seeds_then_candidate_id'
            or protocol.get('reproducibility_gate') != 'all_27_exact_in_independent_session'
            or protocol.get('test_initial_capital') != 1000.
            or protocol.get('test_initial_positions') != 'all_cash_no_carry_from_validation'
            or protocol.get('test_policy') != 'frozen_final_checkpoints_no_training_or_selection'
            or protocol.get('test_repeat_policy') != 'same_seal_only_preserve_all_attempts_first_complete_is_primary'
            or protocol.get('test_access_default') is not False):
        raise ValueError('Unsupported evaluation protocol; do not silently change the frozen estimands')
    state, resolved = read_pilot_batch(root)
    report = verified_report(root, 'batch_report.json')
    if (state['status'] != 'complete' or report['completed_jobs'] != 27
            or state['batch_id'] != protocol['batch_id'] or state['git_sha'] != protocol['training_sha']
            or state['plan_sha256'] != protocol['plan_sha256'] or state['test_used'] is not False
            or resolved['plan']['seeds'] != protocol['seeds']
            or resolved['plan']['stage1_run_id'] != protocol['stage1_run_id']):
        raise ValueError('Use the complete, originally audited pilot batch and protocol')
    for job in resolved['jobs']:
        entry = state['jobs'][job['job_id']]
        if entry['status'] != 'complete':
            raise ValueError('Every planned job must be complete')
        _checked_run(_run_path(root, entry['run_directory']), job['config'], state)
        if (job['config']['initial_capital'] != 1000. or job['config']['annualization'] != 252
                or job['config']['risk_free_annual'] != 0. or job['config']['commission_bps'] != 5.
                or job['config']['device'] != 'cpu'):
            raise ValueError('Evaluation requires the original financial conventions and CPU pilot')
    for evidence in state['attempts']:
        checked = _verified_test_evidence(evidence['directory'], state['git_sha'])
        if checked != {k: evidence[k] for k in checked}:
            raise ValueError('Original training test evidence changed')
    # Recompute the predeclared finite-grid choice; do not trust a hand-edited selection.
    for mode in ('fixed', 'adaptive'):
        scores = {}
        for job in resolved['jobs']:
            if job['mode'] == mode:
                run = _run_path(root, state['jobs'][job['job_id']]['run_directory'])
                value = read_json(run / 'validation_metrics.json')['net_sharpe']
                if value is None or not np.isfinite(value):
                    raise ValueError('Undefined selection metric')
                scores.setdefault(job['candidate_id'], []).append(value)
        winner = sorted(scores, key=lambda c: (-float(np.mean(scores[c])), c))[0]
        if winner != protocol['selected'][mode] or report['selection']['selected'][mode] != winner:
            raise ValueError('Selection differs from the fixed validation rule/protocol')
    return state, resolved


def runtime_record():
    return {'python': sys.version, 'versions': {n: importlib.metadata.version(n) for n in
        ('numpy', 'pandas', 'torch', 'gymnasium', 'stable-baselines3', 'pyarrow')},
        'torch_settings': torch_runtime_info(), 'torch_build': torch.__config__.show(),
        'cpu_capability': getattr(torch.backends.cpu, 'get_cpu_capability', lambda: 'unknown')(),
        'matmul_precision': torch.get_float32_matmul_precision()}


def audit_pilot_validation(batch, validation, protocol, destination, *, evaluator_sha, independent_session=False):
    if validation.provenance.get('split') != 'validation':
        raise ValueError('Independent replay may only use validation')
    if type(independent_session) is not bool or not independent_session:
        raise ValueError('Start an independent Colab runtime and explicitly attest it')
    state, resolved = verified_batch(batch, protocol)
    if _contract(validation) != state['context']['data_contract']:
        raise ValueError('Validation data contract changed')
    configure_torch_runtime()
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)  # Before the first inference in this fresh kernel.
    runtime = runtime_record()
    if runtime['python'] != state['context']['python'] or runtime['versions'] != state['context']['versions']:
        raise ValueError('Use the recorded Python/library versions for exact replay')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'training_sha': state['git_sha'], 'evaluator_sha': evaluator_sha,
        'batch_id': state['batch_id'], 'plan_sha256': state['plan_sha256'], 'runtime': runtime,
        'independent_session_attested': True, 'independent_session_machine_verified': False,
        'thresholds': 'none; all exact flags preserved separately', 'runs': {},
        'training_performed': False, 'test_used': False}
    write_json(root / 'reproducibility.json', report)
    try:
        for job in resolved['jobs']:
            source = _run_path(batch, state['jobs'][job['job_id']]['run_directory'])
            target = root / job['job_id']
            target.mkdir()
            original, teacher, predictions = _teacher_replay(source, validation)
            current, metrics = _closed_loop(source, validation)
            repeated, repeated_metrics = _closed_loop(source, validation)
            teacher.to_csv(target / 'saved_action_finances.csv', index=False)
            predictions.to_csv(target / 'predictions_on_saved_states.csv', index=False)
            current.to_csv(target / 'replayed_finances.csv', index=False)
            write_json(target / 'replayed_metrics.json', metrics)
            comparison = {'saved_action_finances_exact': _exact_tables(original, teacher)['exact'],
                'policy_actions_on_saved_states_exact': bool(predictions.action_exact.all()),
                'closed_loop_finances_exact': _exact_tables(original, current)['exact'],
                'closed_loop_metrics_exact': metrics == read_json(source / 'validation_metrics.json'),
                'same_runtime_finances_exact': _exact_tables(current, repeated)['exact'],
                'same_runtime_metrics_exact': metrics == repeated_metrics,
                'max_absolute_action_difference': float(predictions.max_absolute_action_difference.max()),
                'max_absolute_nav_difference': float(np.max(np.abs(original.portfolio_value - current.portfolio_value))),
                'source_run': source.name, 'source_metadata_sha256': sha256_file(source / 'metadata.json')}
            write_json(target / 'comparison.json', comparison)
            report['runs'][job['job_id']] = comparison
            write_json(root / 'reproducibility.json', report)
            print(job['job_id'], {k: v for k, v in comparison.items() if k.endswith('_exact')}, flush=True)
        report['all_27_exact'] = len(report['runs']) == 27 and all(
            all(v for k, v in r.items() if k.endswith('_exact')) for r in report['runs'].values())
        report['status'] = 'exact_replay_passed' if report['all_27_exact'] else 'diagnostic_complete'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), all_27_exact=False)
        seal_report(root, 'reproducibility.json', report)
        raise
    seal_report(root, 'reproducibility.json', report)
    return report


def selected_jobs(state, resolved, protocol):
    jobs = [j for j in resolved['jobs'] if protocol['selected'].get(j['mode']) == j['candidate_id']]
    if {(j['mode'], j['seed']) for j in jobs} != {(m, s) for m in ('none', 'fixed', 'adaptive') for s in protocol['seeds']}:
        raise ValueError('Selected jobs must cover all three modes and seeds')
    return jobs


def analyze_validation(batch, protocol, destination):
    state, resolved = verified_batch(batch, protocol)
    frames = {}
    for job in selected_jobs(state, resolved, protocol):
        source = _run_path(batch, state['jobs'][job['job_id']]['run_directory'])
        frames[(job['mode'], job['seed'])] = pd.read_csv(source / 'validation_financials.csv', float_precision='round_trip')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    tables = paired_analysis(frames, seeds=protocol['seeds'], sigma_ref=state['context']['data_contract']['sigma_ref'],
                             bootstrap=protocol['bootstrap'])
    for job in selected_jobs(state, resolved, protocol):
        source = _run_path(batch, state['jobs'][job['job_id']]['run_directory'])
        expected = read_json(source / 'validation_metrics.json')
        observed = tables['metrics_by_seed']
        observed = observed[(observed['mode'] == job['mode']) & (observed.seed == job['seed'])].iloc[0]
        for key, value in expected.items():
            if not (pd.isna(observed[key]) if value is None else observed[key] == value):
                raise ValueError('Recomputed validation metric differs from source: ' + job['job_id'] + '/' + key)
    for name, frame in tables.items():
        frame.to_csv(root / (name + '.csv'), index=False)
    report = {'status': 'analysis_complete', 'split': 'validation', 'training_performed': False,
        'test_used': False, 'protocol': protocol, 'protocol_sha256': canonical_digest(protocol),
        'source_batch_report_sha256': sha256_file(Path(batch) / 'batch_report.json'),
        'selected_on_same_validation': True, 'interpretation': 'exploratory; no independent superiority claim',
        'seed_uncertainty': 'paired seed differences and sample std only; no IID day-times-seed inference',
        'time_uncertainty': 'paired stationary bootstrap conditional on this fixed set of policies/seeds',
        'regime_threshold': state['context']['data_contract']['sigma_ref'],
        'regime_path_metrics': 'CAGR/drawdown undefined on nonconsecutive subsets'}
    seal_report(root, 'analysis.json', report)
    return report


def freeze_decisions(batch, stage1_run, session, protocol, *, evaluator_sha, project_root):
    """Create an exclusive, content-addressed decision record before any test read."""
    session, project_root, stage1_run = Path(session), Path(project_root), Path(stage1_run)
    if (session / 'decision_lock.json').exists():
        raise FileExistsError('Decisions already sealed; never overwrite a pre-test lock')
    state, resolved = verified_batch(batch, protocol)
    replay = verified_report(session / 'replay', 'reproducibility.json')
    analysis = verified_report(session / 'analysis', 'analysis.json')
    tests = _verified_test_evidence(session, evaluator_sha)
    if (replay.get('status') != 'exact_replay_passed' or replay.get('all_27_exact') is not True
            or replay.get('independent_session_attested') is not True or replay['evaluator_sha'] != evaluator_sha
            or replay['batch_id'] != state['batch_id'] or replay['plan_sha256'] != state['plan_sha256']
            or analysis.get('status') != 'analysis_complete' or analysis.get('split') != 'validation'
            or analysis.get('test_used') is not False or analysis['protocol_sha256'] != canonical_digest(protocol)
            or analysis['source_batch_report_sha256'] != sha256_file(Path(batch) / 'batch_report.json')):
        raise ValueError('Test remains blocked: exact independent replay and paired analysis must complete first')
    expected_ids = {j['job_id'] for j in resolved['jobs']}
    if set(replay['runs']) != expected_ids or any(
            not all(v for k, v in item.items() if k.endswith('_exact')) for item in replay['runs'].values()):
        raise ValueError('Incomplete or nonexact independent replay')
    for job in resolved['jobs']:
        source = _run_path(batch, state['jobs'][job['job_id']]['run_directory'])
        if replay['runs'][job['job_id']]['source_metadata_sha256'] != sha256_file(source / 'metadata.json'):
            raise ValueError('Models changed after independent replay')
    data = read_json(stage1_run / 'metadata.json')
    if (data['run_id'] != protocol['stage1_run_id'] or sha256_file(stage1_run / 'metadata.json') !=
            state['context']['data_contract']['stage1_metadata_sha256']):
        raise ValueError('Original Stage 1 identity mismatch')
    # Integrity checks hash compressed bytes, without opening test observations.
    data_hashes = {k: v for k, v in data['files_sha256'].items() if k.endswith('.parquet')}
    verify_file_map(stage1_run, data_hashes)
    verify_evaluator_checkout(project_root, evaluator_sha)
    verify_file_map(project_root, {p.relative_to(session / 'provenance').as_posix(): sha256_file(p)
        for p in (session / 'provenance').rglob('*') if p.is_file()})
    runs = []
    for job in selected_jobs(state, resolved, protocol):
        source = _run_path(batch, state['jobs'][job['job_id']]['run_directory'])
        runs.append({**job, 'run_directory': state['jobs'][job['job_id']]['run_directory'],
            'metadata_sha256': sha256_file(source / 'metadata.json'),
            'final_checkpoint': read_json(source / 'checkpoints/final/checkpoint.json')})
    evidence = {p.relative_to(session).as_posix(): sha256_file(p) for p in sorted(session.rglob('*'))
        if p.is_file() and p.name not in {'session.json', 'session.sha256'}}
    lock = {'schema': 1, 'status': 'sealed_before_test', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': protocol, 'protocol_sha256': canonical_digest(protocol), 'evaluator_sha': evaluator_sha,
        'training_sha': state['git_sha'], 'batch_id': state['batch_id'], 'plan_sha256': state['plan_sha256'],
        'batch_report_sha256': sha256_file(Path(batch) / 'batch_report.json'), 'data_contract': state['context']['data_contract'],
        'stage1_metadata_sha256': sha256_file(stage1_run / 'metadata.json'), 'data_files_sha256': data_hashes,
        'runs': runs, 'tests': tests, 'runtime': replay['runtime'], 'test_opened': False,
        'evidence_sha256': evidence, 'evaluation_source_sha256': {p.relative_to(project_root).as_posix(): sha256_file(p)
            for p in sorted(project_root.glob('src/*.py'))}}
    # Exclusive creation refuses a second choice after viewing test. Sidecar is
    # deliberately written last; an interrupted seal is unusable, not repaired silently.
    with (session / 'decision_lock.json').open('x', encoding='utf-8') as stream:
        json.dump(lock, stream, ensure_ascii=False, indent=2, allow_nan=False)
    (session / 'decision_lock.sha256').write_text(sha256_file(session / 'decision_lock.json') + '  decision_lock.json\n')
    return lock


def verify_decision_lock(session, *, evaluator_sha, project_root):
    session = Path(session)
    lock = read_json(session / 'decision_lock.json')
    if (session / 'decision_lock.sha256').read_text().split() != [sha256_file(session / 'decision_lock.json'), 'decision_lock.json']:
        raise ValueError('Decision lock checksum mismatch')
    if (lock.get('status') != 'sealed_before_test' or lock.get('test_opened') is not False
            or lock['evaluator_sha'] != evaluator_sha or lock['protocol_sha256'] != canonical_digest(lock['protocol'])):
        raise ValueError('Use the original sealed decisions and evaluator SHA')
    verify_file_map(session, lock['evidence_sha256'])
    verify_evaluator_checkout(project_root, evaluator_sha)
    verify_file_map(project_root, lock['evaluation_source_sha256'])
    replay = verified_report(session / 'replay', 'reproducibility.json')
    if replay.get('all_27_exact') is not True or replay.get('status') != 'exact_replay_passed':
        raise ValueError('Independent replay gate is not passed')
    return lock
