"""Explicit, sealed out-of-sample evaluation. Default development APIs stay closed."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch

from .data import sha256_file, write_json
from .environment_data import MarketData, load_stage1_market
from .paired_analysis import paired_analysis
from .postpilot import (read_json, verify_decision_lock, verified_batch, runtime_record, seal_report)
from .ppo_training import (load_saved_policy, evaluate_policy_financial, configure_torch_runtime, _contract)
from .pilot import _run_path
from .validation import verify_file_map


def _authorized_lock(session, expected_digest, evaluator_sha, project_root, enable_test):
    if type(enable_test) is not bool or not enable_test:
        raise ValueError('Test access disabled; explicitly enable after sealing decisions')
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError('Copy the externally recorded decision_lock SHA256 before test')
    path = Path(session) / 'decision_lock.json'
    if sha256_file(path) != expected_digest:
        raise ValueError('Decision lock differs from the pre-recorded SHA256')
    return verify_decision_lock(session, evaluator_sha=evaluator_sha, project_root=project_root)


def _load_locked_test(stage1_run, lock):
    """Called only after authorization and the durable test-access marker."""
    root = Path(stage1_run)
    meta = read_json(root / 'metadata.json')
    if sha256_file(root / 'metadata.json') != lock['stage1_metadata_sha256']:
        raise ValueError('Stage 1 metadata changed after sealing')
    if (meta['config']['test_start'], meta['config']['end_exclusive']) != (
            lock['protocol']['test_start'], lock['protocol']['test_end_exclusive']):
        raise ValueError('Frozen test dates differ')
    verify_file_map(root, lock['data_files_sha256'])
    # Reuse the Stage 2 loader: it checks split coverage, causal features, the
    # training-only reference, and the unchanged synthetic adjusted-price basis.
    data = load_stage1_market(root, 'test', acknowledge_synthetic_prices=True)
    contract = lock['data_contract']
    data = MarketData(data.adjusted_open, data.adjusted_close, data.features, data.sigma_ref, data.sigma_column,
        {**data.provenance, 'snapshot_id': contract['snapshot_id'],
         'snapshot_manifest_sha256': contract['snapshot_manifest_sha256']})
    if _contract(data) != contract:
        raise ValueError('Test feature/asset/reference contract differs from training')
    if not ((data.features.index >= lock['protocol']['test_start']) &
            (data.features.index < lock['protocol']['test_end_exclusive'])).all():
        raise ValueError('Out-of-sample dates outside the frozen interval')
    return data


def evaluate_locked_test(batch, stage1_run, decision_session, output_root, *, expected_lock_sha256,
                         evaluator_sha, project_root, enable_test=False, test_evidence_directory=None):
    """Nine frozen models, no retraining, selection, or checkpoint search on test.

    Creates an immutable first-access marker before loading test. Technical
    retries keep the same seal and retain every failed/complete attempt.
    Only the first complete attempt is the primary result; repeats are diagnostic.
    """
    lock = _authorized_lock(decision_session, expected_lock_sha256, evaluator_sha, project_root, enable_test)
    from .pilot import _verified_test_evidence
    if test_evidence_directory is None:
        raise ValueError('Successful full pytest evidence is required before test access')
    test_evidence = _verified_test_evidence(test_evidence_directory, evaluator_sha)
    state, resolved = verified_batch(batch, lock['protocol'])
    if sha256_file(Path(batch) / 'batch_report.json') != lock['batch_report_sha256']:
        raise ValueError('Batch changed since pre-test seal')
    configure_torch_runtime()
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    runtime = runtime_record()
    if runtime != lock['runtime']:
        raise ValueError('Test evaluation runtime differs from the independently verified evaluator')
    ledger = Path(output_root) / expected_lock_sha256
    ledger.mkdir(parents=True, exist_ok=True)
    marker = ledger / 'test_access.json'
    try:
        with marker.open('x', encoding='utf-8') as stream:
            json.dump({'test_opened': True, 'decision_lock_sha256': expected_lock_sha256,
                'evaluator_sha': evaluator_sha, 'first_access_utc': datetime.now(timezone.utc).isoformat()}, stream)
    except FileExistsError:
        previous = read_json(marker)
        if previous['decision_lock_sha256'] != expected_lock_sha256 or previous['evaluator_sha'] != evaluator_sha:
            raise ValueError('First-access marker belongs to different frozen decisions')
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    root = ledger / 'attempts' / attempt_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'test_access_started', 'test_opened': True, 'training_performed': False,
        'decision_lock_sha256': expected_lock_sha256, 'decision_session': str(decision_session),
        'training_sha': lock['training_sha'], 'evaluator_sha': evaluator_sha, 'runtime': runtime,
        'test_evidence': test_evidence, 'protocol': lock['protocol'], 'runs': {}}
    report.update(attempt_id=attempt_id, first_access_marker_sha256=sha256_file(marker),
        primary_result_rule='first_complete_attempt_only; later repeats are reproducibility diagnostics')
    write_json(root / 'test_access.json', report)
    frames = {}
    try:
        data = _load_locked_test(stage1_run, lock)
        for job in lock['runs']:
            source = _run_path(batch, job['run_directory'])
            if (sha256_file(source / 'metadata.json') != job['metadata_sha256'] or
                    read_json(source / 'checkpoints/final/checkpoint.json') != job['final_checkpoint']):
                raise ValueError('Frozen model/normalization/configuration changed')
            target = root / job['job_id']
            target.mkdir()
            current = []
            for _ in range(2):
                model, vector, normalizer, config = load_saved_policy(source, data, evaluation_split='test')
                try:
                    current.append(evaluate_policy_financial(model, vector, normalizer, data, config, evaluation_split='test'))
                finally:
                    vector.close()
            pd.testing.assert_frame_equal(current[0][0], current[1][0], check_exact=True)
            if current[0][1] != current[1][1]:
                raise AssertionError('Two independent test loads produced different metrics')
            records, metrics = current[0]
            records.to_csv(target / 'test_financials.csv', index=False)
            write_json(target / 'test_metrics.json', metrics)
            frames[(job['mode'], job['seed'])] = records
            report['runs'][job['job_id']] = {'source_run': source.name, 'source_metadata_sha256': job['metadata_sha256'],
                'same_runtime_exact': True, 'observations': len(records), 'normalization_fit': 'train_only_frozen',
                'initial_positions': 'all_cash', 'metrics': metrics}
            write_json(root / 'test_access.json', report)
            print('Test completed:', job['job_id'], len(records), flush=True)
        tables = paired_analysis(frames, seeds=lock['protocol']['seeds'], sigma_ref=data.sigma_ref,
                                 bootstrap=lock['protocol']['bootstrap'])
        for name, table in tables.items():
            table.to_csv(root / (name + '.csv'), index=False)
        report.update(status='evaluation_complete', first_date=str(data.features.index.min().date()),
            last_date=str(data.features.index.max().date()), observations=len(data.features),
            inference_scope='time uncertainty conditional on nine validation-selected policies; three paired seeds',
            primary_comparison='adaptive_minus_fixed overall net_sharpe',
            secondary_scope='regime/turnover/baseline comparisons descriptive; no multiplicity-adjusted claims')
    except BaseException as error:
        report.update(status='failed_after_test_access_marker', error_type=type(error).__name__, error=str(error))
        seal_report(root, 'test_report.json', report)
        raise
    seal_report(root, 'test_report.json', report)
    pointer = ledger / 'first_complete.json'
    try:
        with pointer.open('x', encoding='utf-8') as stream:
            json.dump({'attempt_id': attempt_id, 'report': (root / 'test_report.json').relative_to(ledger).as_posix(),
                       'report_sha256': sha256_file(root / 'test_report.json')}, stream, indent=2)
    except FileExistsError:
        primary = read_json(pointer)
        original = (ledger / primary['report']).resolve()
        if not original.is_relative_to(ledger.resolve()) or sha256_file(original) != primary['report_sha256']:
            raise ValueError('Original primary test result changed; preserve and inspect all evidence')
    return root
