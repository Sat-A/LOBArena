import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from LOBArena.evaluate import pipeline


@pytest.fixture
def test_output_root():
    root = Path('/home/s5e/satyamaga.s5e/LOBArena/tests/.batch_eval_outputs') / f"run_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root)


def _base_args(output_root: Path, run_name: str = 'batch_eval'):
    repo_root = Path(__file__).resolve().parents[2]
    return SimpleNamespace(
        world_model='historical',
        policy_mode='random',
        fixed_action=0,
        jaxmarl_root=str(repo_root / 'JaxMARL-HFT'),
        lobs5_root=str(repo_root / 'LOBS5'),
        lobs5_ckpt_path='',
        policy_ckpt_dir='',
        policy_config='',
        policy_handoff='',
        policy_handoff_batch=None,
        policy_handoff_manifest='',
        data_dir=str(repo_root / 'LOBArena'),
        sample_index=0,
        checkpoint_step=None,
        test_split=1.0,
        start_date='',
        end_date='',
        n_cond_msgs=64,
        n_steps=5,
        sample_top_n=1,
        seed=None,
        output_root=str(output_root),
        run_name=run_name,
        fast_startup=True,
        cpu_safe=False,
        device='auto',
        strict_generative=False,
    )


def test_resolve_batch_candidates_manifest_with_fairness(test_output_root):
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'policy_handoff_valid.json'
    manifest_path = test_output_root / 'manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'fairness': {
                    'seed': 77,
                    'start_date': '2024-01-02',
                    'end_date': '2024-01-05',
                },
                'candidates': [
                    {'name': 'alpha', 'policy_handoff': str(fixture)},
                    {'name': 'beta', 'policy_handoff': str(fixture)},
                ],
            }
        )
    )

    args = _base_args(test_output_root)
    args.policy_handoff_manifest = str(manifest_path)

    candidates, fairness = pipeline.resolve_batch_candidates(args)

    assert [c.candidate_id for c in candidates] == ['alpha', 'beta']
    assert all(Path(c.policy_handoff_path).is_absolute() for c in candidates)
    assert fairness == {
        'seed': 77,
        'start_date': '2024-01-02',
        'end_date': '2024-01-05',
    }


def test_run_batch_evaluation_writes_summary_and_applies_shared_fairness(test_output_root):
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'policy_handoff_valid.json'
    args = _base_args(test_output_root, run_name='batch_smoke')
    args.policy_handoff_batch = [str(fixture), str(fixture)]

    seen = []

    def _fake_eval_runner(candidate_args):
        run_dir = Path(candidate_args.output_root) / candidate_args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        seen.append(
            {
                'run_name': candidate_args.run_name,
                'seed': candidate_args.seed,
                'start_date': candidate_args.start_date,
                'end_date': candidate_args.end_date,
                'policy_handoff': candidate_args.policy_handoff,
            }
        )
        if candidate_args.run_name.endswith('candidate_001'):
            total_pnl = 5.0
        else:
            total_pnl = 9.0
        return {
            'run_name': candidate_args.run_name,
            'run_dir': str(run_dir),
            'metrics': {'pnl': {'total_pnl': total_pnl}},
        }

    rc = pipeline.run_batch_evaluation(args, eval_runner=_fake_eval_runner)
    assert rc == 0
    assert len(seen) == 2

    batch_summary = test_output_root / 'batch_smoke' / 'batch_summary.json'
    assert batch_summary.exists()
    data = json.loads(batch_summary.read_text())

    assert data['n_candidates'] == 2
    assert data['shared_fairness']['seed'] is None
    assert data['shared_fairness']['start_date'] == ''
    assert data['shared_fairness']['end_date'] == ''

    candidates = data['candidates']
    assert candidates[0]['total_pnl'] == 9.0
    assert candidates[0]['rank_baseline'] == 1
    assert candidates[1]['total_pnl'] == 5.0
    assert candidates[1]['rank_baseline'] == 2


def test_batch_mode_conflicts_with_single_handoff(test_output_root):
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'policy_handoff_valid.json'
    args = _base_args(test_output_root)
    args.policy_handoff = str(fixture)
    args.policy_handoff_batch = [str(fixture)]

    with pytest.raises(ValueError, match='either --policy_handoff'):
        pipeline.resolve_batch_candidates(args)


def test_manifest_fairness_type_validation(test_output_root):
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'policy_handoff_valid.json'
    manifest_path = test_output_root / 'bad_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'fairness': ['not-an-object'],
                'candidates': [{'name': 'alpha', 'policy_handoff': str(fixture)}],
            }
        )
    )

    args = _base_args(test_output_root)
    args.policy_handoff_manifest = str(manifest_path)

    with pytest.raises(ValueError, match='fairness'):
        pipeline.resolve_batch_candidates(args)


def test_batch_fairness_manifest_seed_applies_when_cli_seed_missing(test_output_root):
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'policy_handoff_valid.json'
    manifest_path = test_output_root / 'manifest_seed.json'
    manifest_path.write_text(
        json.dumps(
            {
                'fairness': {'seed': 91, 'start_date': '2024-04-01', 'end_date': '2024-04-03'},
                'candidates': [{'name': 'alpha', 'policy_handoff': str(fixture)}],
            }
        )
    )

    args = _base_args(test_output_root, run_name='batch_seed_from_manifest')
    args.policy_handoff_manifest = str(manifest_path)
    seen = []

    def _fake_eval_runner(candidate_args):
        seen.append(
            {
                'seed': candidate_args.seed,
                'start_date': candidate_args.start_date,
                'end_date': candidate_args.end_date,
            }
        )
        run_dir = Path(candidate_args.output_root) / candidate_args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return {'run_name': candidate_args.run_name, 'run_dir': str(run_dir), 'metrics': {'pnl': {'total_pnl': 1.0}}}

    rc = pipeline.run_batch_evaluation(args, eval_runner=_fake_eval_runner)
    assert rc == 0
    assert seen == [{'seed': 91, 'start_date': '2024-04-01', 'end_date': '2024-04-03'}]


def test_batch_fairness_cli_overrides_manifest(test_output_root):
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'policy_handoff_valid.json'
    manifest_path = test_output_root / 'manifest_override.json'
    manifest_path.write_text(
        json.dumps(
            {
                'fairness': {'seed': 7, 'start_date': '2024-01-01', 'end_date': '2024-01-02'},
                'candidates': [{'name': 'alpha', 'policy_handoff': str(fixture)}],
            }
        )
    )

    args = _base_args(test_output_root, run_name='batch_cli_override')
    args.policy_handoff_manifest = str(manifest_path)
    args.seed = 123
    args.start_date = '2024-05-01'
    args.end_date = '2024-05-04'
    seen = []

    def _fake_eval_runner(candidate_args):
        seen.append(
            {
                'seed': candidate_args.seed,
                'start_date': candidate_args.start_date,
                'end_date': candidate_args.end_date,
            }
        )
        run_dir = Path(candidate_args.output_root) / candidate_args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return {'run_name': candidate_args.run_name, 'run_dir': str(run_dir), 'metrics': {'pnl': {'total_pnl': 2.0}}}

    rc = pipeline.run_batch_evaluation(args, eval_runner=_fake_eval_runner)
    assert rc == 0
    assert seen == [{'seed': 123, 'start_date': '2024-05-01', 'end_date': '2024-05-04'}]
