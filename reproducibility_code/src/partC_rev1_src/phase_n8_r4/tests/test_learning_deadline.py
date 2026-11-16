"""The parent process enforces the complete D00+D01 clock."""
import pytest
from partC_rev1_src.phase_n8.common import write_json
from partC_rev1_src.phase_n8_r4.supervise import job_deadline


def clock_fixture(root):
    c=dict(start_epoch=1000,compute_cutoff_epoch=80000,deadline_epoch=94400,run_id='test')
    start=dict(task='D0_LEARNING_PREPARE',run_id='test',job_id='D00_LEARNING_PREPARE',start_epoch=2000)
    clock=dict(stage_start_epoch=2001,stage_deadline_epoch=16401,cap_seconds=14400)
    write_json(root/'jobs/D00_LEARNING_PREPARE/START.json',start)
    write_json(root/'jobs/D00_LEARNING_PREPARE/EXIT.json',dict(**start,ended_epoch=2550,exit_code=0,supervisor_stop=None))
    write_json(root/'learning/LEARNING_CLOCK.json',clock)
    return c,clock


def test_supervisor_includes_preparation_and_launch_gap(tmp_path):
    c,_=clock_fixture(tmp_path)
    deadline,binding=job_deadline(c,tmp_path,'D1_LEARNING_STUDY',2600,14400)
    assert deadline==16400
    assert binding['complete_stage_start_epoch']==2000
    assert binding['hard_stage_deadline_epoch']==16400
    assert len(binding['clock_sha256'])==64


def test_supervisor_rejects_reset_learning_clock(tmp_path):
    c,clock=clock_fixture(tmp_path)
    clock['stage_deadline_epoch']+=500
    write_json(tmp_path/'learning/LEARNING_CLOCK.json',clock)
    with pytest.raises(ValueError,match='changed or reset'):
        job_deadline(c,tmp_path,'D1_LEARNING_STUDY',2600,14400)


def test_supervisor_rejects_failed_preparation(tmp_path):
    c,_=clock_fixture(tmp_path)
    write_json(tmp_path/'jobs/D00_LEARNING_PREPARE/EXIT.json',dict(exit_code=1,supervisor_stop=None))
    with pytest.raises(ValueError,match='usable supervised handoff'):
        job_deadline(c,tmp_path,'D1_LEARNING_STUDY',2600,14400)


def test_other_jobs_preserve_existing_deadline_policy(tmp_path):
    c=dict(compute_cutoff_epoch=5000,deadline_epoch=6000)
    assert job_deadline(c,tmp_path,'C2_PHYSICAL_CONTROL',4500,900)==(5000,None)
    assert job_deadline(c,tmp_path,'F1_BUILD_RELEASE',4500,900)==(5400,None)
