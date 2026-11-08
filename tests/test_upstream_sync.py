"""Keep upstream merges separate from installed Home Assistant code."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/sync_upstream.py"
SPEC = importlib.util.spec_from_file_location("sync_upstream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def commit(path, name, contents):
    (path / name).write_text(contents)
    git(path, "add", name)
    git(path, "commit", "-qm", "Change fixture")
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def repositories(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    git(upstream, "config", "user.name", "Fixture Author")
    git(upstream, "config", "user.email", "fixture@example.invalid")
    commit(upstream, "appliance.txt", "base\n")
    fork = tmp_path / "fork"
    subprocess.run(["git", "clone", str(upstream), str(fork)], check=True, capture_output=True)
    git(fork, "config", "user.name", "Fixture Author")
    git(fork, "config", "user.email", "fixture@example.invalid")
    commit(fork, "frame.txt", "working TV support\n")
    return upstream, fork


def test_no_update_does_not_create_branch_or_commit(repositories):
    upstream, fork = repositories
    before = git(fork, "rev-parse", "HEAD")

    assert sync.prepare(fork, str(upstream), "codex/update-fixture", {}) is None

    assert git(fork, "rev-parse", "HEAD") == before
    assert git(fork, "branch", "--show-current") == "main"


def test_upstream_merge_retains_custom_frame_code(repositories):
    upstream, fork = repositories
    upstream_sha = commit(upstream, "appliance.txt", "new appliance fix\n")
    base_sha = git(fork, "rev-parse", "HEAD")

    candidate = sync.prepare(fork, str(upstream), "codex/update-fixture", {})

    assert candidate["base"] == base_sha
    assert candidate["upstream"] == upstream_sha
    assert candidate["head"] == git(fork, "rev-parse", "HEAD")
    assert (fork / "frame.txt").read_text() == "working TV support\n"
    assert (fork / "appliance.txt").read_text() == "new appliance fix\n"
    assert git(fork, "show", "-s", "--format=%an <%ae>|%cn <%ce>") == (
        f"{sync.AUTHOR_NAME} <{sync.AUTHOR_EMAIL}>|{sync.AUTHOR_NAME} <{sync.AUTHOR_EMAIL}>"
    )


def test_conflict_stops_before_a_merge_commit(repositories):
    upstream, fork = repositories
    before = commit(fork, "appliance.txt", "fork setting\n")
    commit(upstream, "appliance.txt", "conflicting upstream setting\n")

    with pytest.raises(subprocess.CalledProcessError):
        sync.prepare(fork, str(upstream), "codex/update-fixture", {})

    assert git(fork, "rev-parse", "HEAD") == before
    assert (fork / "frame.txt").read_text() == "working TV support\n"


def protection():
    return [
        {
            "type": "required_status_checks",
            "parameters": {
                "strict_required_status_checks_policy": True,
                "required_status_checks": [
                    {"context": name, "integration_id": sync.ACTIONS_APP_ID}
                    for name in sync.REQUIRED_CHECKS
                ],
            },
        },
        {"type": "non_fast_forward"},
        {"type": "deletion"},
    ]


class GitHub:
    def __init__(self):
        self.calls = []
        self.base = "base-sha"
        self.head = "candidate-sha"
        self.policy = protection()
        self.run = {
            "id": 42,
            "workflow_id": 7,
            "head_sha": self.head,
            "head_branch": "codex/update-fixture",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://example.invalid/actions/runs/42",
        }
        self.jobs = [{"name": name, "conclusion": "success"} for name in sync.REQUIRED_CHECKS]

    def api(self, path, method="GET", data=None):
        self.calls.append((path, method, data))
        if path.endswith("/actions/workflows/validate.yml"):
            return {"id": 7, "path": ".github/workflows/validate.yml", "state": "active"}
        if path.endswith("/dispatches"):
            return None
        if "/actions/workflows/7/runs?" in path:
            return {"workflow_runs": [self.run]}
        if path.endswith("/actions/runs/42/jobs?per_page=100"):
            return {"jobs": self.jobs, "total_count": len(self.jobs)}
        if path.endswith("/rules/branches/main?per_page=100"):
            return self.policy
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": self.base}}
        if path.endswith("/git/ref/heads/codex/update-fixture"):
            return {"object": {"sha": self.head}}
        if path.endswith("/pulls"):
            return {"number": 9, "html_url": "https://example.invalid/pull/9"}
        if path.endswith("/pulls/9/merge"):
            return {"merged": True}
        raise AssertionError(path)


@pytest.fixture
def candidate():
    return {
        "base": "base-sha",
        "head": "candidate-sha",
        "branch": "codex/update-fixture",
        "upstream": "upstream-sha",
    }


def test_validation_is_dispatched_on_exact_candidate_branch(candidate):
    github = GitHub()

    result = sync.validate(github, candidate)

    assert result["id"] == 42
    dispatch = [call for call in github.calls if call[1] == "POST"]
    assert dispatch == [
        (
            f"{sync.API}/actions/workflows/7/dispatches",
            "POST",
            {"ref": candidate["branch"]},
        )
    ]


@pytest.mark.parametrize("conclusion", ["failure", "skipped", "cancelled", None])
def test_unsuccessful_or_skipped_job_cannot_pass_validation(candidate, conclusion):
    github = GitHub()
    github.jobs[0]["conclusion"] = conclusion

    with pytest.raises(RuntimeError, match="mandatory checks"):
        sync.validate(github, candidate)

    assert not any(call[0].endswith("/pulls") for call in github.calls)


def test_missing_mandatory_job_cannot_pass_validation(candidate):
    github = GitHub()
    github.jobs.pop()

    with pytest.raises(RuntimeError, match="mandatory checks"):
        sync.validate(github, candidate)


@pytest.mark.parametrize("field", ["head_sha", "head_branch", "event", "workflow_id"])
def test_unrelated_successful_run_is_not_accepted(candidate, field, monkeypatch):
    github = GitHub()
    github.run[field] = "wrong"
    clock = iter([0, 0, 2])
    monkeypatch.setattr(
        sync, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None)
    )

    with pytest.raises(RuntimeError, match="did not finish"):
        sync.validate(github, candidate, timeout=1)

    assert any("/actions/workflows/7/runs?" in call[0] for call in github.calls)


def test_failed_workflow_is_not_accepted_even_with_successful_jobs(candidate):
    github = GitHub()
    github.run["conclusion"] = "failure"

    with pytest.raises(RuntimeError, match="validation failed"):
        sync.validate(github, candidate)


@pytest.mark.parametrize("field", ["base", "head"])
def test_changed_revision_stops_before_pr_creation(candidate, field):
    github = GitHub()
    setattr(github, field, "changed-sha")

    with pytest.raises(RuntimeError, match="changed"):
        sync.promote(github, candidate)

    assert not any(call[1] in {"POST", "PUT"} for call in github.calls)


@pytest.mark.parametrize(
    "defect",
    ["not_strict", "missing_check", "unbound_app", "force_push", "deletion", "pagination"],
)
def test_missing_protection_stops_before_pr_creation(candidate, defect):
    github = GitHub()
    if defect == "not_strict":
        github.policy[0]["parameters"]["strict_required_status_checks_policy"] = False
    elif defect == "missing_check":
        github.policy[0]["parameters"]["required_status_checks"].pop()
    elif defect == "unbound_app":
        github.policy[0]["parameters"]["required_status_checks"][0]["integration_id"] = -1
    elif defect == "force_push":
        github.policy.pop(1)
    elif defect == "deletion":
        github.policy.pop(2)
    else:
        github.policy.extend({"type": "unrelated"} for _ in range(97))

    with pytest.raises(RuntimeError, match="branch rules"):
        sync.promote(github, candidate)

    assert not any(call[1] in {"POST", "PUT"} for call in github.calls)


def test_checked_candidate_is_promoted_through_pr_with_expected_head(candidate):
    github = GitHub()

    assert sync.promote(github, candidate) == "https://example.invalid/pull/9"

    mutations = [call for call in github.calls if call[1] in {"POST", "PUT"}]
    assert mutations[0][2]["base"] == "main"
    assert mutations[0][2]["head"] == candidate["branch"]
    assert mutations[1] == (
        f"{sync.API}/pulls/9/merge",
        "PUT",
        {"sha": candidate["head"], "merge_method": "merge"},
    )


def test_changed_base_after_pr_creation_stops_before_merge(candidate):
    class ChangedBase(GitHub):
        def api(self, path, method="GET", data=None):
            result = super().api(path, method, data)
            if path.endswith("/pulls") and method == "POST":
                self.base = "new-main-sha"
            return result

    github = ChangedBase()

    with pytest.raises(RuntimeError, match="changed"):
        sync.promote(github, candidate)

    assert not any(call[1] == "PUT" for call in github.calls)


@pytest.fixture
def run_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_REPOSITORY", sync.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    return summary


def read_summary(path):
    return json.loads(path.read_text().split("```json\n", 1)[1].split("\n```", 1)[0])


def test_add_add_conflict_reports_revisions_and_stops_before_push(
    repositories, monkeypatch, run_environment, capsys
):
    upstream, fork = repositories
    base = commit(fork, "media_player.py", "Frame TV player\n")
    upstream_sha = commit(upstream, "media_player.py", "Soundbar player\n")
    monkeypatch.chdir(fork)
    monkeypatch.setattr(sync, "UPSTREAM", str(upstream))
    monkeypatch.setattr(
        sync, "GitHub", lambda: pytest.fail("Must not call GitHub after a conflict")
    )

    with pytest.raises(subprocess.CalledProcessError):
        sync.main()

    report = read_summary(run_environment)
    assert report["result"] == "failed"
    assert report["stage"] == "merge"
    assert report["error_kind"] == "merge_conflict"
    assert report["conflicts"] == ["media_player.py"]
    assert report["base"] == report["head"] == base
    assert report["upstream"] == upstream_sha
    assert report["branch"] == "codex/upstream-sync-123-1"
    assert "not pushed" in report["next_action"]
    assert git(fork, "rev-parse", "HEAD") == base
    assert not git(upstream, "for-each-ref", "refs/heads/codex/")
    assert "CONFLICT (add/add)" in report["diagnostics"]
    assert "CONFLICT (add/add)" in capsys.readouterr().err


def test_fetch_failure_is_not_reported_as_a_conflict(
    repositories, monkeypatch, run_environment, capsys, tmp_path
):
    _, fork = repositories
    base = git(fork, "rev-parse", "HEAD")
    monkeypatch.chdir(fork)
    monkeypatch.setattr(sync, "UPSTREAM", str(tmp_path / "missing-upstream"))
    monkeypatch.setattr(
        sync, "GitHub", lambda: pytest.fail("Must not call GitHub after fetch fails")
    )

    with pytest.raises(subprocess.CalledProcessError):
        sync.main()

    report = read_summary(run_environment)
    assert report["stage"] == "fetch"
    assert report["error_kind"] == "command_failed"
    assert report["conflicts"] == []
    assert report["base"] == report["head"] == base
    assert "upstream" not in report
    assert git(fork, "branch", "--show-current") == "main"
    assert "does not appear to be a git repository" in capsys.readouterr().err


def test_merge_command_failure_without_unmerged_files_is_not_a_conflict(
    repositories, monkeypatch, run_environment
):
    upstream, fork = repositories
    commit(upstream, "appliance.txt", "new appliance fix\n")
    monkeypatch.chdir(fork)
    monkeypatch.setattr(sync, "UPSTREAM", str(upstream))
    original_git = sync.git

    def fail_merge(root, *args, **kwargs):
        if args[0] == "merge":
            raise subprocess.CalledProcessError(128, ["git", "merge"], stderr="fatal: disk error")
        return original_git(root, *args, **kwargs)

    monkeypatch.setattr(sync, "git", fail_merge)
    with pytest.raises(subprocess.CalledProcessError):
        sync.main()

    report = read_summary(run_environment)
    assert report["stage"] == "merge"
    assert report["error_kind"] == "command_failed"
    assert report["conflicts"] == []


def test_merge_base_failure_preserves_text_diagnostics(repositories, monkeypatch, run_environment):
    upstream, fork = repositories
    commit(upstream, "appliance.txt", "new appliance fix\n")
    monkeypatch.chdir(fork)
    monkeypatch.setattr(sync, "UPSTREAM", str(upstream))
    original_run = sync.subprocess.run

    def fail_merge_base(args, **kwargs):
        if "merge-base" in args:
            assert kwargs["text"] is True
            return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: bad object\n")
        return original_run(args, **kwargs)

    monkeypatch.setattr(sync.subprocess, "run", fail_merge_base)
    with pytest.raises(subprocess.CalledProcessError):
        sync.main()

    report = read_summary(run_environment)
    assert report["error_kind"] == "command_failed"
    assert report["diagnostics"] == "fatal: bad object\n"
    assert report["conflicts"] == []


def test_up_to_date_summary_includes_both_revisions(repositories, monkeypatch, run_environment):
    upstream, fork = repositories
    monkeypatch.chdir(fork)
    monkeypatch.setattr(sync, "UPSTREAM", str(upstream))
    monkeypatch.setattr(sync, "GitHub", lambda: pytest.fail("Up-to-date sync must not call GitHub"))

    sync.main()

    report = read_summary(run_environment)
    assert report["result"] == "up_to_date"
    assert report["stage"] == "complete"
    assert report["base"] == report["head"] == git(fork, "rev-parse", "HEAD")
    assert report["upstream"] == git(upstream, "rev-parse", "HEAD")
    assert report["conflicts"] == []


def test_command_diagnostics_redact_github_tokens(repositories, monkeypatch, capsys):
    _, fork = repositories
    monkeypatch.setenv("GH_TOKEN", "synthetic-gh-token")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-github-token")
    error = subprocess.CalledProcessError(
        128,
        ["git", "fetch"],
        output="remote: synthetic-gh-token\n",
        stderr="fatal: synthetic-github-token\n",
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(sync.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        sync.git(fork, "fetch", "upstream", "main")

    diagnostics = capsys.readouterr().err
    assert "Git fetch failed (exit 128)" in diagnostics
    assert "remote: ***" in diagnostics
    assert "fatal: ***" in diagnostics
    assert "synthetic-" not in diagnostics


@pytest.mark.parametrize("stage", ["push", "validation", "promotion"])
def test_failed_candidate_stage_reports_failure_and_does_not_continue(
    candidate, monkeypatch, run_environment, stage
):
    calls = []
    github = GitHub()

    def prepare(*args):
        args[-1].update(candidate)
        return candidate

    def push(*args):
        calls.append("push")
        if stage == "push":
            raise subprocess.CalledProcessError(128, ["git", "push"])

    monkeypatch.setattr(sync, "prepare", prepare)
    monkeypatch.setattr(sync, "git", push)
    monkeypatch.setattr(sync, "GitHub", lambda: github)
    if stage == "validation":
        github.run["conclusion"] = "failure"
    if stage == "promotion":
        github.base = "changed-main-sha"

    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        sync.main()

    report = read_summary(run_environment)
    assert report["result"] == "failed"
    assert report["stage"] == stage
    assert report["base"] == candidate["base"]
    assert report["upstream"] == candidate["upstream"]
    assert report["head"] == candidate["head"]
    assert report["next_action"]
    assert calls == ["push"]
    assert not any(call[0].endswith("/pulls") or call[1] == "PUT" for call in github.calls)
    if stage == "push":
        assert not github.calls
    if stage == "promotion":
        assert report["validation_run"] == github.run["html_url"]


def test_successful_sync_summary_identifies_validated_commit(
    candidate, monkeypatch, run_environment
):
    github = GitHub()

    def prepare(*args):
        args[-1].update(candidate)
        return candidate

    monkeypatch.setattr(sync, "prepare", prepare)
    monkeypatch.setattr(sync, "git", lambda *_: None)
    monkeypatch.setattr(sync, "GitHub", lambda: github)

    sync.main()

    report = read_summary(run_environment)
    assert report["result"] == "merged"
    assert report["stage"] == "complete"
    assert report["head"] == candidate["head"]
    assert report["validation_run"] == github.run["html_url"]
    assert report["pr"] == "https://example.invalid/pull/9"


def test_github_failure_diagnostics_and_summary_redact_token(
    candidate, monkeypatch, run_environment, capsys
):
    monkeypatch.setenv("GH_TOKEN", "synthetic-token")

    def prepare(*args):
        args[-1].update(candidate)
        return candidate

    class DeniedGitHub:
        def api(self, *args):
            raise subprocess.CalledProcessError(
                1, ["gh", "api"], output="", stderr="HTTP 403: synthetic-token denied"
            )

    monkeypatch.setattr(sync, "prepare", prepare)
    monkeypatch.setattr(sync, "git", lambda *_: None)
    monkeypatch.setattr(sync, "GitHub", DeniedGitHub)

    with pytest.raises(subprocess.CalledProcessError):
        sync.main()

    report = read_summary(run_environment)
    assert report["stage"] == "validation"
    assert report["error_kind"] == "command_failed"
    assert report["diagnostics"] == "HTTP 403: *** denied"
    output = capsys.readouterr()
    assert "HTTP 403: *** denied" in output.err
    assert "synthetic-token" not in output.err + output.out + run_environment.read_text()


def test_context_rejection_still_writes_failure_summary(monkeypatch, run_environment):
    monkeypatch.setenv("GITHUB_REF", "refs/heads/untrusted")
    with pytest.raises(RuntimeError, match="maintained fork"):
        sync.main()
    report = read_summary(run_environment)
    assert report["stage"] == "context"
    assert report["result"] == "failed"


def test_scheduled_controller_uses_trusted_main_and_narrow_permissions():
    workflow = SCRIPT.parents[1] / "workflows/sync-upstream.yml"
    data = yaml.load(workflow.read_text(), Loader=yaml.BaseLoader)
    assert data["on"]["schedule"] == [{"cron": "30 8 * * 0"}]
    assert "workflow_dispatch" in data["on"]
    job = data["jobs"]["sync"]
    assert "github.repository == 'kevinbaijnath/localthings'" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert job["permissions"] == {
        "contents": "write",
        "actions": "write",
        "pull-requests": "write",
    }
    assert job["steps"][0]["with"]["ref"] == "${{ github.sha }}"


def test_existing_validation_checks_include_controller():
    workflow = SCRIPT.parents[1] / "workflows/validate.yml"
    data = yaml.load(workflow.read_text(), Loader=yaml.BaseLoader)
    commands = [step.get("run", "") for step in data["jobs"]["lint"]["steps"]]
    for command in ["ruff format --check", "ruff check", "ty check"]:
        assert f"{command} custom_components tests .github/scripts" in commands
