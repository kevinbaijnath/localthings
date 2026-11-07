"""Keep upstream merges separate from installed Home Assistant code."""

import importlib.util
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

    assert sync.prepare(fork, str(upstream), "codex/update-fixture") is None

    assert git(fork, "rev-parse", "HEAD") == before
    assert git(fork, "branch", "--show-current") == "main"


def test_upstream_merge_retains_custom_frame_code(repositories):
    upstream, fork = repositories
    upstream_sha = commit(upstream, "appliance.txt", "new appliance fix\n")
    base_sha = git(fork, "rev-parse", "HEAD")

    candidate = sync.prepare(fork, str(upstream), "codex/update-fixture")

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
        sync.prepare(fork, str(upstream), "codex/update-fixture")

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


def test_failed_validation_never_promotes_candidate(candidate, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("GITHUB_REPOSITORY", sync.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setattr(sync, "prepare", lambda *_: candidate)
    monkeypatch.setattr(sync, "git", lambda *args: calls.append(args))
    github = GitHub()
    github.run["conclusion"] = "failure"
    monkeypatch.setattr(sync, "GitHub", lambda: github)

    with pytest.raises(RuntimeError, match="validation failed"):
        sync.main()

    assert len(calls) == 1
    assert calls[0][1] == "push"
    assert not any(call[0].endswith("/pulls") for call in github.calls)


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
